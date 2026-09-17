"""Offline evaluation of recorded episodes; never imports a simulator or controls actors."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path


VERSION = '1.0.0'
DEFAULT_CONFIG = Path(__file__).with_name('evaluator_config_v1.json')
METRICS = (
    'min_center_distance_m', 'min_clearance_m', 'closest_time_s', 'min_ttc_s',
    'min_ttc_time_s', 'min_speed_during_interaction_mps', 'max_deceleration_mps2',
    'stop_duration_s', 'braking_duration_s', 'time_to_goal_s', 'path_length_m',
    'episode_duration_s', 'first_detection_time_s', 'reaction_latency_s',
)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def mapping(value):
    return value if isinstance(value, dict) else {}


def reject_constant(value):
    raise ValueError('nonfinite JSON constant: ' + value)


def read_json(path, issues, required=False):
    if not path.exists():
        if required:
            issues.append('missing_file:' + path.name)
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8-sig'), parse_constant=reject_constant)
        if not isinstance(data, dict):
            raise ValueError('expected a JSON object')
        return data
    except (ValueError, OSError) as error:
        issues.append('unreadable_file:' + path.name + ':' + str(error))
        return {}


def read_jsonl(path, issues, required=False):
    if not path.exists():
        if required:
            issues.append('missing_file:' + path.name)
        return []
    rows = []
    try:
        with path.open(encoding='utf-8-sig') as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line, parse_constant=reject_constant)
                    if not isinstance(row, dict):
                        raise ValueError('expected a JSON object')
                    rows.append(row)
                except ValueError as error:
                    issues.append(f'invalid_jsonl:{path.name}:{number}:{error}')
    except OSError as error:
        issues.append('unreadable_file:' + path.name + ':' + str(error))
    return rows


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(config):
    for key in ('config_version', 'threshold_version'):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError('Missing config version: ' + key)
    for key in ('target_clearance_m', 'ttc_horizon_s', 'legacy_carter_radius_m',
                'legacy_person_radius_m', 'interaction_center_distance_m',
                'goal_approach_exclusion_m', 'movement_threshold_mps', 'stop_threshold_mps',
                'deceleration_threshold_mps2', 'minimum_stop_duration_s',
                'unknown_cadence_max_gap_s', 'timestamp_tolerance_s'):
        if not finite(config.get(key)) or config[key] <= 0:
            raise ValueError('Expected positive finite config: ' + key)
    if config['stop_threshold_mps'] >= config['movement_threshold_mps']:
        raise ValueError('stop threshold must be below movement threshold')


def controller(navigation):
    navigation = mapping(navigation)
    return mapping(navigation.get('controller') or navigation.get('nav2'))


def navigation_mode(rows, resolved, scenario):
    explicit = scenario.get('controller_mode') or resolved.get('controller_mode')
    if explicit:
        return explicit
    modes = {mapping(r.get('navigation')).get('controller_mode') for r in rows}
    modes.discard(None)
    if len(modes) == 1:
        return modes.pop()
    if any('nav2' in mapping(r.get('navigation')) for r in rows):
        return 'nav2'
    description = resolved.get('robot_control', '').lower()
    if 'nav2' in description:
        return 'nav2'
    if 'fixed' in description:
        return 'fixed'
    return None


def classify_outcome(rows, legacy, issues):
    controls = [(r.get('timestamp'), controller(r.get('navigation'))) for r in rows]
    final = controller(legacy.get('navigation'))
    if final:
        controls.append((None, final))
    succeeded = [t for t, c in controls if c.get('status') == 'succeeded']
    aborted = any(c.get('status') in ('aborted', 'canceled', 'cancelled') or
                  mapping(c.get('detail')).get('action_status') in (5, 6)
                  for _, c in controls)
    terminated = legacy.get('termination')
    if terminated in ('nav_abort', 'navigation_aborted'):
        aborted = True
    # Older runners wrap a recorded FollowPath abort in RuntimeError.
    error = legacy.get('error')
    wrapped_abort = aborted and isinstance(error, str) and (
        error.startswith("RuntimeError({'status': 'failed'") and 'action_status' in error)
    program_error = bool(error and not wrapped_abort) or terminated in (
        'setup_or_save_error', 'unresolvable_scenario', 'program_error')
    if terminated == 'runtime_error' and not wrapped_abort:
        program_error = True
    complete = terminated in ('both_goals_reached', 'goal_reached')
    timeout = terminated == 'timeout'
    collision_end = terminated in ('collision', 'physical_contact', 'proxy_collision')
    known = complete or timeout or aborted or program_error or collision_end or bool(succeeded)
    if not known:
        issues.append('task_terminal_state_missing')
    if complete and aborted:
        issues.append('contradictory_task_outcome')
    if program_error:
        issues.append('program_error')
    goal = True if complete or succeeded else (False if known else None)
    result = ('PROGRAM_ERROR' if program_error else 'NAV_ABORT' if aborted else
              'TIMEOUT' if timeout else 'COLLISION' if collision_end else
              'GOAL_REACHED' if goal else 'UNKNOWN')
    terminal_times = [t for t, c in controls if finite(t) and
                      (c.get('status') in ('succeeded', 'failed', 'aborted', 'canceled', 'cancelled'))]
    return {
        'navigation_result': result, 'goal_reached': goal,
        'timeout': timeout if known else None, 'nav_abort': aborted if known else None,
        'program_error': program_error if known else None,
        'recorded_termination': terminated, 'recorded_error': error,
        'collision_flags': {}, 'safety_result': 'UNAVAILABLE',
        'behavior': {'decelerated': None, 'stopped': None, 'detoured': None},
    }, min(terminal_times) if terminal_times else None, min(
        (t for t in succeeded if finite(t)), default=None)


def disc_ttc(relative, velocity, radius, horizon):
    c = sum(x*x for x in relative) - radius*radius
    if c <= 0:
        return 0.0
    a = sum(x*x for x in velocity)
    b = 2 * sum(x*y for x, y in zip(relative, velocity))
    discriminant = b*b - 4*a*c
    if a < 1e-12 or b >= 0 or discriminant < 0:
        return None
    t = (-b - math.sqrt(max(0.0, discriminant))) / (2*a)
    return t if 0 <= t <= horizon else None


def swept_distance(first, second):
    delta = [b-a for a, b in zip(first, second)]
    squared = sum(v*v for v in delta)
    fraction = max(0.0, min(1.0, -sum(a*b for a, b in zip(first, delta))/squared)) if squared else 0.0
    return math.hypot(*(a+fraction*b for a, b in zip(first, delta))), fraction


def fraction_below(first, second, threshold):
    if first <= threshold and second <= threshold:
        return 1.0
    if first > threshold and second > threshold:
        return 0.0
    crossing = (threshold-first)/(second-first)
    return crossing if first <= threshold else 1.0-crossing


def pair_statistics(rows, radius, horizon):
    closest, closest_time, ttc_min, ttc_time = None, None, None, None
    previous = None
    for row in rows:
        if not row['person']['present']:
            previous = None
            continue
        car, person, t = row['carter'], row['person'], row['timestamp']
        relative = [person[k]-car[k] for k in ('x', 'y')]
        distance, at = math.hypot(*relative), t
        if previous is not None:
            old_relative, old_time = previous
            candidate, fraction = swept_distance(old_relative, relative)
            if candidate < distance:
                distance, at = candidate, old_time + fraction*(t-old_time)
        if closest is None or distance < closest:
            closest, closest_time = distance, at
        if all(finite(actor.get(k)) for actor in (car, person) for k in ('vx', 'vy')):
            ttc = disc_ttc(relative, [person[k]-car[k] for k in ('vx', 'vy')], radius, horizon)
            if ttc is not None and (ttc_min is None or ttc < ttc_min):
                ttc_min, ttc_time = ttc, t
        previous = relative, t
    return {'min_center_distance_m': closest, 'min_clearance_m': None if closest is None else closest-radius,
            'closest_time_s': closest_time, 'min_ttc_s': ttc_min, 'min_ttc_time_s': ttc_time}


def validate_rows(rows, resolved, issues, missing, config):
    if len(rows) < 2:
        issues.append('insufficient_trajectory_samples')
    times = [r.get('timestamp') for r in rows]
    valid_time = bool(rows) and all(finite(t) and t >= 0 for t in times)
    if not valid_time or any(b <= a for a, b in zip(times, times[1:])):
        issues.append('invalid_or_nonmonotonic_simulation_time')
        valid_time = False
    if valid_time:
        cadence = resolved.get('state_dt')
        max_gap = 1.5*cadence if finite(cadence) and cadence > 0 else config['unknown_cadence_max_gap_s']
        if any(b-a > max_gap + config['timestamp_tolerance_s'] for a, b in zip(times, times[1:])):
            issues.append('trajectory_sampling_gap')
    position_ok, velocity_ok = bool(rows), bool(rows)
    for actor in ('carter', 'person'):
        chosen = rows if actor == 'carter' else [r for r in rows if mapping(r.get(actor)).get('present') is True]
        for key in ('x', 'y', 'vx', 'vy', 'heading'):
            count = sum(not finite(mapping(r.get(actor)).get(key)) for r in chosen)
            if count:
                missing[actor+'.'+key] = f'missing or nonfinite in {count} required samples'
                if key in ('x', 'y'):
                    issues.append(actor+'_position_missing')
                    position_ok = False
                elif key in ('vx', 'vy'):
                    issues.append(actor+'_velocity_missing')
                    velocity_ok = False
        if actor == 'carter' and any(not finite(mapping(r.get(actor)).get('angular_velocity')) for r in chosen):
            missing['carter.angular_velocity'] = 'not recorded for every sample'
    if any(type(mapping(r.get('person')).get('present')) is not bool for r in rows):
        issues.append('person_presence_missing')
        position_ok = False
    return valid_time, position_ok, velocity_ok


def response_statistics(rows, path, config, missing):
    empty = {k: None for k in ('min_speed_during_interaction_mps', 'max_deceleration_mps2',
                              'stop_duration_s', 'braking_duration_s')}
    goal = path[-1] if isinstance(path, list) and path else None
    if not isinstance(goal, list) or len(goal) != 2 or not all(finite(v) for v in goal):
        missing['response_window'] = 'Carter goal unavailable; cannot exclude goal braking'
        return empty, [], None
    speeds = [math.hypot(r['carter']['vx'], r['carter']['vy']) for r in rows]
    started = next((i for i, speed in enumerate(speeds) if speed > config['movement_threshold_mps']), None)
    selected = [False]*len(rows)
    if started is not None:
        for i, row in enumerate(rows):
            car, person = row['carter'], row['person']
            selected[i] = (i >= started and person['present'] and
                           math.dist([car['x'], car['y']], goal) > config['goal_approach_exclusion_m'] and
                           math.dist([car['x'], car['y']], [person['x'], person['y']]) <= config['interaction_center_distance_m'])
    if not any(selected):
        missing['response_window'] = 'no samples inside the configured active interaction window'
        return empty, [], None
    minimum = min(speed for speed, selected_i in zip(speeds, selected) if selected_i)
    max_deceleration, stopped, braking, eligible_time = 0.0, 0.0, 0.0, 0.0
    for i in range(1, len(rows)):
        if not (selected[i-1] and selected[i]):
            continue
        dt = rows[i]['timestamp'] - rows[i-1]['timestamp']
        acceleration = (speeds[i] - speeds[i-1]) / dt
        max_deceleration = max(max_deceleration, -acceleration)
        stopped += dt*fraction_below(speeds[i-1], speeds[i], config['stop_threshold_mps'])
        braking += dt if acceleration <= -config['deceleration_threshold_mps2'] else 0.0
        eligible_time += dt
    intervals = []
    for i, active in enumerate(selected):
        if active and (i == 0 or not selected[i-1]):
            intervals.append([rows[i]['timestamp'], rows[i]['timestamp']])
        if active:
            intervals[-1][1] = rows[i]['timestamp']
    if not eligible_time:
        missing['response_derivatives'] = 'no adjacent samples in the interaction window'
    return {'min_speed_during_interaction_mps': minimum,
            'max_deceleration_mps2': max_deceleration if eligible_time else None,
            'stop_duration_s': stopped if eligible_time else None,
            'braking_duration_s': braking if eligible_time else None}, intervals, eligible_time


def physical_contacts(rows):
    flags = {}
    for key in ('person_carter', 'carter_environment', 'person_environment'):
        values = [mapping(r.get('contacts')).get(key) for r in rows]
        flags[key] = (True if any(v is True for v in values) else
                      False if values and all(v is False for v in values) else None)
    return flags


def evaluate_episode(episode, config):
    validate_config(config)
    episode = Path(episode).resolve()
    issues, warnings, missing = [], [], {}
    resolved = read_json(episode/'resolved.json', issues)
    scenario = read_json(episode/'scenario.json', issues)
    legacy = read_json(episode/'metrics.json', issues)
    rows = read_jsonl(episode/'trajectory.jsonl', issues, required=True)
    observations = read_jsonl(episode/'observations.jsonl', issues)
    spec = mapping(resolved.get('spec')) or mapping(scenario.get('spec')) or scenario
    compiled = mapping(resolved.get('resolved')) or mapping(scenario.get('resolved'))
    path = compiled.get('path') or mapping(spec.get('carter')).get('route')
    if not path:
        carter_spec = mapping(spec.get('carter'))
        if carter_spec.get('start') is not None and carter_spec.get('goal') is not None:
            path = [carter_spec['start'], carter_spec['goal']]
    if not spec:
        issues.append('scenario_configuration_missing')
    outcome, terminal_time, goal_time = classify_outcome(rows, legacy, issues)
    valid_time, valid_position, valid_velocity = validate_rows(rows, resolved, issues, missing, config)
    if isinstance(legacy.get('frames'), int) and legacy['frames'] != len(rows):
        issues.append('recorded_frame_count_mismatch')
    if legacy.get('carter_on_walkable') is False:
        issues.append('carter_outside_recorded_walkable_area')
    if legacy.get('person_on_walkable') is False and any(mapping(r.get('person')).get('present') for r in rows):
        issues.append('person_outside_recorded_walkable_area')
    mode = navigation_mode(rows, resolved, scenario)
    if mode is None:
        missing['controller_mode'] = 'historical log does not identify the controller; no inference from folder names'
    motion_mode = scenario.get('motion_mode') or resolved.get('motion_mode')
    if motion_mode is None and 'hunav' in resolved.get('backend', '').lower():
        motion_mode = 'hunav'
    if motion_mode is None:
        missing['motion_mode'] = 'not recorded'
    if scenario.get('seed', spec.get('seed')) is None:
        missing['seed'] = 'not recorded; cannot reconstruct an RNG seed retrospectively'
    legacy_profile = ('HuNav' in resolved.get('backend', '') and
                      mapping(spec.get('event')).get('type') in ('crossing', 'occlusion'))
    proxy = mapping(scenario.get('proxy')) or mapping(resolved.get('proxy'))
    carter_radius = proxy.get('carter_radius_m')
    person_radius = proxy.get('person_radius_m', mapping(compiled.get('person')).get('radius'))
    if legacy_profile:
        if carter_radius is None:
            carter_radius = config['legacy_carter_radius_m']
            warnings.append('carter_radius_from_versioned_legacy_profile')
        if person_radius is None:
            person_radius = config['legacy_person_radius_m']
            warnings.append('person_radius_from_versioned_legacy_profile')
    radii_ok = all(finite(r) and r > 0 for r in (carter_radius, person_radius))
    if not radii_ok:
        missing['proxy_radii'] = 'unavailable for this input format'
        issues.append('proxy_radii_unavailable')
    raw_files = ('trajectory.jsonl', 'observations.jsonl', 'metrics.json', 'resolved.json',
                 'scenario.json', 'scenario.yaml', 'nav2_params.yaml')
    hashes = {name: sha256(episode/name) for name in raw_files if (episode/name).is_file()}
    nav_hash = hashes.get('nav2_params.yaml') or resolved.get('nav2_params_sha256')
    if hashes.get('nav2_params.yaml') and resolved.get('nav2_params_sha256') not in (None, hashes['nav2_params.yaml']):
        issues.append('nav2_configuration_hash_mismatch')
    if mode == 'nav2' and nav_hash is None:
        missing['nav2_config_version'] = 'not preserved in this historical episode'
    stale_running = sum(mapping(r.get('navigation')).get('command_stale') is True and
                        controller(r.get('navigation')).get('status') == 'running' for r in rows)
    if stale_running:
        issues.append('stale_command_during_navigation')
    if any(mapping(r.get('navigation')).get('ok') is False or
           mapping(r.get('sensor_health')).get('valid') is False for r in rows):
        issues.append('recorded_control_or_sensor_failure')
    command_count = sum(isinstance(mapping(r.get('navigation')).get('cmd_vel'), list) and
                        len(r['navigation']['cmd_vel']) == 2 and
                        all(finite(v) for v in r['navigation']['cmd_vel']) for r in rows)
    if command_count != len(rows) or not rows:
        missing['commands'] = 'complete applied cmd_vel snapshots unavailable'
    missing['command_receive_time'] = 'legacy snapshots have simulation sample time, not original ROS receive time'
    if mode == 'nav2' and not observations:
        missing['sensor_observations'] = 'not recorded'
        warnings.append('sensor_health_not_verifiable')
    if observations:
        if len(observations) != len(rows):
            issues.append('observation_frame_count_mismatch')
        offset = None
        for row, obs in zip(rows, observations):
            t, ot = row.get('timestamp'), obs.get('timestamp')
            if not (finite(t) and finite(ot)) or abs(t-ot) > config['timestamp_tolerance_s']:
                issues.append('observation_state_time_mismatch')
                break
            sensor = mapping(obs.get('observation'))
            stamp = sensor.get('time')
            if not finite(stamp):
                issues.append('sensor_simulation_time_missing')
                break
            offset = stamp-t if offset is None else offset
            if abs(stamp-t-offset) > config['timestamp_tolerance_s']:
                issues.append('sensor_simulation_clock_drift')
                break
            ranges = sensor.get('ranges')
            if not isinstance(ranges, list) or not ranges or any(
                    v is not None and (not finite(v) or v < 0) for v in ranges):
                issues.append('invalid_lidar_observation')
                break
    metrics = dict.fromkeys(METRICS)
    intervals, duration = [], None
    task_rows = rows
    if valid_time and terminal_time is not None:
        task_rows = [r for r in rows if r['timestamp'] <= terminal_time]
    geometry_ready = valid_time and valid_position and bool(task_rows)
    contacts = physical_contacts(task_rows)
    proxy_collision = None
    if valid_time:
        metrics['episode_duration_s'] = rows[-1]['timestamp']-rows[0]['timestamp']
        if goal_time is not None and outcome['goal_reached']:
            metrics['time_to_goal_s'] = goal_time-rows[0]['timestamp']
    if metrics['time_to_goal_s'] is None:
        missing['time_to_goal_s'] = 'no timestamped successful task status; episode end is not arrival time'
    if geometry_ready:
        metrics['path_length_m'] = sum(math.dist([a['carter']['x'], a['carter']['y']],
                                                [b['carter']['x'], b['carter']['y']])
                                       for a, b in zip(task_rows, task_rows[1:]))
        if radii_ok:
            metrics.update(pair_statistics(task_rows, carter_radius+person_radius, config['ttc_horizon_s']))
            gap = metrics['min_clearance_m']
            proxy_collision = None if gap is None else gap <= 0
        if valid_velocity:
            response, intervals, duration = response_statistics(task_rows, path, config, missing)
            metrics.update(response)
    if not valid_velocity:
        metrics['min_ttc_s'], metrics['min_ttc_time_s'] = None, None
        missing['min_ttc_s'] = 'incomplete velocity records'
    elif metrics['min_ttc_s'] is None:
        missing['min_ttc_s'] = 'no predicted contact within the configured horizon, or no valid pair samples'
    if metrics['min_clearance_m'] is None:
        missing['min_clearance_m'] = 'no complete active pair geometry'
    if contacts['person_environment']:
        issues.append('person_contact_with_static_environment')
    if contacts['person_carter'] is None:
        missing['physical_contact'] = 'no complete physical contact sensor record'
    missing['first_detection_time_s'] = 'reliable navigation-sensor detection event not recorded'
    missing['reaction_latency_s'] = 'detection event and causal response onset unavailable'
    missing['behavior.detoured'] = 'no versioned detour classifier in this evaluator'
    outcome['collision_flags'] = {'proxy_person_carter': proxy_collision, 'physical_contacts': contacts}
    gap = metrics['min_clearance_m']
    outcome['safety_result'] = ('PHYSICAL_CONTACT' if contacts['person_carter'] else
                                'PROXY_OVERLAP' if proxy_collision else
                                'CLEARANCE_BELOW_TARGET' if gap is not None and gap < config['target_clearance_m'] else
                                'NO_PROXY_VIOLATION' if gap is not None else 'UNAVAILABLE')
    deceleration, stop = metrics['max_deceleration_mps2'], metrics['stop_duration_s']
    outcome['behavior'].update(
        decelerated=None if deceleration is None else deceleration >= config['deceleration_threshold_mps2'],
        stopped=None if stop is None else stop >= config['minimum_stop_duration_s'])
    valid = not issues
    rho = gap-config['target_clearance_m'] if valid and gap is not None else None
    ineligible = []
    if not valid:
        ineligible.append('invalid_episode')
    if gap is None:
        ineligible.append('no_valid_pair_measurement')
    if mode != 'nav2':
        ineligible.append('not_nav2_closed_loop')
    if mode == 'nav2' and nav_hash is None:
        ineligible.append('nav2_config_version_missing')
    if mode == 'nav2' and not observations:
        ineligible.append('sensor_health_not_verifiable')
    if mode == 'nav2' and command_count != len(rows):
        ineligible.append('command_snapshots_incomplete')
    return {
        'schema_version': 1,
        'evaluator': {'version': VERSION, 'config': config, 'code_sha256': sha256(Path(__file__))},
        'scenario': {'scenario_id': scenario.get('scenario_id', spec.get('scenario_id', episode.name)),
                     'episode_id': episode.parent.name+'/'+episode.name,
                     'source': scenario.get('source', 'parameterized' if legacy_profile else None),
                     'parent_id': scenario.get('parent_id'), 'seed': scenario.get('seed', spec.get('seed')),
                     'motion_mode': motion_mode, 'controller_mode': mode,
                     'event_type': scenario.get('event_type', mapping(spec.get('event')).get('type')),
                     'carter_path': path, 'person': compiled.get('person', spec.get('person')),
                     'event_parameters': spec.get('event'),
                     'variant_parameters': scenario.get('variant_parameters'),
                     'search_round': scenario.get('search_round'), 'nav2_config_sha256': nav_hash},
        'quality': {'valid': valid, 'invalid_reasons': sorted(set(issues)),
                    'warnings': sorted(set(warnings)), 'missing_fields': missing,
                    'frames': len(rows), 'command_snapshot_frames': command_count,
                    'stale_running_frames': stale_running,
                    'observation_frames': len(observations),
                    'event_valid_legacy': legacy.get('event_valid')},
        'outcome': outcome, 'metrics': metrics,
        'search_feedback': {'objective_name': 'minimum_proxy_clearance',
                            'threshold_version': config['threshold_version'],
                            'target_clearance_m': config['target_clearance_m'], 'rho': rho,
                            'eligible': not ineligible, 'ineligible_reasons': ineligible},
        'definitions': {'time': 'episode-relative simulation seconds',
                        'proxy': {'shape': 'swept_2d_discs', 'carter_radius_m': carter_radius,
                                  'person_radius_m': person_radius},
                        'safety_window': 'person present from episode start through recorded controller termination',
                        'response_intervals_s': intervals, 'response_observed_duration_s': duration,
                        'stop_integration': 'piecewise-linear speed threshold crossings; adjacent selected samples only',
                        'path_length': 'sum of XY distances through recorded controller termination',
                        'command_time': 'simulation time of applied command snapshot; not ROS receive timestamp'},
        'provenance': {'input_directory': str(episode), 'input_sha256': hashes,
                       'recorded_runner_source_sha256': resolved.get('source_sha256'),
                       'recorded_scene_config_sha256': resolved.get('base_config_sha256')},
    }


def atomic_write(path, content):
    temp = path.with_name(path.name+'.tmp-'+str(os.getpid()))
    try:
        with temp.open('x', encoding='utf-8', newline='') as stream:
            stream.write(content)
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def summary_row(result):
    scene, quality, outcome, feedback = (result[k] for k in ('scenario', 'quality', 'outcome', 'search_feedback'))
    person = mapping(scene['person'])
    variants = mapping(scene['variant_parameters'])
    return {
        'episode_id': scene['episode_id'], 'scenario_id': scene['scenario_id'],
        'source': scene['source'], 'parent_id': scene['parent_id'], 'seed': scene['seed'],
        'motion_mode': scene['motion_mode'], 'controller_mode': scene['controller_mode'],
        'event_type': scene['event_type'], 'search_round': scene['search_round'],
        'person_speed_mps': person.get('speed'), 'spawn_time_s': person.get('spawn_time'),
        'time_shift_s': variants.get('time_shift_s'), 'speed_scale': variants.get('speed_scale'),
        'variant_parameters': scene['variant_parameters'], 'event_parameters': scene['event_parameters'],
        'valid': quality['valid'], 'invalid_reasons': quality['invalid_reasons'],
        'navigation_result': outcome['navigation_result'], 'goal_reached': outcome['goal_reached'],
        'timeout': outcome['timeout'], 'nav_abort': outcome['nav_abort'],
        'safety_result': outcome['safety_result'],
        'proxy_collision': outcome['collision_flags']['proxy_person_carter'],
        'physical_person_carter_contact': outcome['collision_flags']['physical_contacts']['person_carter'],
        **result['metrics'], 'decelerated': outcome['behavior']['decelerated'],
        'stopped': outcome['behavior']['stopped'], 'rho': feedback['rho'],
        'feedback_eligible': feedback['eligible'], 'feedback_ineligible_reasons': feedback['ineligible_reasons'],
        'threshold_version': feedback['threshold_version'], 'config_version': result['evaluator']['config']['config_version'],
        'target_clearance_m': feedback['target_clearance_m'],
        'nav2_config_sha256': scene['nav2_config_sha256'],
        'result_path': str(Path(result['provenance']['input_directory'])/'result.json'),
    }


def discover(root, event):
    parents = {p.parent for pattern in ('trajectory.jsonl', 'metrics.json', 'scenario.json', 'resolved.json')
               for p in root.rglob(pattern)}
    selected = []
    for parent in sorted(parents):
        if event == 'all':
            selected.append(parent)
            continue
        resolved = read_json(parent/'resolved.json', [])
        scenario = read_json(parent/'scenario.json', [])
        kind = scenario.get('event_type', mapping(mapping(resolved.get('spec')).get('event')).get('type'))
        if kind == event:
            selected.append(parent)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--input-root', type=Path)
    inputs.add_argument('--episode', type=Path, action='append')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--event', choices=('crossing', 'occlusion', 'all'), default='crossing')
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--replace-generated', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    validate_config(config)
    episodes = args.episode or discover(args.input_root.resolve(), args.event)
    if not episodes:
        parser.error('No matching episodes; no files written')
    if not args.summary.parent.is_dir():
        parser.error('Summary parent must already exist')
    for episode in episodes:
        target = episode/'result.json'
        if target.exists():
            previous = read_json(target, [])
            if not args.replace_generated or mapping(previous.get('evaluator')).get('version') is None:
                parser.error('Existing result requires --replace-generated and evaluator provenance: '+str(target))
    if args.summary.exists() and not args.replace_generated:
        parser.error('Existing summary requires --replace-generated')
    results = []
    for episode in episodes:
        result = evaluate_episode(episode, config)
        atomic_write(episode/'result.json', json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
        results.append(result)
    buffer = io.StringIO(newline='')
    records = [summary_row(result) for result in results]
    writer = csv.DictWriter(buffer, fieldnames=list(records[0]))
    writer.writeheader()
    for record in records:
        writer.writerow({key: ('null' if value is None else json.dumps(value, ensure_ascii=False)
                              if isinstance(value, (dict, list, bool)) else value) for key, value in record.items()})
    atomic_write(args.summary, buffer.getvalue())
    print(json.dumps({'episodes': len(results), 'valid': sum(r['quality']['valid'] for r in results),
                      'feedback_eligible': sum(r['search_feedback']['eligible'] for r in results),
                      'below_target': sum(r['search_feedback']['rho'] is not None and r['search_feedback']['rho'] < 0 for r in results),
                      'summary': str(args.summary.resolve())}, ensure_ascii=False))


if __name__ == '__main__':
    main()
