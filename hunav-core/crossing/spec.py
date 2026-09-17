import math
from pathlib import Path

import yaml


def number(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a number')
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be finite and in [{minimum}, {maximum}]')
    return float(value)


def fields(value, required, optional, name):
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be a mapping')
    if required - value.keys() or value.keys() - required - optional:
        raise ValueError(f'{name}: missing {required-value.keys()}, unknown {value.keys()-required-optional}')


def load_spec(path):
    path = Path(path)
    if path.stat().st_size > 65536:
        raise ValueError('Scenario YAML must be smaller than 64 KiB')
    spec = yaml.safe_load(path.read_text(encoding='utf-8'))
    fields(spec, {'event', 'carter', 'person'}, {'schema_version', 'max_seconds', 'occluder'}, 'scenario')
    if type(spec.get('schema_version', 1)) is not int or spec.get('schema_version', 1) != 1:
        raise ValueError('Only schema_version 1 is supported')
    if isinstance(spec.get('event'), dict) and spec['event'].get('type') == 'occlusion':
        from occlusion import validate_spec
        return validate_spec(spec)
    if 'occluder' in spec:
        raise ValueError('occluder belongs to occlusion events only')
    fields(spec['event'], {'type', 'conflict_point', 'crossing_angle_deg', 'arrival_gap'}, set(), 'event')
    fields(spec['carter'], {'path', 'speed'}, set(), 'carter')
    fields(spec['person'], {'direction', 'speed', 'pre_conflict_distance', 'post_conflict_distance'}, set(), 'person')
    if spec['event']['type'] != 'crossing' or spec['event']['crossing_angle_deg'] != 90:
        raise ValueError('This version supports perpendicular crossing only')
    if spec['carter']['path'] != 'main_corridor':
        raise ValueError('scene55 currently exposes main_corridor only')
    if spec['person']['direction'] != 'left_to_right':
        raise ValueError('This version fixes left_to_right relative to Carter forward')
    xy = spec['event']['conflict_point']
    if not isinstance(xy, list) or len(xy) != 2:
        raise ValueError('conflict_point must contain [x, y] in metres')
    spec['event']['conflict_point'] = [number(v, 'conflict_point', -1000, 1000) for v in xy]
    spec['event']['arrival_gap'] = number(spec['event']['arrival_gap'], 'arrival_gap', -30, 30)
    spec['carter']['speed'] = number(spec['carter']['speed'], 'carter.speed', .05, 1.5)
    spec['person']['speed'] = number(spec['person']['speed'], 'person.speed', .05, 2.)
    for key in ['pre_conflict_distance', 'post_conflict_distance']:
        spec['person'][key] = number(spec['person'][key], key, .25, 20.)
    spec['schema_version'] = 1
    spec['max_seconds'] = number(spec.get('max_seconds', 45), 'max_seconds', 1, 120)
    return spec


def resolve(spec, path):
    import numpy as np
    path = np.asarray(path, dtype=float)
    if path.shape != (2, 2):
        raise ValueError('main_corridor must be a straight two-point path')
    point = np.array(spec['event']['conflict_point'])
    forward = path[1]-path[0]
    length = np.linalg.norm(forward)
    forward /= length
    along = float((point-path[0])@forward)
    if not 0. < along < length or np.linalg.norm(point-(path[0]+along*forward)) > .01:
        raise ValueError('conflict_point must lie inside the Carter path')
    left = np.array([-forward[1], forward[0]])
    person = spec['person']
    start = point+left*person['pre_conflict_distance']
    goal = point-left*person['post_conflict_distance']
    carter_arrival = along/spec['carter']['speed']
    spawn = carter_arrival+spec['event']['arrival_gap']-person['pre_conflict_distance']/person['speed']
    if spawn < -1e-9:
        raise ValueError('Requested arrival_gap requires negative spawn_time; change event geometry or timing')
    if spawn >= spec['max_seconds']:
        raise ValueError('Derived spawn_time lies after episode timeout')
    if carter_arrival+spec['event']['arrival_gap'] >= spec['max_seconds']:
        raise ValueError('Nominal conflict lies after episode timeout')
    return {'path': path.tolist(), 'person': {'start': start.tolist(), 'goal': goal.tolist(),
            'speed': person['speed'], 'spawn_time': max(0., spawn), 'radius': .23,
            'yaw': math.atan2(-left[1], -left[0]), 'social_force_factor': 10.},
            'nominal_carter_arrival_s': carter_arrival,
            'nominal_person_arrival_s': carter_arrival+spec['event']['arrival_gap'],
            'arrival_gap_definition': 'nominal Person arrival minus nominal Carter arrival',
            'motion_assumption': 'constant nominal speed, no acceleration or interaction; actual states recorded separately'}
