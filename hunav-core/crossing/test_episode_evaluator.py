import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from episode_evaluator import DEFAULT_CONFIG, disc_ttc, evaluate_episode, fraction_below, swept_distance


def state(t, x=0.0, y=0.0, vx=1.0, vy=0.0, present=True, px=0.2, py=2.0, status='running'):
    return {'timestamp': t,
            'carter': {'x': x, 'y': y, 'vx': vx, 'vy': vy, 'speed': math.hypot(vx, vy),
                       'heading': 0.0, 'angular_velocity': 0.0},
            'person': {'present': present, 'x': px if present else None, 'y': py if present else None,
                       'vx': 0.0 if present else None, 'vy': 0.0 if present else None,
                       'heading': 0.0 if present else None},
            'navigation': {'controller_mode': 'nav2', 'ok': True, 'cmd_vel': [vx, 0.0],
                           'command_stale': False, 'controller': {'status': status}}}


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.episode = self.root/'run'/'episode'
        self.episode.mkdir(parents=True)
        self.config = json.loads(DEFAULT_CONFIG.read_text())
        self.rows = [state(0, x=0), state(.1, x=.1), state(.2, x=.2)]
        self.legacy = {'termination': 'timeout', 'status': 'failed', 'error': None}
        self.resolved = {'spec': {'event': {'type': 'crossing', 'arrival_gap': 0}, 'carter': {'speed': 1}},
                         'resolved': {'path': [[0, 0], [10, 0]], 'person': {'radius': .23}},
                         'state_dt': .1, 'backend': 'Isaac + official HuNavSim',
                         'robot_control': 'Nav2 DWB',
                         'nav2_params_sha256': hashlib.sha256(b'test navigation config').hexdigest()}
        self.write()

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        (self.episode/'resolved.json').write_text(json.dumps(self.resolved))
        (self.episode/'metrics.json').write_text(json.dumps({**self.legacy, 'frames': len(self.rows)}))
        (self.episode/'trajectory.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        observations = [{'timestamp': r['timestamp'], 'observation': {'time': r['timestamp']+3,
                         'ranges': [5.0, None]}, 'control': r.get('navigation')} for r in self.rows]
        (self.episode/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in observations))
        (self.episode/'nav2_params.yaml').write_bytes(b'test navigation config')

    def evaluate(self):
        self.write()
        return evaluate_episode(self.episode, self.config)

    def test_target_and_missing_contacts(self):
        result = self.evaluate()
        self.assertTrue(result['quality']['valid'])
        self.assertAlmostEqual(result['metrics']['min_clearance_m'], 1.22)
        self.assertAlmostEqual(result['search_feedback']['rho'], .97)
        self.assertTrue(result['search_feedback']['eligible'])
        self.assertIsNone(result['outcome']['collision_flags']['physical_contacts']['person_carter'])
        self.assertIsNone(result['metrics']['reaction_latency_s'])
        self.assertEqual(result['outcome']['navigation_result'], 'TIMEOUT')

    def test_no_person_is_not_zero_clearance(self):
        self.rows = [state(0, present=False), state(.1, present=False)]
        result = self.evaluate()
        self.assertTrue(result['quality']['valid'])
        for key in ('min_clearance_m', 'min_ttc_s', 'stop_duration_s'):
            self.assertIsNone(result['metrics'][key])
        self.assertIsNone(result['search_feedback']['rho'])
        self.assertFalse(result['search_feedback']['eligible'])

    def test_ttc_analytic_cases(self):
        self.assertAlmostEqual(disc_ttc([2, 0], [-1, 0], .78, 10), 1.22)
        self.assertIsNone(disc_ttc([2, 0], [1, 0], .78, 10))
        self.assertIsNone(disc_ttc([2, 0], [0, 0], .78, 10))
        self.assertIsNone(disc_ttc([20, 0], [-1, 0], .78, 10))
        self.assertEqual(disc_ttc([.5, 0], [1, 0], .78, 10), 0)
        self.assertAlmostEqual(disc_ttc([2, .78], [-1, 0], .78, 10), 2)

    def test_swept_contact_between_samples(self):
        self.rows = [state(0, x=-1, px=0, py=0), state(.1, x=1, px=0, py=0)]
        result = self.evaluate()
        self.assertAlmostEqual(result['metrics']['min_clearance_m'], -.78)
        self.assertAlmostEqual(result['metrics']['closest_time_s'], .05)
        self.assertTrue(result['outcome']['collision_flags']['proxy_person_carter'])
        self.assertIsNone(result['outcome']['collision_flags']['physical_contacts']['person_carter'])

    def test_no_interpolation_through_absence(self):
        self.rows = [state(0, x=-2, px=0, py=0), state(.1, present=False),
                     state(.2, x=2, px=0, py=0)]
        result = self.evaluate()
        self.assertAlmostEqual(result['metrics']['min_clearance_m'], 1.22)
        self.assertFalse(result['outcome']['collision_flags']['proxy_person_carter'])

    def test_actual_time_delta_and_xy_path(self):
        self.rows = [state(0, x=0, vx=1), state(.05, x=.1, vx=.5), state(.15, x=.2, vx=.4)]
        self.rows[-1]['carter']['heading'] = 2.0
        result = self.evaluate()
        self.assertTrue(result['quality']['valid'])
        self.assertAlmostEqual(result['metrics']['max_deceleration_mps2'], 10)
        self.assertAlmostEqual(result['metrics']['path_length_m'], .2)
        self.assertAlmostEqual(result['metrics']['braking_duration_s'], .15)

    def test_startup_and_terminal_idle_excluded(self):
        self.rows = [state(0, x=0, vx=0), state(.1, x=.1, vx=1),
                     state(.2, x=.2, vx=.8, status='succeeded'), state(.3, x=.2, vx=0, status='succeeded')]
        self.legacy = {'termination': 'both_goals_reached', 'error': None}
        result = self.evaluate()
        self.assertAlmostEqual(result['metrics']['min_speed_during_interaction_mps'], .8)
        self.assertEqual(result['metrics']['stop_duration_s'], 0)
        self.assertAlmostEqual(result['metrics']['time_to_goal_s'], .2)

    def test_goal_approach_excluded_from_response(self):
        self.resolved['resolved']['path'][-1] = [.3, 0]
        result = self.evaluate()
        self.assertIsNone(result['metrics']['min_speed_during_interaction_mps'])
        self.assertIsNotNone(result['metrics']['min_clearance_m'])

    def test_safe_stop_is_success_and_has_positive_rho(self):
        self.rows = []
        x = 0
        for i in range(101):
            speed = 0 if 20 <= i <= 24 else 1
            self.rows.append(state(i*.1, x=x, vx=speed, px=2, py=2,
                                   status='succeeded' if i == 100 else 'running'))
            x += speed*.1
        self.resolved['resolved']['path'][-1] = [self.rows[-1]['carter']['x'], 0]
        self.legacy = {'termination': 'both_goals_reached', 'error': None}
        result = self.evaluate()
        self.assertTrue(result['quality']['valid'])
        self.assertTrue(result['outcome']['behavior']['stopped'])
        self.assertEqual(result['outcome']['navigation_result'], 'GOAL_REACHED')
        self.assertEqual(result['outcome']['safety_result'], 'NO_PROXY_VIOLATION')
        self.assertGreater(result['search_feedback']['rho'], 0)

    def test_stop_threshold_fraction(self):
        self.assertAlmostEqual(fraction_below(0, .1, .03), .3)
        self.assertAlmostEqual(fraction_below(.1, 0, .03), .3)
        self.assertEqual(fraction_below(0, 0, .03), 1)

    def test_nav_abort_separate_from_runtime_failure(self):
        final = {'status': 'failed', 'detail': {'action_status': 6}}
        self.legacy = {'termination': 'runtime_error', 'navigation': {'nav2': final},
                       'error': "RuntimeError({'status': 'failed', 'detail': {'action_status': 6}})"}
        result = self.evaluate()
        self.assertTrue(result['quality']['valid'])
        self.assertTrue(result['outcome']['nav_abort'])
        self.assertFalse(result['outcome']['program_error'])
        self.assertEqual(result['outcome']['navigation_result'], 'NAV_ABORT')

    def test_program_error_never_rewarded(self):
        self.legacy = {'termination': 'runtime_error', 'error': 'RuntimeError(lost floor)'}
        result = self.evaluate()
        self.assertFalse(result['quality']['valid'])
        self.assertIsNone(result['search_feedback']['rho'])
        self.assertFalse(result['search_feedback']['eligible'])

    def test_missing_terminal_state_not_success(self):
        self.legacy = {}
        result = self.evaluate()
        self.assertIsNone(result['outcome']['goal_reached'])
        self.assertFalse(result['quality']['valid'])

    def test_legacy_completion_not_used_as_arrival_timestamp(self):
        self.legacy = {'termination': 'both_goals_reached', 'error': None}
        result = self.evaluate()
        self.assertTrue(result['outcome']['goal_reached'])
        self.assertIsNone(result['metrics']['time_to_goal_s'])

    def test_duplicate_and_reversed_timestamps(self):
        for timestamp in (0, -.1):
            with self.subTest(timestamp=timestamp):
                self.rows[1]['timestamp'] = timestamp
                result = self.evaluate()
                self.assertFalse(result['quality']['valid'])
                self.assertIsNone(result['search_feedback']['rho'])
                self.assertIsNone(result['metrics']['path_length_m'])

    def test_sampling_gap_invalidates_feedback(self):
        self.rows[-1]['timestamp'] = 1
        result = self.evaluate()
        self.assertIn('trajectory_sampling_gap', result['quality']['invalid_reasons'])
        self.assertIsNone(result['search_feedback']['rho'])

    def test_missing_velocity_preserves_geometric_diagnostic(self):
        del self.rows[1]['person']['vx']
        result = self.evaluate()
        self.assertFalse(result['quality']['valid'])
        self.assertIsNotNone(result['metrics']['min_clearance_m'])
        self.assertIsNone(result['metrics']['min_ttc_s'])
        self.assertIsNone(result['search_feedback']['rho'])

    def test_malformed_and_nonfinite_json(self):
        for bad in ('{bad}\n', '{"timestamp": NaN}\n'):
            self.write()
            with (self.episode/'trajectory.jsonl').open('a') as stream:
                stream.write(bad)
            result = evaluate_episode(self.episode, self.config)
            self.assertFalse(result['quality']['valid'])
            self.assertIsNone(result['search_feedback']['rho'])

    def test_empty_input_and_missing_actor(self):
        self.rows = []
        result = self.evaluate()
        self.assertFalse(result['quality']['valid'])
        self.rows = [state(0), state(.1)]
        del self.rows[1]['carter']
        result = self.evaluate()
        self.assertFalse(result['quality']['valid'])

    def test_physical_contact_is_independent(self):
        self.rows[1]['contacts'] = {'person_carter': True}
        result = self.evaluate()
        self.assertFalse(result['outcome']['collision_flags']['proxy_person_carter'])
        self.assertTrue(result['outcome']['collision_flags']['physical_contacts']['person_carter'])
        self.assertEqual(result['outcome']['safety_result'], 'PHYSICAL_CONTACT')

    def test_static_person_contact_is_invalid_and_false_requires_complete_coverage(self):
        self.rows[0]['contacts'] = {'person_carter': False}
        result = self.evaluate()
        self.assertIsNone(result['outcome']['collision_flags']['physical_contacts']['person_carter'])
        for row in self.rows:
            row['contacts'] = {'person_carter': False, 'person_environment': False}
        result = self.evaluate()
        self.assertFalse(result['outcome']['collision_flags']['physical_contacts']['person_carter'])
        self.rows[1]['contacts']['person_environment'] = True
        result = self.evaluate()
        self.assertFalse(result['quality']['valid'])
        self.assertIsNone(result['search_feedback']['rho'])

    def test_applied_command_staleness_and_post_goal(self):
        self.rows[1]['navigation']['command_stale'] = True
        result = self.evaluate()
        self.assertFalse(result['quality']['valid'])
        self.rows[1]['navigation']['controller']['status'] = 'succeeded'
        self.legacy = {'termination': 'both_goals_reached'}
        result = self.evaluate()
        self.assertTrue(result['quality']['valid'])

    def test_sensor_timestamp_and_invalid_scan(self):
        for altered in ({'time': 10, 'ranges': [5]}, {'time': 3, 'ranges': [-1]}):
            self.write()
            p = self.episode/'observations.jsonl'
            observations = [json.loads(line) for line in p.read_text().splitlines()]
            observations[1]['observation'] = altered
            p.write_text(''.join(json.dumps(r)+'\n' for r in observations))
            result = evaluate_episode(self.episode, self.config)
            self.assertFalse(result['quality']['valid'])

    def test_nav_config_mismatch(self):
        self.resolved['nav2_params_sha256'] = 'bad'
        result = self.evaluate()
        self.assertIn('nav2_configuration_hash_mismatch', result['quality']['invalid_reasons'])

    def test_missing_reproducibility_or_sensor_channels_excludes_search(self):
        self.write()
        (self.episode/'observations.jsonl').unlink()
        result = evaluate_episode(self.episode, self.config)
        self.assertTrue(result['quality']['valid'])
        self.assertFalse(result['search_feedback']['eligible'])
        self.write()
        (self.episode/'nav2_params.yaml').unlink()
        del self.resolved['nav2_params_sha256']
        (self.episode/'resolved.json').write_text(json.dumps(self.resolved))
        result = evaluate_episode(self.episode, self.config)
        self.assertIsNone(result['scenario']['nav2_config_sha256'])
        self.assertFalse(result['search_feedback']['eligible'])

    def test_missing_commands_excludes_search(self):
        self.rows[1]['navigation']['cmd_vel'] = None
        result = self.evaluate()
        self.assertIn('commands', result['quality']['missing_fields'])
        self.assertIn('command_snapshots_incomplete', result['search_feedback']['ineligible_reasons'])

    def test_fixed_baseline_separate(self):
        for row in self.rows:
            row['navigation']['controller_mode'] = 'fixed'
        result = self.evaluate()
        self.assertTrue(result['quality']['valid'])
        self.assertIsNotNone(result['search_feedback']['rho'])
        self.assertFalse(result['search_feedback']['eligible'])
        self.assertIn('not_nav2_closed_loop', result['search_feedback']['ineligible_reasons'])

    def test_replay_protocol_preserves_parent(self):
        (self.episode/'scenario.json').write_text(json.dumps({
            'scenario_id': 'variant_001', 'source': 'augmented', 'parent_id': 'crowdes_001',
            'motion_mode': 'trajectory_replay', 'seed': 5, 'variant_parameters': {'time_shift_s': .2, 'speed_scale': 1.1},
            'search_round': 2}))
        result = evaluate_episode(self.episode, self.config)
        self.assertEqual(result['scenario']['motion_mode'], 'trajectory_replay')
        self.assertEqual(result['scenario']['parent_id'], 'crowdes_001')
        self.assertEqual(result['scenario']['variant_parameters']['speed_scale'], 1.1)

    def test_cli_batch_idempotence_and_no_raw_mutation(self):
        original = {p.name: p.read_bytes() for p in self.episode.iterdir()}
        summary = self.root/'summary.csv'
        command = [sys.executable, '-B', str(Path(__file__).with_name('episode_evaluator.py')),
                   '--input-root', str(self.root), '--summary', str(summary)]
        first = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        data = (self.episode/'result.json').read_bytes()
        with summary.open() as stream:
            records = list(csv.DictReader(stream))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['physical_person_carter_contact'], 'null')
        self.assertEqual(records[0]['target_clearance_m'], '0.25')
        again = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(again.returncode, 0)
        updated = subprocess.run(command+['--replace-generated'], capture_output=True, text=True)
        self.assertEqual(updated.returncode, 0, updated.stderr)
        self.assertEqual(data, (self.episode/'result.json').read_bytes())
        for name, content in original.items():
            self.assertEqual(content, (self.episode/name).read_bytes())

    def test_configuration_rejects_nonpositive_or_unversioned(self):
        for key, value in [('target_clearance_m', float('nan')), ('threshold_version', ''), ('stop_threshold_mps', 1)]:
            config = copy.deepcopy(self.config)
            config[key] = value
            with self.assertRaises(ValueError):
                evaluate_episode(self.episode, config)


if __name__ == '__main__':
    unittest.main(verbosity=2)
