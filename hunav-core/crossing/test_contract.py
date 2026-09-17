import copy
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from spec import load_spec, resolve
from metrics import disc_ttc, intersections, pair_metrics, summarize


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.spec = {'event': {'type': 'crossing', 'conflict_point': [0., 4.],
                              'crossing_angle_deg': 90, 'arrival_gap': 0.},
                     'carter': {'path': 'main_corridor', 'speed': .8},
                     'person': {'direction': 'left_to_right', 'speed': 1.,
                                'pre_conflict_distance': 3., 'post_conflict_distance': 2.5}}

    def load(self, value):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'scenario.yaml'
            path.write_text(yaml.safe_dump(value))
            return load_spec(path)

    def test_event_compiler(self):
        r = resolve(self.load(self.spec), [[0., 0.], [0., 10.]])
        self.assertEqual(r['person']['start'], [-3., 4.])
        self.assertEqual(r['person']['goal'], [2.5, 4.])
        self.assertEqual(r['person']['spawn_time'], 2.)
        self.spec['event']['arrival_gap'] = 1.
        self.assertEqual(resolve(self.load(self.spec), [[0.,0.],[0.,10.]])['person']['spawn_time'], 3.)

    def test_invalid_configuration(self):
        for value in [float('nan'), float('inf'), 0., -1., True, 'fast']:
            bad = copy.deepcopy(self.spec); bad['person']['speed'] = value
            with self.assertRaises(ValueError):
                self.load(bad)
        bad = copy.deepcopy(self.spec); bad['person']['speeed'] = 1.
        with self.assertRaises(ValueError): self.load(bad)
        bad = copy.deepcopy(self.spec); bad['schema_version'] = True
        with self.assertRaises(ValueError): self.load(bad)
        bad = copy.deepcopy(self.spec); bad['carter']['path'] = 'missing'
        with self.assertRaises(ValueError): self.load(bad)
        bad = self.load(self.spec); bad['event']['conflict_point'] = [1., 4.]
        with self.assertRaises(ValueError): resolve(bad, [[0.,0.],[0.,10.]])
        bad = self.load(self.spec); bad['event']['arrival_gap'] = -4.
        with self.assertRaises(ValueError): resolve(bad, [[0.,0.],[0.,10.]])

    def test_ttc(self):
        self.assertAlmostEqual(disc_ttc([2.,0.], [-1.,0.], 1.), 1.)
        self.assertIsNone(disc_ttc([2.,0.], [1.,0.], 1.))
        self.assertIsNone(disc_ttc([2.,0.], [0.,0.], 1.))
        self.assertIsNone(disc_ttc([2.,2.], [-1.,0.], 1.))
        self.assertEqual(disc_ttc([.5,0.], [0.,0.], 1.), 0.)
        self.assertIsNone(disc_ttc([20.,0.], [-1.,0.], 1.))

    def test_crossing_geometry(self):
        hits = intersections([[-1,0],[1,0]], [[0,-1],[0,1]], [0,2], [2,4])
        self.assertEqual(hits[0]['xy'], [0.,0.])
        self.assertEqual(hits[0]['time_separation'], 2.)
        self.assertFalse(intersections([[-1,0],[1,0]], [[-1,1],[1,1]]))

    def test_swept_collision_and_absence(self):
        car = {'x':0., 'y':0., 'vx':0., 'vy':0.}
        self.assertIsNone(pair_metrics(car, {'present':False})['ttc'])
        rows = []
        for t,x in [(0.,-2.),(1.,2.)]:
            person = {'present':True, 'x':x, 'y':0., 'vx':4., 'vy':0.}
            rows.append({'timestamp':t,'carter':car,'person':person,'pair':pair_metrics(car,person)})
        result = summarize(rows, {'person': {'start':[-2,0], 'goal':[2,0]}}, [[0,-1],[0,1]])
        self.assertTrue(result['collision'])
        self.assertAlmostEqual(result['minimum_distance_m'], 0.)
        self.assertAlmostEqual(result['first_collision_time_s'], (2-.78)/4)
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
