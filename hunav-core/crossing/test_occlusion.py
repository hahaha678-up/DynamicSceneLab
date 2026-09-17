import copy
import json
import unittest
from types import SimpleNamespace
from visibility import visibility, physx_static_raycast
from occlusion import validate_spec, summarize_visibility


class OcclusionTests(unittest.TestCase):
    def test_visible_hidden_outside_and_absent(self):
        c={'x':0.,'y':0.,'heading':0.}
        p={'x':3.,'y':0.,'present':True}
        clear=visibility(c,p,lambda *a:None)
        hidden=visibility(c,p,lambda *a:[1.,0.,.6])
        self.assertEqual(clear['clear_samples'],9)
        self.assertTrue(clear['visible'])
        self.assertEqual(hidden['blocked_samples'],9)
        self.assertFalse(hidden['visible'])
        self.assertEqual(visibility(c,dict(p,x=-3.),lambda *a:None)['in_fov_samples'],0)
        self.assertFalse(visibility(c,{'present':False},lambda *a:None)['visible'])

    def test_partial_visibility(self):
        def ray(o,d,n):return None if (o+d*n)[2]>1.2 else [1.,0.,.6]
        v=visibility({'x':0.,'y':0.,'heading':0.},{'x':3.,'y':0.,'present':True},ray)
        self.assertTrue(v['visible'])
        self.assertEqual(v['clear_samples'],3)

    def test_physx_binding_and_static_filter(self):
        class Query:
            def raycast_all(self,o,d,n,callback):
                for path,distance in [('/World/Carter/chassis_link',.1),('/World/Environment/CollisionMesh',1.)]:
                    callback(SimpleNamespace(rigid_body=path,distance=distance,position=[distance,0.,.6]))
        self.assertEqual(physx_static_raycast(Query())([0,0,0],[1,0,0],2),[1.,0.,.6])

    def test_metric_time_origin_and_never_visible(self):
        rows=[]
        for t,visible in [(2.,False),(2.1,False),(2.2,True)]:
            rows.append({'timestamp':t,'person':{'present':True},'pair':{'ttc':.4,'clearance':.8},
                         'visibility':{'visible':visible,'in_fov_samples':9,'blocked_samples':0 if visible else 9,
                                       'blocker_points':[[3.5,12.,.6]]}})
        r={'nominal_carter_arrival_s':2.1,'occluder':{'bounds_xy':[[3.2,11.7],[4.25,12.7]]}}
        m=summarize_visibility(rows,r)
        self.assertTrue(m['initially_occluded'])
        self.assertAlmostEqual(m['occlusion_duration'],.2)
        self.assertAlmostEqual(m['reaction_window_actual'],-.1)
        self.assertIsNone(summarize_visibility(rows[:2],r)['first_visible_time'])
        self.assertIsNone(summarize_visibility([],r)['reaction_window_actual'])
        json.dumps(m,allow_nan=False)

    def test_invalid_spec(self):
        s={'event':{'type':'occlusion','conflict_point':[2.75,12.85],'reaction_window':1.},
           'occluder':{'region':'pillar_corner_01'},'carter':{'path':'pillar_corridor','speed':.8},
           'person':{'direction':'hidden_to_corridor','speed':1.}}
        self.assertEqual(validate_spec(copy.deepcopy(s))['event']['reaction_window'],1.)
        for v in [True,float('nan'),-.1,0.,'one']:
            bad=copy.deepcopy(s);bad['event']['reaction_window']=v
            with self.assertRaises(ValueError):validate_spec(bad)
        bad=copy.deepcopy(s);bad['occluder']['region']='missing'
        with self.assertRaises(ValueError):validate_spec(bad)


if __name__=='__main__':unittest.main()
