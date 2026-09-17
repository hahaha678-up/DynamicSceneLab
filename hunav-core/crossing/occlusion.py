import math
import numpy as np
from visibility import SENSOR, visibility


REGIONS = {'pillar_corner_01': {'bounds_xy': [[3.2,11.7],[4.25,12.7]],
           'description': 'existing tall column at the right side of scene55 corridor'}}
PATHS = {'pillar_corridor': [[2.75,8.5],[2.75,15.5]]}
WINDOW_TOLERANCE = .06


def validate_spec(spec):
    from spec import fields, number
    fields(spec['event'], {'type','conflict_point','reaction_window'},set(),'event')
    fields(spec.get('occluder'),{'region'},set(),'occluder')
    fields(spec['carter'],{'path','speed'},set(),'carter')
    fields(spec['person'],{'direction','speed'},set(),'person')
    if spec['occluder']['region'] not in REGIONS:
        raise ValueError('Unknown scene55 occluder region')
    if spec['carter']['path'] not in PATHS:
        raise ValueError('Occlusion requires a registered scene55 occlusion path')
    if spec['person']['direction'] != 'hidden_to_corridor':
        raise ValueError('Occlusion direction must be hidden_to_corridor')
    xy=spec['event']['conflict_point']
    if not isinstance(xy,list) or len(xy)!=2:
        raise ValueError('conflict_point must contain [x,y]')
    spec['event']['conflict_point']=[number(v,'conflict_point',-1000,1000) for v in xy]
    spec['event']['reaction_window']=number(spec['event']['reaction_window'],'reaction_window',.1,10.)
    spec['carter']['speed']=number(spec['carter']['speed'],'carter.speed',.05,1.5)
    spec['person']['speed']=number(spec['person']['speed'],'person.speed',.05,2.)
    spec['max_seconds']=number(spec.get('max_seconds',45),'max_seconds',1,120)
    spec['schema_version']=1
    return spec


def blocked_by_region(result, region):
    bounds=np.array(region['bounds_xy'])
    return any(np.all(np.asarray(p[:2])>=bounds[0]) and np.all(np.asarray(p[:2])<=bounds[1])
               for p in result['blocker_points'])


def summarize_visibility(rows, resolved):
    active=[r for r in rows if r['person']['present']]
    first=next((r for r in active if r['visibility']['visible']),None)
    initial=active[0] if active else None
    hidden=bool(initial and initial['visibility']['in_fov_samples']>0 and
                initial['visibility']['blocked_samples']>0 and not initial['visibility']['visible'] and
                blocked_by_region(initial['visibility'],resolved['occluder']))
    first_time=first['timestamp'] if first else None
    duration=first_time-initial['timestamp'] if first and hidden else None
    before=[r for r in active if first is None or r['timestamp']<first_time]
    return {'first_visible_time':first_time, 'occlusion_duration':duration,
            'reaction_window_actual':resolved['nominal_carter_arrival_s']-first_time if first else None,
            'initially_occluded':hidden,
            'pre_visibility_out_of_fov_frames':sum(r['visibility']['in_fov_samples']==0 for r in before),
            'first_visible_ttc_s':first['pair']['ttc'] if first else None,
            'first_visible_clearance_m':first['pair']['clearance'] if first else None,
            'visibility_definition':'Any of 9 body samples inside virtual forward camera FOV and unobstructed by scene mesh',
            'reaction_window_definition':'Fixed nominal simultaneous conflict time minus measured first-visible time; may be negative',
            'behavior_observation':'HuNav retains its original access to Carter state; camera visibility does not gate SFM forces'}


class UnresolvableOcclusion(ValueError):
    def __init__(self, report):
        self.report=report
        super().__init__(f"Requested reaction_window {report['requested']} s is outside the sampled feasible set; closest {report['closest']} s")


class OcclusionSpecResolver:
    catalogs={}

    def __init__(self, static_raycast, walkable):
        self.raycast=static_raycast
        self.walkable=walkable

    def resolve(self,spec,path):
        path=np.asarray(path,dtype=float)
        point=np.asarray(spec['event']['conflict_point'])
        forward=(path[-1]-path[0])/np.linalg.norm(path[-1]-path[0])
        along=float((point-path[0])@forward)
        if along<=0 or along>=np.linalg.norm(path[-1]-path[0]) or np.linalg.norm(point-path[0]-along*forward)>.01:
            raise ValueError('conflict_point must lie inside the Carter path')
        vc,vp=spec['carter']['speed'],spec['person']['speed']
        tc=along/vc
        if tc>=spec['max_seconds']:
            raise ValueError('Nominal conflict occurs after timeout')
        region=REGIONS[spec['occluder']['region']]
        key=(tuple(path.ravel()),tuple(point),vc,vp,spec['occluder']['region'])
        if key not in self.catalogs:
            candidates=[]
            yaw=math.atan2(forward[1],forward[0])
            right=np.array([forward[1],-forward[0]])
            for angle_deg in np.arange(-20.,80.01,.5):
                a=math.radians(angle_deg)
                approach=math.cos(a)*right+math.sin(a)*forward
                goal=point-1.5*approach
                if not self.walkable(np.linspace(point,goal,40),.28).all():
                    continue
                def check(w):
                    c=point-forward*vc*w
                    p=point+approach*vp*w
                    return visibility({'x':c[0],'y':c[1],'heading':yaw},
                        {'present':True,'x':p[0],'y':p[1]},self.raycast)
                windows=np.arange(0.,min(tc-.05,4./vp),.05)
                observations=[check(w) for w in windows]
                for i in range(1,len(windows)):
                    if not observations[i-1]['visible'] or observations[i]['visible'] or observations[i]['in_fov_samples']==0:
                        continue
                    lo,hi=windows[i-1],windows[i]
                    for _ in range(5):
                        middle=(lo+hi)/2
                        if check(middle)['visible']:lo=middle
                        else:hi=middle
                    rw=(lo+hi)/2
                    for lead in [.5,.3,.15]:
                        start_window=rw+lead
                        if start_window>=tc or vp*start_window>4.:
                            continue
                        start=point+approach*vp*start_window
                        if not self.walkable(np.linspace(start,goal,120),.28).all():
                            continue
                        checks=[check(w) for w in np.linspace(rw+.003,start_window,12)]
                        if any(v['visible'] or v['in_fov_samples']==0 for v in checks):
                            continue
                        if not blocked_by_region(checks[-1],region):
                            continue
                        candidates.append({'window':float(rw),'start':start.tolist(),'goal':goal.tolist(),
                            'spawn_time':float(tc-start_window),'angle_deg':float(angle_deg),
                            'nominal_occlusion_duration':lead,'initial_visibility':checks[-1]})
                        break
            self.catalogs[key]=candidates
        candidates=self.catalogs[key]
        target=spec['event']['reaction_window']
        selected=min(candidates,key=lambda r:(abs(r['window']-target),-r['nominal_occlusion_duration'])) if candidates else None
        if selected is None or abs(selected['window']-target)>WINDOW_TOLERANCE:
            raise UnresolvableOcclusion({'requested':target,'closest':selected['window'] if selected else None,
                'sampled_window_range':[min(c['window'] for c in candidates),max(c['window'] for c in candidates)] if candidates else None,
                'candidate_count':len(candidates),'search_angles_deg':[-20,80,.5],
                'note':'Infeasible within this scene binding and finite search; not a proof for every possible route.'})
        delta=np.asarray(selected['goal'])-selected['start']
        return {'path':path.tolist(),'person':{'start':selected['start'],'goal':selected['goal'],
            'spawn_time':selected['spawn_time'],'speed':vp,'radius':.23,
            'yaw':math.atan2(delta[1],delta[0]),'social_force_factor':10.},
            'nominal_carter_arrival_s':tc,'nominal_person_arrival_s':tc,
            'reaction_window_requested':target,'reaction_window_resolved':selected['window'],
            'nominal_first_visible_time':tc-selected['window'],'resolver_tolerance_s':WINDOW_TOLERANCE,
            'occluder':dict(region,name=spec['occluder']['region']),'sensor':SENSOR,
            'derived_approach_angle_deg':selected['angle_deg'],
            'nominal_occlusion_duration':selected['nominal_occlusion_duration'],
            'motion_assumption':'constant nominal speeds and simultaneous arrival at conflict point; HuNav may change actual arrival/visibility'}
