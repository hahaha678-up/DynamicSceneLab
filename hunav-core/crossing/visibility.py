import math
import numpy as np


SENSOR = {'forward_offset_m': .25, 'height_m': .6, 'horizontal_fov_deg': 120.,
          'vertical_fov_deg': 100., 'range_m': 20., 'target_radius_m': .23,
          'target_heights_m': [.35, .9, 1.5]}


def visibility(car, person, static_raycast, sensor=SENSOR):
    if not person['present']:
        return {'visible': False, 'present': False, 'in_fov_samples': 0,
                'clear_samples': 0, 'blocked_samples': 0, 'blocker_points': []}
    yaw=car['heading']
    forward=np.array([math.cos(yaw),math.sin(yaw),0.])
    right=np.array([forward[1],-forward[0],0.])
    eye=np.array([car['x'],car['y'],sensor['height_m']])+forward*sensor['forward_offset_m']
    total, clear, blocked, hits=0,0,0,[]
    for height in sensor['target_heights_m']:
        for lateral in [-sensor['target_radius_m'],0.,sensor['target_radius_m']]:
            target=np.array([person['x'],person['y'],height])+lateral*right
            delta=target-eye
            distance=float(np.linalg.norm(delta))
            along=float(delta@forward)
            in_fov=(0.02<distance<=sensor['range_m'] and along>0 and
                abs(math.atan2(delta@right,along))<=math.radians(sensor['horizontal_fov_deg']/2) and
                abs(math.atan2(delta[2],along))<=math.radians(sensor['vertical_fov_deg']/2))
            if not in_fov:
                continue
            total+=1
            hit=static_raycast(eye,delta/distance,distance-.02)
            if hit is None:
                clear+=1
            else:
                blocked+=1
                hits.append([float(v) for v in hit])
    return {'visible': clear>0, 'present': True, 'in_fov_samples': total,
            'clear_samples': clear, 'blocked_samples': blocked, 'blocker_points': hits}


def physx_static_raycast(query):
    def raycast(origin,direction,distance):
        nearest=[]
        errors=[]
        def report(hit):
            try:
                if '/World/Environment/CollisionMesh' in str(hit.rigid_body):
                    nearest.append((float(hit.distance),list(hit.position)))
            except Exception as exc:
                errors.append(exc)
            return True
        query.raycast_all(tuple(float(v) for v in origin),tuple(float(v) for v in direction),float(distance),report)
        if errors:
            raise RuntimeError('PhysX raycast callback failed') from errors[0]
        return min(nearest,key=lambda pair:pair[0])[1] if nearest else None
    return raycast
