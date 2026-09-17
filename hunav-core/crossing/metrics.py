import math

import numpy as np


def disc_ttc(relative_position, relative_velocity, radius, horizon=10.):
    r, v = np.asarray(relative_position), np.asarray(relative_velocity)
    c = float(r @ r - radius**2)
    if c <= 0:
        return 0.
    a, b = float(v @ v), 2*float(r @ v)
    discriminant = b*b-4*a*c
    if a < 1e-12 or b >= 0 or discriminant < 0:
        return None
    t = (-b-math.sqrt(max(0., discriminant)))/(2*a)
    return float(t) if 0 <= t <= horizon else None


def cross2(a, b):
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


def intersections(first, second, first_times=None, second_times=None):
    a, b = np.asarray(first), np.asarray(second)
    hits = []
    if len(a) < 2 or len(b) < 2:
        return hits
    q, s = b[:-1], np.diff(b, axis=0)
    for i, (p, r) in enumerate(zip(a[:-1], np.diff(a, axis=0))):
        den = cross2(r, s)
        good = np.abs(den) > 1e-10
        u, v = np.zeros(len(s)), np.zeros(len(s))
        u[good] = cross2(q-p, s)[good]/den[good]
        v[good] = cross2(q-p, r)[good]/den[good]
        good &= (u >= 0)&(u <= 1)&(v >= 0)&(v <= 1)
        for j in np.flatnonzero(good):
            angle = math.degrees(math.acos(np.clip(abs(float(r@s[j]))/(np.linalg.norm(r)*np.linalg.norm(s[j])), 0, 1)))
            if angle < 15.:
                continue
            hit = {'xy': (p+u[j]*r).tolist(), 'angle_degrees': angle}
            if first_times is not None:
                ta = float(first_times[i]+u[j]*(first_times[i+1]-first_times[i]))
                tb = float(second_times[j]+v[j]*(second_times[j+1]-second_times[j]))
                hit.update(carter_time=ta, person_time=tb, time_separation=abs(ta-tb))
            hits.append(hit)
    return hits


def pair_metrics(carter, person, radius=.78):
    if not person['present']:
        return {'center_distance': None, 'clearance': None, 'ttc': None, 'collision': False}
    r = np.array([person['x']-carter['x'], person['y']-carter['y']])
    v = np.array([person['vx']-carter['vx'], person['vy']-carter['vy']])
    d = float(np.linalg.norm(r))
    return {'center_distance': d, 'clearance': d-radius,
            'ttc': disc_ttc(r, v, radius), 'collision': d <= radius}


def summarize(rows, spec, path):
    active = [r for r in rows if r['person']['present']]
    minimum, closest_time = None, None
    first_collision = None
    for i, row in enumerate(active):
        distance = row['pair']['center_distance']
        t = row['timestamp']
        if i:
            previous = active[i-1]
            r0 = np.array([previous['person'][k]-previous['carter'][k] for k in ('x', 'y')])
            r1 = np.array([row['person'][k]-row['carter'][k] for k in ('x', 'y')])
            delta = r1-r0
            # Swept relative motion catches contacts between saved samples.
            u = float(np.clip(-(r0@delta)/(delta@delta), 0, 1)) if delta@delta > 1e-12 else 0.
            candidate = float(np.linalg.norm(r0+u*delta))
            if candidate < distance:
                distance = candidate
                t = previous['timestamp']+u*(row['timestamp']-previous['timestamp'])
            contact = disc_ttc(r0, delta/(row['timestamp']-previous['timestamp']), .78,
                               horizon=row['timestamp']-previous['timestamp']+1e-9)
            if first_collision is None and contact is not None:
                first_collision = previous['timestamp']+contact
        if first_collision is None and row['pair']['collision']:
            first_collision = row['timestamp']
        if minimum is None or distance < minimum:
            minimum, closest_time = distance, t
    times = [r['timestamp'] for r in active]
    carter = [[r['carter'][k] for k in ('x', 'y')] for r in rows]
    person = [[r['person'][k] for k in ('x', 'y')] for r in active]
    hits = intersections(carter, person, [r['timestamp'] for r in rows], times)
    finite = [(r['pair']['ttc'], r['timestamp']) for r in active if r['pair']['ttc'] is not None]
    gap = None if minimum is None else minimum-.78
    min_ttc = min(finite) if finite else (None, None)
    band = 'unavailable'
    if gap is not None:
        band = ('high' if gap < .25 or (min_ttc[0] is not None and min_ttc[0] < 1.) else
                'medium' if gap < .8 or (min_ttc[0] is not None and min_ttc[0] < 2.5) else 'low')
    return {'planned_path_crossing': bool(intersections(path, [spec['person']['start'], spec['person']['goal']])),
            'actual_path_crossing': bool(hits),
            'crossing': min(hits, key=lambda x:x['time_separation']) if hits else None,
            'minimum_distance_m': minimum, 'minimum_clearance_m': gap, 'closest_time_s': closest_time,
            'minimum_ttc_s': min_ttc[0], 'minimum_ttc_timestamp': min_ttc[1], 'ttc_horizon_s': 10.,
            'collision': first_collision is not None, 'first_collision_time_s': first_collision,
            'collision_definition': 'swept 2D discs: Carter radius 0.55 m + Person radius 0.23 m',
            'observed_conflict_band': band,
            'band_definition': 'high: clearance<0.25m or TTC<1s; medium: clearance<0.8m or TTC<2.5s; otherwise low',
            'difficulty_note': 'Post-response conflict proxy; not intrinsic scenario difficulty or dataset labels.'}
