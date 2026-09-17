import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, '/repo/hunav-core/crossing')
from metrics import intersections


ROOT = Path('/work/crowdes-b')
parser = argparse.ArgumentParser()
parser.add_argument('run')
args = parser.parse_args()
run = ROOT / 'runs' / args.run
scene_name = json.loads((run / 'provenance.json').read_text())['scene']
inputs = np.load(ROOT / 'inputs' / scene_name / 'input.npz')
origin, resolution = inputs['world_origin'], float(inputs['H'][0, 0])
route = inputs['route']
route_length = float(np.linalg.norm(route[1] - route[0]))
route_times = np.array([0., route_length / .8])
policy = {
    'minimum_duration_s': 2., 'minimum_distance_travelled_m': 1.,
    'maximum_speed_mps': 3., 'p95_speed_limit_mps': 2.2,
    'pedestrian_radius_m': .23, 'raster_clearance_tolerance_m': .04,
    'minimum_crossing_angle_degrees': 15., 'maximum_nominal_arrival_gap_s': 2.,
    'duplicate_mean_path_distance_m': .3, 'carter_speed_mps': .8,
    'carter_radius_m': .55, 'segment_sample_spacing_m': resolution / 2,
}
(run / 'validation_policy.json').write_text(json.dumps(policy, indent=2))


def dense_positions(xy):
    return np.concatenate([np.linspace(a, b, max(2, int(np.ceil(np.linalg.norm(b-a) / (resolution/2)))+1))
                           for a, b in zip(xy[:-1], xy[1:])])


def path_descriptor(xy):
    distance = np.r_[0, np.linalg.norm(np.diff(xy, axis=0), axis=1).cumsum()]
    unique = np.r_[True, np.diff(distance) > 1e-8]
    samples = np.linspace(0, distance[-1], 32)
    return np.column_stack([np.interp(samples, distance[unique], xy[unique, axis]) for axis in [0, 1]])


rows, trajectories, descriptors = [], {}, []
selected = []
for csv in sorted(run.glob('seed_*_world.csv')):
    seed = int(csv.stem.split('_')[1])
    frame = pd.read_csv(csv)
    for agent_id, traj in frame.groupby('agent_id', sort=True):
        traj = traj.sort_values('time_s')
        xy, times = traj[['x', 'y']].to_numpy(), traj.time_s.to_numpy()
        key = f'seed_{seed:03d}_agent_{int(agent_id):03d}'
        record = {'id': key, 'seed': seed, 'agent_id': int(agent_id), 'samples': len(traj)}
        reasons = []
        if len(xy) < 2 or not np.isfinite(np.c_[xy, times]).all() or not (np.diff(times) > 0).all():
            record.update(valid=False, reasons=['invalid_samples'])
            rows.append(record)
            continue
        steps, dt = np.linalg.norm(np.diff(xy, axis=0), axis=1), np.diff(times)
        speeds = steps / dt
        duration, length = float(times[-1]-times[0]), float(steps.sum())
        dense = dense_positions(xy)
        cells = np.rint((dense-origin) / resolution).astype(int)
        height, width = inputs['walkable'].shape
        inside = (cells[:, 0] >= 0) & (cells[:, 0] < width) & (cells[:, 1] >= 0) & (cells[:, 1] < height)
        safe_cells = np.clip(cells, [0, 0], [width-1, height-1])
        nav_valid = inside & (inputs['source_walkable'][safe_cells[:, 1], safe_cells[:, 0]] > 0)
        clearance = inputs['mesh_clearance'][safe_cells[:, 1], safe_cells[:, 0]]
        body_valid = inside & (clearance >= policy['pedestrian_radius_m']-policy['raster_clearance_tolerance_m'])
        if duration < policy['minimum_duration_s']: reasons.append('short_duration')
        if length < policy['minimum_distance_travelled_m']: reasons.append('little_motion')
        if speeds.max() > policy['maximum_speed_mps'] or np.quantile(speeds, .95) > policy['p95_speed_limit_mps']:
            reasons.append('implausible_speed')
        if not nav_valid.all(): reasons.append('outside_navmesh')
        if not body_valid.all(): reasons.append('body_static_overlap')
        hits = intersections(route, xy, route_times, times)
        timed_hits = [h for h in hits if h['time_separation'] <= policy['maximum_nominal_arrival_gap_s']]
        progress = np.clip(times / route_times[-1], 0, 1)
        carter = route[0] + progress[:, None] * (route[1] - route[0])
        relative = xy - carter
        delta = np.diff(relative, axis=0)
        fraction = np.clip(-np.sum(relative[:-1]*delta, axis=1) / np.maximum(np.sum(delta*delta, axis=1), 1e-12), 0, 1)
        nearest = np.linalg.norm(relative[:-1]+fraction[:, None]*delta, axis=1)
        record.update(start_time_s=float(times[0]), end_time_s=float(times[-1]), duration_s=duration,
                      length_m=length, start_xy=xy[0].tolist(), end_xy=xy[-1].tolist(),
                      mean_speed_mps=length/duration, p95_speed_mps=float(np.quantile(speeds, .95)),
                      maximum_speed_mps=float(speeds.max()), minimum_static_clearance_m=float(clearance.min()),
                      navmesh_valid_fraction=float(nav_valid.mean()), body_valid_fraction=float(body_valid.mean()),
                      valid=not reasons, reasons=reasons, geometric_crossings=hits,
                      timed_crossings=timed_hits, nominal_minimum_pair_clearance_m=float(nearest.min()-.78))
        trajectories[key] = traj
        if not reasons:
            descriptor = path_descriptor(xy)
            duplicate = next((other for other, desc in descriptors if np.linalg.norm(desc-descriptor, axis=1).mean() < .3), None)
            record['near_duplicate_of'] = duplicate
            if duplicate is None:
                descriptors.append((key, descriptor))
                if timed_hits: selected.append(record)
        rows.append(record)

selected.sort(key=lambda r: min(h['time_separation'] for h in r['timed_crossings']))
export = run / 'selected'
export.mkdir(exist_ok=True)
for record in selected[:5]:
    trajectories[record['id']].to_csv(export / (record['id']+'.csv'), index=False)
    (export / (record['id']+'.json')).write_text(json.dumps(record, indent=2))
from collections import Counter
summary = {
    'candidate_count': len(rows), 'valid_count': sum(r['valid'] for r in rows),
    'distinct_valid_paths': len(descriptors),
    'valid_geometric_crossing_count': sum(bool(r.get('geometric_crossings')) and r['valid'] for r in rows),
    'valid_timed_crossing_count': sum(bool(r.get('timed_crossings')) and r['valid'] for r in rows),
    'distinct_timed_crossing_count': len(selected), 'selected': [r['id'] for r in selected[:5]],
    'rejection_reasons': dict(Counter(reason for r in rows for reason in r['reasons'])),
    'policy': policy, 'candidates': rows,
    'metrics_source_sha256': hashlib.sha256(Path('/repo/hunav-core/crossing/metrics.py').read_bytes()).hexdigest(),
}
(run / 'analysis.json').write_text(json.dumps(summary, indent=2))
canvas = np.full((*inputs['walkable'].shape, 3), 25, np.uint8)
canvas[inputs['source_walkable'] > 0] = (65, 65, 65)
canvas[inputs['walkable'] > 0] = (210, 210, 210)
for row in rows:
    if row['id'] not in trajectories: continue
    xy = trajectories[row['id']][['x', 'y']].to_numpy()
    cells = np.rint((xy-origin) / resolution).astype(np.int32)
    color = (40, 160, 40) if row['valid'] else (80, 80, 210)
    cv2.polylines(canvas, [cells], False, color, 1)
cv2.polylines(canvas, [np.rint((route-origin)/resolution).astype(np.int32)], False, (255, 90, 20), 4)
canvas = np.flipud(canvas).copy()
cv2.putText(canvas, f"valid {summary['valid_count']}/{len(rows)}; timed crossing {summary['valid_timed_crossing_count']}",
            (15, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
cv2.imwrite(str(run / 'trajectories.png'), canvas)
print(json.dumps({k: v for k, v in summary.items() if k not in ['candidates', 'policy']}), flush=True)
