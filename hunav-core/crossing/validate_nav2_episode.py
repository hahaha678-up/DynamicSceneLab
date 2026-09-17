import argparse
import json
import math
from pathlib import Path

import numpy as np

from metrics import pair_metrics

parser = argparse.ArgumentParser()
parser.add_argument('tag')
args = parser.parse_args()
if not args.tag.startswith('nav2_crossing_') or '/' in args.tag:
    raise ValueError('Expected one navigation run tag')
episode = Path('/work/output')/args.tag/'00_gap0_person1.0'
rows = [json.loads(s) for s in (episode/'trajectory.jsonl').read_text().splitlines()]
observations = [json.loads(s) for s in (episode/'observations.jsonl').read_text().splitlines()]
metrics = json.loads((episode/'metrics.json').read_text())
resolved = json.loads((episode/'resolved.json').read_text())
maps = [json.loads(s) for s in (Path('/repo/hunav-core/runtime')/args.tag/'costmaps.jsonl').read_text().splitlines()]
maps = [m for m in maps if m['kind']=='costmap']
detections, costmap_hits = [], []
for row, observation in zip(rows, observations):
    if not row['person']['present']:
        continue
    c, p, scan = row['carter'], row['person'], observation['observation']
    angles = scan['angle_min']+np.arange(len(scan['ranges']))*scan['angle_increment']+c['heading']
    ranges = np.array([v if v is not None else np.nan for v in scan['ranges']])
    eye = np.array([c['x']+.25*math.cos(c['heading']), c['y']+.25*math.sin(c['heading'])])
    hits = eye+np.c_[np.cos(angles), np.sin(angles)]*ranges[:, None]
    count = int(np.sum(np.linalg.norm(hits-[p['x'], p['y']], axis=1)<.30))
    if count:
        detections.append({'time': row['timestamp'], 'returns': count})
    if 10.<row['timestamp']<13. and abs(row['navigation']['cmd_vel'][0])<.01:
        t = row['navigation']['nav2']['costmap_time']
        m = min(maps, key=lambda m: abs(m['time']-t))
        grid = np.array(m['data']).reshape(m['height'], m['width'])
        y, x = np.where(grid==100)
        points = np.c_[x+.5, y+.5]*m['resolution']+m['origin']
        in_footprint = np.linalg.norm(points-[c['x'], c['y']], axis=1)<.85
        near_human = np.linalg.norm(points-[p['x'], p['y']], axis=1)<.4
        costmap_hits.append({'time': row['timestamp'], 'human_lethal_cells': int((in_footprint&near_human).sum()),
                             'other_lethal_cells': int((in_footprint&~near_human).sum())})
goal = resolved['resolved']['path'][-1]
interaction = [r for r in rows if 10.<r['timestamp']<13. and math.dist([r['carter']['x'],r['carter']['y']],goal)>1.]
slow = min(interaction, key=lambda r: r['carter']['speed'])
recovery = [r['carter']['speed'] for r in rows if slow['timestamp']<r['timestamp']<slow['timestamp']+2.]
checks = {
    'nav2_action_succeeded': metrics['navigation']['nav2']['status']=='succeeded',
    'both_goals_completed': metrics['status']=='completed' and metrics['person_goal_reached'],
    'collision_free': not metrics['collision'],
    'within_goal_tolerance': metrics['carter_goal_error_m']<.15,
    'crossing_valid': metrics['event_valid'],
    'observations_recorded_30hz': len(rows)==len(observations) and
        np.max(np.abs(np.diff([r['timestamp'] for r in rows])-1/30))<1e-6,
    'person_detected_by_lidar': bool(detections),
    'person_marked_in_costmap_during_stop_command': any(h['human_lethal_cells']>0 for h in costmap_hits),
    'slowed_before_goal_approach': slow['carter']['speed']<.3,
    'resumed_after_slowdown': max(recovery)>.6,
    'scenario_timing_frozen': resolved['resolved']['person']['spawn_time']==9. and resolved['spec']['event']['arrival_gap']==0.,
    'pair_metrics_consistent': all(pair_metrics(r['carter'],r['person'])==r['pair'] for r in rows),
}
report = {'success': bool(all(checks.values())), 'checks': {k:bool(v) for k,v in checks.items()},
    'first_lidar_person_time_s': detections[0]['time'] if detections else None,
    'interaction_min_speed_mps': slow['carter']['speed'], 'slowdown_time_s': slow['timestamp'],
    'recovered_speed_mps': max(recovery), 'physical_full_stop_during_interaction': slow['carter']['speed']<.03,
    'costmap_evidence': costmap_hits, 'minimum_clearance_m': metrics['minimum_clearance_m'],
    'minimum_ttc_s': metrics['minimum_ttc_s'], 'frames':len(rows), 'costmap_frames':len(maps),
    'scope': 'One reactive Crossing. Ground truth is used only in offline association and evaluation; full stop-and-yield was not observed.'}
(episode/'navigation_acceptance.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
if not report['success']:
    raise SystemExit('Navigation acceptance failed')
