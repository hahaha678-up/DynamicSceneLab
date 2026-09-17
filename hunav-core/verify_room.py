import argparse
import json
from pathlib import Path

import numpy as np

root = Path('/work')
parser = argparse.ArgumentParser()
parser.add_argument('--suffix', default='')
parser.add_argument('--prefix', default='hunav_')
parser.add_argument('--cases', nargs='+', default=['stop', 'cross', 'bypass'])
parser.add_argument('--require-reaction', action='store_true')
args = parser.parse_args()
geometry = np.load(root / 'output/geometry.npz')
summaries = {}
for case in args.cases:
    report = json.loads((root / f'output/{args.prefix}{case}{args.suffix}_result.json').read_text())
    rows = report['trajectory']
    human = np.array([row['human']['xy'] for row in rows])
    proxy = np.array([row['proxy_xy'] for row in rows])
    pixels = np.rint((human-geometry['origin'])/float(geometry['resolution'])).astype(int)
    h, w = geometry['walkable'].shape
    inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h))
    assert inside.all(), (case, 'human outside scene bounds')
    supported = geometry['walkable'][pixels[:, 1], pixels[:, 0]].astype(bool)
    arrived = [row for row in rows if row['human'].get('arrived')]
    direction = np.array(report['human_config']['goal'])-report['human_config']['start']
    normal = np.array([-direction[1],direction[0]])/np.linalg.norm(direction)
    lateral_offset = float(np.abs((human-report['human_config']['start'])@normal).max())
    moving = [row for row in rows if row['human']['speed'] > .01 and not row['human']['arrived']]
    response_window = [row for row in moving if row['t'] >= moving[0]['t']+.8]
    minimum_active_speed = min(row['human']['speed'] for row in response_window)
    summary = {'success': report['success'], 'robot_goal_error_m': report['final_goal_distance'],
               'human_goal_error_m': report['human_goal_distance'], 'minimum_disc_gap_m': report['minimum_disc_gap'],
               'proxy_error_m': float(np.linalg.norm(human-proxy, axis=1).max()),
               'human_on_walkable_fraction': float(supported.mean()),
               'obstacle_samples_max': max(row['obstacle_samples'] for row in rows),
               'human_arrival_s': arrived[0]['t'] if arrived else None,
               'service_mean_ms': float(np.mean([row['human']['service_ms'] for row in rows if not row['human'].get('arrived')])),
               'wall_seconds': report['wall_seconds'], 'video_frames': len(rows),
               'max_lateral_offset_m': lateral_offset,
               'minimum_active_speed_mps': minimum_active_speed}
    assert report['success'] and report['robot_goal_reached'] and report['human_goal_reached'], summary
    assert report['minimum_disc_gap'] >= 0., summary
    assert summary['proxy_error_m'] < 1e-6 and supported.all(), summary
    assert all(row['human']['behavior'] == 1 for row in rows), summary
    assert all(row['human']['speed'] == 0. for row in arrived), summary
    assert summary['obstacle_samples_max'] > 0, summary
    if args.require_reaction and case == 'cross':
        assert lateral_offset > .3, summary
        assert minimum_active_speed < report['human_config']['speed']*.6, summary
    summaries[case] = summary
output = Path('/repo/hunav-core/output')
print(json.dumps(summaries, indent=2), flush=True)
