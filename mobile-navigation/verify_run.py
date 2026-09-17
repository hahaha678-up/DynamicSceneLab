import json
from pathlib import Path
import cv2
import numpy as np

root = Path('/work')
name = 'carter_scene64_final'
result = json.loads((root / f'output/{name}_result.json').read_text())
geometry = np.load(root / 'output/geometry.npz')
mesh_map = np.load(root / 'output/mesh_clearance.npz')
origin, cell = geometry['origin'], float(geometry['resolution'])
overlaps, outside = [], []
positions, speed = [], []
footprint = np.array([[-.622, -.457], [.622, -.457], [.622, .457], [-.622, .457]])
for row in result['trajectory']:
    center, yaw = np.array(row['center']), row['yaw']
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    pixels = np.rint(((rotation @ footprint.T).T+center-origin)/cell).astype(np.int32)
    mask = np.zeros_like(mesh_map['obstacles'], np.uint8)
    cv2.fillPoly(mask, [pixels], 1)
    overlaps.append(int(np.count_nonzero(mask & mesh_map['obstacles'])))
    outside.append(int(np.count_nonzero(mask & (1-geometry['walkable']))))
    positions.append(row['position'])
    speed.append(row['speed'])
positions = np.array(positions)
video = cv2.VideoCapture(str(root / f'output/{name}.mp4'))
frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = float(video.get(cv2.CAP_PROP_FPS))
decoded = 0
while True:
    ok, image = video.read()
    if not ok:
        break
    assert image.shape == (600, 960, 3)
    if decoded in (0, frames//2, frames-1):
        cv2.imwrite(str(root / f'output/{name}_frame_{decoded}.png'), image)
    decoded += 1
video.release()
summary = {'success': result['success'], 'goal_distance_m': result['final_goal_distance'],
           'final_speed_m_s': result['final_speed'],
           'travel_distance_m': float(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1).sum()),
           'height_range_m': [float(positions[:, 2].min()), float(positions[:, 2].max())],
           'max_obstacle_overlap_cells': max(overlaps), 'max_outside_navmesh_cells': max(outside),
           'frames': frames, 'decoded_frames': decoded, 'fps': fps,
           'sim_seconds': result['sim_seconds'], 'wall_seconds_without_startup': result['wall_seconds']}
summary['passed'] = bool(result['success'] and result['final_goal_distance'] < .12
                         and result['final_speed'] < .02 and max(overlaps) == 0
                         and max(outside) == 0 and decoded == frames and frames > 100)
(root / 'output/verification.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
if not summary['passed']:
    raise SystemExit(2)
