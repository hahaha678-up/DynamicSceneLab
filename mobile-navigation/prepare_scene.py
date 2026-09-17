import argparse
import heapq
import json
import math
import struct
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
from plyfile import PlyData
from scipy.ndimage import distance_transform_edt, label

parser = argparse.ArgumentParser()
parser.add_argument('--scene', choices=['scene64', 'scene55'], default='scene55')
args = parser.parse_args()
scene_id = args.scene
ROOT = Path('/work')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
nav = (ROOT / f'assets/{scene_id}/{scene_id}.navmesh').read_bytes()
offset = nav.index(b'VAND')
header = struct.unpack_from('<2I4i9i3f6ff', nav, offset)
vertices = np.frombuffer(nav, '<f4', header[7] * 3, offset + 100).reshape(-1, 3).copy()
floor = float(np.median(vertices[:, 1]) - 0.02)
if scene_id == 'scene55':
    # This NavMesh sits above the carpet; use the measured collision-mesh floor.
    floor = -0.1801194303101908
# Habitat is Y-up; the same rigid transform is applied to geometry and cameras.
gs_to_world = np.array([[1., 0., 0., 0.], [0., 0., -1., 0.],
                        [0., 1., 0., -floor], [0., 0., 0., 1.]])
world_vertices = (gs_to_world[:3, :3] @ vertices.T).T + gs_to_world[:3, 3]
poly_offset = offset + 100 + header[7] * 12
polygons = []
for index in range(header[6]):
    values = struct.unpack_from('<I6H6HHBB', nav, poly_offset + 32 * index)
    count, kind = values[-2:]
    if kind >> 6 == 0:
        polygons.append(list(values[1:1 + count]))
resolution = 0.05
origin = world_vertices[:, :2].min(0) - 0.75
shape = np.ceil((world_vertices[:, :2].max(0) + 0.75 - origin) / resolution).astype(int)
walkable = np.zeros((shape[1], shape[0]), np.uint8)
for indices in polygons:
    pixels = np.rint((world_vertices[indices, :2] - origin) / resolution).astype(np.int32)
    cv2.fillPoly(walkable, [pixels], 1)
clearance = distance_transform_edt(walkable) * resolution
safe = clearance >= 0.90
components, count = label(safe)
if count == 0:
    raise RuntimeError('No connected space with Carter clearance')
largest = 1 + int(np.argmax(np.bincount(components.ravel())[1:]))
free = components == largest
ys, xs = np.nonzero(free)
cells = np.column_stack([xs, ys])
start_cell = cells[np.argmin(cells[:, 1] + .1 * cells[:, 0])]
goal_cell = cells[np.argmax(np.linalg.norm(cells - start_cell, axis=1))]


def astar(start, goal):
    start, goal = tuple(start), tuple(goal)
    queue = [(0., start)]
    costs, parents = {start: 0.}, {}
    while queue:
        _, current = heapq.heappop(queue)
        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parents[path[-1]])
            return np.array(path[::-1])
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nxt = (current[0] + dx, current[1] + dy)
            x, y = nxt
            if not (0 <= y < free.shape[0] and 0 <= x < free.shape[1] and free[y, x]):
                continue
            if dx and dy and not (free[current[1], x] and free[y, current[0]]):
                continue
            cost = costs[current] + math.hypot(dx, dy)
            if cost < costs.get(nxt, float('inf')):
                costs[nxt], parents[nxt] = cost, current
                heapq.heappush(queue, (cost + math.dist(nxt, goal), nxt))
    raise RuntimeError('No route between automatically selected points')


path_cells = astar(start_cell, goal_cell)
path = origin + path_cells * resolution
route = path[np.unique(np.r_[0, np.arange(8, len(path), 8), len(path)-1])]
if scene_id == 'scene55':
    route = np.array([[1.3, 5.], [1.3, 15.]])
    path_cells = np.rint((route - origin) / resolution).astype(int)
mesh = PlyData.read(ROOT / f'assets/{scene_id}/{scene_id}.mesh.ply')
points = np.column_stack([mesh['vertex'][k] for k in ('x', 'y', 'z')])
points = (gs_to_world[:3, :3] @ points.T).T + gs_to_world[:3, 3]
faces = np.array(list(mesh['face']['vertex_indices']), dtype=np.int32)
np.savez(OUT / 'geometry.npz', points=points, faces=faces, walkable=walkable,
         free=free, origin=origin, resolution=resolution, route=route)
config = {'scene': f'Habitat-GS {scene_id}', 'gs_asset': f'assets/{scene_id}/{scene_id}.gs.ply', 'gs_to_world': gs_to_world.tolist(),
          'floor_gs_y': floor, 'resolution_m': resolution, 'clearance_m': .90,
          'navigation_center_offset': [-.48, 0.],
          'route': route.tolist(), 'start': route[0].tolist(), 'goal': route[-1].tolist(),
          'mesh_vertices': len(points), 'mesh_faces': len(faces),
          'navigation_area_m2': float(walkable.sum() * resolution ** 2),
          'safe_area_m2': float(free.sum() * resolution ** 2)}
(ROOT / 'scene_config.json').write_text(json.dumps(config, indent=2))
canvas = np.full((*walkable.shape, 3), 35, np.uint8)
canvas[walkable > 0] = (145, 145, 145)
canvas[free] = (175, 220, 175)
cv2.polylines(canvas, [path_cells.reshape(-1, 1, 2)], False, (255, 90, 20), 2)
cv2.circle(canvas, tuple(start_cell), 3, (0, 180, 0), -1)
cv2.circle(canvas, tuple(goal_cell), 3, (0, 0, 255), -1)
cv2.imwrite(str(OUT / 'navigation_map.png'), cv2.resize(np.flipud(canvas), None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))

sys.path.insert(0, '/repo/re3sim/gaussian_splatting')
from gaussian_renderer import GaussianModel, render
from scene.cameras import Camera
model = GaussianModel(0)
model.load_ply(str(ROOT / f'assets/{scene_id}/{scene_id}.gs.ply'))
pipe = types.SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
world_to_gs = np.linalg.inv(gs_to_world)
target = np.r_[route.mean(0)-np.array([.6, .5]), .20]
eye = np.r_[route[0]+np.array([-.43, -.87]), 2.90]
if scene_id == 'scene55':
    eye = np.array([.3, 2.5, 2.3])
    target = np.array([1.3, 12., 1.0])
    config['human'] = {'start': [5., 10.], 'goal': [0., 10.], 'yaw': math.pi,
                       'speed': 1.5205219696308003, 'radius': .23, 'social_force_factor': 10.}
    config['human_start_delay'] = 6.8
    config['routes'] = {'cross': route.tolist()}
config['overview_eye'], config['overview_target'] = eye.tolist(), target.tolist()
(ROOT / 'scene_config.json').write_text(json.dumps(config, indent=2))
for name, eye, target in [('overview', eye, target),
                          ('robot_view', np.r_[route[0], .65], np.r_[route[-1], .65])]:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.]); right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    c2w = np.eye(4); c2w[:3, :3] = np.column_stack([right, down, forward]); c2w[:3, 3] = eye
    w2c = np.linalg.inv(world_to_gs @ c2w)
    camera = Camera(0, w2c[:3, :3].T, w2c[:3, 3], math.radians(85),
                    2*math.atan(600/960*math.tan(math.radians(85)/2)),
                    torch.zeros(3, 600, 960, device='cuda'), None, name, 0)
    with torch.no_grad():
        rgb = render(camera, model, pipe, torch.zeros(3, device='cuda'))['render']
    rgb = (rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
    cv2.imwrite(str(OUT / f'{scene_id}_{name}.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
print(json.dumps(config), flush=True)
