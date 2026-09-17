import json
from pathlib import Path
import cv2
import numpy as np

root = Path('/work')
data = np.load(root / 'output/geometry.npz')
points, faces, route = data['points'], data['faces'], data['route']
triangles = points[faces]
floor_candidates = triangles[(triangles[:, :, 2].max(1) < .20) & (triangles[:, :, 2].min(1) > -.30)]
samples = np.concatenate([np.linspace(a, b, max(2, int(np.linalg.norm(b-a)/.05))) for a, b in zip(route[:-1], route[1:])])
heights = []
for sample in samples:
    a, b, c = floor_candidates[:, 0], floor_candidates[:, 1], floor_candidates[:, 2]
    v0, v1, v2 = b[:, :2]-a[:, :2], c[:, :2]-a[:, :2], sample-a[:, :2]
    det = v0[:, 0]*v1[:, 1]-v0[:, 1]*v1[:, 0]
    good = np.abs(det) > 1e-10
    u = np.zeros(len(det)); v = u.copy()
    u[good] = (v2[good, 0]*v1[good, 1]-v2[good, 1]*v1[good, 0])/det[good]
    v[good] = (v0[good, 0]*v2[good, 1]-v0[good, 1]*v2[good, 0])/det[good]
    inside = good & (u >= 0) & (v >= 0) & (u+v <= 1)
    z = a[:, 2]+u*(b[:, 2]-a[:, 2])+v*(c[:, 2]-a[:, 2])
    heights.append(float(z[inside].max()) if inside.any() else None)
valid = [h for h in heights if h is not None]
print(json.dumps({'samples': len(samples), 'floor_supported': len(valid),
                  'floor_range': [min(valid), max(valid)] if valid else None}), flush=True)
origin, resolution = data['origin'], float(data['resolution'])
obstacles = np.zeros_like(data['walkable'], np.uint8)


def clip(poly, height, keep_above):
    output = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        ia = a[2] >= height if keep_above else a[2] <= height
        ib = b[2] >= height if keep_above else b[2] <= height
        if ia:
            output.append(a)
        if ia != ib:
            output.append(a+(height-a[2])/(b[2]-a[2])*(b-a))
    return np.array(output)


for tri in triangles:
    if tri[:, 2].max() < .15 or tri[:, 2].min() > 1.50:
        continue
    poly = clip(tri, .15, True)
    if len(poly) < 3:
        continue
    poly = clip(poly, 1.50, False)
    if len(poly) < 3:
        continue
    pixels = np.rint((poly[:, :2]-origin)/resolution).astype(np.int32)
    cv2.fillPoly(obstacles, [pixels], 1)
from scipy.ndimage import distance_transform_edt
clearance = distance_transform_edt(1-obstacles)*resolution
cells = np.rint((samples-origin)/resolution).astype(int)
route_clearance = clearance[cells[:, 1], cells[:, 0]]
print('MESH_ROUTE_CLEARANCE', float(route_clearance.min()), flush=True)
np.savez(root / 'output/mesh_clearance.npz', obstacles=obstacles, clearance=clearance,
         sample_positions=samples, floor_heights=np.array([np.nan if h is None else h for h in heights]))
canvas = np.full((*obstacles.shape, 3), 35, np.uint8)
canvas[data['walkable'] > 0] = (160, 160, 160)
canvas[obstacles > 0] = (40, 40, 240)
cv2.polylines(canvas, [cells.reshape(-1, 1, 2)], False, (255, 200, 20), 1)
cv2.imwrite(str(root / 'output/mesh_map.png'), cv2.resize(np.flipud(canvas), None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
