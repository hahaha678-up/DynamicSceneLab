import argparse
import hashlib
import json
import struct
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import label


ROOT = Path('/work/crowdes-b')
parser = argparse.ArgumentParser()
parser.add_argument('--exact-navmesh', action='store_true')
args = parser.parse_args()
scene = json.loads(Path('/work/scene_config.json').read_text())
assert scene['scene'] == 'Habitat-GS scene55', scene['scene']
geometry = np.load('/work/output/geometry.npz')
clearance = np.load('/work/output/mesh_clearance.npz')['clearance']
origin = geometry['origin']
resolution = float(geometry['resolution'])
mask = (geometry['walkable'] > 0) & (clearance >= 0.23)
components, _ = label(mask)
start_cell = np.rint((np.array(scene['start']) - origin) / resolution).astype(int)
component = components[start_cell[1], start_cell[0]]
assert component > 0, 'Carter start is outside the pedestrian walkable component'
raw_area = float(mask.sum() * resolution ** 2)
mask &= components == component

nav_path = Path('/work/assets/scene55/scene55.navmesh')
nav = nav_path.read_bytes()
offset = nav.index(b'VAND')
header = struct.unpack_from('<2I4i9i3f6ff', nav, offset)
vertices = np.frombuffer(nav, '<f4', header[7] * 3, offset + 100).reshape(-1, 3).copy()
transform = np.asarray(scene['gs_to_world'])
world_vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
pixel_vertices = (world_vertices[:, :2] - origin) / resolution
polygons = []
poly_offset = offset + 100 + header[7] * 12
for index in range(header[6]):
    values = struct.unpack_from('<I6H6HHBB', nav, poly_offset + 32 * index)
    count, kind = values[-2:]
    if kind >> 6 != 0:
        continue
    indices = list(values[1:1 + count])
    xy = pixel_vertices[indices]
    area = np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - xy[:, 1] * np.roll(xy[:, 0], -1))
    # PathFinder embeds pixels as (u, 0, v), so upward faces need clockwise UV.
    polygons.append(indices[::-1] if area > 0 else indices)

exact_check = None
if args.exact_navmesh:
    from utils.navmesh import PathFinderNew

    pf = PathFinderNew([(float(x), 0., float(y)) for x, y in pixel_vertices], polygons)
    start = (float(start_cell[0]), 0., float(start_cell[1]))
    group = pf._navmesh.sample_polygon(start).get_group()
    before = int(mask.sum())
    # Rounded polygon rasterization can mark pixels outside the pathfinder's mesh.
    for row, col in np.argwhere(mask):
        node = pf._navmesh.sample_polygon((float(col), 0., float(row)))
        if node is None or node.get_group() != group:
            mask[row, col] = False
    components, _ = label(mask)
    mask &= components == components[start_cell[1], start_cell[0]]
    cells = np.argwhere(mask)
    probes = cells[np.random.default_rng(47).choice(len(cells), min(100, len(cells)), replace=False)]
    reachable = sum(bool(pf.search_path(start, (float(col), 0., float(row)))) for row, col in probes)
    assert reachable == len(probes), 'Exact-mask path connectivity check failed'
    exact_check = {'initial_pixels': before, 'retained_pixels': int(mask.sum()),
                   'reachable_probes': reachable, 'total_probes': len(probes),
                   'rule': 'Pixel centers inside exact navmesh and Carter connected group'}

# CrowdES uses a local metric frame. The Isaac translation is applied at export.
homography = np.diag([resolution, resolution, 1.0])
out = ROOT / 'inputs' / ('scene55_exact' if args.exact_navmesh else 'scene55')
out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / 'input.npz', walkable=mask.astype(np.uint8), H=homography,
                    world_origin=origin, mesh_clearance=clearance,
                    source_walkable=geometry['walkable'], route=np.asarray(scene['route']))
(out / 'navmesh.json').write_text(json.dumps({'vertices': pixel_vertices.tolist(), 'polygons': polygons}))
cv2.imwrite(str(out / 'walkable.png'), mask.astype(np.uint8) * 255)
route_cells = np.rint((np.asarray(scene['route']) - origin) / resolution).astype(np.int32)
canvas = np.full((*mask.shape, 3), 30, np.uint8)
canvas[geometry['walkable'] > 0] = (70, 70, 70)
canvas[mask] = (195, 195, 195)
cv2.polylines(canvas, [route_cells], False, (50, 100, 255), 3)
cv2.imwrite(str(out / 'overview.png'), np.flipud(canvas))
metadata = {
    'scene': scene['scene'], 'resolution_m': resolution, 'world_origin_xy': origin.tolist(),
    'pixel_to_isaac': 'world_xy = world_origin_xy + 0.05 * [u, v]',
    'row_axis': 'Stored image rows increase with Isaac world Y; overview.png alone is flipped for display.',
    'model_H': homography.tolist(), 'model_input': 'seven-class one-hot segmentation; RGB disabled by pretrained config',
    'walkable_area_before_component_m2': raw_area, 'walkable_area_m2': float(mask.sum() * resolution ** 2),
    'pedestrian_radius_m': 0.23, 'component_rule': 'connected traversable area containing Carter start',
    'manual_density_maps': False, 'navmesh_polygons': len(polygons), 'route': scene['route'],
    'exact_navmesh_check': exact_check,
    'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     [Path('/work/scene_config.json'), Path('/work/output/geometry.npz'),
                      Path('/work/output/mesh_clearance.npz'), nav_path]},
}
(out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
print(json.dumps(metadata), flush=True)
