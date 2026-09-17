import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path('/work/crowdes-b')
POLICY = {
    'name': 'clean_walking_v1', 'target': 40,
    'minimum_duration_s': 5., 'minimum_path_m': 2.,
    'minimum_net_displacement_m': 1., 'minimum_net_to_path_ratio': .2,
    'maximum_speed_mps': 3., 'p95_speed_mps': 2.2,
    'maximum_acceleration_mps2': 8., 'p95_acceleration_mps2': 4.,
    'maximum_fraction_below_0_1_mps': .6,
    'stalled_window_s': 10., 'stalled_window_path_m': 1.,
    'stalled_window_net_m': .25,
    'human_radius_m': .23, 'minimum_raster_clearance_m': .28,
    'sample_spacing_m': .025, 'navmesh_numeric_tolerance_m': 1e-7,
    'duplicate_path_distance_m': .5,
    'trajectory_edits_allowed': False,
    'isaac_replay_verified': False,
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


class MeshCoverage:
    def __init__(self, vertices, polygons):
        self.faces = []
        for indices in polygons:
            face = np.asarray(vertices)[indices]
            area = np.sum(face[:, 0] * np.roll(face[:, 1], -1) - face[:, 1] * np.roll(face[:, 0], -1))
            if area > 0:
                face = face[::-1]
            edges = np.roll(face, -1, axis=0) - face
            norms = np.linalg.norm(edges, axis=1)
            keep = norms > 1e-12
            self.faces.append((face[keep], edges[keep] / norms[keep, None]))
        self.lower = np.array([p.min(axis=0) for p, _ in self.faces])
        self.upper = np.array([p.max(axis=0) for p, _ in self.faces])

    def covers_segment(self, start, finish):
        eps = POLICY['navmesh_numeric_tolerance_m']
        start, finish = np.asarray(start), np.asarray(finish)
        direction = finish - start
        candidates = np.flatnonzero(np.all(self.upper + eps >= np.minimum(start, finish), axis=1)
                                    & np.all(self.lower - eps <= np.maximum(start, finish), axis=1))
        intervals = []
        # Clip against every convex face, then cover the entire segment with their union.
        for index in candidates:
            points, edges = self.faces[index]
            offset = start - points
            a = edges[:, 0] * offset[:, 1] - edges[:, 1] * offset[:, 0]
            slope = edges[:, 0] * direction[1] - edges[:, 1] * direction[0]
            parallel = np.abs(slope) < 1e-12
            if np.any(parallel & (a > eps)):
                continue
            positive, negative = slope > 1e-12, slope < -1e-12
            lo = max(0., float(np.max((eps-a[negative])/slope[negative])) if negative.any() else 0.)
            hi = min(1., float(np.min((eps-a[positive])/slope[positive])) if positive.any() else 1.)
            if lo <= hi:
                intervals.append((lo, hi))
        reach = 0.
        for lo, hi in sorted(intervals):
            if lo > reach + 1e-9:
                return False
            reach = max(reach, hi)
            if reach >= 1. - 1e-9:
                return True
        return False


def self_test():
    v = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [2, 0], [2, 1], [0, 2], [1, 2]], float)
    mesh = MeshCoverage(v, [[0, 1, 2, 3], [3, 2, 5, 4], [1, 6, 7, 2]])
    assert mesh.covers_segment([.1, .5], [1.9, .5])
    assert not mesh.covers_segment([.8, 1.8], [1.8, .8])
    assert mesh.covers_segment([.5, .5], [.5, .5])
    assert not mesh.covers_segment([3., 3.], [3., 3.])
    separate = MeshCoverage(np.array([[0, 0], [0, 1], [1, 1], [1, 0],
                                      [1.0001, 0], [1.0001, 1], [2, 1], [2, 0]]),
                            [[0, 1, 2, 3], [4, 5, 6, 7]])
    assert not separate.covers_segment([.5, .5], [1.5, .5])
    data = {'H': np.diag([.05, .05, 1.]), 'world_origin': np.zeros(2),
            'walkable': np.ones((160, 160)), 'source_walkable': np.ones((160, 160)),
            'mesh_clearance': np.ones((160, 160))}
    room = MeshCoverage(np.array([[0, 0], [0, 8], [8, 8], [8, 0]]), [[0, 1, 2, 3]])
    frames = np.arange(251)
    walking = pd.DataFrame({'frame': frames, 'time_s': frames/25., 'x': 1+frames/50., 'y': 1.})
    pixels = walking.copy()
    pixels[['x', 'y']] /= .05
    assert evaluate(walking, pixels, data, room)['valid']
    static = walking.copy()
    static['x'] = 1.
    static_pixels = static.copy()
    static_pixels[['x', 'y']] /= .05
    assert not evaluate(static, static_pixels, data, room)['valid']
    broken = walking.copy()
    broken.loc[100, 'time_s'] = broken.loc[99, 'time_s']
    assert evaluate(broken, pixels, data, room)['reasons'] == ['invalid_samples']
    data['mesh_clearance'][:, 60:70] = .1
    assert 'insufficient_body_clearance' in evaluate(walking, pixels, data, room)['reasons']
    data['mesh_clearance'][:] = 1.
    jump = walking.copy()
    jump.loc[100, 'x'] += .1
    jump_pixels = jump.copy()
    jump_pixels[['x', 'y']] /= .05
    assert 'abrupt_velocity_change' in evaluate(jump, jump_pixels, data, room)['reasons']
    print('PASS: geometry seams/gaps/corners, valid walking, stationary, timestamps, body clearance and velocity discontinuity')


def evaluate(traj, pixel, data, mesh):
    xy, times = traj[['x', 'y']].to_numpy(), traj.time_s.to_numpy()
    reasons = []
    if len(xy) < 2 or not np.isfinite(np.c_[xy, times]).all() or not (np.diff(times) > 0).all():
        return {'reasons': ['invalid_samples'], 'valid': False}
    if not np.array_equal(traj.frame.to_numpy(), pixel.frame.to_numpy()):
        return {'reasons': ['pixel_world_frames_mismatch'], 'valid': False}
    mapped = pixel[['x', 'y']].to_numpy() * data['H'][0, 0] + data['world_origin']
    if not np.allclose(xy, mapped, atol=1e-7, rtol=0) or not np.allclose(times, traj.frame/25., atol=1e-9, rtol=0):
        reasons.append('coordinate_time_mapping')
    if not np.allclose(np.diff(times), .04, atol=1e-8, rtol=0):
        reasons.append('missing_frames')
    delta, dt = np.diff(xy, axis=0), np.diff(times)
    ds = np.linalg.norm(delta, axis=1)
    speed = ds / dt
    velocity = delta / dt[:, None]
    accel = np.linalg.norm(np.diff(velocity, axis=0), axis=1) / ((dt[:-1]+dt[1:])/2)
    duration, length = float(times[-1]-times[0]), float(ds.sum())
    net = float(np.linalg.norm(xy[-1]-xy[0]))
    fraction_slow = float(np.mean(speed < .1))
    if duration < POLICY['minimum_duration_s']: reasons.append('short_duration')
    if length < POLICY['minimum_path_m']: reasons.append('little_motion')
    if net < POLICY['minimum_net_displacement_m'] or net/max(length, 1e-9) < POLICY['minimum_net_to_path_ratio']:
        reasons.append('poor_progress')
    if speed.max() > POLICY['maximum_speed_mps'] or np.quantile(speed, .95) > POLICY['p95_speed_mps']:
        reasons.append('implausible_speed')
    if len(accel) and (accel.max() > POLICY['maximum_acceleration_mps2'] or np.quantile(accel, .95) > POLICY['p95_acceleration_mps2']):
        reasons.append('abrupt_velocity_change')
    if fraction_slow > POLICY['maximum_fraction_below_0_1_mps']: reasons.append('mostly_stationary')
    cumulative = np.r_[0., ds.cumsum()]
    begin = np.flatnonzero(times <= times[-1]-POLICY['stalled_window_s'])
    end = np.searchsorted(times, times[begin]+POLICY['stalled_window_s'])
    stalled = ((cumulative[end]-cumulative[begin] < POLICY['stalled_window_path_m'])
               & (np.linalg.norm(xy[end]-xy[begin], axis=1) < POLICY['stalled_window_net_m']))
    if stalled.any(): reasons.append('stalled_window')
    dense = np.concatenate([np.linspace(a, b, max(2, int(math.ceil(np.linalg.norm(b-a)/POLICY['sample_spacing_m']))+1))
                            for a, b in zip(xy[:-1], xy[1:])])
    cells = np.rint((dense-data['world_origin']) / data['H'][0, 0]).astype(int)
    h, w = data['walkable'].shape
    inside = (cells[:, 0]>=0) & (cells[:, 0]<w) & (cells[:, 1]>=0) & (cells[:, 1]<h)
    cells = np.clip(cells, [0, 0], [w-1, h-1])
    clear = data['mesh_clearance'][cells[:, 1], cells[:, 0]]
    if not (inside & (clear >= POLICY['minimum_raster_clearance_m'])).all():
        reasons.append('insufficient_body_clearance')
    if not (inside & (data['source_walkable'][cells[:, 1], cells[:, 0]] > 0)).all():
        reasons.append('outside_navmesh_raster')
    exact_ok = all(mesh.covers_segment(a, b) for a, b in zip(xy[:-1], xy[1:]))
    if not exact_ok: reasons.append('outside_exact_navmesh')
    unique = np.r_[True, np.diff(cumulative) > 1e-9]
    samples = np.linspace(0., length, 32)
    descriptor = np.column_stack([np.interp(samples, cumulative[unique], xy[unique, axis]) for axis in [0, 1]])
    return {'valid': not reasons, 'reasons': reasons, 'duration_s': duration,
            'length_m': length, 'net_displacement_m': net,
            'maximum_speed_mps': float(speed.max()), 'p95_speed_mps': float(np.quantile(speed, .95)),
            'maximum_acceleration_mps2': float(accel.max()) if len(accel) else 0.,
            'p95_acceleration_mps2': float(np.quantile(accel, .95)) if len(accel) else 0.,
            'fraction_slow': fraction_slow, 'minimum_clearance_m': float(clear.min()),
            'exact_navmesh_all_segments': exact_ok, 'descriptor': descriptor.tolist(),
            'start_xy': xy[0].tolist(), 'end_xy': xy[-1].tolist(),
            'start_time_s': float(times[0]), 'end_time_s': float(times[-1])}


def original_rows(source, agent_id):
    lines = source.read_bytes().splitlines(keepends=True)
    header = next(csv.reader([lines[0].decode()]))
    column = header.index('agent_id')
    return lines[0] + b''.join(line for line in lines[1:]
                              if int(float(next(csv.reader([line.decode()]))[column])) == agent_id)


def main():
    inputs = ROOT / 'inputs/scene55_exact'
    data = np.load(inputs/'input.npz')
    nav = json.loads((inputs/'navmesh.json').read_text())
    mesh = MeshCoverage(np.asarray(nav['vertices'])*data['H'][0, 0]+data['world_origin'], nav['polygons'])
    input_hashes = {name: digest((inputs/name).read_bytes()) for name in ['input.npz', 'navmesh.json']}
    cache_path = ROOT/'runs/clean40_audit.json'
    fingerprint = digest(Path(__file__).read_bytes()+json.dumps(input_hashes, sort_keys=True).encode())
    prior = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    cache = prior.get('records', {}) if prior.get('validator_fingerprint') == fingerprint else {}
    records = []
    completed_seeds = set()
    for run in sorted((ROOT/'runs').glob('scene55_exact_*')):
        if not run.is_dir() or not (run/'status.json').exists(): continue
        provenance = json.loads((run/'provenance.json').read_text())
        if provenance['scene'] != 'scene55_exact': continue
        assert all(provenance.get('input_sha256', {}).get(k) == v for k, v in input_hashes.items()), 'Input changed across runs'
        state = json.loads((run/'status.json').read_text())
        for result in state['results']:
            if result['status'] != 'ok': continue
            seed = result['seed']
            assert seed not in completed_seeds, f'Duplicate seed {seed}'
            completed_seeds.add(seed)
            source = run/f'seed_{seed:03d}_world.csv'
            pixels = run/f'seed_{seed:03d}_pixels.csv'
            source_hash, pixel_hash = digest(source.read_bytes()), digest(pixels.read_bytes())
            frame, pixel = pd.read_csv(source), pd.read_csv(pixels)
            for aid, traj in frame.groupby('agent_id'):
                key = f'{run.name}/seed_{seed:03d}_agent_{int(aid):03d}'
                record = cache.get(key)
                if record is None or record['source_sha256'] != source_hash or record['pixel_sha256'] != pixel_hash:
                    record = evaluate(traj, pixel[pixel.agent_id == aid], data, mesh)
                    if result.get('emission_path_failures', 0):
                        record['reasons'].append('emission_path_failure')
                        record['valid'] = False
                    record.update(id=key, seed=seed, agent_id=int(aid), source=str(source),
                                  source_sha256=source_hash, pixel_sha256=pixel_hash)
                cache[key] = record
                records.append(record)
    records.sort(key=lambda r: (r['seed'], r['agent_id']))
    chosen = []
    duplicate_count = 0
    for record in records:
        if not record['valid']: continue
        desc = np.asarray(record['descriptor'])
        distance = [min(np.linalg.norm(desc-np.asarray(other['descriptor']), axis=1).mean(),
                        np.linalg.norm(desc-np.asarray(other['descriptor'])[::-1], axis=1).mean()) for other in chosen]
        if distance and min(distance) < POLICY['duplicate_path_distance_m']:
            duplicate_count += 1
            continue
        chosen.append(record)
    output = ROOT/'datasets/clean40_v1'
    output.mkdir(parents=True, exist_ok=True)
    accepted = []
    for index, record in enumerate(chosen[:POLICY['target']], 1):
        raw = original_rows(Path(record['source']), record['agent_id'])
        name = f'person_{index:03d}.csv'
        target = output/name
        if target.exists():
            assert target.read_bytes() == raw, 'Published candidate order changed'
        else:
            target.write_bytes(raw)
        accepted.append({**{k: v for k, v in record.items() if k != 'descriptor'},
                         'file': name, 'sha256': digest(raw)})
    summary = {'target': POLICY['target'], 'accepted': len(accepted),
               'stage': 'ready_for_review' if len(accepted) == POLICY['target'] else 'collecting',
               'completed_seeds': sorted(completed_seeds), 'candidate_count': len(records),
               'strict_valid_count': sum(r['valid'] for r in records),
               'distinct_valid_count': len(chosen), 'duplicate_count': duplicate_count,
               'rejections': dict(Counter(reason for record in records for reason in record['reasons']))}
    manifest = {**summary, 'policy': POLICY, 'input_sha256': input_hashes,
                'validator_sha256': digest(Path(__file__).read_bytes()),
                'coordinates': {'x_y_units': 'Isaac world meters', 'time_s': 'original frame / 25',
                                'world_origin_xy': data['world_origin'].tolist(), 'meters_per_pixel': .05},
                'samples': accepted, 'isaac_replay_verified': False}
    write_json(output/'manifest.json', manifest)
    write_json(cache_path, {**summary, 'validator_fingerprint': fingerprint, 'records': cache})
    write_json(ROOT/'runs/clean40_progress.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    self_test() if args.self_test else main()
