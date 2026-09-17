import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


class Track:
    def __init__(self, spec):
        self.spec = spec
        self.source_hash = None
        if 'csv' in spec:
            path = Path(spec['csv'])
            self.source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            rows = list(csv.DictReader(path.open()))
            values = [[float(r[k]) for k in ('time_s', 'x', 'y')] for r in rows]
        elif 'episode' in spec:
            path = Path(spec['episode']) / 'trajectory.jsonl'
            self.source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            values = [[r['timestamp'], r['person']['x'], r['person']['y']]
                      for r in rows if r['person']['present']]
        else:
            values = spec['points']
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError('Track needs at least two finite [time_s, x_m, y_m] samples')
        if np.any(np.diff(values[:, 0]) <= 0):
            raise ValueError('Track timestamps must increase')
        shift, scale = spec.get('time_shift_s', 0.), spec.get('speed_scale', 1.)
        if not math.isfinite(shift) or not math.isfinite(scale) or scale <= 0:
            raise ValueError('Invalid timing transform')
        self.t = values[0, 0] + shift + (values[:, 0] - values[0, 0]) / scale
        self.xy = values[:, 1:]
        if self.t[0] < 0:
            raise ValueError('Person cannot spawn before episode zero')
        self.velocity = np.diff(self.xy, axis=0) / np.diff(self.t)[:, None]
        if np.linalg.norm(self.velocity, axis=1).max() > 3.:
            raise ValueError('Demo walking speed exceeds 3 m/s')

    def state(self, t):
        absent = {'present': False, 'x': None, 'y': None, 'heading': None,
                  'speed': None, 'vx': None, 'vy': None, 'arrived': bool(t >= self.t[-1])}
        if t < self.t[0] - 1e-7 or (t > self.t[-1] + 1e-7 and self.spec.get('end', 'hold') == 'hide'):
            return absent
        i = int(np.clip(np.searchsorted(self.t, t, side='right') - 1, 0, len(self.t) - 2))
        f = float(np.clip((t - self.t[i]) / (self.t[i + 1] - self.t[i]), 0, 1))
        xy = self.xy[i] * (1 - f) + self.xy[i + 1] * f
        velocity = self.velocity[i] if t < self.t[-1] else np.zeros(2)
        direction = self.velocity[i]
        return {'present': True, 'x': float(xy[0]), 'y': float(xy[1]),
                'heading': math.atan2(direction[1], direction[0]),
                'speed': float(np.linalg.norm(velocity)), 'vx': float(velocity[0]),
                'vy': float(velocity[1]), 'arrived': bool(t >= self.t[-1] - 1e-7)}

    def resolved(self):
        v = self.velocity[0]
        return {'start': self.xy[0].tolist(), 'goal': self.xy[-1].tolist(),
                'spawn_time': float(self.t[0]), 'end_time': float(self.t[-1]),
                'speed': float(np.linalg.norm(self.velocity, axis=1).max()),
                'radius': .23, 'yaw': math.atan2(v[1], v[0]), 'source_sha256': self.source_hash}

    def sampled_path(self, spacing=.025):
        return np.concatenate([np.linspace(a, b, max(2, math.ceil(np.linalg.norm(b-a) / spacing) + 1))
                               for a, b in zip(self.xy, self.xy[1:])])
