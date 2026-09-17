import json
import os
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

os.chdir('/work/LHM')
sys.path.insert(0, '/work/LHM')
avatar = torch.load('/work/output/habitat_state.pt', map_location='cpu')
source = avatar['params']
start, end, count = 40, 69, 120
times = np.linspace(start, end, count, endpoint=False)
result = {}
for key, values in source.items():
    if key in ('betas', 'transform_mat_neutral_pose'):
        result[key] = values
        continue
    data = values[0].numpy()
    if key.endswith('pose'):
        shape = data.shape[1:]
        sequence = data[start:end+1].reshape(end-start+1, -1, 3).copy()
        # Close a single gait cycle before resampling, so repeated steps have no pose jump.
        u = np.linspace(0, 1, len(sequence))
        sequence -= (3*u*u-2*u*u*u)[:, None, None]*(sequence[-1]-sequence[0])
        output = np.stack([Slerp(np.arange(start, end+1), Rotation.from_rotvec(sequence[:, j]))(times).as_rotvec()
                           for j in range(sequence.shape[1])], axis=1).reshape(count, *shape)
    elif key == 'trans':
        # HuNav owns planar locomotion. Retain only the source cycle's small vertical variation.
        output = np.zeros((count, 3))
        y = data[start:end+1, 1].copy()
        y -= np.linspace(y[0], y[-1], len(y))
        output[:, 1] = np.interp(times, np.arange(start, end+1), y)
    else:
        output = np.repeat(data[:1], count, axis=0)
    result[key] = torch.tensor(output[None], dtype=values.dtype)
avatar['params'] = result
torch.save(avatar, '/work/output/social_state.pt')
stride = float(np.linalg.norm(np.diff(source['trans'][0, start:end+1, [0, 2]].numpy(), axis=0), axis=1).sum())
metadata = {'cycle_frames': count, 'source_interval': [start, end], 'stride_m': stride,
            'source': 'Habitat-GS supplied SMPL-X walk', 'root_translation': 'HuNavSim'}
open('/work/output/social_walk.json', 'w').write(json.dumps(metadata, indent=2))
print('SOCIAL_WALK_READY', metadata, flush=True)
