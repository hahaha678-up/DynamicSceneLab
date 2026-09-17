import os
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

os.chdir('/work/LHM')
sys.path.insert(0, '/work/LHM')
data = np.load('/work/downloads/habitat_walk.npz')
assert str(data['smpl_type']) == 'smplx'
avatar = torch.load('/work/output/avatar_state.pt', map_location='cpu')
source_times = np.arange(len(data['body_pose']))/float(data['fps'])
target_times = np.arange(int(source_times[-1]*30)+1)/30
count = len(target_times)


def rotations(key):
    source = data[key].reshape(len(source_times),-1,3)
    result = np.stack([Slerp(source_times,Rotation.from_rotvec(source[:,j]))(target_times).as_rotvec() for j in range(source.shape[1])],axis=1)
    return result


params = {key:(value if key in ('betas','transform_mat_neutral_pose') else value[:,:1].repeat(1,count,*([1]*(value.ndim-2)))) for key,value in avatar['params'].items()}
initial_heading = Rotation.from_rotvec(data['global_orient'][0]).inv()
to_cv = Rotation.from_euler('x',180,degrees=True)*initial_heading
root = (to_cv*Rotation.from_rotvec(rotations('global_orient')[:,0])).as_rotvec()
trans = np.stack([np.interp(target_times,source_times,data['transl'][:,j]) for j in range(3)],axis=1)
trans = to_cv.apply(trans-data['transl'][0])
params['root_pose'] = torch.tensor(root[None],dtype=torch.float32)
params['trans'] = torch.tensor(trans[None],dtype=torch.float32)
# Keep the target avatar's shape and bind transform; source skinning matrices belong to another body.
for source_key,target_key in [('body_pose','body_pose'),('left_hand_pose','lhand_pose'),('right_hand_pose','rhand_pose')]:
    params[target_key] = torch.tensor(rotations(source_key)[None],dtype=torch.float32)
avatar['params'] = params
torch.save(avatar, '/work/output/habitat_state.pt')
print('HABITAT_NATIVE_SMPLX_READY',count,'frames',float(source_times[-1]),'seconds',flush=True)
