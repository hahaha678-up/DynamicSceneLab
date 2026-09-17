import numpy as np
import torch
import os
import sys
os.chdir('/work/LHM')
sys.path.insert(0, '/work/LHM')
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d

source = np.load('/work/downloads/walking.npz')
adapted_poses = source['pose_aa'].astype(np.float64).copy()
raw = adapted_poses.reshape(-1,24,3).copy()
phase = gaussian_filter1d(raw[:,2,0]-raw[:,1,0], sigma=2.)
phase /= np.quantile(np.abs(phase[600:816]), .98)
phase = np.clip(phase,-1.,1.)
spine = Rotation.from_rotvec(raw[:,3])*Rotation.from_rotvec(raw[:,6])*Rotation.from_rotvec(raw[:,9])
for shoulder, collar, sign in ((16,13,1.),(17,14,-1.)):
    parent = spine*Rotation.from_rotvec(raw[:,collar])
    neutral = Rotation.from_rotvec(np.median(raw[600:800,shoulder], axis=0))
    # Apply opposite swings about the torso's lateral axis, compensating the collar frame.
    swing = Rotation.from_euler('x', sign*30.*phase, degrees=True)
    adjusted = parent.inv()*swing*parent*neutral
    adapted_poses[:,shoulder*3:shoulder*3+3] = adjusted.as_rotvec()
for elbow, sign in ((18,-1.),(19,1.)):
    base = np.median(raw[600:800,elbow], axis=0)
    values = np.repeat(base[None],len(raw),axis=0)
    values[:,1] = np.deg2rad(sign*(30.+5.*sign*phase))
    adapted_poses[:,elbow*3:elbow*3+3] = values
avatar = torch.load('/work/output/avatar_state.pt', map_location='cpu')
fps = 30
times = np.arange(130) / fps
u = np.clip((times-3.7)/.6, 0, 1)
progress = np.where(times < 3.7, times, 3.7 + .6*(u-.5*u*u))
indices = 600 + progress * float(source['fps'])
lower = np.floor(indices).astype(int)
upper = lower+1
ratio = (indices-lower)[:,None]
poses = adapted_poses[lower]*(1-ratio)+adapted_poses[upper]*ratio
translation = source['transl'][lower]*(1-ratio)+source['transl'][upper]*ratio
blend = (u*u*(3-2*u))[:,None]
# SMPL and SMPL-X share the first 21 body joints; keep this avatar's face and hands.
idle = np.median(adapted_poses[600:800], axis=0)
for joint in (1,2,4,5,7,8,10,11):
    idle[joint*3:joint*3+3] = 0
poses[:,3:66] = poses[:,3:66]*(1-blend)+idle[3:66]*blend
cv_rotation = np.diag([1.,-1.,-1.])
root = Rotation.from_matrix(cv_rotation @ Rotation.from_rotvec(poses[:,:3]).as_matrix()).as_rotvec()
translation = (translation-translation[0]) @ cv_rotation.T
params = {key:(value if key in ('betas','transform_mat_neutral_pose') else value[:,:1].repeat(1,len(times),*([1]*(value.ndim-2)))) for key,value in avatar['params'].items()}
params['root_pose'] = torch.tensor(root[None], dtype=torch.float32)
params['body_pose'] = torch.tensor(poses[:,3:66].reshape(1,-1,21,3), dtype=torch.float32)
params['trans'] = torch.tensor(translation[None], dtype=torch.float32)
avatar['params'] = params
torch.save(avatar, '/work/output/walking_state.pt')
print('WALK_READY',len(times),'frames', 'translation_cv',translation[-1].tolist(), flush=True)
