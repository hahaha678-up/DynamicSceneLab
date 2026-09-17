import os
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['PIP_NO_INDEX'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

import json
import argparse
import socket
import struct
import sys
import time
from pathlib import Path

ROOT = Path('/work')
os.chdir(ROOT / 'LHM')
sys.path.insert(0, str(ROOT / 'LHM'))
import torch
from accelerate import Accelerator
from safetensors import safe_open
from LHM.models.rendering.gs_renderer import GS3DRenderer

Accelerator()
torch.set_num_threads(8)
torch._dynamo.config.disable = True


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--walking', action='store_true')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--habitat', action='store_true')
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    config = json.loads((ROOT / 'downloads/config.json').read_text())
    renderer = GS3DRenderer(
        human_model_path=config['human_model_path'], subdivide_num=config['smplx_subdivide_num'],
        smpl_type=config['smplx_type'], feat_dim=config['transformer_dim'], query_dim=config['gs_query_dim'],
        use_rgb=config['gs_use_rgb'], sh_degree=config['gs_sh'], xyz_offset_max_step=config['gs_xyz_offset_max_step'],
        mlp_network_config=config['gs_mlp_network_config'], expr_param_dim=config['expr_param_dim'],
        shape_param_dim=config['shape_param_dim'], clip_scaling=config['gs_clip_scaling'],
        cano_pose_type=config['cano_pose_type'], skip_decoder=True, dense_sample_pts=config['dense_sample_pts'])
    with safe_open(str(ROOT / 'downloads/model.safetensors'), framework='pt', device='cpu') as checkpoint:
        weights = {key[len('renderer.'):]:checkpoint.get_tensor(key) for key in checkpoint.keys() if key.startswith('renderer.')}
    renderer.load_state_dict(weights, strict=True)
    renderer.cuda().eval()
    state_name = 'social_state.pt'
    avatar = torch.load(ROOT / 'output' / state_name, map_location='cuda')
    params = avatar['params']
    idle = torch.load(ROOT / 'output/habitat_state.pt', map_location='cuda')['params']
    idle_frame = renderer.get_single_view_smpl_data(idle, 0)
    count = params['body_pose'].shape[1]
    if args.preview:
        import numpy as np
        import imageio.v2 as imageio
        from LHM.models.rendering.gs_renderer import Camera
        renderer.device = torch.device('cuda')
        target = ROOT / 'output/habitat_walk_preview.mp4'
        with imageio.get_writer(target,fps=30,codec='libx264',quality=9,macro_block_size=1) as writer:
            for index in range(count):
                frame = renderer.get_single_view_smpl_data(params,index)
                animated,_ = renderer.animate_gs_model(avatar['attrs'][0],avatar['query'][0],renderer.get_single_batch_smpl_data(frame,0),debug=False)
                gs = animated[0]
                center = params['trans'][0,index].cpu().numpy()
                images = []
                for offset in ([.8,-.1,-3.5],[3.5,-.1,-.8]):
                    eye = center+offset
                    forward = center-eye; forward /= np.linalg.norm(forward)
                    right = np.cross(forward,[0.,-1.,0.]); right /= np.linalg.norm(right)
                    down = np.cross(forward,right)
                    c2w = np.eye(4); c2w[:3,:3] = np.column_stack([right,down,forward]); c2w[:3,3] = eye
                    intr = np.eye(4); intr[0,0]=intr[1,1]=850; intr[0,2]=192; intr[1,2]=320
                    camera = Camera.from_c2w(torch.tensor(c2w,dtype=torch.float32,device='cuda'),torch.tensor(intr,dtype=torch.float32,device='cuda'),640,384)
                    result = renderer.forward_single_view(gs,camera,torch.full((3,),.85,device='cuda'))
                    images.append((result['comp_rgb'].clamp(0,1).cpu().numpy()*255).astype(np.uint8))
                writer.append_data(np.concatenate(images,axis=1))
        print('HABITAT_PREVIEW_READY',target,flush=True)
        return
    if args.export:
        import numpy as np
        if args.walking:
            p = params
            joints = renderer.smplx_model.smplx_layer(
                global_orient=torch.zeros(count,3,device='cuda'),
                body_pose=p['body_pose'][0].reshape(count,-1),
                left_hand_pose=p['lhand_pose'][0].reshape(count,-1),
                right_hand_pose=p['rhand_pose'][0].reshape(count,-1),
                jaw_pose=p['jaw_pose'][0], leye_pose=p['leye_pose'][0], reye_pose=p['reye_pose'][0],
                expression=p['expr'][0], betas=p['betas'].expand(count,-1),
                face_offset=None,joint_offset=None).joints[:100].cpu().numpy()
            left = joints[:,20,2]-joints[:,16,2]
            right = joints[:,21,2]-joints[:,17,2]
            left_leg = joints[:,7,2]-joints[:,0,2]
            alternating = np.corrcoef(left,right)[0,1]
            contralateral = np.corrcoef(right,left_leg)[0,1]
            print('ARM_PHASE_CHECK', {'left_right_correlation':float(alternating), 'right_arm_left_leg_correlation':float(contralateral), 'wrist_fore_aft_ranges_m':[float(np.ptp(left)),float(np.ptp(right))]}, flush=True)
            assert alternating < -.85 and contralateral > .5, 'Arm swing does not match contralateral walking phase'
        frames = []
        for index in range(1):
            frame = renderer.get_single_view_smpl_data(params, index)
            animated, _ = renderer.animate_gs_model(avatar['attrs'][0], avatar['query'][0], renderer.get_single_batch_smpl_data(frame, 0), debug=False)
            gs = animated[0]
            frames.append(torch.cat((gs.xyz, gs.rotation, gs.scaling, gs.shs.reshape(-1,3), gs.opacity), dim=1).float().cpu().numpy())
        output_name = 'social_reference.npy'
        np.save(ROOT / 'output' / output_name, np.stack(frames))
        print('EXPORTED_MOTION', count, flush=True)
        return
    endpoint = ROOT / 'output/social_avatar.sock'
    assert not endpoint.exists(), f'Socket already exists: {endpoint}'
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    served, times = 0, []
    try:
        server.bind(str(endpoint))
        server.listen(1)
        print(f'AVATAR_SERVICE_READY frames={count} gpu_memory_gib={torch.cuda.memory_allocated()/2**30:.3f}', flush=True)
        connection, _ = server.accept()
        with connection:
            while True:
                request = bytearray()
                while len(request) < 4:
                    chunk = connection.recv(4-len(request))
                    if not chunk:
                        if request:
                            raise ConnectionError('Incomplete frame request')
                        return
                    request.extend(chunk)
                encoded = struct.unpack('<I', request)[0]
                frame_index, blend_level = encoded % count, encoded // count
                if not 0 <= blend_level <= 100:
                    raise ValueError(f'Frame {frame_index} outside motion length {count}')
                blend = blend_level/100.
                start = time.perf_counter()
                frame = renderer.get_single_view_smpl_data(params, frame_index)
                for key in ('body_pose','lhand_pose','rhand_pose','trans'):
                    frame[key] = idle_frame[key]*(1.-blend) + frame[key]*blend
                animated, _ = renderer.animate_gs_model(avatar['attrs'][0], avatar['query'][0], renderer.get_single_batch_smpl_data(frame, 0), debug=False)
                gs = animated[0]
                # Activated values, wxyz rotations, float32; the client converts to its renderer's representation.
                values = torch.cat((gs.xyz, gs.rotation, gs.scaling, gs.shs.reshape(-1,3), gs.opacity), dim=1)
                assert torch.isfinite(values).all()
                payload = values.float().cpu().numpy().tobytes()
                connection.sendall(struct.pack('<I', len(payload)) + payload)
                times.append(time.perf_counter()-start)
                served += 1
                if served % 100 == 0:
                    print(f'AVATAR_SERVICE frames={served} mean_ms={1000*sum(times)/len(times):.1f}', flush=True)
    finally:
        server.close()
        if endpoint.is_socket():
            endpoint.unlink()
        (ROOT / 'output/social_avatar_service_result.json').write_text(json.dumps({'frames':served, 'mean_seconds':sum(times)/len(times) if times else None}))
        print(f'AVATAR_SERVICE_CLOSED frames={served}', flush=True)


if __name__ == '__main__':
    main()
