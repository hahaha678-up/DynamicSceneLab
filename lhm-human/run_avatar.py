import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['PIP_NO_INDEX'] = '1'

import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path('/work')
SOURCE = ROOT / 'LHM'
os.chdir(SOURCE)
sys.path.insert(0, str(SOURCE))

import cv2
import numpy as np
import torch
from PIL import Image
from accelerate import Accelerator
from safetensors.torch import load_file

Accelerator()
torch._dynamo.config.disable = True
torch.set_num_threads(8)

import gfpgan.utils
gfpgan.utils.ROOT_DIR = str(SOURCE)
from LHM.models import ModelHumanLRMSapdinoBodyHeadSD3_5
from LHM.models.encoders.dinov2_fusion_wrapper import Dinov2FusionWrapper
import importlib.util

motion_spec = importlib.util.spec_from_file_location('lhm_motion_utils', SOURCE / 'LHM/runners/infer/utils.py')
motion_utils = importlib.util.module_from_spec(motion_spec)
motion_spec.loader.exec_module(motion_utils)
for function_name in ('prepare_motion_seqs', 'calc_new_tgt_size_by_aspect', 'center_crop_according_to_mask', 'resize_image_keepaspect_np', 'scale_intrs'):
    globals()[function_name] = getattr(motion_utils, function_name)
from engine.SegmentAPI.base import Bbox
import ast

# Reuse the official preprocessing functions without importing its automatic installer.
preprocess_source = ast.parse((SOURCE / 'LHM/runners/infer/human_lrm.py').read_text())
preprocess_functions = [node for node in preprocess_source.body if isinstance(node, ast.FunctionDef) and node.name in ('get_bbox', 'infer_preprocess_image')]
exec(compile(ast.Module(body=preprocess_functions, type_ignores=[]), '<LHM official preprocessing>', 'exec'))
from LHM.utils.face_detector import FaceDetector
from engine.pose_estimation.pose_estimator import PoseEstimator


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=90)
    args = parser.parse_args()
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    image_path = SOURCE / 'train_data/example_imgs/video_image_20240913__-videos_clips__-data__-326594731337_0.png'
    rgb = np.array(Image.open(image_path).convert('RGB'))
    log('Estimating source body shape and face crop')
    pose = PoseEstimator(str(SOURCE / 'pretrained_models/human_model_files'))
    body = pose(str(image_path))
    betas = np.asarray(body.beta, dtype=np.float32).reshape(1, -1)
    log(f'Body shape {betas.shape}, full-body ratio={body.ratio}')
    del pose
    gc.collect()
    torch.cuda.empty_cache()
    detector = FaceDetector(str(SOURCE / 'pretrained_models/gagatracker/vgghead/vgg_heads_l.trcd'), 'cuda')
    bbox = detector(torch.from_numpy(rgb.copy()).permute(2, 0, 1))
    x0, y0, x1, y1 = [int(x) for x in bbox]
    head = cv2.resize(rgb[max(0,y0):y1, max(0,x0):x1], (112, 112))
    Image.fromarray(head).save(out / 'source_face.png')
    del detector
    gc.collect()
    torch.cuda.empty_cache()

    # This official sample has a white background; retain its largest foreground component.
    mask = (rgb.min(axis=2) < 240).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    assert count > 1, 'No foreground in the selected source image'
    mask = (labels == (1 + stats[1:, cv2.CC_STAT_AREA].argmax())).astype(np.uint8) * 255
    image, _, _ = infer_preprocess_image(str(image_path), mask, None, 0, 1., 896, 5/3, [1.,1.], 1024, 14)
    Image.fromarray((image[0].permute(1,2,0).numpy()*255).astype(np.uint8)).save(out / 'source_preprocessed.png')
    head = torch.from_numpy(head.copy()).permute(2,0,1).float()[None,None] / 255.
    motions = prepare_motion_seqs(str(SOURCE / 'train_data/motion_video/mimo6/smplx_params'), None, str(out), 30, 1., 5/3, [1.,1.,0], 512, False, motion_size=args.frames)
    params = {k:v.cuda() for k,v in motions['smplx_params'].items()}
    params['betas'] = torch.from_numpy(betas).cuda()

    # The LHM checkpoint includes DINO weights; prevent a redundant network download.
    original_builder = Dinov2FusionWrapper._build_dinov2
    Dinov2FusionWrapper._build_dinov2 = staticmethod(lambda model_name, modulation_dim=None, pretrained=True: original_builder(model_name, modulation_dim, pretrained=False))
    log('Loading LHM-500M and local priors')
    model = ModelHumanLRMSapdinoBodyHeadSD3_5(**json.loads((ROOT / 'downloads/config.json').read_text()))
    state = load_file(str(ROOT / 'downloads/model.safetensors'))
    result = model.load_state_dict(state, strict=False)
    log(f'Checkpoint separately loaded prior prefixes: {sorted(set(k.split(".")[0] for k in result.missing_keys))}; unexpected: {result.unexpected_keys}')
    assert not result.unexpected_keys
    core = ('encoder.', 'transformer.', 'renderer.', 'pcl_embed.', 'motion_embed_mlp.')
    assert not any(k.startswith(core) for k in result.missing_keys), 'Core LHM weights missing'
    del state
    model.cuda()
    model.eval()
    model.renderer.device = torch.device('cuda')
    log('Reconstructing one Gaussian human')
    attrs, query, neutral = model.infer_single_view(image[None].cuda(), head.cuda(), None, None, None, None, None, smplx_params=params)
    params['transform_mat_neutral_pose'] = neutral
    log(f'Avatar reconstructed; allocated GPU memory={torch.cuda.memory_allocated()/2**30:.2f} GiB')
    torch.save({'attrs':attrs, 'query':query, 'params':params}, out / 'avatar_state.pt')
    from LHM.models.rendering.gs_renderer import Camera
    frame_arrays = []
    for index in range(len(motions['motion_seqs'])):
        frame = {k:(v if k in ('betas','transform_mat_neutral_pose') else v[:,index:index+1]) for k,v in params.items()}
        gs = model.animation_infer_gs(attrs, query, frame)
        values = torch.cat((gs.xyz, gs.rotation, gs.scaling, gs.shs.reshape(-1,3), gs.opacity), dim=1)
        assert torch.isfinite(values).all(), f'Invalid Gaussians in frame {index}'
        frame_arrays.append(values.float().cpu().numpy())
        if index in (0, min(30,args.frames-1), args.frames-1):
            gs.save_ply(str(out / f'human_{index:03}.ply'))
            intr = motions['render_intrs'][0,index].clone().cuda()
            width, height = [int(x) for x in frame['img_size_wh'][0,0]]
            ratio = 720 / height
            intr[:2] *= ratio
            camera = Camera.from_c2w(motions['render_c2ws'][0,index].cuda(), intr, 720, int(width*ratio))
            rendered = model.renderer.forward_single_view(gs, camera, torch.ones(3,device='cuda'))
            preview = (rendered['comp_rgb'].clamp(0,1).cpu().numpy()*255).astype(np.uint8)
            Image.fromarray(preview).save(out / f'human_{index:03}.png')
            log(f'Frame {index}: {values.shape}, bounds={gs.xyz.amin(0).tolist()}..{gs.xyz.amax(0).tolist()}')
    np.save(out / 'human_motion.npy', np.stack(frame_arrays))
    log(f'AVATAR_MOTION_READY frames={len(frame_arrays)}')


if __name__ == '__main__':
    main()
