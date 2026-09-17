import argparse
import copy
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

from utils.config import get_config
from utils.homography import image2world
from utils.navmesh import PathFinderNew
import CrowdES.inference_model as inference


ROOT = Path('/work/crowdes-b')
parser = argparse.ArgumentParser()
parser.add_argument('--scene', default='scene55')
parser.add_argument('--seeds', type=int, default=1)
parser.add_argument('--first-seed', type=int, default=0)
parser.add_argument('--seconds', type=float, default=35.0)
parser.add_argument('--run', required=True)
parser.add_argument('--gt-maps', action='store_true')
args = parser.parse_args()
out = ROOT / 'runs' / args.run
out.mkdir(parents=True, exist_ok=False)
os.chdir(ROOT / 'vendor')
torch.set_num_threads(4)
config = get_config('./configs/model/CrowdES_eth.yaml')
checkpoint = ROOT / 'models/checkpoints/eth'
config.crowd_emitter.emitter_pre.checkpoint_dir = str(checkpoint / 'emitter_pre')
config.crowd_emitter.emitter.checkpoint_dir = str(checkpoint / 'emitter')
config.crowd_simulator.simulator.checkpoint_dir = str(checkpoint / 'simulator')

# Keep the pretrained timing and pace; only restrict the requested population/type.
inference.CONTROL_POPULATION_SIZE = 1
inference.CONTROL_AGENT_TYPE = 0
inference.CONTROL_WALKING_PACE = None
framework = inference.CrowdESFramework(config)
inputs = ROOT / 'inputs' / args.scene
data = np.load(inputs / 'input.npz')
walkable = data['walkable']
H = data['H']
world_origin = data['world_origin'] if 'world_origin' in data else np.zeros(2)
labels = json.loads((ROOT / 'vendor/datasets/segmentation_classes.json').read_text())
seg_ids = np.where(walkable > 0, labels['sidewalk'] - 1, labels['grass'] - 1)
seg = np.eye(len(labels), dtype=np.float32)[seg_ids]
img = data['img'] if 'img' in data else np.repeat((walkable * 180 + 35)[..., None], 3, axis=2)
navmesh = json.loads((inputs / 'navmesh.json').read_text())
pathfinder = PathFinderNew([(x, 0., y) for x, y in navmesh['vertices']], navmesh['polygons'])
appearance = data['appearance'] if args.gt_maps else None
population = data['population'] if args.gt_maps else None
fps = config.dataset.dataset_fps
provenance = {
    'source_commit': '0e663594604533d77ca98998bb93b63f1d509982', 'checkpoint': str(checkpoint),
    'scene': args.scene, 'seconds': args.seconds, 'fps': fps,
    'seeds': list(range(args.first_seed, args.first_seed + args.seeds)),
    'controls': {'population_size': 1, 'agent_type': 0, 'walking_pace': None},
    'ground_truth_density_maps': args.gt_maps, 'config': config,
    'world_origin_xy': world_origin.tolist(), 'H': H.tolist(),
    'inference_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'input_sha256': {name: hashlib.sha256((inputs / name).read_bytes()).hexdigest()
                     for name in ['input.npz', 'navmesh.json', 'metadata.json'] if (inputs / name).is_file()},
}
(out / 'provenance.json').write_text(json.dumps(provenance, indent=2))
results = []
started = time.monotonic()
for seed in provenance['seeds']:
    before = time.monotonic()
    try:
        framework.initialize_scene(img, seg, walkable, copy.deepcopy(navmesh), H,
                                   appearance_gt=appearance, population_gt=population)
        with torch.inference_mode():
            generated = framework.generate(round(args.seconds * fps), seed=seed)
        generated.to_csv(out / f'seed_{seed:03d}_pixels.csv', index=False)
        converted = generated.copy()
        xy = image2world(converted[['x', 'y']].to_numpy(), H) + world_origin
        converted['x'] = xy[:, 0]
        converted['y'] = xy[:, 1]
        converted['time_s'] = converted['frame'] / fps
        converted.to_csv(out / f'seed_{seed:03d}_world.csv', index=False)
        emitted = {str(k): {key: value.tolist() if isinstance(value, np.ndarray) else
                           value.item() if isinstance(value, np.generic) else value
                           for key, value in parameter.items()}
                   for k, parameter in framework.agent_parameter.items()}
        (out / f'seed_{seed:03d}_emissions.json').write_text(json.dumps(emitted, indent=2))
        missing_paths = sum(not pathfinder.search_path(
            (p['origin_xy'][0], 0., p['origin_xy'][1]),
            (p['goal_xy'][0], 0., p['goal_xy'][1])) for p in emitted.values())
        if seed == args.first_seed:
            for name, density in [('appearance', framework.appearance_density_map),
                                  ('population', framework.population_density_map)]:
                np.save(out / f'{name}.npy', density)
                cv2.imwrite(str(out / f'{name}.png'), (np.clip(density, 0, 1) * 255).astype(np.uint8))
        result = {'seed': seed, 'status': 'ok', 'agents': int(generated.agent_id.nunique()),
                  'emission_count': len(emitted), 'emission_path_failures': missing_paths,
                  'rows': len(generated), 'wall_seconds': time.monotonic() - before}
    except Exception as error:
        (out / f'seed_{seed:03d}_error.txt').write_text(traceback.format_exc())
        result = {'seed': seed, 'status': 'error', 'error': str(error),
                  'wall_seconds': time.monotonic() - before}
    results.append(result)
    status = {'complete': len(results) == args.seeds, 'results': results,
              'elapsed_seconds': time.monotonic() - started,
              'cuda_peak_allocated_mb': torch.cuda.max_memory_allocated() / 1024 ** 2}
    (out / 'status.json').write_text(json.dumps(status, indent=2))
    print(json.dumps(result), flush=True)
    if len(results) >= 3 and all(r['status'] == 'error' for r in results[-3:]):
        raise RuntimeError('Three consecutive seeds failed; inspect the recorded errors before continuing.')
print(json.dumps(status), flush=True)
