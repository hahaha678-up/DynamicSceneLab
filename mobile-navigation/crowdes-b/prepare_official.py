import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from utils.config import get_config
from utils.dataloader.synthetic_dataloader import SyntheticDataset


root = Path('/work/crowdes-b')
os.chdir(root / 'vendor')
config = get_config('./configs/model/CrowdES_eth.yaml')
# The synthetic loader inherits a benchmark directory scan before replacing its scenes.
(Path(config.dataset.dataset_preprocessed_path) / 'test/image').mkdir(parents=True, exist_ok=True)
Path(config.dataset.dataset_path).mkdir(parents=True, exist_ok=True)
scene = 'synth_scurve'
print(json.dumps({'image_size': Image.open(f'datasets/Synthetic/image_terrain/{scene}_bg.png').size}), flush=True)
dataset = SyntheticDataset(config, [scene])
data = dataset[0]
out = root / 'inputs' / scene
out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / 'input.npz', walkable=data['walkable'], H=data['H'], img=data['img'],
                    appearance=data['appearance'], population=data['population'], world_origin=np.zeros(2))
(out / 'navmesh.json').write_text(json.dumps(data['navmesh']))
(out / 'metadata.json').write_text(json.dumps({
    'source': 'Official real-world_datasets.zip release v1.0-dataset, synth_scurve members',
    'density_maps': 'official supplied maps', 'navmesh': 'official SyntheticDataset baker defaults',
    'segmentation': 'binary walkable converted into the declared sidewalk/grass classes by infer.py',
    'H': data['H'].tolist(), 'shape': list(data['walkable'].shape),
}, indent=2))
print('official input ready', flush=True)
