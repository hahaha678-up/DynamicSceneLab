import json
import sys
from pathlib import Path
sys.path.insert(0, '/isaac-sim/kit/exts/omni.usd.libs')
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

path = '/work/assets/isaac-4.0/Isaac/Robots/Carter/nova_carter.usd'
layer = Sdf.Layer.FindOrOpen(path)
lines = layer.ExportToString().splitlines()
for i, line in enumerate(lines):
    if '@' in line or 'variantSet' in line or 'variants =' in line:
        print('\n'.join(lines[max(0, i-2):i+2]), flush=True)
print('EXTERNAL_REFS', UsdUtils.ExtractExternalReferences(path), flush=True)
