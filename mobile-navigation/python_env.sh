#!/usr/bin/env bash
set -e
source /root/miniconda/etc/profile.d/conda.sh
conda activate py10
source /isaac-sim/setup_python_env.sh
export LD_LIBRARY_PATH=/isaac-sim/kit/exts/omni.usd.libs/bin:${LD_LIBRARY_PATH}
exec python "$@"
