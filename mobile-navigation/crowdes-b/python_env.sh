#!/usr/bin/env bash
set -eo pipefail
source /root/miniconda/etc/profile.d/conda.sh
conda activate py10
source /isaac-sim/setup_python_env.sh
B=/work/crowdes-b
export PYTHONPATH="$B/python-deps:$B/vendor:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/isaac-sim/kit/exts/omni.usd.libs/bin:${LD_LIBRARY_PATH:-}"
export HF_HOME="$B/cache/huggingface" TORCH_HOME="$B/cache/torch" XDG_CACHE_HOME="$B/cache" MPLCONFIGDIR="$B/cache/matplotlib" TMPDIR="$B/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
exec python "$@"
