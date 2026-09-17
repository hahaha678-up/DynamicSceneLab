#!/usr/bin/env bash
set -euo pipefail
B=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN=${1:-scene55_batch_20260916_seed00_19}
SCENE=${2:-scene55}
SEEDS=${3:-20}
FIRST_SEED=${4:-0}
mkdir -p "$B/runs"
test ! -e "$B/runs/$RUN"
trap 'result=$?; printf "%s\n" "$result" > "$B/runs/$RUN.exitcode"' EXIT
printf '{"phase":"room_inference","run":"%s","seeds":%s,"first_seed":%s,"seconds_per_scene":35}\n' "$RUN" "$SEEDS" "$FIRST_SEED" > "$B/runs/batch_state.json"
timeout --signal=TERM --kill-after=30s 2400 docker run --rm --pull never --gpus device=1 --network none \
  --name re3sim-crowdes-batch -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$B/..":/work:ro \
  -v "$B":/work/crowdes-b:rw --entrypoint /bin/bash re3sim:1.0.0-cuda118 \
  /work/crowdes-b/python_env.sh -B /work/crowdes-b/infer.py \
  --scene "$SCENE" --seeds "$SEEDS" --first-seed "$FIRST_SEED" --seconds 35 --run "$RUN" > "$B/runs/$RUN.log" 2>&1
printf '{"phase":"analysis","run":"%s"}\n' "$RUN" > "$B/runs/batch_state.json"
docker run --rm --pull never --network none --name re3sim-crowdes-batch-analysis \
  -v "$B/../..":/repo:ro \
  -v "$B/..":/work:ro \
  -v "$B":/work/crowdes-b:rw --entrypoint /bin/bash re3sim:1.0.0-cuda118 \
  /work/crowdes-b/python_env.sh -B /work/crowdes-b/analyze.py "$RUN" > "$B/runs/$RUN.analysis.log" 2>&1
printf '{"phase":"finished","run":"%s"}\n' "$RUN" > "$B/runs/batch_state.json"
