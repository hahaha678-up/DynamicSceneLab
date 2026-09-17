#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
isaac_dir=$(cd -- "$project_dir/../.." && pwd)
if [[ -n "${ISAAC_DOCKER_ENV:-}" ]]; then source "$ISAAC_DOCKER_ENV"; fi
container=re3sim-mobile-scene64
if ! docker inspect "$container" >/dev/null 2>&1; then
    docker run -d --name "$container" --device nvidia.com/gpu=0 --shm-size=4g \
        -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e NVIDIA_DRIVER_CAPABILITIES=all \
        -e VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json \
        -v "$project_dir:/work" -v "$project_dir/..:/repo:ro" \
        --entrypoint /bin/bash re3sim:1.0.0-cuda118 -lc 'sleep infinity' >/dev/null
elif [ "$(docker inspect -f '{{.State.Running}}' "$container")" != true ]; then
    docker start "$container" >/dev/null
fi
docker exec "$container" bash /work/python_env.sh /work/run_demo.py "$@"
