#!/bin/bash
set -eo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CORE="$ROOT/hunav-core"
if [[ -n "${ISAAC_DOCKER_ENV:-}" ]]; then source "$ISAAC_DOCKER_ENV"; fi
case "${1:-cross}" in
  all) cases=(cross) ;;
  cross) cases=("$1") ;;
  *) echo 'Usage: bash run_demo.sh [cross|all]' >&2; exit 2 ;;
esac
exec 9>"$CORE/runtime/demo.lock"
flock -n 9 || { echo 'A room demo is already running' >&2; exit 1; }
for name in re3sim-hunav-core re3sim-lhm-human re3sim-mobile-scene64; do
  test "$(docker inspect -f '{{.State.Running}}' "$name")" = true || docker start "$name"
done
if ! test -S "$CORE/runtime/hunav.sock"; then
  docker exec -d re3sim-hunav-core bash -c 'exec bash /work/run_bridge.sh > /work/runtime/bridge.log 2>&1'
  for attempt in $(seq 1 30); do
    test -S "$CORE/runtime/hunav.sock" && break
    sleep 1
  done
  test -S "$CORE/runtime/hunav.sock" || { echo 'HuNav bridge failed; see runtime/bridge.log' >&2; exit 1; }
fi
for case_name in "${cases[@]}"; do
  test ! -S "$ROOT/lhm-human/output/social_avatar.sock" || { echo 'Avatar service already has an endpoint' >&2; exit 1; }
  docker exec -d re3sim-lhm-human bash -c 'exec /work/venv/bin/python /work/serve_social_avatar.py > /work/output/social_avatar.log 2>&1'
  for attempt in $(seq 1 45); do
    test -S "$ROOT/lhm-human/output/social_avatar.sock" && break
    sleep 1
  done
  test -S "$ROOT/lhm-human/output/social_avatar.sock" || { echo 'LHM service failed; see output/social_avatar.log' >&2; exit 1; }
  echo "Running HuNav room case: $case_name"
  docker exec re3sim-mobile-scene64 bash /work/python_env.sh /repo/lhm-human/run_social_room.py \
    --seconds 40 --name "hunav_scene55_$case_name" --case "$case_name" --record --live-avatar --habitat --motion-preview \
    > "$CORE/runtime/room_$case_name.log" 2>&1
  python3 - "$ROOT/mobile-navigation/output/hunav_scene55_${case_name}_result.json" <<'PY'
import json
import sys
from pathlib import Path
result = json.loads(Path(sys.argv[1]).read_text())
if not result['success']:
    raise SystemExit('Room demo did not pass: ' + str(result.get('failure') or 'actor goal/stop checks'))
PY
  echo "Finished: $ROOT/mobile-navigation/output/hunav_scene55_$case_name.mp4"
done
