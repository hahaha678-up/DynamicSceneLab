#!/bin/bash
set -euo pipefail
CORE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -n "${ISAAC_DOCKER_ENV:-}" ]]; then source "$ISAAC_DOCKER_ENV"; fi
mkdir -p "$CORE/runtime" "$CORE/../mobile-navigation/output"
exec 9>"$CORE/runtime/demo.lock"
flock -n 9 || { echo 'An episode or room demo is already running' >&2; exit 1; }
if [ "$#" -eq 0 ]; then
  set -- "$CORE/crossing/scenarios/gap0_person1.0.yaml"
fi
scenarios=()
for item in "$@"; do
  if [[ "$item" != /* ]]; then item="$CORE/$item"; fi
  item=$(realpath -e "$item")
  [[ "$item" == "$CORE/"* ]] || { echo 'Scenario must be inside hunav-core' >&2; exit 2; }
  scenarios+=("/repo/hunav-core/${item#"$CORE/"}")
done
for container in re3sim-hunav-core re3sim-mobile-scene64; do
  if [ "$(docker inspect -f '{{.State.Running}}' "$container")" != true ]; then docker start "$container"; fi
done
bridge_ready() {
python3 - "$CORE/runtime/hunav.sock" "$CORE/runtime" <<'PY'
import socket
import sys
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
    client.settimeout(2)
    try:
        client.connect(sys.argv[1])
    except OSError:
        sys.exit(1)
PY
}
if ! bridge_ready; then
  python3 - "$CORE/runtime/hunav.sock" "$CORE/runtime" <<'PY'
import errno
import socket
import sys
from pathlib import Path
path = Path(sys.argv[1])
assert path.parent.resolve() == Path(sys.argv[2]).resolve()
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
    client.settimeout(2)
    try:
        client.connect(str(path))
    except OSError as exc:
        if exc.errno not in (errno.ENOENT, errno.ECONNREFUSED):
            raise
        if path.is_socket():
            path.unlink()
PY
  docker exec -d re3sim-hunav-core bash -c 'exec bash /work/run_bridge.sh > /work/runtime/bridge.log 2>&1'
  for attempt in $(seq 1 30); do bridge_ready && break; sleep 1; done
  bridge_ready || { echo 'HuNav bridge failed; inspect runtime/bridge.log' >&2; exit 1; }
fi
batch="crossing_$(date -u +%Y%m%dT%H%M%S)_$$"
output="/work/output/$batch"
log="$CORE/runtime/$batch.log"
trap 'echo "Episode runner failed; log: $log" >&2; tail -n 20 "$log" >&2' ERR
docker exec re3sim-mobile-scene64 bash /work/python_env.sh /repo/hunav-core/crossing/runner.py \
  --output "$output" "${scenarios[@]}" > "$CORE/runtime/$batch.log" 2>&1
result="$CORE/../mobile-navigation/output/$batch/summary.json"
python3 - "$result" "${#scenarios[@]}" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
results = json.loads(path.read_text())
print('OUTPUT', path.parent)
for result in results:
    print(result['scenario'], result['status'], 'crossing=', result['actual_path_crossing'],
          'clearance=', result['minimum_clearance_m'], 'TTC=', result['minimum_ttc_s'],
          'collision=', result['collision'])
if len(results) != int(sys.argv[2]) or any(r['status'] != 'completed' or not r.get('event_valid', False) for r in results):
    raise SystemExit('One or more episodes failed; inspect saved metrics and runner log')
PY
