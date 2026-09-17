#!/bin/bash
set -eo pipefail
source /opt/ros/humble/setup.bash
set -u
tag="$1"
mode="${2:-nav2}"
route="${3:-[[1.3,5.0],[1.3,15.0]]}"
[[ "$tag" =~ ^(nav2_crossing|ab)_[a-zA-Z0-9_.-]+$ ]] || exit 2
export ROS_LOG_DIR="/work/runtime/$tag/ros"
controller_pid=''
manager_pid=''
cleanup() {
  for pid in "$controller_pid" "$manager_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then kill -TERM "$pid"; fi
  done
  wait || true
}
trap cleanup EXIT
if [[ "$mode" == nav2 ]]; then
/opt/ros/humble/lib/nav2_controller/controller_server --ros-args \
  --params-file /work/crossing/nav2_params.yaml > "/work/runtime/$tag/controller.log" 2>&1 &
controller_pid=$!
/opt/ros/humble/lib/nav2_lifecycle_manager/lifecycle_manager --ros-args \
  -r __node:=navigation_lifecycle_manager -p autostart:=true \
  -p use_sim_time:=true -p 'node_names:=[controller_server]' \
  > "/work/runtime/$tag/lifecycle.log" 2>&1 &
manager_pid=$!
fi
python3 /work/crossing/ros_endpoint.py --"$mode" --route "$route" \
  --nav-log "/work/runtime/$tag/costmaps.jsonl"
