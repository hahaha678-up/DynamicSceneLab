#!/bin/bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /work/ws/install/setup.bash
exec python3 -u /work/bridge.py
