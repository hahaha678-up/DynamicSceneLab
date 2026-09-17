#!/bin/bash
set -eo pipefail
source /opt/ros/humble/setup.bash
cd /work/vendor/lightsfm
make install
cd /work/ws
export CMAKE_BUILD_PARALLEL_LEVEL=6
export MAKEFLAGS=-j6
colcon build --executor sequential --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBTCPP_UNIT_TESTS=OFF -DBTCPP_EXAMPLES=OFF -DBTCPP_BUILD_TOOLS=OFF -DBTCPP_GROOT_INTERFACE=OFF -DHUNAV_GROOT_INTERFACE=OFF
