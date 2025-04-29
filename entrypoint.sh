#!/bin/bash
set -e

# Source ROS2
source /opt/ros/humble/setup.bash

# Execute command passed to entrypoint
exec "$@" 