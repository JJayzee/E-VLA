#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Teleoperate the SO-100 at the DAVIS recording rate. The SO-100 config supplies
# the default DAVIS camera; pass --robot.cameras='{}' manually for motor-only
# bench work.
ROBOT_TYPE="so100"
FPS=30
DISPLAY_DATA=true

exec python lerobot/scripts/control_robot.py \
  --control.fps="$FPS" \
  --robot.type="$ROBOT_TYPE" \
  --control.display_data="$DISPLAY_DATA" \
  --control.type=teleoperate
