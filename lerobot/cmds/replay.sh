#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Replay one episode from a published subset.
ROBOT_TYPE="so100"
REPO_ID="pap/200lux"
DATASET_ROOT="$ROOT_DIR/dataset_task_light_raw/pap/200lux"
EPISODE=0
FPS=30

exec python lerobot/scripts/control_robot.py \
  --robot.type="$ROBOT_TYPE" \
  --control.type=replay \
  --control.fps="$FPS" \
  --control.repo_id="$REPO_ID" \
  --control.root="$DATASET_ROOT" \
  --control.episode="$EPISODE"
