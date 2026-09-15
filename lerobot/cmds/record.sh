#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Image-only SO-100/DAVIS recording example. This branch deliberately exposes
# only image/state/action features even though DAVIS also produces events.
ROBOT_TYPE="so100"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
REPO_ID="evla/record_image_only_${RUN_ID}"
DATASET_ROOT="$ROOT_DIR/outputs/recordings/image_only_${RUN_ID}"
TASK="Grasp a cube and put it in the cup."
FPS=30
WARMUP_TIME_S=30
EPISODE_TIME_S=30
RESET_TIME_S=30
NUM_EPISODES=30
PUSH_TO_HUB=false
VIDEO=true
BATCH_ENCODING_SIZE=1
TAGS='["so100", "image-only"]'

exec python lerobot/scripts/control_robot.py \
  --robot.type="$ROBOT_TYPE" \
  --control.type=record \
  --control.fps="$FPS" \
  --control.single_task="$TASK" \
  --control.repo_id="$REPO_ID" \
  --control.root="$DATASET_ROOT" \
  --control.tags="$TAGS" \
  --control.warmup_time_s="$WARMUP_TIME_S" \
  --control.episode_time_s="$EPISODE_TIME_S" \
  --control.reset_time_s="$RESET_TIME_S" \
  --control.num_episodes="$NUM_EPISODES" \
  --control.push_to_hub="$PUSH_TO_HUB" \
  --control.video="$VIDEO" \
  --control.batch_encoding_size="$BATCH_ENCODING_SIZE"
