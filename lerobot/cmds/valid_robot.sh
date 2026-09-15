#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ROBOT_TYPE="so100"
FPS=30
RUN_ID="$(date +%Y%m%d_%H%M%S)"
REPO_ID="evla/eval_image_only_${RUN_ID}"
DATASET_ROOT="$ROOT_DIR/outputs/eval_image_only_${RUN_ID}"
TASK_1="Grasp a cube and put it in the cup."
TASK_2="Pick up the red cube and put it on top of the green cube."
TASK_3="Put the red cube in the right cup and the green cube in the left cup."
# Set POLICY_PATH to the image-only checkpoint selected for this robot evaluation.
POLICY_PATH="${POLICY_PATH:-}"
if [[ -z "$POLICY_PATH" ]]; then
  echo "Set POLICY_PATH to an image-only checkpoint for robot evaluation." >&2
  exit 2
fi
DEVICE="cuda"
USE_AMP=false
WARMUP_TIME_S=80
EPISODE_TIME_S=65
RESET_TIME_S=60
NUM_EPISODES=10
VIDEO=false
SAVE_DATA=false
RECORD_EVERY_CONTROL_FRAME=false

exec python lerobot/scripts/control_robot.py \
  --robot.type="$ROBOT_TYPE" \
  --control.type=valid \
  --control.fps="$FPS" \
  --control.repo_id="$REPO_ID" \
  --control.root="$DATASET_ROOT" \
  --control.single_task="$TASK_1" \
  --control.policy.path="$POLICY_PATH" \
  --control.policy.device="$DEVICE" \
  --control.policy.use_amp="$USE_AMP" \
  --control.warmup_time_s="$WARMUP_TIME_S" \
  --control.episode_time_s="$EPISODE_TIME_S" \
  --control.reset_time_s="$RESET_TIME_S" \
  --control.num_episodes="$NUM_EPISODES" \
  --control.video="$VIDEO" \
  --control.save_data="$SAVE_DATA" \
  --control.record_every_control_frame="$RECORD_EVERY_CONTROL_FRAME"
