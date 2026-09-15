#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Dataset validation only checks inference and training convergence. Action MSE/RMSE
# is not a meaningful performance metric; threshold pass rate is not task success.

DATASET_REPO_ID="pap/200lux"
DATASET_ROOT="$ROOT_DIR/dataset_task_light_raw/pap/200lux"
# Set POLICY_PATH to the checkpoint to evaluate on the pap/200lux subset.
POLICY_PATH="${POLICY_PATH:-}"
if [[ -z "$POLICY_PATH" ]]; then
  echo "Set POLICY_PATH to an image-only checkpoint for evaluation on pap/200lux." >&2
  exit 2
fi
DEVICE="cuda"
USE_AMP=false
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$ROOT_DIR/outputs/validation_image_only_${RUN_ID}"
JOB_NAME="validation_image_only_${RUN_ID}"
NUM_WORKERS=4
BATCH_SIZE=16
ACTION_SUCCESS_THRESHOLD=2
SAVE_PREDICTIONS=true

exec python lerobot/scripts/valid_dataset.py \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --policy.path="$POLICY_PATH" \
  --policy.device="$DEVICE" \
  --policy.use_amp="$USE_AMP" \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --num_workers="$NUM_WORKERS" \
  --batch_size="$BATCH_SIZE" \
  --action_success_threshold="$ACTION_SUCCESS_THRESHOLD" \
  --save_predictions="$SAVE_PREDICTIONS"
