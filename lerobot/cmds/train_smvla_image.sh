#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Image-only multitask fine-tuning over all published task/light subsets.
DATASET_REPO_ID="pap/200lux pap/lowlight sorting/200lux sorting/lowlight stack/200lux stack/lowlight"
DATASET_ROOT="$ROOT_DIR/dataset_task_light_raw"
# Set POLICY_PATH to a compatible SmolVLA base checkpoint obtained separately.
POLICY_PATH="${POLICY_PATH:-}"
if [[ -z "$POLICY_PATH" ]]; then
  echo "Set POLICY_PATH to a compatible SmolVLA base checkpoint; this repository does not include one." >&2
  exit 2
fi
OUTPUT_DIR="$ROOT_DIR/outputs/retrain_image_only"
JOB_NAME="retrain_image_only"
DEVICE="cuda"
USE_AMP=false
POLICY_LR=2e-5
SCHEDULER_WARMUP_STEPS=300
SCHEDULER_DECAY_STEPS=5400
SCHEDULER_DECAY_LR=5e-6
NUM_WORKERS=8
BATCH_SIZE=256
STEPS=6000
LOG_FREQ=100
SAVE_CHECKPOINT=true
SAVE_FREQ=3000
WANDB_ENABLE=false
WANDB_PROJECT="smvla_image"
WANDB_MODE="offline"

exec python lerobot/scripts/train_smvla_image.py \
  --job_name="$JOB_NAME" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --policy.path="$POLICY_PATH" \
  --policy.device="$DEVICE" \
  --policy.use_amp="$USE_AMP" \
  --policy.load_ignore_keys='[]' \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true \
  --policy.train_action_proj=true \
  --policy.use_grouped_lr=true \
  --policy.optimizer_lr="$POLICY_LR" \
  --policy.scheduler_warmup_steps="$SCHEDULER_WARMUP_STEPS" \
  --policy.scheduler_decay_steps="$SCHEDULER_DECAY_STEPS" \
  --policy.scheduler_decay_lr="$SCHEDULER_DECAY_LR" \
  --output_dir="$OUTPUT_DIR" \
  --resume=false \
  --num_workers="$NUM_WORKERS" \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --log_freq="$LOG_FREQ" \
  --save_checkpoint="$SAVE_CHECKPOINT" \
  --save_freq="$SAVE_FREQ" \
  --wandb.enable="$WANDB_ENABLE" \
  --wandb.project="$WANDB_PROJECT" \
  --wandb.notes="$JOB_NAME" \
  --wandb.mode="$WANDB_MODE"
