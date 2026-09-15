#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Event-adapter training over all published task/light subsets. This example
# selects count windows; use exactly one event-window mode.
DATASET_NAME="task_light_raw"
DATASET_REPO_ID="pap/200lux pap/lowlight sorting/200lux sorting/lowlight stack/200lux stack/lowlight"
DATASET_ROOT="$ROOT_DIR/dataset_task_light_raw"
# Set POLICY_PATH to a compatible image-only SmolVLA checkpoint obtained or trained separately.
POLICY_PATH="${POLICY_PATH:-}"
if [[ -z "$POLICY_PATH" ]]; then
  echo "Set POLICY_PATH to a compatible image-only checkpoint before training the event adapter." >&2
  exit 2
fi
EVENT_REPR="accu_count"
EVENT_WINDOW_SIZE=2000
OUTPUT_DIR="$ROOT_DIR/outputs/${DATASET_NAME}_event_2000_train"
JOB_NAME="${DATASET_NAME}_event_2000"
DEVICE="cuda"
USE_AMP=false
TRAIN_EXPERT_ONLY=false
TRAIN_EVENT_BRANCH_ONLY=true
TRAIN_EXPERT_AND_EVENT_BRANCH=false
TRAIN_STATE_PROJ=false
TRAIN_ACTION_PROJ=false
USE_GROUPED_LR=true
LR_EVENT_ENCODER=2e-4
LR_FUSION_MODULES=2e-4
WD_EVENT_ENCODER=1e-6
WD_FUSION_MODULES=1e-6
SCHEDULER_WARMUP_STEPS=625
SCHEDULER_DECAY_STEPS=11250
SCHEDULER_DECAY_LR=5e-6
NUM_WORKERS=8
BATCH_SIZE=64
STEPS=12500
LOG_FREQ=50
SAVE_CHECKPOINT=true
SAVE_FREQ=1000
WANDB_ENABLE=true
WANDB_PROJECT="smolvla_image_event"
WANDB_NOTES="${DATASET_NAME}_event_2000"
WANDB_MODE="offline"

exec python lerobot/scripts/train_smvla_image_event.py \
  --job_name="$JOB_NAME" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --dataset.event_window_size="$EVENT_WINDOW_SIZE" \
  --policy.path="$POLICY_PATH" \
  --policy.device="$DEVICE" \
  --policy.use_amp="$USE_AMP" \
  --policy.use_event_encoder=true \
  --policy.event_repr="$EVENT_REPR" \
  --policy.train_expert_only="$TRAIN_EXPERT_ONLY" \
  --policy.train_event_branch_only="$TRAIN_EVENT_BRANCH_ONLY" \
  --policy.train_expert_and_event_branch="$TRAIN_EXPERT_AND_EVENT_BRANCH" \
  --policy.train_state_proj="$TRAIN_STATE_PROJ" \
  --policy.train_action_proj="$TRAIN_ACTION_PROJ" \
  --policy.use_grouped_lr="$USE_GROUPED_LR" \
  --policy.lr_event_encoder="$LR_EVENT_ENCODER" \
  --policy.lr_fusion_modules="$LR_FUSION_MODULES" \
  --policy.wd_event_encoder="$WD_EVENT_ENCODER" \
  --policy.wd_fusion_modules="$WD_FUSION_MODULES" \
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
  --wandb.notes="$WANDB_NOTES" \
  --wandb.mode="$WANDB_MODE"
