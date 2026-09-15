#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# RealSense camera inspection example. Values are deliberately concrete so the
# command can be run immediately; edit this block for a different camera setup.
IMAGES_DIR="outputs/realsense_images"
WIDTH=640
HEIGHT=480
FPS=30
EXPOSURE_DURATION_MS=20
COLOR_MODE="rgb"
WHITE_BALANCE=4650
GAMMA=300
GAIN=128
CONTRAST=50
ROTATION=0
SYNC_MODE=1
SET_EMITTER=0
RECORD_TIME_S=2

exec python lerobot/cameras/intelrealsense.py \
  --images-dir="$IMAGES_DIR" \
  --width="$WIDTH" \
  --height="$HEIGHT" \
  --fps="$FPS" \
  --exposure-duration-ms="$EXPOSURE_DURATION_MS" \
  --color-mode="$COLOR_MODE" \
  --white-balance="$WHITE_BALANCE" \
  --gamma="$GAMMA" \
  --gain="$GAIN" \
  --contrast="$CONTRAST" \
  --rotation="$ROTATION" \
  --sync-mode="$SYNC_MODE" \
  --set-emitter="$SET_EMITTER" \
  --record-time-s="$RECORD_TIME_S"
