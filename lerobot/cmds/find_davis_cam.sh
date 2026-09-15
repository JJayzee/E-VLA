#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# DAVIS camera inspection example. Edit the values below for another lighting
# condition or exposure, then run this file directly.
IMAGES_DIR="outputs/davis_cam_images"
DAVIS_WORK_MODE="davis"
FPS=30
EXPOSURE_DURATION_MS=10
RECORD_TIME_S=5
WHITE_BALANCE=0
FIRST_FRAME_WB=0

exec python lerobot/cameras/inivationdavis.py \
  --images-dir="$IMAGES_DIR" \
  --work-mode="$DAVIS_WORK_MODE" \
  --fps="$FPS" \
  --exposure-duration-ms="$EXPOSURE_DURATION_MS" \
  --record-time-s="$RECORD_TIME_S" \
  --white-balance="$WHITE_BALANCE" \
  --first-frame-wb="$FIRST_FRAME_WB"
