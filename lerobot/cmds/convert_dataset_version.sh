#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# This utility is only for an external LeRobotDataset v2.0 input. The published
# dataset is already v2.1 and does not require conversion.
LOCAL_PATH="${1:?Usage: $0 LOCAL_PATH REPO_ID}"
REPO_ID="${2:?Usage: $0 LOCAL_PATH REPO_ID}"
NUM_WORKERS=4
SKIP_DATA_VALIDATION=false

CONVERSION_ARGS=(
  --local-path="$LOCAL_PATH"
  --repo-id="$REPO_ID"
  --num-workers="$NUM_WORKERS"
)
if [[ "$SKIP_DATA_VALIDATION" == "true" ]]; then
  CONVERSION_ARGS+=(--skip-data-validation)
fi

exec python lerobot/datasets/v21/convert_dataset_v20_to_v21_local.py "${CONVERSION_ARGS[@]}"
