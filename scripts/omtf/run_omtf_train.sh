#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   run_omtf_train.sh <model> <datasets_csv> <epochs> <batch_size> <max_files> <num_workers> [cache_dir]
# Example:
#   run_omtf_train.sh edge_compat S1+S2+S3+B1+B2+B3+B4+B5 50 128 0 4 build/omtf/cache/schema_v1_graph

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <model> <datasets_csv> <epochs> <batch_size> <max_files> <num_workers> [cache_dir]"
  exit 2
fi

MODEL="$1"
DATASETS_CSV="$2"
EPOCHS="$3"
BATCH_SIZE="$4"
MAX_FILES="$5"
NUM_WORKERS="$6"
CACHE_DIR="${7:-}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x "$ROOT_DIR/venv/bin/python" ]]; then
  echo "Missing venv at $ROOT_DIR/venv"
  exit 3
fi

DATASETS_CLI=()
IFS='+' read -r -a DATASET_ARRAY <<< "$DATASETS_CSV"
for ds in "${DATASET_ARRAY[@]}"; do
  DATASETS_CLI+=("$ds")
done

EXTRA_ARGS=()
if [[ -n "$CACHE_DIR" ]]; then
  EXTRA_ARGS+=(--cache-dir "$ROOT_DIR/$CACHE_DIR")
fi
if [[ "$MAX_FILES" -gt 0 ]]; then
  EXTRA_ARGS+=(--max-files "$MAX_FILES")
fi

ulimit -n 65536 2>/dev/null || true

exec "$ROOT_DIR/venv/bin/python" -u src/omtf/train.py \
  --model "$MODEL" \
  --datasets "${DATASETS_CLI[@]}" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --pin-memory \
  "${EXTRA_ARGS[@]}"
