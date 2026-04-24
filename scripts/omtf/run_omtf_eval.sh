#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   run_omtf_eval.sh <checkpoint_a> <checkpoint_b>
# Example:
#   run_omtf_eval.sh build/omtf/checkpoints/edge_compat_best.pt \
#                    build/omtf/checkpoints/slot_model_best.pt

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <checkpoint_a> <checkpoint_b>"
  exit 2
fi

CKPT_A="$1"
CKPT_B="$2"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x "$ROOT_DIR/venv/bin/python" ]]; then
  echo "Missing venv at $ROOT_DIR/venv"
  exit 3
fi

exec "$ROOT_DIR/venv/bin/python" -u src/omtf/eval.py \
  --checkpoint  "$CKPT_A" \
  --checkpoint2 "$CKPT_B" \
  --cache-dir   "$ROOT_DIR/build/omtf/cache/schema_v1_graph" \
  --all-datasets \
  --save
