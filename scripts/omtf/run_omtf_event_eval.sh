#!/usr/bin/env bash
set -euo pipefail

# Usage: run_omtf_event_eval.sh <checkpoint>
CKPT="$1"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x "$ROOT_DIR/venv/bin/python" ]]; then
  echo "Missing venv at $ROOT_DIR/venv"
  exit 3
fi

exec "$ROOT_DIR/venv/bin/python" -u src/omtf/eval_event_level.py \
  --checkpoint  "$CKPT" \
  --cache-dir   "$ROOT_DIR/build/omtf/cache/schema_v1_graph" \
  --all-datasets \
  --save
