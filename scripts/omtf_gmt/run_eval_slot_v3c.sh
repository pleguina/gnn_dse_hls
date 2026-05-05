#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
exec "$ROOT_DIR/venv/bin/python" -u scripts/omtf_gmt/eval_gmt.py \
    --checkpoint build/omtf_gmt/checkpoints/slot_model_v3c/gmt_slot_model_best.pt \
    --cache-dir  build/omtf_gmt/cache \
    --datasets   S1 S2 S3 S4 S5 B1 B2 B3 B4 \
    --threshold  0.0 \
    --output     build/omtf_gmt/eval/slot_model_v3c_eval.md \
    --device     cuda
