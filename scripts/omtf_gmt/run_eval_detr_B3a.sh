#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
exec "$ROOT_DIR/venv/bin/python" -u scripts/omtf_gmt/eval_gmt.py \
    --checkpoint build/omtf_gmt/checkpoints/detr_B3a/gmt_detr_model_best.pt \
    --cache-dir  build/omtf_gmt/cache_v2 \
    --datasets   G1 G2 G3 G4 G5 G6 G7 G8 B4 \
    --threshold  0.0 \
    --output     build/omtf_gmt/eval/detr_B3a_eval.md \
    --device     cuda
