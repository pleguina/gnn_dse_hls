#!/usr/bin/env bash
# DETR h64 — Phase B3a: cache_v2 / G-dataset, Hungarian matching slot competition.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
exec "$ROOT_DIR/venv/bin/python" -u src/omtf_gmt/train.py \
    --cache-dir  build/omtf_gmt/cache_v2 \
    --datasets   G1 G2 G3 G4 G5 G6 G7 G8 B4 \
    --repeat     B4:8 G7:2 G8:4 \
    --model      detr_model \
    --hidden     64 \
    --epochs     50 \
    --batch-size 4096 \
    --num-workers 4 \
    --amp \
    --output-dir build/omtf_gmt/checkpoints/detr_B3a \
    --device     cuda
