#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
exec "$ROOT_DIR/venv/bin/python" -u src/omtf_gmt/train.py \
    --cache-dir  build/omtf_gmt/cache \
    --datasets   S1 S2 S3 S4 S5 B1 B2 B3 B4 \
    --repeat     B4:8 \
    --model      slot_model \
    --hidden     64 \
    --epochs     50 \
    --batch-size 4096 \
    --num-workers 4 \
    --amp \
    --w-count    0.5 \
    --w-div      0.05 \
    --output-dir build/omtf_gmt/checkpoints/slot_model_B1a \
    --device     cuda
