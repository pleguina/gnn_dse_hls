#!/usr/bin/env bash
# detr_model h64 retrain with stub_in_overlap feature (cache_v2).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
exec "$ROOT_DIR/venv/bin/python" -u src/omtf_gmt/train.py \
    --cache-dir  build/omtf_gmt/cache_v2 \
    --datasets   S1 S2 S3 S4 S5 B1 B2 B3 B4 \
    --repeat     B4:8 \
    --model      detr_model \
    --hidden     64 \
    --epochs     50 \
    --batch-size 4096 \
    --num-workers 4 \
    --amp \
    --w-no-obj   0.1 \
    --output-dir build/omtf_gmt/checkpoints/detr_model_B2b \
    --device     cuda
