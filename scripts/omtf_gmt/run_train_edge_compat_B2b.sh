#!/usr/bin/env bash
# EdgeCompat h64 baseline retrain with stub_in_overlap feature (cache_v2).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
exec "$ROOT_DIR/venv/bin/python" -u src/omtf_gmt/train.py \
    --cache-dir  build/omtf_gmt/cache_v2 \
    --datasets   S1 S2 S3 S4 S5 B1 B2 B3 B4 \
    --repeat     B4:8 \
    --model      edge_compat \
    --hidden     64 \
    --epochs     50 \
    --batch-size 4096 \
    --num-workers 4 \
    --amp \
    --output-dir build/omtf_gmt/checkpoints/edge_compat_B2b \
    --device     cuda
