#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
exec "$ROOT_DIR/venv/bin/python" -u src/omtf_gmt/train.py \
    --cache-dir  build/omtf_gmt/cache \
    --datasets   S1 S3 B1 B4 \
    --model      deepsets \
    --hidden     64 \
    --epochs     50 \
    --output-dir build/omtf_gmt/checkpoints \
    --device     cuda
