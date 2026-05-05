#!/usr/bin/env bash
# Build GMT cache (schema v2, N_FEATURES=14) for one physical dataset directory.
# Usage: run_make_gmt_dataset_g.sh <DATASET>   e.g. G1_pos, G7, B4
set -euo pipefail
DATASET="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
exec "$ROOT_DIR/venv/bin/python" -u scripts/omtf_gmt/make_gmt_dataset.py \
    --data-dir   data/prod \
    --output-dir build/omtf_gmt/cache_v2 \
    --datasets "$DATASET"
