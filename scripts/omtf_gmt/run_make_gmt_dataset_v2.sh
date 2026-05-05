#!/usr/bin/env bash
# Rebuild GMT cache with stub_in_overlap feature (N_FEATURES=12, cache_v2).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
exec "$ROOT_DIR/venv/bin/python" -u scripts/omtf_gmt/make_gmt_dataset.py \
    --data-dir   data/prod \
    --output-dir build/omtf_gmt/cache_v2 \
    --datasets S1 S2 S3 S4 S5 B1 B2 B3 B4 \
    "$@"
