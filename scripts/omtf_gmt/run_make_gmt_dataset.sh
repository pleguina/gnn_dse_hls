#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
exec "$ROOT_DIR/venv/bin/python" -u scripts/omtf_gmt/make_gmt_dataset.py \
    --data-dir   data/prod \
    --output-dir build/omtf_gmt/cache \
    "$@"
