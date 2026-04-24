#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
exec "$ROOT_DIR/venv/bin/python" -u scripts/omtf_gmt/validate_truth_transfer.py \
    --data-dir data/prod \
    --files-per-dataset 20 \
    --output   build/omtf_gmt/TRUTH_TRANSFER_SIGNOFF.md
