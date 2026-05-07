#!/usr/bin/env bash
# TPS input audit — Phase B4.
# Runs on all datasets with up to 20 files each (~5 min on a single CPU node).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
exec "$ROOT_DIR/venv/bin/python" -u scripts/omtf_gmt/audit_tps_inputs.py \
    --data-dir   data/prod \
    --output     build/omtf_gmt/eval/TPS_INPUT_AUDIT.md \
    --max-files  20 \
    "$@"
