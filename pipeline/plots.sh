#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
exec "$ROOT_DIR/venv/bin/python" -u scripts/make_das_plots.py \
    --checkpoint  build/omtf_gmt/checkpoints/frozen_fp32_tps_edgecompat_h64/model.pt \
    --cache-dir   build/omtf_gmt/cache_das_tps \
    --unfiltered  build/omtf_gmt/eval/frozen_fp32_das_eval.json \
    --filtered    build/omtf_gmt/eval/frozen_fp32_das_filtered_eval.json \
    --output-dir  build/omtf_gmt/plots/das_validation \
    --device      cuda \
    "$@"
