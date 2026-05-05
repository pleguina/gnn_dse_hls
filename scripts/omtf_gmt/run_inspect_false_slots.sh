#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
exec "$ROOT_DIR/venv/bin/python" -u scripts/omtf_gmt/inspect_false_slots.py \
    --seq-slot-ckpt    build/omtf_gmt/checkpoints/seq_slot_B2a/gmt_seq_slot_best.pt \
    --count-model-ckpt build/omtf_gmt/checkpoints/count_model_B2a/gmt_count_model_best.pt \
    --cache-dir        build/omtf_gmt/cache \
    --datasets         S2 B2 \
    --output           build/omtf_gmt/eval/false_slot_diagnostic.md \
    --device           cuda
