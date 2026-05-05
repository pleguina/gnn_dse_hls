#!/usr/bin/env bash
# EdgeCompat h64 — Phase B3e: B4×6, G7×4, G8×3 — train then eval in one job.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
PYTHON="$ROOT_DIR/venv/bin/python"

echo "=== TRAIN ==="
"$PYTHON" -u src/omtf_gmt/train.py \
    --cache-dir  build/omtf_gmt/cache_v2 \
    --datasets   G1 G2 G3 G4 G5 G6 G7 G8 B4 \
    --repeat     B4:6 G7:4 G8:3 \
    --model      edge_compat \
    --hidden     64 \
    --epochs     50 \
    --batch-size 4096 \
    --num-workers 4 \
    --amp \
    --output-dir build/omtf_gmt/checkpoints/edge_compat_B3e \
    --device     cuda

echo "=== EVAL ==="
"$PYTHON" -u scripts/omtf_gmt/eval_gmt.py \
    --checkpoint build/omtf_gmt/checkpoints/edge_compat_B3e/gmt_edge_compat_best.pt \
    --cache-dir  build/omtf_gmt/cache_v2 \
    --datasets   G1 G2 G3 G4 G5 G6 G7 G8 B4 \
    --threshold  0.0 \
    --output     build/omtf_gmt/eval/edge_compat_B3e_eval.md \
    --device     cuda
