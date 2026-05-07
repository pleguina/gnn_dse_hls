#!/usr/bin/env bash
# Run 6 — TPS-h128 safe schedule: 50 epochs, lower LR, no cosine tail.
# Avoids the training collapse seen with cosine annealing in the 100ep run.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
PYTHON="$ROOT_DIR/venv/bin/python"
OUTDIR="build/omtf_gmt/checkpoints/edge_compat_h128_tps_safe50"
EVALDIR="build/omtf_gmt/eval"

echo "=== TRAIN edge_compat h128 TPS safe50 (lr=5e-4, no cosine) ==="
"$PYTHON" -u src/omtf_gmt/train.py \
    --cache-dir  build/omtf_gmt/cache_v2_tps \
    --datasets   G1 G2 G3 G4 G5 G6 G7 G8 B4 \
    --repeat     B4:6 G7:4 G8:4 \
    --model      edge_compat \
    --hidden     128 \
    --epochs     50 \
    --batch-size 4096 \
    --lr         5e-4 \
    --num-workers 4 \
    --amp \
    --output-dir "$OUTDIR" \
    --device     cuda

echo "=== EVAL best ==="
"$PYTHON" -u scripts/omtf_gmt/eval_gmt.py \
    --checkpoint "$OUTDIR/gmt_edge_compat_best.pt" \
    --cache-dir  build/omtf_gmt/cache_v2_tps \
    --datasets   G1 G2 G3 G4 G5 G6 G7 G8 B4 \
    --threshold  0.0 \
    --output     "$EVALDIR/edge_compat_h128_tps_safe50_best_eval.md" \
    --device     cuda

echo "=== EVAL last ==="
"$PYTHON" -u scripts/omtf_gmt/eval_gmt.py \
    --checkpoint "$OUTDIR/gmt_edge_compat_last.pt" \
    --cache-dir  build/omtf_gmt/cache_v2_tps \
    --datasets   G1 G2 G3 G4 G5 G6 G7 G8 B4 \
    --threshold  0.0 \
    --output     "$EVALDIR/edge_compat_h128_tps_safe50_last_eval.md" \
    --device     cuda
