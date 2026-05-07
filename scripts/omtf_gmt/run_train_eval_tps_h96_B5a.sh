#!/usr/bin/env bash
# Run 5 — TPS-h96 baseline (intermediate capacity between h64 and h128).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
PYTHON="$ROOT_DIR/venv/bin/python"
OUTDIR="build/omtf_gmt/checkpoints/edge_compat_h96_tps_B5a"
EVALDIR="build/omtf_gmt/eval"

echo "=== TRAIN edge_compat h96 TPS B5a ==="
"$PYTHON" -u src/omtf_gmt/train.py \
    --cache-dir  build/omtf_gmt/cache_v2_tps \
    --datasets   G1 G2 G3 G4 G5 G6 G7 G8 B4 \
    --repeat     B4:6 G7:4 G8:4 \
    --model      edge_compat \
    --hidden     96 \
    --epochs     50 \
    --batch-size 4096 \
    --lr         1e-3 \
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
    --output     "$EVALDIR/edge_compat_h96_tps_B5a_best_eval.md" \
    --device     cuda

echo "=== EVAL last ==="
"$PYTHON" -u scripts/omtf_gmt/eval_gmt.py \
    --checkpoint "$OUTDIR/gmt_edge_compat_last.pt" \
    --cache-dir  build/omtf_gmt/cache_v2_tps \
    --datasets   G1 G2 G3 G4 G5 G6 G7 G8 B4 \
    --threshold  0.0 \
    --output     "$EVALDIR/edge_compat_h96_tps_B5a_last_eval.md" \
    --device     cuda
