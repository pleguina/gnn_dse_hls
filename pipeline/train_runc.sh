#!/usr/bin/env bash
# G9/G10 high-eta hard-neg retraining: Run C — w_hard_neg=0.10 (softer pressure)
# Use only if Run A hurts G1-G6 signal efficiency.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON="$ROOT_DIR/venv/bin/python"
OUTDIR="build/omtf_gmt/checkpoints/edge_compat_h64_tps_g9g10_runc"
EVALDIR="build/omtf_gmt/eval"

echo "=== TRAIN edge_compat h64 TPS g9g10 Run C (w_hard_neg=0.10) ==="
"$PYTHON" -u src/omtf_gmt/train.py \
    --cache-dir  build/omtf_gmt/cache_v2_tps \
    --datasets   G1 G2 G3 G4 G5 G6 G7 G8 G9 G10 B4 \
    --repeat     B4:4 G7:4 G8:3 G9:4 G10:1 \
    --model      edge_compat \
    --hidden     64 \
    --epochs     100 \
    --lr         5e-4 \
    --batch-size 4096 \
    --num-workers 4 \
    --amp \
    --scheduler  cosine \
    --save-epochs 25 50 75 100 \
    --w-hard-neg 0.10 \
    --output-dir "$OUTDIR" \
    --device     cuda

echo "=== EVAL best ==="
"$PYTHON" -u scripts/eval.py \
    --checkpoint "$OUTDIR/gmt_edge_compat_best.pt" \
    --cache-dir  build/omtf_gmt/cache_v2_tps \
    --datasets   G1 G2 G3 G4 G5 G6 G7 G8 G9 G10 B4 \
    --threshold  0.0 \
    --output     "$EVALDIR/edge_compat_h64_tps_g9g10_runc_best_eval.md" \
    --device     cuda
