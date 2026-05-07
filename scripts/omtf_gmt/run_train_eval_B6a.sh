#!/usr/bin/env bash
# B6a — EdgeCompatAssign h64, w_assign=0.2, w_hard_neg=0.25
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
PYTHON="$ROOT_DIR/venv/bin/python"

OUTDIR="build/omtf_gmt/checkpoints/edge_compat_assign_B6a"
EVALDIR="build/omtf_gmt/eval"

echo "=== TRAIN EdgeCompatAssign h64 B6a (w_assign=0.2) ==="
"$PYTHON" -u src/omtf_gmt/train.py \
    --cache-dir  build/omtf_gmt/cache_v2_tps \
    --datasets   G1 G2 G3 G4 G5 G6 G7 G8 B4 \
    --repeat     G7:2 G8:4 B4:8 \
    --model      edge_compat_assign \
    --hidden     64 \
    --epochs     50 \
    --batch-size 4096 \
    --num-workers 4 \
    --lr         1e-3 \
    --w-node     1.0 \
    --w-cand     1.0 \
    --w-pt       0.5 \
    --w-assign   0.2 \
    --w-hard-neg 0.25 \
    --amp \
    --scheduler  cosine \
    --save-epochs 25 50 \
    --output-dir "$OUTDIR" \
    --device     cuda

echo "=== EVAL best ==="
"$PYTHON" -u scripts/omtf_gmt/eval_gmt.py \
    --checkpoint "$OUTDIR/gmt_edge_compat_assign_best.pt" \
    --cache-dir  build/omtf_gmt/cache_v2_tps \
    --datasets   G1 G2 G3 G4 G5 G6 G7 G8 B4 \
    --threshold  0.0 \
    --output     "$EVALDIR/edge_compat_assign_B6a_best_eval.md" \
    --device     cuda
