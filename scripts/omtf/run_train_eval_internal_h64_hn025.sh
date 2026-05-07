#!/usr/bin/env bash
# OMTF-internal Phase C — EdgeCompatG h64 + hard-negative loss w=0.25.
# Mirrors the TPS B5 hn025 setup for direct comparison.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
PYTHON="$ROOT_DIR/venv/bin/python"

OUTDIR="build/omtf/checkpoints/internal_h64_hn025"
EVALDIR="build/omtf/eval"

echo "=== TRAIN OMTF-internal h64 hn025 ==="
"$PYTHON" -u src/omtf/train_g.py \
    --cache-dir  build/omtf/cache_internal_g_v1 \
    --datasets   G1 G2 G3 G4 G5 G6 G7 G8 B4 \
    --repeat     B4:6 G7:4 G8:4 \
    --hidden     64 \
    --epochs     100 \
    --batch-size 4096 \
    --num-workers 4 \
    --amp \
    --scheduler  cosine \
    --save-epochs 50 75 100 \
    --w-hard-neg 0.25 \
    --output-dir "$OUTDIR" \
    --device     cuda

echo "=== EVAL best ==="
mkdir -p "$EVALDIR"
"$PYTHON" -u scripts/omtf/eval_internal.py \
    --checkpoint "$OUTDIR/omtf_internal_best.pt" \
    --cache-dir  build/omtf/cache_internal_g_v1 \
    --datasets   G1 G2 G3 G4 G5 G6 G7 G8 B4 \
    --threshold  0.0 \
    --output     "$EVALDIR/internal_h64_hn025_best_eval.md" \
    --device     cuda
