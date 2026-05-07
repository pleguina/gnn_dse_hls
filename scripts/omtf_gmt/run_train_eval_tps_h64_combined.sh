#!/usr/bin/env bash
# Run 4 — TPS-h64 combined: best hard-neg weight + best unmatched-stub weight.
# Fill in W_HARD_NEG and USW after inspecting runs 2a/2b/2c and 3a/3b.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
PYTHON="$ROOT_DIR/venv/bin/python"

W_HARD_NEG=0.50   # set after reviewing Run 2 results
USW=0.25          # set after reviewing Run 3 results

OUTDIR="build/omtf_gmt/checkpoints/edge_compat_h64_tps_combined"
EVALDIR="build/omtf_gmt/eval"

echo "=== TRAIN edge_compat h64 TPS combined (hn=${W_HARD_NEG}, usw=${USW}) ==="
"$PYTHON" -u src/omtf_gmt/train.py \
    --cache-dir  build/omtf_gmt/cache_v2_tps \
    --datasets   G1 G2 G3 G4 G5 G6 G7 G8 B4 \
    --repeat     B4:6 G7:4 G8:4 \
    --model      edge_compat \
    --hidden     64 \
    --epochs     100 \
    --batch-size 4096 \
    --num-workers 4 \
    --amp \
    --scheduler  cosine \
    --save-epochs 50 75 100 \
    --w-hard-neg "$W_HARD_NEG" \
    --unmatched-stub-weight "$USW" \
    --output-dir "$OUTDIR" \
    --device     cuda

echo "=== EVAL best ==="
"$PYTHON" -u scripts/omtf_gmt/eval_gmt.py \
    --checkpoint "$OUTDIR/gmt_edge_compat_best.pt" \
    --cache-dir  build/omtf_gmt/cache_v2_tps \
    --datasets   G1 G2 G3 G4 G5 G6 G7 G8 B4 \
    --threshold  0.0 \
    --output     "$EVALDIR/edge_compat_h64_tps_combined_best_eval.md" \
    --device     cuda
