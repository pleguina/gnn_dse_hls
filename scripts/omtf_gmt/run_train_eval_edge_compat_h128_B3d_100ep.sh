#!/usr/bin/env bash
# EdgeCompat h128 B3d — 100 epochs, cosine LR, periodic checkpoints.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
PYTHON="$ROOT_DIR/venv/bin/python"
OUTDIR="build/omtf_gmt/checkpoints/edge_compat_h128_B3d_100ep"
EVALDIR="build/omtf_gmt/eval"

echo "=== TRAIN ==="
"$PYTHON" -u src/omtf_gmt/train.py \
    --cache-dir  build/omtf_gmt/cache_v2 \
    --datasets   G1 G2 G3 G4 G5 G6 G7 G8 B4 \
    --repeat     B4:6 G7:4 G8:4 \
    --model      edge_compat \
    --hidden     128 \
    --epochs     100 \
    --batch-size 4096 \
    --num-workers 4 \
    --amp \
    --scheduler  cosine \
    --save-epochs 50 75 100 \
    --output-dir "$OUTDIR" \
    --device     cuda

echo "=== EVAL (best, ep50, ep75, ep100, last) ==="
for CKPT in \
    "$OUTDIR/gmt_edge_compat_best.pt" \
    "$OUTDIR/gmt_edge_compat_epoch_0050.pt" \
    "$OUTDIR/gmt_edge_compat_epoch_0075.pt" \
    "$OUTDIR/gmt_edge_compat_epoch_0100.pt" \
    "$OUTDIR/gmt_edge_compat_last.pt"; do
    TAG=$(basename "$CKPT" .pt | sed 's/gmt_edge_compat_//')
    "$PYTHON" -u scripts/omtf_gmt/eval_gmt.py \
        --checkpoint "$CKPT" \
        --cache-dir  build/omtf_gmt/cache_v2 \
        --datasets   G1 G2 G3 G4 G5 G6 G7 G8 B4 \
        --threshold  0.0 \
        --output     "$EVALDIR/edge_compat_h128_B3d_100ep_${TAG}_eval.md" \
        --device     cuda
done
