#!/usr/bin/env bash
# EdgeCompat h64 — Phase B4 TPS: train then eval.
# Uses same dataset mix as KMTF B3d baseline (B4×6, G7×4, G8×4).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
PYTHON="$ROOT_DIR/venv/bin/python"

echo "=== TRAIN edge_compat h64 B4_tps ==="
"$PYTHON" -u src/omtf_gmt/train.py \
    --cache-dir  build/omtf_gmt/cache_v2_tps \
    --datasets   G1 G2 G3 G4 G5 G6 G7 G8 B4 \
    --repeat     B4:6 G7:4 G8:4 \
    --model      edge_compat \
    --hidden     64 \
    --epochs     50 \
    --batch-size 4096 \
    --num-workers 4 \
    --amp \
    --output-dir build/omtf_gmt/checkpoints/edge_compat_h64_B4_tps \
    --device     cuda

echo "=== EVAL edge_compat h64 B4_tps ==="
"$PYTHON" -u scripts/omtf_gmt/eval_gmt.py \
    --checkpoint build/omtf_gmt/checkpoints/edge_compat_h64_B4_tps/gmt_edge_compat_best.pt \
    --cache-dir  build/omtf_gmt/cache_v2_tps \
    --datasets   G1 G2 G3 G4 G5 G6 G7 G8 B4 \
    --threshold  0.0 \
    --output     build/omtf_gmt/eval/edge_compat_h64_B4_tps_eval.md \
    --device     cuda
