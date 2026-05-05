#!/usr/bin/env bash
# Threshold scan: EdgeCompat h128 B3d epoch_0050 at thr = -0.5, 0.0, 0.2, 0.5, 1.0
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
PYTHON="$ROOT_DIR/venv/bin/python"
CKPT="build/omtf_gmt/checkpoints/edge_compat_h128_B3d_100ep/gmt_edge_compat_epoch_0050.pt"
EVALDIR="build/omtf_gmt/eval/threshold_scan_h128_B3d_ep50"
mkdir -p "$EVALDIR"

for THR in -0.5 0.0 0.2 0.5 1.0; do
    TAG="thr$(echo $THR | sed 's/-/neg/' | sed 's/\./p/')"
    echo "=== threshold = $THR ==="
    "$PYTHON" -u scripts/omtf_gmt/eval_gmt.py \
        --checkpoint "$CKPT" \
        --cache-dir  build/omtf_gmt/cache_v2 \
        --datasets   G1 G2 G3 G4 G5 G6 G7 G8 B4 \
        --threshold  $THR \
        --output     "$EVALDIR/eval_${TAG}.md" \
        --device     cuda
done
