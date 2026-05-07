#!/usr/bin/env bash
# Run 1 — threshold scan summary for TPS-h64 best checkpoint.
# Extracts operating-point table from the existing eval JSON (no re-eval needed).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
PYTHON="$ROOT_DIR/venv/bin/python"
EVALDIR="build/omtf_gmt/eval"

echo "=== Generating threshold scan summary from existing eval JSON ==="
"$PYTHON" -u scripts/omtf_gmt/summarize_threshold_scan.py \
    --eval-json "$EVALDIR/edge_compat_h64_B4_tps_100ep_best_eval.json" \
    --output    "$EVALDIR/tps_h64_threshold_scan_summary.md"

echo "=== Done: $EVALDIR/tps_h64_threshold_scan_summary.md ==="
