#!/usr/bin/env bash
# Build OMTF-internal G-dataset cache (all datasets, CPU job).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
PYTHON="$ROOT_DIR/venv/bin/python"

echo "=== BUILD OMTF-internal G-dataset cache ==="
"$PYTHON" -u scripts/omtf/build_cache_g_internal.py \
    --data-dir   data/prod \
    --output-dir build/omtf/cache_internal_g_v1

echo "=== DONE ==="
