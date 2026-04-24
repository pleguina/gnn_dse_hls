#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x "$ROOT_DIR/venv/bin/python" ]]; then
  echo "Missing venv at $ROOT_DIR/venv"
  exit 3
fi

exec "$ROOT_DIR/venv/bin/python" -u scripts/omtf/dataset_audit.py \
  --cache-dir   build/omtf/cache/schema_v1_graph \
  --checkpoint  build/omtf/checkpoints/edge_compat_best_trig.pt \
  --output-dir  build/omtf/audit \
  --device      cuda \
  --phases 0 1 2 3d 3c 3b 5 6 7
