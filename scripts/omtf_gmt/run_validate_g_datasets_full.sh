#!/usr/bin/env bash
set -euo pipefail
DATASET="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
exec "$ROOT_DIR/venv/bin/python" -u scripts/omtf_gmt/validate_new_gmt_datasets.py \
    --data-dir /lustre/ific.uv.es/ml/uovi156/data/prod \
    --datasets "$DATASET" \
    --all-files \
    --output "build/omtf_gmt/eval/new_dataset_validation_full_${DATASET}.md"
