#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase113_nist_ammt_melt_pool_focused_review}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/build_phase113_nist_ammt_melt_pool_focused_review.py \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase113_nist_ammt_melt_pool_focused_review_a100_manifest.json"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 - <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase113_nist_ammt_melt_pool_focused_review_a100_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("melt_pool_focused_review_gate", gate["status"])
print("melt_pool_focused_review_gate mechanism_allowed_targets", gate["mechanism_allowed_targets"])
print("melt_pool_focused_review_gate model_mechanism_allowed", gate["phase113_model_mechanism_allowed"])
print("melt_pool_focused_review_gate model_training_allowed", gate["phase113_model_training_allowed"])
print("melt_pool_focused_review_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
print("melt_pool_focused_review_gate a100_80gb_request_now", gate["a100_80gb_request_now"])
PY
