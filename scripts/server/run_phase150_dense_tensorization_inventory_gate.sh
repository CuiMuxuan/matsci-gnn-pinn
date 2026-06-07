#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase150_dense_tensorization_inventory_gate}"
LOG_DIR="${LOG_DIR:-logs}"
MAX_PREVIEW_ROWS="${MAX_PREVIEW_ROWS:-20000}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/build_phase150_dense_tensorization_inventory_gate.py \
  --output-dir "$OUTPUT_DIR" \
  --max-preview-rows "$MAX_PREVIEW_ROWS" \
  > "$LOG_DIR/phase150_dense_tensorization_inventory_manifest.json"

SUMMARY_SCRIPT="$(mktemp)"
trap 'rm -f "$SUMMARY_SCRIPT"' EXIT
cat > "$SUMMARY_SCRIPT" <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase150_dense_tensorization_inventory_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("dense_tensorization_phase150_gate", gate["status"])
print(
    "dense_tensorization_phase150_gate tensorizable_candidate_rows",
    gate["tensorizable_candidate_rows"],
)
print(
    "dense_tensorization_phase150_gate operator_gap_ready_rows",
    gate["operator_gap_ready_rows"],
)
print(
    "dense_tensorization_phase150_gate phase151_fixed_grid_baseline_review_allowed",
    gate["phase151_fixed_grid_baseline_review_allowed"],
)
print(
    "dense_tensorization_phase150_gate operator_training_allowed_now",
    gate["operator_training_allowed_now"],
)
print(
    "dense_tensorization_phase150_gate phase150_model_training_allowed",
    gate["phase150_model_training_allowed"],
)
print(
    "dense_tensorization_phase150_gate a100_training_allowed_now",
    gate["a100_training_allowed_now"],
)
print(
    "dense_tensorization_phase150_gate a100_80gb_request_now",
    gate["a100_80gb_request_now"],
)
if gate["operator_training_allowed_now"] or gate["phase150_model_training_allowed"]:
    raise SystemExit("Phase 150 must not open operator/model training")
if gate["a100_training_allowed_now"] or gate["a100_80gb_request_now"]:
    raise SystemExit("Phase 150 must not request A100 training or 80GB")
PY

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 "$SUMMARY_SCRIPT"
