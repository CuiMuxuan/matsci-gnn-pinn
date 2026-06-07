#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase151_fixed_grid_dense_baseline_review}"
LOG_DIR="${LOG_DIR:-logs}"
MIN_POINTS_PER_FRAME="${MIN_POINTS_PER_FRAME:-10}"
MIN_SUMMARY_ROWS="${MIN_SUMMARY_ROWS:-9}"
N_ESTIMATORS="${N_ESTIMATORS:-80}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/build_phase151_fixed_grid_dense_baseline_review.py \
  --output-dir "$OUTPUT_DIR" \
  --min-points-per-frame "$MIN_POINTS_PER_FRAME" \
  --min-summary-rows "$MIN_SUMMARY_ROWS" \
  --n-estimators "$N_ESTIMATORS" \
  > "$LOG_DIR/phase151_fixed_grid_dense_baseline_manifest.json"

SUMMARY_SCRIPT="$(mktemp)"
trap 'rm -f "$SUMMARY_SCRIPT"' EXIT
cat > "$SUMMARY_SCRIPT" <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase151_fixed_grid_dense_baseline_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("fixed_grid_dense_phase151_gate", gate["status"])
print(
    "fixed_grid_dense_phase151_gate leakage_safe_source_rows",
    gate["leakage_safe_source_rows"],
)
print(
    "fixed_grid_dense_phase151_gate phase152_low_capacity_dense_design_candidates",
    gate["phase152_low_capacity_dense_design_candidates"],
)
print(
    "fixed_grid_dense_phase151_gate phase151_model_mechanism_allowed",
    gate["phase151_model_mechanism_allowed"],
)
print(
    "fixed_grid_dense_phase151_gate phase151_model_training_allowed",
    gate["phase151_model_training_allowed"],
)
print(
    "fixed_grid_dense_phase151_gate operator_training_allowed_now",
    gate["operator_training_allowed_now"],
)
print(
    "fixed_grid_dense_phase151_gate a100_training_allowed_now",
    gate["a100_training_allowed_now"],
)
print(
    "fixed_grid_dense_phase151_gate a100_80gb_request_now",
    gate["a100_80gb_request_now"],
)
if gate["phase151_model_training_allowed"] or gate["operator_training_allowed_now"]:
    raise SystemExit("Phase 151 must not open model/operator training")
if gate["a100_training_allowed_now"] or gate["a100_80gb_request_now"]:
    raise SystemExit("Phase 151 must not request A100 training or 80GB")
PY

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 "$SUMMARY_SCRIPT"
