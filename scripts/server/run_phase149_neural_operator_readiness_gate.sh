#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase149_neural_operator_readiness_gate}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/build_phase149_neural_operator_readiness_gate.py \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase149_neural_operator_readiness_manifest.json"

SUMMARY_SCRIPT="$(mktemp)"
trap 'rm -f "$SUMMARY_SCRIPT"' EXIT
cat > "$SUMMARY_SCRIPT" <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase149_neural_operator_readiness_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("neural_operator_readiness_phase149_gate", gate["status"])
print("neural_operator_readiness_phase149_gate blocker_rows", gate["blocker_rows"])
print(
    "neural_operator_readiness_phase149_gate phase150_dense_tensorization_inventory_allowed",
    gate["phase150_dense_tensorization_inventory_allowed"],
)
print(
    "neural_operator_readiness_phase149_gate operator_training_allowed_now",
    gate["operator_training_allowed_now"],
)
print(
    "neural_operator_readiness_phase149_gate phase149_model_training_allowed",
    gate["phase149_model_training_allowed"],
)
print(
    "neural_operator_readiness_phase149_gate a100_training_allowed_now",
    gate["a100_training_allowed_now"],
)
print(
    "neural_operator_readiness_phase149_gate a100_80gb_request_now",
    gate["a100_80gb_request_now"],
)
if gate["operator_training_allowed_now"] or gate["phase149_model_training_allowed"]:
    raise SystemExit("Phase 149 must not open operator/model training")
if gate["a100_training_allowed_now"] or gate["a100_80gb_request_now"]:
    raise SystemExit("Phase 149 must not request A100 training or 80GB")
PY

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 "$SUMMARY_SCRIPT"
