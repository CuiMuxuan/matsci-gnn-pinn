#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase117_battery_failure_databank_gate}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/build_phase117_battery_failure_databank_gate.py \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase117_battery_failure_databank_gate_manifest.json"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 - <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase117_battery_failure_databank_gate_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("battery_failure_databank_gate", gate["status"])
print("battery_failure_databank_gate candidates", gate["candidate_targets"])
print("battery_failure_databank_gate phase117_model_mechanism_allowed", gate["phase117_model_mechanism_allowed"])
print("battery_failure_databank_gate phase117_model_training_allowed", gate["phase117_model_training_allowed"])
print("battery_failure_databank_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
print("battery_failure_databank_gate a100_80gb_request_now", gate["a100_80gb_request_now"])
if gate["phase117_model_training_allowed"] or gate["a100_training_allowed_now"] or gate["a100_80gb_request_now"]:
    raise SystemExit("Phase 117 must remain a no-training baseline gate")
PY
