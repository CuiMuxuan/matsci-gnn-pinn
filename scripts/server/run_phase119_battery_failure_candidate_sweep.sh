#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase119_battery_failure_candidate_sweep}"
PHASE117_DIR="${PHASE117_DIR:-docs/results/phase117_battery_failure_databank_gate}"
PHASE118_DIR="${PHASE118_DIR:-docs/results/phase118_battery_failure_focused_review}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/build_phase119_battery_failure_candidate_sweep.py \
  --phase117-dir "$PHASE117_DIR" \
  --phase118-dir "$PHASE118_DIR" \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase119_battery_failure_candidate_sweep_manifest.json"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 - <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase119_battery_failure_candidate_sweep_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("battery_failure_candidate_sweep_gate", gate["status"])
print("battery_failure_candidate_sweep_gate allowed_targets", gate["allowed_candidate_targets"])
print("battery_failure_candidate_sweep_gate phase119_model_mechanism_allowed", gate["phase119_model_mechanism_allowed"])
print("battery_failure_candidate_sweep_gate phase119_model_training_allowed", gate["phase119_model_training_allowed"])
print("battery_failure_candidate_sweep_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
print("battery_failure_candidate_sweep_gate a100_80gb_request_now", gate["a100_80gb_request_now"])
if gate["phase119_model_training_allowed"] or gate["a100_training_allowed_now"] or gate["a100_80gb_request_now"]:
    raise SystemExit("Phase 119 must remain a no-training candidate sweep")
PY
