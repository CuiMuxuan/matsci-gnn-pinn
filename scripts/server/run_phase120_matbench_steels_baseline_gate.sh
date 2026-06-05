#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase120_matbench_steels_baseline_gate}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/build_phase120_matbench_steels_baseline_gate.py \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase120_matbench_steels_baseline_gate_manifest.json"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 - <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase120_matbench_steels_baseline_gate_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("matbench_steels_baseline_gate", gate["status"])
print("matbench_steels_baseline_gate selected_profile", gate["selected_profile"])
print("matbench_steels_baseline_gate phase120_focused_review_allowed", gate["phase120_focused_review_allowed"])
print("matbench_steels_baseline_gate phase120_model_mechanism_allowed", gate["phase120_model_mechanism_allowed"])
print("matbench_steels_baseline_gate phase120_model_training_allowed", gate["phase120_model_training_allowed"])
print("matbench_steels_baseline_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
print("matbench_steels_baseline_gate a100_80gb_request_now", gate["a100_80gb_request_now"])
if gate["phase120_model_training_allowed"] or gate["a100_training_allowed_now"] or gate["a100_80gb_request_now"]:
    raise SystemExit("Phase 120 must remain a no-training baseline gate")
PY
