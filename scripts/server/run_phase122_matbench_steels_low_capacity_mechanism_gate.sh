#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase122_matbench_steels_low_capacity_mechanism_gate}"
PHASE120_DIR="${PHASE120_DIR:-docs/results/phase120_matbench_steels_baseline_gate}"
PHASE121_DIR="${PHASE121_DIR:-docs/results/phase121_matbench_steels_focused_review}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/build_phase122_matbench_steels_low_capacity_mechanism_gate.py \
  --phase120-dir "$PHASE120_DIR" \
  --phase121-dir "$PHASE121_DIR" \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase122_matbench_steels_low_capacity_mechanism_manifest.json"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 - <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase122_matbench_steels_low_capacity_mechanism_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("matbench_steels_low_capacity_mechanism_gate", gate["status"])
print("matbench_steels_low_capacity_mechanism_gate blocking_audits", gate["blocking_audits"])
print("matbench_steels_low_capacity_mechanism_gate selected_profile", gate["selected_low_capacity_profile"])
print("matbench_steels_low_capacity_mechanism_gate selected_model", gate["selected_low_capacity_model_label"])
print("matbench_steels_low_capacity_mechanism_gate phase122_model_mechanism_allowed", gate["phase122_model_mechanism_allowed"])
print("matbench_steels_low_capacity_mechanism_gate phase122_model_training_allowed", gate["phase122_model_training_allowed"])
print("matbench_steels_low_capacity_mechanism_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
print("matbench_steels_low_capacity_mechanism_gate a100_80gb_request_now", gate["a100_80gb_request_now"])
if gate["phase122_model_training_allowed"] or gate["a100_training_allowed_now"] or gate["a100_80gb_request_now"]:
    raise SystemExit("Phase 122 must remain a no-training low-capacity mechanism gate")
PY
