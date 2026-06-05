#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase111_nist_ammt_registered_target_closure_package}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/build_phase111_nist_ammt_registered_target_closure_package.py \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase111_nist_ammt_registered_target_closure_package_a100_manifest.json"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 - <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase111_nist_ammt_registered_target_closure_package_a100_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("registered_target_closure_gate", gate["status"])
print("registered_target_closure_gate sequence_branch_closed", gate["nist_ammt_sequence_branch_closed"])
print("registered_target_closure_gate appendix_diagnostic_package_ready", gate["appendix_diagnostic_package_ready"])
print("registered_target_closure_gate main_paper_floor", gate["main_paper_floor"])
print("registered_target_closure_gate phase111_model_training_allowed", gate["phase111_model_training_allowed"])
print("registered_target_closure_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
print("registered_target_closure_gate a100_80gb_request_now", gate["a100_80gb_request_now"])
PY
