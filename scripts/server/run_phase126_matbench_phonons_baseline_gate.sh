#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase126_matbench_phonons_baseline_gate}"
RAW_PATH="${RAW_PATH:-data/raw/external/matbench_phonons/matbench_phonons.json.gz}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/build_phase126_matbench_phonons_baseline_gate.py \
  --raw-path "$RAW_PATH" \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase126_matbench_phonons_manifest.json"

SUMMARY_SCRIPT="$(mktemp)"
trap 'rm -f "$SUMMARY_SCRIPT"' EXIT
cat > "$SUMMARY_SCRIPT" <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase126_matbench_phonons_manifest.json").read_text(encoding="utf-8")
)
gate = manifest["gate"]
print("matbench_phonons_gate", gate["status"])
print("matbench_phonons_gate selected_profile", gate["selected_profile"])
print("matbench_phonons_gate selected_method", gate["selected_method"])
print("matbench_phonons_gate focused_review_allowed", gate["phase126_focused_review_allowed"])
print("matbench_phonons_gate phase126_model_mechanism_allowed", gate["phase126_model_mechanism_allowed"])
print("matbench_phonons_gate phase126_model_training_allowed", gate["phase126_model_training_allowed"])
print("matbench_phonons_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
print("matbench_phonons_gate a100_80gb_request_now", gate["a100_80gb_request_now"])
if gate["phase126_model_training_allowed"] or gate["a100_training_allowed_now"] or gate["a100_80gb_request_now"]:
    raise SystemExit("Phase 126 must remain a no-training baseline gate")
PY

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 "$SUMMARY_SCRIPT"
