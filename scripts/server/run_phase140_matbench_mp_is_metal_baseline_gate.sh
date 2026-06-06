#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase140_matbench_mp_is_metal_baseline_gate}"
RAW_PATH="${RAW_PATH:-data/raw/external/matbench_mp_is_metal/matbench_mp_is_metal.json.gz}"
MAX_ROWS="${MAX_ROWS:-12000}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/build_phase140_matbench_mp_is_metal_baseline_gate.py \
  --raw-path "$RAW_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --max-rows "$MAX_ROWS" \
  > "$LOG_DIR/phase140_matbench_mp_is_metal_manifest.json"

SUMMARY_SCRIPT="$(mktemp)"
trap 'rm -f "$SUMMARY_SCRIPT"' EXIT
cat > "$SUMMARY_SCRIPT" <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("logs/phase140_matbench_mp_is_metal_manifest.json").read_text(encoding="utf-8"))
gate = manifest["gate"]
print("matbench_mp_is_metal_gate", gate["status"])
print("matbench_mp_is_metal_gate row_cap_applied", gate["row_cap_applied"])
print("matbench_mp_is_metal_gate selected_raw_row_count", gate["selected_raw_row_count"])
print("matbench_mp_is_metal_gate selected_profile", gate["selected_profile"])
print("matbench_mp_is_metal_gate selected_method", gate["selected_method"])
print("matbench_mp_is_metal_gate focused_review_allowed", gate["phase140_focused_review_allowed"])
print("matbench_mp_is_metal_gate phase140_model_mechanism_allowed", gate["phase140_model_mechanism_allowed"])
print("matbench_mp_is_metal_gate phase140_model_training_allowed", gate["phase140_model_training_allowed"])
print("matbench_mp_is_metal_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
print("matbench_mp_is_metal_gate a100_80gb_request_now", gate["a100_80gb_request_now"])
if gate["phase140_model_mechanism_allowed"] or gate["phase140_model_training_allowed"]:
    raise SystemExit("Phase 140 must remain a no-training baseline triage gate")
if gate["a100_training_allowed_now"] or gate["a100_80gb_request_now"]:
    raise SystemExit("Phase 140 must not request A100 training or 80GB")
PY

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 "$SUMMARY_SCRIPT"
