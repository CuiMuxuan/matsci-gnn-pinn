#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase134_matbench_log_gvrh_focused_review}"
PHASE129_DIR="${PHASE129_DIR:-docs/results/phase129_matbench_log_gvrh_baseline_gate}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/build_phase134_matbench_log_gvrh_focused_review.py \
  --phase129-dir "$PHASE129_DIR" \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase134_matbench_log_gvrh_focused_review_manifest.json"

SUMMARY_SCRIPT="$(mktemp)"
trap 'rm -f "$SUMMARY_SCRIPT"' EXIT
cat > "$SUMMARY_SCRIPT" <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase134_matbench_log_gvrh_focused_review_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("matbench_log_gvrh_focused_review_gate", gate["status"])
print("matbench_log_gvrh_focused_review_gate blocking_audits", gate["blocking_audits"])
print("matbench_log_gvrh_focused_review_gate phase134_model_mechanism_allowed", gate["phase134_model_mechanism_allowed"])
print("matbench_log_gvrh_focused_review_gate phase134_model_training_allowed", gate["phase134_model_training_allowed"])
print("matbench_log_gvrh_focused_review_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
print("matbench_log_gvrh_focused_review_gate a100_80gb_request_now", gate["a100_80gb_request_now"])
if gate["phase134_model_training_allowed"] or gate["a100_training_allowed_now"] or gate["a100_80gb_request_now"]:
    raise SystemExit("Phase 134 must remain a no-training focused review")
PY

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 "$SUMMARY_SCRIPT"
