#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
PHASE144_DIR="${PHASE144_DIR:-docs/results/phase144_mpea_mechanical_baseline_gate}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase145_mpea_mechanical_focused_review}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/build_phase145_mpea_mechanical_focused_review.py \
  --phase144-dir "$PHASE144_DIR" \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase145_mpea_mechanical_focused_review_manifest.json"

SUMMARY_SCRIPT="$(mktemp)"
trap 'rm -f "$SUMMARY_SCRIPT"' EXIT
cat > "$SUMMARY_SCRIPT" <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("logs/phase145_mpea_mechanical_focused_review_manifest.json").read_text(encoding="utf-8"))
gate = manifest["gate"]
print("mpea_mechanical_phase145_gate", gate["status"])
print("mpea_mechanical_phase145_gate selected_target", gate["selected_target"])
print("mpea_mechanical_phase145_gate viable_split_reviews", gate["viable_split_reviews"])
print("mpea_mechanical_phase145_gate split_pass_rate", gate["split_pass_rate"])
print("mpea_mechanical_phase145_gate phase145_model_mechanism_allowed", gate["phase145_model_mechanism_allowed"])
print("mpea_mechanical_phase145_gate phase145_model_training_allowed", gate["phase145_model_training_allowed"])
print("mpea_mechanical_phase145_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
print("mpea_mechanical_phase145_gate a100_80gb_request_now", gate["a100_80gb_request_now"])
if gate["phase145_model_training_allowed"]:
    raise SystemExit("Phase 145 must not open model training")
if gate["a100_training_allowed_now"] or gate["a100_80gb_request_now"]:
    raise SystemExit("Phase 145 must not request A100 training or 80GB")
PY

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 "$SUMMARY_SCRIPT"
