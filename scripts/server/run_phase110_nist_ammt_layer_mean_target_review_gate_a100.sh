#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase110_nist_ammt_layer_mean_target_review_gate}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/phase110_nist_ammt_layer_mean_target_review_gate.py \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase110_nist_ammt_layer_mean_target_review_gate_a100_manifest.json"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 - <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase110_nist_ammt_layer_mean_target_review_gate_a100_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("layer_mean_target_review_gate", gate["status"])
print("layer_mean_target_review_gate reviewed_target", gate["reviewed_target"])
print("layer_mean_target_review_gate layer_time_shortcut_detected", gate["layer_time_shortcut_detected"])
print("layer_mean_target_review_gate model_mechanism_allowed", gate["phase110_model_mechanism_allowed"])
print("layer_mean_target_review_gate model_training_allowed", gate["phase110_model_training_allowed"])
print("layer_mean_target_review_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
PY
