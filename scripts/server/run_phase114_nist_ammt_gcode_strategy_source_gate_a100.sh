#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase114_nist_ammt_gcode_strategy_source_gate}"
LOG_DIR="${LOG_DIR:-logs}"
DATA_ROOT="${DATA_ROOT:-data/raw/nist_ammt/mds2_2044}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/phase114_nist_ammt_gcode_strategy_source_gate.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase114_nist_ammt_gcode_strategy_source_gate_a100_manifest.json"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 - <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase114_nist_ammt_gcode_strategy_source_gate_a100_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("gcode_strategy_source_gate", gate["status"])
print("gcode_strategy_source_gate row_count", gate["row_count"])
print("gcode_strategy_source_gate selected_target", gate["selected_target"])
print("gcode_strategy_source_gate focused_review_allowed", gate["phase114_focused_review_allowed"])
print("gcode_strategy_source_gate model_training_allowed", gate["phase114_model_training_allowed"])
print("gcode_strategy_source_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
print("gcode_strategy_source_gate a100_80gb_request_now", gate["a100_80gb_request_now"])
PY
