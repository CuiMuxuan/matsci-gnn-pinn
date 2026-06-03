#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase105_nist_ammt_source_path_feature_gate}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/phase105_nist_ammt_source_path_feature_gate.py \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase105_nist_ammt_source_path_feature_gate_a100_manifest.json"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 - <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase105_nist_ammt_source_path_feature_gate_a100_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("source_path_gate", gate["status"])
print("source_path_gate selected_profile", gate["selected_feature_profile"])
print("source_path_gate phase105_cpu_smoke_allowed", gate["phase105_cpu_smoke_allowed"])
print("source_path_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
PY
