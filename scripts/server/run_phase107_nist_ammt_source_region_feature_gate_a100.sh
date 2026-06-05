#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase107_nist_ammt_source_region_feature_gate}"
LOG_DIR="${LOG_DIR:-logs}"
DATA_ROOT="${DATA_ROOT:-data/raw/nist_ammt/mds2_2044}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/phase107_nist_ammt_source_region_feature_gate.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase107_nist_ammt_source_region_feature_gate_a100_manifest.json"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 - <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase107_nist_ammt_source_region_feature_gate_a100_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("source_region_gate", gate["status"])
print("source_region_gate selected_profile", gate["selected_feature_profile"])
print("source_region_gate focused_review_allowed", gate["phase107_focused_review_allowed"])
print("source_region_gate model_training_allowed", gate["phase107_model_training_allowed"])
print("source_region_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
PY
