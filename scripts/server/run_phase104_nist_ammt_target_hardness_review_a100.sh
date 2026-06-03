#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase104_nist_ammt_target_hardness_review}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/phase104_nist_ammt_target_hardness_review.py \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase104_nist_ammt_target_hardness_review_a100_manifest.json"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 - <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase104_nist_ammt_target_hardness_review_a100_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("target_hardness", gate["status"])
print("target_hardness selected_target", gate["selected_target"])
print("target_hardness phase105_model_mechanism_allowed", gate["phase105_model_mechanism_allowed"])
print("target_hardness a100_training_allowed_now", gate["a100_training_allowed_now"])
PY
