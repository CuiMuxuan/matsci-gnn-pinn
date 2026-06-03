#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
DATA_ROOT="${DATA_ROOT:-data/raw/nist_ammt/mds2_2044}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase104_nist_ammt_baseline_smoke}"
LOG_DIR="${LOG_DIR:-logs}"
MAX_SOURCE_ROWS="${MAX_SOURCE_ROWS:-0}"
MIN_ROWS_FOR_MECHANISM="${MIN_ROWS_FOR_MECHANISM:-100}"
REGISTERED_ROWS_PER_TARGET_TYPE="${REGISTERED_ROWS_PER_TARGET_TYPE:-64}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

run_python() {
  PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
    "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 "$@"
}

run_python scripts/server/phase103_nist_ammt_tiny_registered_table_builder.py \
  --output-dir "$OUTPUT_DIR" \
  --join-candidates-csv "docs/results/phase103_nist_ammt_registered_intake/phase103_nist_ammt_source_target_join_candidates.csv" \
  --rows-per-target-type "$REGISTERED_ROWS_PER_TARGET_TYPE" \
  > "$LOG_DIR/phase104_nist_ammt_expanded_registered_table_a100_manifest.json"

run_python scripts/server/phase104_nist_ammt_tiny_numeric_field_builder.py \
  --data-root "$DATA_ROOT" \
  --tiny-table "$OUTPUT_DIR/phase103_nist_ammt_tiny_registered_source_target_table.csv" \
  --output-dir "$OUTPUT_DIR" \
  --max-source-rows "$MAX_SOURCE_ROWS" \
  > "$LOG_DIR/phase104_nist_ammt_tiny_numeric_field_a100_manifest.json"

run_python scripts/server/phase104_nist_ammt_baseline_smoke.py \
  --output-dir "$OUTPUT_DIR" \
  --min-rows-for-mechanism "$MIN_ROWS_FOR_MECHANISM" \
  > "$LOG_DIR/phase104_nist_ammt_baseline_smoke_a100_manifest.json"

run_python - <<'PY'
import json
from pathlib import Path
for label, path in {
    "registered": Path("logs/phase104_nist_ammt_expanded_registered_table_a100_manifest.json"),
    "numeric": Path("logs/phase104_nist_ammt_tiny_numeric_field_a100_manifest.json"),
    "baseline": Path("logs/phase104_nist_ammt_baseline_smoke_a100_manifest.json"),
}.items():
    manifest = json.loads(path.read_text(encoding="utf-8"))
    gate = manifest["gate"]
    print(label, gate["status"])
    print(label, "phase105_model_mechanism_allowed", gate["phase105_model_mechanism_allowed"])
    print(label, "a100_training_allowed_now", gate["a100_training_allowed_now"])
PY
