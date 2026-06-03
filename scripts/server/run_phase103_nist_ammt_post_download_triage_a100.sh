#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
DATA_ROOT="${DATA_ROOT:-data/raw/nist_ammt/mds2_2044}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase103_nist_ammt_registered_intake}"
LOG_DIR="${LOG_DIR:-logs}"
MAX_MEMBERS="${MAX_MEMBERS:-100000}"
MAX_CANDIDATES_PER_ROLE="${MAX_CANDIDATES_PER_ROLE:-200}"
MAX_PER_ROLE="${MAX_PER_ROLE:-8}"
MAX_BYTES="${MAX_BYTES:-4096}"
DEEP_MAX_TARGET_GROUPS="${DEEP_MAX_TARGET_GROUPS:-8}"
DEEP_MAX_SAMPLES_PER_GROUP="${DEEP_MAX_SAMPLES_PER_GROUP:-2}"
DEEP_MAX_BINARY_HEADER_BYTES="${DEEP_MAX_BINARY_HEADER_BYTES:-4096}"
DEEP_MAX_TIMING_ROWS="${DEEP_MAX_TIMING_ROWS:-32}"
DEEP_MAX_TEXT_SCAN_BYTES="${DEEP_MAX_TEXT_SCAN_BYTES:-65536}"
JOIN_MIN_TARGET_COVERAGE="${JOIN_MIN_TARGET_COVERAGE:-0.95}"
JOIN_MIN_LAYER_PAIRS="${JOIN_MIN_LAYER_PAIRS:-20}"
JOIN_MIN_MELT_POOL_PAIRS="${JOIN_MIN_MELT_POOL_PAIRS:-5}"
TINY_ROWS_PER_TARGET_TYPE="${TINY_ROWS_PER_TARGET_TYPE:-12}"
INTAKE_MANIFEST="${INTAKE_MANIFEST:-logs/phase103_nist_ammt_intake_a100_manifest.json}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

if command -v tmux >/dev/null 2>&1 && tmux has-session -t phase103_nist_large 2>/dev/null; then
  echo "phase103_nist_large is still active; wait for large-download audit to finish before triage" >&2
  exit 2
fi

if [[ ! -s "$INTAKE_MANIFEST" ]]; then
  echo "missing non-empty intake manifest: $INTAKE_MANIFEST" >&2
  exit 2
fi

run_python() {
  PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
    "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 "$@"
}

run_python scripts/server/phase103_nist_ammt_schema_scout.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --max-members "$MAX_MEMBERS" \
  --max-candidates-per-role "$MAX_CANDIDATES_PER_ROLE" \
  > "$LOG_DIR/phase103_nist_ammt_schema_scout_a100_manifest.json"

run_python scripts/server/phase103_nist_ammt_member_schema_sampler.py \
  --data-root "$DATA_ROOT" \
  --candidates-csv "$OUTPUT_DIR/phase103_nist_ammt_schema_scout_candidates.csv" \
  --output-dir "$OUTPUT_DIR" \
  --max-per-role "$MAX_PER_ROLE" \
  --max-bytes "$MAX_BYTES" \
  > "$LOG_DIR/phase103_nist_ammt_member_schema_sampler_a100_manifest.json"

run_python scripts/server/phase103_nist_ammt_deep_registration_probe.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --max-members "$MAX_MEMBERS" \
  --max-target-groups "$DEEP_MAX_TARGET_GROUPS" \
  --max-samples-per-group "$DEEP_MAX_SAMPLES_PER_GROUP" \
  --max-binary-header-bytes "$DEEP_MAX_BINARY_HEADER_BYTES" \
  --max-timing-rows "$DEEP_MAX_TIMING_ROWS" \
  --max-text-scan-bytes "$DEEP_MAX_TEXT_SCAN_BYTES" \
  > "$LOG_DIR/phase103_nist_ammt_deep_registration_probe_a100_manifest.json"

run_python scripts/server/phase103_nist_ammt_join_probe.py \
  --sequence-groups-csv "$OUTPUT_DIR/phase103_nist_ammt_deep_sequence_groups.csv" \
  --output-dir "$OUTPUT_DIR" \
  --min-target-coverage "$JOIN_MIN_TARGET_COVERAGE" \
  --min-layer-pairs "$JOIN_MIN_LAYER_PAIRS" \
  --min-melt-pool-pairs "$JOIN_MIN_MELT_POOL_PAIRS" \
  > "$LOG_DIR/phase103_nist_ammt_join_probe_a100_manifest.json"

run_python scripts/server/phase103_nist_ammt_tiny_table_feasibility_gate.py \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase103_nist_ammt_tiny_table_feasibility_a100_manifest.json"

run_python scripts/server/phase103_nist_ammt_tiny_registered_table_builder.py \
  --output-dir "$OUTPUT_DIR" \
  --rows-per-target-type "$TINY_ROWS_PER_TARGET_TYPE" \
  > "$LOG_DIR/phase103_nist_ammt_tiny_registered_table_a100_manifest.json"

run_python - <<'PY'
import json
from pathlib import Path

paths = {
    "schema": Path("logs/phase103_nist_ammt_schema_scout_a100_manifest.json"),
    "sampler": Path("logs/phase103_nist_ammt_member_schema_sampler_a100_manifest.json"),
    "deep": Path("logs/phase103_nist_ammt_deep_registration_probe_a100_manifest.json"),
    "join": Path("logs/phase103_nist_ammt_join_probe_a100_manifest.json"),
    "tiny": Path("logs/phase103_nist_ammt_tiny_table_feasibility_a100_manifest.json"),
    "tiny_table": Path("logs/phase103_nist_ammt_tiny_registered_table_a100_manifest.json"),
}
for label, path in paths.items():
    manifest = json.loads(path.read_text(encoding="utf-8"))
    gate = manifest["gate"]
    print(label, gate["status"])
    print(label, "phase104_baseline_smoke_allowed", gate["phase104_baseline_smoke_allowed"])
    print(label, "a100_training_allowed_now", gate["a100_training_allowed_now"])
PY
