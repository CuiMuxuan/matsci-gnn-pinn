#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
DATA_ROOT="${DATA_ROOT:-data/raw/nist_ammt/mds2_2044}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase103_nist_ammt_registered_intake}"
LOG_DIR="${LOG_DIR:-logs}"
DOWNLOAD_BACKEND="${DOWNLOAD_BACKEND:-wget}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-5}"
DOWNLOAD_TIMEOUT_SECONDS="${DOWNLOAD_TIMEOUT_SECONDS:-900}"
MAX_MEMBERS="${MAX_MEMBERS:-2000}"
PHASE103_LARGE_DOWNLOADS="${PHASE103_LARGE_DOWNLOADS:-0}"

mkdir -p "$LOG_DIR" "$DATA_ROOT" "$OUTPUT_DIR"

large_args=()
if [[ "$PHASE103_LARGE_DOWNLOADS" == "1" ]]; then
  large_args+=(--large-downloads)
fi

run_python() {
  PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
    "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 "$@"
}

run_python scripts/server/phase103_nist_ammt_registered_intake_audit.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --download \
  "${large_args[@]}" \
  --download-backend "$DOWNLOAD_BACKEND" \
  --retries "$DOWNLOAD_RETRIES" \
  --timeout-seconds "$DOWNLOAD_TIMEOUT_SECONDS" \
  --max-members "$MAX_MEMBERS" \
  > "$LOG_DIR/phase103_nist_ammt_intake_a100_manifest.json"

run_python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("logs/phase103_nist_ammt_intake_a100_manifest.json").read_text(encoding="utf-8"))
gate = manifest["gate"]
print("phase103_status", gate["status"])
print("metadata_ready", gate["metadata_ready"])
print("registration_hits", gate["registration_keyword_hits"])
print("timing_hits", gate["timing_keyword_hits"])
print("phase104_baseline_smoke_allowed", gate["phase104_baseline_smoke_allowed"])
print("a100_training_allowed_now", gate["a100_training_allowed_now"])
PY
