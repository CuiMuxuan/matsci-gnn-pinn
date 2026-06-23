#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
RAW_PATH="${RAW_PATH:-data/raw/external/phase162_uci_steel_industry_energy/steel_industry_energy_consumption.zip}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase162_uci_steel_industry_energy_baseline_gate}"
LOG_DIR="${LOG_DIR:-logs}"
ALLOW_DOWNLOAD="${ALLOW_DOWNLOAD:-0}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

DOWNLOAD_FLAG=()
if [[ "$ALLOW_DOWNLOAD" == "1" ]]; then
  DOWNLOAD_FLAG=(--allow-download)
fi

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/build_phase162_uci_steel_industry_energy_baseline_gate.py \
  --raw-path "$RAW_PATH" \
  --output-dir "$OUTPUT_DIR" \
  "${DOWNLOAD_FLAG[@]}" \
  > "$LOG_DIR/phase162_uci_steel_industry_energy_baseline_manifest.json"

SUMMARY_SCRIPT="$(mktemp)"
trap 'rm -f "$SUMMARY_SCRIPT"' EXIT
cat > "$SUMMARY_SCRIPT" <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase162_uci_steel_industry_energy_baseline_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("uci_steel_energy_phase162_gate", gate["status"])
print("uci_steel_energy_phase162_gate selected_profile", gate["selected_profile"])
print("uci_steel_energy_phase162_gate selected_method", gate["selected_method"])
print("uci_steel_energy_phase162_gate phase163_focused_review_allowed", gate["phase163_focused_review_allowed"])
print("uci_steel_energy_phase162_gate phase162_model_mechanism_allowed", gate["phase162_model_mechanism_allowed"])
print("uci_steel_energy_phase162_gate phase162_model_training_allowed", gate["phase162_model_training_allowed"])
print("uci_steel_energy_phase162_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
print("uci_steel_energy_phase162_gate a100_80gb_request_now", gate["a100_80gb_request_now"])
if gate["phase162_model_training_allowed"]:
    raise SystemExit("Phase 162 must not open neural/model training")
if gate["a100_training_allowed_now"] or gate["a100_80gb_request_now"]:
    raise SystemExit("Phase 162 must not request A100 training or A100-SXM4-80GB")
PY

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 "$SUMMARY_SCRIPT"
