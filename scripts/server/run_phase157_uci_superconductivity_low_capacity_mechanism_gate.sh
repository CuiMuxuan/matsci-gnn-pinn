#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
PHASE156_DIR="${PHASE156_DIR:-docs/results/phase156_uci_superconductivity_focused_review}"
RAW_PATH="${RAW_PATH:-data/raw/external/phase155_uci_superconductivity/superconductivty_data.zip}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase157_uci_superconductivity_low_capacity_mechanism_gate}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/build_phase157_uci_superconductivity_low_capacity_mechanism_gate.py \
  --phase156-dir "$PHASE156_DIR" \
  --raw-path "$RAW_PATH" \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase157_uci_superconductivity_low_capacity_mechanism_manifest.json"

SUMMARY_SCRIPT="$(mktemp)"
trap 'rm -f "$SUMMARY_SCRIPT"' EXIT
cat > "$SUMMARY_SCRIPT" <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase157_uci_superconductivity_low_capacity_mechanism_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("uci_superconductivity_phase157_gate", gate["status"])
print("uci_superconductivity_phase157_gate selected_profile", gate["selected_low_capacity_profile"])
print("uci_superconductivity_phase157_gate selected_model", gate["selected_low_capacity_model_label"])
print("uci_superconductivity_phase157_gate focused_validation_allowed", gate["phase157_focused_validation_allowed"])
print("uci_superconductivity_phase157_gate phase157_model_mechanism_allowed", gate["phase157_model_mechanism_allowed"])
print("uci_superconductivity_phase157_gate phase157_model_training_allowed", gate["phase157_model_training_allowed"])
print("uci_superconductivity_phase157_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
print("uci_superconductivity_phase157_gate a100_80gb_request_now", gate["a100_80gb_request_now"])
if gate["phase157_model_training_allowed"]:
    raise SystemExit("Phase 157 must not open neural/model training")
if gate["a100_training_allowed_now"] or gate["a100_80gb_request_now"]:
    raise SystemExit("Phase 157 must not request A100 training or A100-SXM4-80GB")
PY

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 "$SUMMARY_SCRIPT"
