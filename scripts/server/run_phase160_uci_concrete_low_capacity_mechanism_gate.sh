#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
PHASE159_DIR="${PHASE159_DIR:-docs/results/phase159_uci_concrete_focused_review}"
RAW_PATH="${RAW_PATH:-data/raw/external/phase158_uci_concrete/concrete_compressive_strength.zip}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase160_uci_concrete_low_capacity_mechanism_gate}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/build_phase160_uci_concrete_low_capacity_mechanism_gate.py \
  --phase159-dir "$PHASE159_DIR" \
  --raw-path "$RAW_PATH" \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase160_uci_concrete_low_capacity_mechanism_manifest.json"

SUMMARY_SCRIPT="$(mktemp)"
trap 'rm -f "$SUMMARY_SCRIPT"' EXIT
cat > "$SUMMARY_SCRIPT" <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase160_uci_concrete_low_capacity_mechanism_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("uci_concrete_phase160_gate", gate["status"])
print("uci_concrete_phase160_gate selected_profile", gate["selected_low_capacity_profile"])
print("uci_concrete_phase160_gate selected_model", gate["selected_low_capacity_model_label"])
print("uci_concrete_phase160_gate focused_validation_allowed", gate["phase160_focused_validation_allowed"])
print("uci_concrete_phase160_gate phase160_model_mechanism_allowed", gate["phase160_model_mechanism_allowed"])
print("uci_concrete_phase160_gate phase160_model_training_allowed", gate["phase160_model_training_allowed"])
print("uci_concrete_phase160_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
print("uci_concrete_phase160_gate a100_80gb_request_now", gate["a100_80gb_request_now"])
if gate["phase160_model_training_allowed"]:
    raise SystemExit("Phase 160 must not open neural/model training")
if gate["a100_training_allowed_now"] or gate["a100_80gb_request_now"]:
    raise SystemExit("Phase 160 must not request A100 training or A100-SXM4-80GB")
PY

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 "$SUMMARY_SCRIPT"
