#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase155_uci_superconductivity_baseline_gate}"
RAW_PATH="${RAW_PATH:-data/raw/external/phase155_uci_superconductivity/superconductivty_data.zip}"
LOG_DIR="${LOG_DIR:-logs}"
ALLOW_DOWNLOAD="${ALLOW_DOWNLOAD:-0}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

BUILD_ARGS=(
  scripts/server/build_phase155_uci_superconductivity_baseline_gate.py
  --raw-path "$RAW_PATH"
  --output-dir "$OUTPUT_DIR"
)
if [[ "$ALLOW_DOWNLOAD" == "1" ]]; then
  BUILD_ARGS+=(--allow-download)
fi

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  "${BUILD_ARGS[@]}" \
  > "$LOG_DIR/phase155_uci_superconductivity_baseline_manifest.json"

SUMMARY_SCRIPT="$(mktemp)"
trap 'rm -f "$SUMMARY_SCRIPT"' EXIT
cat > "$SUMMARY_SCRIPT" <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase155_uci_superconductivity_baseline_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("uci_superconductivity_phase155_gate", gate["status"])
print("uci_superconductivity_phase155_gate selected_profile", gate["selected_profile"])
print("uci_superconductivity_phase155_gate selected_method", gate["selected_method"])
print("uci_superconductivity_phase155_gate phase156_focused_review_allowed", gate["phase156_focused_review_allowed"])
print("uci_superconductivity_phase155_gate phase155_model_mechanism_allowed", gate["phase155_model_mechanism_allowed"])
print("uci_superconductivity_phase155_gate phase155_model_training_allowed", gate["phase155_model_training_allowed"])
print("uci_superconductivity_phase155_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
print("uci_superconductivity_phase155_gate a100_80gb_request_now", gate["a100_80gb_request_now"])
if gate["phase155_model_mechanism_allowed"] or gate["phase155_model_training_allowed"]:
    raise SystemExit("Phase 155 must remain a no-training baseline-first source intake")
if gate["a100_training_allowed_now"] or gate["a100_80gb_request_now"]:
    raise SystemExit("Phase 155 must not request A100 training or A100-SXM4-80GB")
PY

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 "$SUMMARY_SCRIPT"
