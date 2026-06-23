#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase169_hidden_source_closure_identifiability_gate}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/build_phase169_hidden_source_closure_identifiability_gate.py \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase169_hidden_source_closure_identifiability_manifest.json"

SUMMARY_SCRIPT="$(mktemp)"
trap 'rm -f "$SUMMARY_SCRIPT"' EXIT
cat > "$SUMMARY_SCRIPT" <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase169_hidden_source_closure_identifiability_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("hidden_source_closure_phase169_gate", gate["status"])
print("hidden_source_closure_phase169_gate candidate_method", gate["candidate_method"])
print("hidden_source_closure_phase169_gate best_control_method", gate["best_control_method"])
print("hidden_source_closure_phase169_gate phase170_low_budget_mechanism_design_allowed", gate["phase170_low_budget_mechanism_design_allowed"])
print("hidden_source_closure_phase169_gate phase169_model_mechanism_allowed", gate["phase169_model_mechanism_allowed"])
print("hidden_source_closure_phase169_gate phase169_model_training_allowed", gate["phase169_model_training_allowed"])
print("hidden_source_closure_phase169_gate phase170_training_allowed_now", gate["phase170_training_allowed_now"])
print("hidden_source_closure_phase169_gate bayesian_pinn_training_allowed_now", gate["bayesian_pinn_training_allowed_now"])
print("hidden_source_closure_phase169_gate adaptive_sampling_training_allowed_now", gate["adaptive_sampling_training_allowed_now"])
print("hidden_source_closure_phase169_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
print("hidden_source_closure_phase169_gate a100_80gb_request_now", gate["a100_80gb_request_now"])
if gate["phase169_model_mechanism_allowed"] or gate["phase169_model_training_allowed"]:
    raise SystemExit("Phase 169 must not open model mechanism or model training")
if gate["phase170_training_allowed_now"]:
    raise SystemExit("Phase 169 may design Phase 170 but must not start training now")
if gate["bayesian_pinn_training_allowed_now"] or gate["adaptive_sampling_training_allowed_now"]:
    raise SystemExit("Phase 169 must not open Bayesian/adaptive PINN training")
if gate["a100_training_allowed_now"] or gate["a100_80gb_request_now"]:
    raise SystemExit("Phase 169 must not request A100 training or 80GB")
PY

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 "$SUMMARY_SCRIPT"
