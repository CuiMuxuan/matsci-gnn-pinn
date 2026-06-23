#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase173_trainable_hidden_closure_low_budget_smoke}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/build_phase173_trainable_hidden_closure_low_budget_smoke.py \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase173_trainable_hidden_closure_low_budget_smoke_manifest.json"

SUMMARY_SCRIPT="$(mktemp)"
trap 'rm -f "$SUMMARY_SCRIPT"' EXIT
cat > "$SUMMARY_SCRIPT" <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase173_trainable_hidden_closure_low_budget_smoke_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("trainable_hidden_closure_phase173_gate", gate["status"])
print("trainable_hidden_closure_phase173_gate selected_variant", gate["selected_variant"])
print("trainable_hidden_closure_phase173_gate best_control_variant", gate["best_control_variant"])
print("trainable_hidden_closure_phase173_gate phase174_low_capacity_hidden_closure_design_allowed", gate["phase174_low_capacity_hidden_closure_design_allowed"])
print("trainable_hidden_closure_phase173_gate phase173_model_mechanism_allowed", gate["phase173_model_mechanism_allowed"])
print("trainable_hidden_closure_phase173_gate phase173_model_training_allowed", gate["phase173_model_training_allowed"])
print("trainable_hidden_closure_phase173_gate phase174_training_allowed_now", gate["phase174_training_allowed_now"])
print("trainable_hidden_closure_phase173_gate bayesian_pinn_training_allowed_now", gate["bayesian_pinn_training_allowed_now"])
print("trainable_hidden_closure_phase173_gate adaptive_sampling_training_allowed_now", gate["adaptive_sampling_training_allowed_now"])
print("trainable_hidden_closure_phase173_gate gcn_pinn_training_allowed_now", gate["gcn_pinn_training_allowed_now"])
print("trainable_hidden_closure_phase173_gate cnn_operator_training_allowed_now", gate["cnn_operator_training_allowed_now"])
print("trainable_hidden_closure_phase173_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
print("trainable_hidden_closure_phase173_gate a100_80gb_request_now", gate["a100_80gb_request_now"])
if gate["phase173_model_mechanism_allowed"] or gate["phase173_model_training_allowed"]:
    raise SystemExit("Phase 173 must not open model mechanism or model training")
if gate["phase174_training_allowed_now"]:
    raise SystemExit("Phase 173 may open only a later design gate, not training now")
if gate["bayesian_pinn_training_allowed_now"] or gate["adaptive_sampling_training_allowed_now"]:
    raise SystemExit("Phase 173 must not open Bayesian/adaptive PINN training")
if gate["gcn_pinn_training_allowed_now"] or gate["cnn_operator_training_allowed_now"]:
    raise SystemExit("Phase 173 must not reopen graph/CNN/operator training")
if gate["a100_training_allowed_now"] or gate["a100_80gb_request_now"]:
    raise SystemExit("Phase 173 must not request A100 training or 80GB")
PY

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 "$SUMMARY_SCRIPT"
