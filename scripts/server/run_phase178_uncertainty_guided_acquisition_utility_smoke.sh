#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase178_uncertainty_guided_acquisition_utility_smoke}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/build_phase178_uncertainty_guided_acquisition_utility_smoke.py \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase178_uncertainty_guided_acquisition_utility_manifest.json"

SUMMARY_SCRIPT="$(mktemp)"
trap 'rm -f "$SUMMARY_SCRIPT"' EXIT
cat > "$SUMMARY_SCRIPT" <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase178_uncertainty_guided_acquisition_utility_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("uncertainty_guided_acquisition_phase178_gate", gate["status"])
print("uncertainty_guided_acquisition_phase178_gate selected_policy", gate["selected_policy"])
print("uncertainty_guided_acquisition_phase178_gate best_candidate_policy", gate["best_candidate_policy"])
print("uncertainty_guided_acquisition_phase178_gate best_control_policy", gate["best_control_policy"])
print("uncertainty_guided_acquisition_phase178_gate phase179_training_design_allowed", gate["phase179_training_design_allowed"])
print("uncertainty_guided_acquisition_phase178_gate phase178_model_mechanism_allowed", gate["phase178_model_mechanism_allowed"])
print("uncertainty_guided_acquisition_phase178_gate phase178_model_training_allowed", gate["phase178_model_training_allowed"])
print("uncertainty_guided_acquisition_phase178_gate phase179_training_allowed_now", gate["phase179_training_allowed_now"])
print("uncertainty_guided_acquisition_phase178_gate bayesian_pinn_training_allowed_now", gate["bayesian_pinn_training_allowed_now"])
print("uncertainty_guided_acquisition_phase178_gate adaptive_sampling_training_allowed_now", gate["adaptive_sampling_training_allowed_now"])
print("uncertainty_guided_acquisition_phase178_gate gcn_pinn_training_allowed_now", gate["gcn_pinn_training_allowed_now"])
print("uncertainty_guided_acquisition_phase178_gate cnn_operator_training_allowed_now", gate["cnn_operator_training_allowed_now"])
print("uncertainty_guided_acquisition_phase178_gate am_bench_training_allowed_now", gate["am_bench_training_allowed_now"])
print("uncertainty_guided_acquisition_phase178_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
print("uncertainty_guided_acquisition_phase178_gate a100_80gb_request_now", gate["a100_80gb_request_now"])
if gate["phase178_model_mechanism_allowed"] or gate["phase178_model_training_allowed"]:
    raise SystemExit("Phase 178 must remain a no-training utility smoke")
if gate["phase179_training_allowed_now"]:
    raise SystemExit("Phase 178 may open only design, not training")
if gate["bayesian_pinn_training_allowed_now"] or gate["adaptive_sampling_training_allowed_now"]:
    raise SystemExit("Phase 178 must not open Bayesian/adaptive PINN training")
if gate["gcn_pinn_training_allowed_now"] or gate["cnn_operator_training_allowed_now"]:
    raise SystemExit("Phase 178 must not reopen graph/CNN/operator training")
if gate["am_bench_training_allowed_now"]:
    raise SystemExit("Phase 178 must not open AM-Bench training")
if gate["a100_training_allowed_now"] or gate["a100_80gb_request_now"]:
    raise SystemExit("Phase 178 must not request A100 training or 80GB")
PY

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 "$SUMMARY_SCRIPT"
