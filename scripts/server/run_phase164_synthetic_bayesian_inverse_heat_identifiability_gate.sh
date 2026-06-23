#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase164_synthetic_bayesian_inverse_heat_identifiability_gate}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/build_phase164_synthetic_bayesian_inverse_heat_identifiability_gate.py \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase164_synthetic_bayesian_inverse_heat_identifiability_manifest.json"

SUMMARY_SCRIPT="$(mktemp)"
trap 'rm -f "$SUMMARY_SCRIPT"' EXIT
cat > "$SUMMARY_SCRIPT" <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase164_synthetic_bayesian_inverse_heat_identifiability_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("synthetic_bayesian_inverse_heat_phase164_gate", gate["status"])
print("synthetic_bayesian_inverse_heat_phase164_gate selected_method", gate["selected_method"])
print("synthetic_bayesian_inverse_heat_phase164_gate best_control_method", gate["best_control_method"])
print("synthetic_bayesian_inverse_heat_phase164_gate phase165_adaptive_sampler_gate_allowed", gate["phase165_adaptive_sampler_gate_allowed"])
print("synthetic_bayesian_inverse_heat_phase164_gate phase164_low_capacity_training_allowed", gate["phase164_low_capacity_training_allowed"])
print("synthetic_bayesian_inverse_heat_phase164_gate phase164_model_mechanism_allowed", gate["phase164_model_mechanism_allowed"])
print("synthetic_bayesian_inverse_heat_phase164_gate phase164_model_training_allowed", gate["phase164_model_training_allowed"])
print("synthetic_bayesian_inverse_heat_phase164_gate bayesian_pinn_training_allowed_now", gate["bayesian_pinn_training_allowed_now"])
print("synthetic_bayesian_inverse_heat_phase164_gate adaptive_sampling_training_allowed_now", gate["adaptive_sampling_training_allowed_now"])
print("synthetic_bayesian_inverse_heat_phase164_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
print("synthetic_bayesian_inverse_heat_phase164_gate a100_80gb_request_now", gate["a100_80gb_request_now"])
if gate["phase164_low_capacity_training_allowed"]:
    raise SystemExit("Phase 164 must not open low-capacity training")
if gate["phase164_model_mechanism_allowed"] or gate["phase164_model_training_allowed"]:
    raise SystemExit("Phase 164 must not open model mechanism or model training")
if gate["bayesian_pinn_training_allowed_now"] or gate["adaptive_sampling_training_allowed_now"]:
    raise SystemExit("Phase 164 must not open Bayesian/adaptive PINN training")
if gate["a100_training_allowed_now"] or gate["a100_80gb_request_now"]:
    raise SystemExit("Phase 164 must not request A100 training or 80GB")
PY

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 "$SUMMARY_SCRIPT"
