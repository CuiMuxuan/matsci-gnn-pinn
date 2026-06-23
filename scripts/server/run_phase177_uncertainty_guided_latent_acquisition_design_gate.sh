#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase177_uncertainty_guided_latent_acquisition_design_gate}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/build_phase177_uncertainty_guided_latent_acquisition_design_gate.py \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase177_uncertainty_guided_latent_acquisition_design_manifest.json"

SUMMARY_SCRIPT="$(mktemp)"
trap 'rm -f "$SUMMARY_SCRIPT"' EXIT
cat > "$SUMMARY_SCRIPT" <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase177_uncertainty_guided_latent_acquisition_design_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("uncertainty_guided_latent_acquisition_phase177_gate", gate["status"])
print("uncertainty_guided_latent_acquisition_phase177_gate candidate_mechanism", gate["candidate_mechanism"])
print("uncertainty_guided_latent_acquisition_phase177_gate materially_different_from_phase175", gate["materially_different_from_phase175"])
print("uncertainty_guided_latent_acquisition_phase177_gate phase178_no_training_acquisition_smoke_allowed", gate["phase178_no_training_acquisition_smoke_allowed"])
print("uncertainty_guided_latent_acquisition_phase177_gate phase177_model_mechanism_allowed", gate["phase177_model_mechanism_allowed"])
print("uncertainty_guided_latent_acquisition_phase177_gate phase177_model_training_allowed", gate["phase177_model_training_allowed"])
print("uncertainty_guided_latent_acquisition_phase177_gate phase178_training_allowed_now", gate["phase178_training_allowed_now"])
print("uncertainty_guided_latent_acquisition_phase177_gate bayesian_pinn_training_allowed_now", gate["bayesian_pinn_training_allowed_now"])
print("uncertainty_guided_latent_acquisition_phase177_gate adaptive_sampling_training_allowed_now", gate["adaptive_sampling_training_allowed_now"])
print("uncertainty_guided_latent_acquisition_phase177_gate gcn_pinn_training_allowed_now", gate["gcn_pinn_training_allowed_now"])
print("uncertainty_guided_latent_acquisition_phase177_gate cnn_operator_training_allowed_now", gate["cnn_operator_training_allowed_now"])
print("uncertainty_guided_latent_acquisition_phase177_gate am_bench_training_allowed_now", gate["am_bench_training_allowed_now"])
print("uncertainty_guided_latent_acquisition_phase177_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
print("uncertainty_guided_latent_acquisition_phase177_gate a100_80gb_request_now", gate["a100_80gb_request_now"])
if not gate["materially_different_from_phase175"]:
    raise SystemExit("Phase 177 must be materially different from Phase 175")
if gate["phase177_model_mechanism_allowed"] or gate["phase177_model_training_allowed"]:
    raise SystemExit("Phase 177 must remain a design-only gate")
if gate["phase178_training_allowed_now"]:
    raise SystemExit("Phase 177 may open only a later no-training smoke, not training")
if gate["bayesian_pinn_training_allowed_now"] or gate["adaptive_sampling_training_allowed_now"]:
    raise SystemExit("Phase 177 must not open Bayesian/adaptive PINN training")
if gate["gcn_pinn_training_allowed_now"] or gate["cnn_operator_training_allowed_now"]:
    raise SystemExit("Phase 177 must not reopen graph/CNN/operator training")
if gate["am_bench_training_allowed_now"]:
    raise SystemExit("Phase 177 must not open AM-Bench training")
if gate["a100_training_allowed_now"] or gate["a100_80gb_request_now"]:
    raise SystemExit("Phase 177 must not request A100 training or 80GB")
PY

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 "$SUMMARY_SCRIPT"
