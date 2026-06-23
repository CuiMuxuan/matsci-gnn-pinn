#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase166_low_budget_pinn_smoke_design_gate}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/build_phase166_low_budget_pinn_smoke_design_gate.py \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase166_low_budget_pinn_smoke_design_manifest.json"

SUMMARY_SCRIPT="$(mktemp)"
trap 'rm -f "$SUMMARY_SCRIPT"' EXIT
cat > "$SUMMARY_SCRIPT" <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase166_low_budget_pinn_smoke_design_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("low_budget_pinn_smoke_phase166_gate", gate["status"])
print("low_budget_pinn_smoke_phase166_gate selected_sampler", gate["selected_sampler"])
print("low_budget_pinn_smoke_phase166_gate phase167_local_low_budget_pinn_smoke_allowed", gate["phase167_local_low_budget_pinn_smoke_allowed"])
print("low_budget_pinn_smoke_phase166_gate phase166_model_mechanism_allowed", gate["phase166_model_mechanism_allowed"])
print("low_budget_pinn_smoke_phase166_gate phase166_model_training_allowed", gate["phase166_model_training_allowed"])
print("low_budget_pinn_smoke_phase166_gate phase167_training_allowed_now", gate["phase167_training_allowed_now"])
print("low_budget_pinn_smoke_phase166_gate bayesian_pinn_training_allowed_now", gate["bayesian_pinn_training_allowed_now"])
print("low_budget_pinn_smoke_phase166_gate adaptive_sampling_training_allowed_now", gate["adaptive_sampling_training_allowed_now"])
print("low_budget_pinn_smoke_phase166_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
print("low_budget_pinn_smoke_phase166_gate a100_80gb_request_now", gate["a100_80gb_request_now"])
if gate["phase166_model_mechanism_allowed"] or gate["phase166_model_training_allowed"]:
    raise SystemExit("Phase 166 must not open model mechanism or model training")
if gate["phase167_training_allowed_now"]:
    raise SystemExit("Phase 166 may design Phase 167 but must not start training now")
if gate["bayesian_pinn_training_allowed_now"] or gate["adaptive_sampling_training_allowed_now"]:
    raise SystemExit("Phase 166 must not open Bayesian/adaptive PINN training now")
if gate["a100_training_allowed_now"] or gate["a100_80gb_request_now"]:
    raise SystemExit("Phase 166 must not request A100 training or 80GB")
PY

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 "$SUMMARY_SCRIPT"
