#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase167_low_budget_pinn_smoke}"
LOG_DIR="${LOG_DIR:-logs}"
STEPS="${STEPS:-650}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/build_phase167_low_budget_pinn_smoke.py \
  --output-dir "$OUTPUT_DIR" \
  --steps "$STEPS" \
  --device "$DEVICE" \
  --local-torch-blocked \
  > "$LOG_DIR/phase167_low_budget_pinn_smoke_manifest.json"

SUMMARY_SCRIPT="$(mktemp)"
trap 'rm -f "$SUMMARY_SCRIPT"' EXIT
cat > "$SUMMARY_SCRIPT" <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase167_low_budget_pinn_smoke_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("low_budget_pinn_smoke_phase167_gate", gate["status"])
print("low_budget_pinn_smoke_phase167_gate selected_variant", gate["selected_variant"])
print("low_budget_pinn_smoke_phase167_gate best_control_variant", gate["best_control_variant"])
print("low_budget_pinn_smoke_phase167_gate phase168_focused_review_allowed", gate["phase168_focused_review_allowed"])
print("low_budget_pinn_smoke_phase167_gate phase167_model_mechanism_allowed", gate["phase167_model_mechanism_allowed"])
print("low_budget_pinn_smoke_phase167_gate phase167_model_claim_allowed", gate["phase167_model_claim_allowed"])
print("low_budget_pinn_smoke_phase167_gate bayesian_pinn_training_allowed_now", gate["bayesian_pinn_training_allowed_now"])
print("low_budget_pinn_smoke_phase167_gate adaptive_sampling_training_allowed_now", gate["adaptive_sampling_training_allowed_now"])
print("low_budget_pinn_smoke_phase167_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
print("low_budget_pinn_smoke_phase167_gate a100_80gb_request_now", gate["a100_80gb_request_now"])
if gate["phase167_model_mechanism_allowed"] or gate["phase167_model_claim_allowed"]:
    raise SystemExit("Phase 167 smoke must not open model mechanism or model claim")
if gate["bayesian_pinn_training_allowed_now"] or gate["adaptive_sampling_training_allowed_now"]:
    raise SystemExit("Phase 167 must not open Bayesian/adaptive PINN training now")
if gate["a100_training_allowed_now"] or gate["a100_80gb_request_now"]:
    raise SystemExit("Phase 167 must not request A100 training or 80GB")
PY

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 "$SUMMARY_SCRIPT"
