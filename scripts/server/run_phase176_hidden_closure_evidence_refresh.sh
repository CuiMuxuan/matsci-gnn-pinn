#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase176_hidden_closure_evidence_refresh}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/build_phase176_hidden_closure_evidence_refresh.py \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase176_hidden_closure_evidence_refresh_manifest.json"

SUMMARY_SCRIPT="$(mktemp)"
trap 'rm -f "$SUMMARY_SCRIPT"' EXIT
cat > "$SUMMARY_SCRIPT" <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase176_hidden_closure_evidence_refresh_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("hidden_closure_evidence_phase176_gate", gate["status"])
print("hidden_closure_evidence_phase176_gate synthetic_hidden_closure_claim_allowed_now", gate["synthetic_hidden_closure_claim_allowed_now"])
print("hidden_closure_evidence_phase176_gate second_paper_core_claim_ready", gate["second_paper_core_claim_ready"])
print("hidden_closure_evidence_phase176_gate low_capacity_head_claim_ready", gate["low_capacity_head_claim_ready"])
print("hidden_closure_evidence_phase176_gate phase177_materially_different_mechanism_design_allowed", gate["phase177_materially_different_mechanism_design_allowed"])
print("hidden_closure_evidence_phase176_gate phase176_model_mechanism_allowed", gate["phase176_model_mechanism_allowed"])
print("hidden_closure_evidence_phase176_gate phase176_model_training_allowed", gate["phase176_model_training_allowed"])
print("hidden_closure_evidence_phase176_gate phase177_training_allowed_now", gate["phase177_training_allowed_now"])
print("hidden_closure_evidence_phase176_gate bayesian_pinn_training_allowed_now", gate["bayesian_pinn_training_allowed_now"])
print("hidden_closure_evidence_phase176_gate adaptive_sampling_training_allowed_now", gate["adaptive_sampling_training_allowed_now"])
print("hidden_closure_evidence_phase176_gate gcn_pinn_training_allowed_now", gate["gcn_pinn_training_allowed_now"])
print("hidden_closure_evidence_phase176_gate cnn_operator_training_allowed_now", gate["cnn_operator_training_allowed_now"])
print("hidden_closure_evidence_phase176_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
print("hidden_closure_evidence_phase176_gate a100_80gb_request_now", gate["a100_80gb_request_now"])
if gate["second_paper_core_claim_ready"]:
    raise SystemExit("Phase 176 must not declare the second-paper core claim ready")
if gate["low_capacity_head_claim_ready"]:
    raise SystemExit("Phase 176 must keep the low-capacity head closed")
if gate["phase176_model_mechanism_allowed"] or gate["phase176_model_training_allowed"]:
    raise SystemExit("Phase 176 must remain an evidence refresh")
if gate["phase177_training_allowed_now"]:
    raise SystemExit("Phase 176 may open only design, not Phase 177 training")
if gate["bayesian_pinn_training_allowed_now"] or gate["adaptive_sampling_training_allowed_now"]:
    raise SystemExit("Phase 176 must not open Bayesian/adaptive PINN training")
if gate["gcn_pinn_training_allowed_now"] or gate["cnn_operator_training_allowed_now"]:
    raise SystemExit("Phase 176 must not reopen graph/CNN/operator training")
if gate["a100_training_allowed_now"] or gate["a100_80gb_request_now"]:
    raise SystemExit("Phase 176 must not request A100 training or 80GB")
PY

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 "$SUMMARY_SCRIPT"
