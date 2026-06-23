#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase163_pinn_bayesian_hybrid_roadmap}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/build_phase163_pinn_bayesian_hybrid_roadmap.py \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase163_pinn_bayesian_hybrid_roadmap_manifest.json"

SUMMARY_SCRIPT="$(mktemp)"
trap 'rm -f "$SUMMARY_SCRIPT"' EXIT
cat > "$SUMMARY_SCRIPT" <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase163_pinn_bayesian_hybrid_roadmap_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("pinn_bayesian_hybrid_roadmap_phase163_gate", gate["status"])
print("pinn_bayesian_hybrid_roadmap_phase163_gate verified_literature_rows", gate["verified_literature_rows"])
print("pinn_bayesian_hybrid_roadmap_phase163_gate recommended_next_phase", gate["recommended_next_phase"])
print("pinn_bayesian_hybrid_roadmap_phase163_gate phase164_no_training_design_allowed", gate["phase164_no_training_design_allowed"])
print("pinn_bayesian_hybrid_roadmap_phase163_gate phase163_model_mechanism_allowed", gate["phase163_model_mechanism_allowed"])
print("pinn_bayesian_hybrid_roadmap_phase163_gate phase163_model_training_allowed", gate["phase163_model_training_allowed"])
print("pinn_bayesian_hybrid_roadmap_phase163_gate bayesian_pinn_training_allowed_now", gate["bayesian_pinn_training_allowed_now"])
print("pinn_bayesian_hybrid_roadmap_phase163_gate adaptive_sampling_training_allowed_now", gate["adaptive_sampling_training_allowed_now"])
print("pinn_bayesian_hybrid_roadmap_phase163_gate gcn_pinn_training_allowed_now", gate["gcn_pinn_training_allowed_now"])
print("pinn_bayesian_hybrid_roadmap_phase163_gate cnn_pinn_training_allowed_now", gate["cnn_pinn_training_allowed_now"])
print("pinn_bayesian_hybrid_roadmap_phase163_gate operator_training_allowed_now", gate["operator_training_allowed_now"])
print("pinn_bayesian_hybrid_roadmap_phase163_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
print("pinn_bayesian_hybrid_roadmap_phase163_gate a100_80gb_request_now", gate["a100_80gb_request_now"])
if gate["phase163_model_mechanism_allowed"] or gate["phase163_model_training_allowed"]:
    raise SystemExit("Phase 163 must not open model mechanism or model training")
if gate["bayesian_pinn_training_allowed_now"] or gate["adaptive_sampling_training_allowed_now"]:
    raise SystemExit("Phase 163 must not open Bayesian/adaptive PINN training")
if gate["gcn_pinn_training_allowed_now"] or gate["cnn_pinn_training_allowed_now"]:
    raise SystemExit("Phase 163 must not open GCN/CNN PINN training")
if gate["operator_training_allowed_now"]:
    raise SystemExit("Phase 163 must not reopen operator training")
if gate["a100_training_allowed_now"] or gate["a100_80gb_request_now"]:
    raise SystemExit("Phase 163 must not request A100 training or 80GB")
PY

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 "$SUMMARY_SCRIPT"
