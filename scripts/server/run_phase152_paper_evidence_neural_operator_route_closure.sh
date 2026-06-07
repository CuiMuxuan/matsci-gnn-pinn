#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase152_paper_evidence_neural_operator_route_closure}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/build_phase152_paper_evidence_neural_operator_route_closure.py \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase152_paper_evidence_neural_operator_route_closure_manifest.json"

SUMMARY_SCRIPT="$(mktemp)"
trap 'rm -f "$SUMMARY_SCRIPT"' EXIT
cat > "$SUMMARY_SCRIPT" <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase152_paper_evidence_neural_operator_route_closure_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("paper_evidence_refresh_phase152_gate", gate["status"])
print("paper_evidence_refresh_phase152_gate first_paper_draft_allowed_now", gate["first_paper_draft_allowed_now"])
print("paper_evidence_refresh_phase152_gate neural_operator_route_closed_as_diagnostic", gate["neural_operator_route_closed_as_diagnostic"])
print("paper_evidence_refresh_phase152_gate new_neural_operator_model_claim_ready", gate["new_neural_operator_model_claim_ready"])
print("paper_evidence_refresh_phase152_gate phase152_model_mechanism_allowed", gate["phase152_model_mechanism_allowed"])
print("paper_evidence_refresh_phase152_gate phase152_model_training_allowed", gate["phase152_model_training_allowed"])
print("paper_evidence_refresh_phase152_gate operator_training_allowed_now", gate["operator_training_allowed_now"])
print("paper_evidence_refresh_phase152_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
print("paper_evidence_refresh_phase152_gate a100_80gb_request_now", gate["a100_80gb_request_now"])
if gate["new_neural_operator_model_claim_ready"]:
    raise SystemExit("Phase 152 must not promote a neural-operator model claim")
if gate["phase152_model_mechanism_allowed"] or gate["phase152_model_training_allowed"]:
    raise SystemExit("Phase 152 must remain a no-training evidence refresh")
if gate["operator_training_allowed_now"]:
    raise SystemExit("Phase 152 must not open operator training")
if gate["a100_training_allowed_now"] or gate["a100_80gb_request_now"]:
    raise SystemExit("Phase 152 must not request A100 training or 80GB")
PY

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 "$SUMMARY_SCRIPT"
