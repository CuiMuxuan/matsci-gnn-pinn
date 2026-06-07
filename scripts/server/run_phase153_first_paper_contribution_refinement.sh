#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase153_first_paper_contribution_refinement}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/build_phase153_first_paper_contribution_refinement.py \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase153_first_paper_contribution_refinement_manifest.json"

SUMMARY_SCRIPT="$(mktemp)"
trap 'rm -f "$SUMMARY_SCRIPT"' EXIT
cat > "$SUMMARY_SCRIPT" <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase153_first_paper_contribution_refinement_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("first_paper_contribution_refinement_phase153_gate", gate["status"])
print("first_paper_contribution_refinement_phase153_gate contribution_refinement_ready", gate["contribution_refinement_ready"])
print("first_paper_contribution_refinement_phase153_gate first_paper_draft_allowed_now", gate["first_paper_draft_allowed_now"])
print("first_paper_contribution_refinement_phase153_gate first_paper_submission_ready", gate["first_paper_submission_ready"])
print("first_paper_contribution_refinement_phase153_gate new_model_claim_ready", gate["new_model_claim_ready"])
print("first_paper_contribution_refinement_phase153_gate phase153_model_mechanism_allowed", gate["phase153_model_mechanism_allowed"])
print("first_paper_contribution_refinement_phase153_gate phase153_model_training_allowed", gate["phase153_model_training_allowed"])
print("first_paper_contribution_refinement_phase153_gate operator_training_allowed_now", gate["operator_training_allowed_now"])
print("first_paper_contribution_refinement_phase153_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
print("first_paper_contribution_refinement_phase153_gate a100_80gb_request_now", gate["a100_80gb_request_now"])
if gate["new_model_claim_ready"]:
    raise SystemExit("Phase 153 must not open a new model claim")
if gate["phase153_model_mechanism_allowed"] or gate["phase153_model_training_allowed"]:
    raise SystemExit("Phase 153 must remain a no-training contribution refinement package")
if gate["operator_training_allowed_now"]:
    raise SystemExit("Phase 153 must not open operator training")
if gate["a100_training_allowed_now"] or gate["a100_80gb_request_now"]:
    raise SystemExit("Phase 153 must not request A100 training or 80GB")
PY

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 "$SUMMARY_SCRIPT"
