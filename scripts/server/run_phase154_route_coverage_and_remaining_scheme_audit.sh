#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase154_route_coverage_and_remaining_scheme_audit}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/build_phase154_route_coverage_and_remaining_scheme_audit.py \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase154_route_coverage_and_remaining_scheme_audit_manifest.json"

SUMMARY_SCRIPT="$(mktemp)"
trap 'rm -f "$SUMMARY_SCRIPT"' EXIT
cat > "$SUMMARY_SCRIPT" <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase154_route_coverage_and_remaining_scheme_audit_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("route_coverage_phase154_gate", gate["status"])
print("route_coverage_phase154_gate currently_executable_model_routes_verified", gate["currently_executable_model_routes_verified"])
print("route_coverage_phase154_gate all_possible_future_schemes_exhausted", gate["all_possible_future_schemes_exhausted"])
print("route_coverage_phase154_gate first_paper_draft_allowed_now", gate["first_paper_draft_allowed_now"])
print("route_coverage_phase154_gate first_paper_submission_ready", gate["first_paper_submission_ready"])
print("route_coverage_phase154_gate new_model_claim_ready", gate["new_model_claim_ready"])
print("route_coverage_phase154_gate phase154_model_mechanism_allowed", gate["phase154_model_mechanism_allowed"])
print("route_coverage_phase154_gate phase154_model_training_allowed", gate["phase154_model_training_allowed"])
print("route_coverage_phase154_gate operator_training_allowed_now", gate["operator_training_allowed_now"])
print("route_coverage_phase154_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
print("route_coverage_phase154_gate a100_80gb_request_now", gate["a100_80gb_request_now"])
if gate["new_model_claim_ready"]:
    raise SystemExit("Phase 154 must not open a new model claim")
if gate["phase154_model_mechanism_allowed"] or gate["phase154_model_training_allowed"]:
    raise SystemExit("Phase 154 must remain a no-training route-coverage audit")
if gate["operator_training_allowed_now"]:
    raise SystemExit("Phase 154 must not open operator training")
if gate["a100_training_allowed_now"] or gate["a100_80gb_request_now"]:
    raise SystemExit("Phase 154 must not request A100 training or 80GB")
PY

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 "$SUMMARY_SCRIPT"
