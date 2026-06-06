#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase137_paper_evidence_refresh}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/build_phase137_paper_evidence_refresh.py \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase137_paper_evidence_refresh_manifest.json"

SUMMARY_SCRIPT="$(mktemp)"
trap 'rm -f "$SUMMARY_SCRIPT"' EXIT
cat > "$SUMMARY_SCRIPT" <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase137_paper_evidence_refresh_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("paper_evidence_refresh_gate", gate["status"])
print("paper_evidence_refresh_gate first_paper_draft_allowed_now", gate["first_paper_draft_allowed_now"])
print("paper_evidence_refresh_gate phase137_model_mechanism_allowed", gate["phase137_model_mechanism_allowed"])
print("paper_evidence_refresh_gate phase137_model_training_allowed", gate["phase137_model_training_allowed"])
print("paper_evidence_refresh_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
print("paper_evidence_refresh_gate a100_80gb_request_now", gate["a100_80gb_request_now"])
if gate["phase137_model_training_allowed"] or gate["a100_training_allowed_now"] or gate["a100_80gb_request_now"]:
    raise SystemExit("Phase 137 must remain a no-training evidence refresh")
PY

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 "$SUMMARY_SCRIPT"
