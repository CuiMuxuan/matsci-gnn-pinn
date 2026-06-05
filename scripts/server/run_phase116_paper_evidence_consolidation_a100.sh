#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase116_paper_evidence_consolidation}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/build_phase116_paper_evidence_consolidation.py \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase116_paper_evidence_consolidation_a100_manifest.json"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 - <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase116_paper_evidence_consolidation_a100_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("paper_evidence_consolidation_gate", gate["status"])
print("paper_evidence_consolidation_gate evidence_consolidated", gate["paper_evidence_consolidated"])
print("paper_evidence_consolidation_gate phase116_model_mechanism_allowed", gate["phase116_model_mechanism_allowed"])
print("paper_evidence_consolidation_gate phase116_model_training_allowed", gate["phase116_model_training_allowed"])
print("paper_evidence_consolidation_gate a100_training_allowed_now", gate["a100_training_allowed_now"])
print("paper_evidence_consolidation_gate a100_80gb_request_now", gate["a100_80gb_request_now"])
if gate["phase116_model_training_allowed"] or gate["a100_training_allowed_now"] or gate["a100_80gb_request_now"]:
    raise SystemExit("Phase 116 must remain a no-training consolidation package")
PY
