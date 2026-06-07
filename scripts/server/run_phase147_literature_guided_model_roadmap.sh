#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase147_literature_guided_model_roadmap}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/build_phase147_literature_guided_model_roadmap.py \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase147_literature_guided_model_roadmap_manifest.json"

SUMMARY_SCRIPT="$(mktemp)"
trap 'rm -f "$SUMMARY_SCRIPT"' EXIT
cat > "$SUMMARY_SCRIPT" <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase147_literature_guided_model_roadmap_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("literature_guided_model_roadmap_phase147_gate", gate["status"])
print(
    "literature_guided_model_roadmap_phase147_gate phase148_no_training_design_allowed",
    gate["phase148_no_training_design_allowed"],
)
print(
    "literature_guided_model_roadmap_phase147_gate recommended_phase148_route",
    gate["recommended_phase148_route"],
)
print(
    "literature_guided_model_roadmap_phase147_gate phase147_model_mechanism_allowed",
    gate["phase147_model_mechanism_allowed"],
)
print(
    "literature_guided_model_roadmap_phase147_gate phase147_model_training_allowed",
    gate["phase147_model_training_allowed"],
)
print(
    "literature_guided_model_roadmap_phase147_gate a100_training_allowed_now",
    gate["a100_training_allowed_now"],
)
print(
    "literature_guided_model_roadmap_phase147_gate a100_80gb_request_now",
    gate["a100_80gb_request_now"],
)
if gate["phase147_model_mechanism_allowed"] or gate["phase147_model_training_allowed"]:
    raise SystemExit("Phase 147 must remain a no-training roadmap")
if gate["a100_training_allowed_now"] or gate["a100_80gb_request_now"]:
    raise SystemExit("Phase 147 must not request A100 training or 80GB")
PY

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 "$SUMMARY_SCRIPT"
