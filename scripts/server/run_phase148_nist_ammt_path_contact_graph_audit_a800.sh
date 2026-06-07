#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/results/phase148_nist_ammt_path_contact_graph_audit}"
LOG_DIR="${LOG_DIR:-logs}"
DATA_ROOT="${DATA_ROOT:-data/raw/nist_ammt/mds2_2044}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 \
  scripts/server/phase148_nist_ammt_path_contact_graph_audit.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/phase148_nist_ammt_path_contact_graph_audit_manifest.json"

SUMMARY_SCRIPT="$(mktemp)"
trap 'rm -f "$SUMMARY_SCRIPT"' EXIT
cat > "$SUMMARY_SCRIPT" <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("logs/phase148_nist_ammt_path_contact_graph_audit_manifest.json").read_text(
        encoding="utf-8"
    )
)
gate = manifest["gate"]
print("path_contact_graph_audit_phase148_gate", gate["status"])
print(
    "path_contact_graph_audit_phase148_gate selected_feature_profile",
    gate["selected_feature_profile"],
)
print(
    "path_contact_graph_audit_phase148_gate best_control_profile",
    gate["best_control_profile"],
)
print(
    "path_contact_graph_audit_phase148_gate focused_review_allowed",
    gate["phase148_focused_review_allowed"],
)
print(
    "path_contact_graph_audit_phase148_gate model_mechanism_allowed",
    gate["phase148_model_mechanism_allowed"],
)
print(
    "path_contact_graph_audit_phase148_gate model_training_allowed",
    gate["phase148_model_training_allowed"],
)
print(
    "path_contact_graph_audit_phase148_gate a100_training_allowed_now",
    gate["a100_training_allowed_now"],
)
print(
    "path_contact_graph_audit_phase148_gate a100_80gb_request_now",
    gate["a100_80gb_request_now"],
)
if gate["phase148_model_mechanism_allowed"] or gate["phase148_model_training_allowed"]:
    raise SystemExit("Phase 148 must not open graph mechanisms or training")
if gate["a100_training_allowed_now"] or gate["a100_80gb_request_now"]:
    raise SystemExit("Phase 148 must not request A100 training or 80GB")
PY

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 "$SUMMARY_SCRIPT"
