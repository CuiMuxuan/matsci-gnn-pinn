#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/matsci-gnn-pinn}"
CONDA="${CONDA:-/home/vipuser/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-gnnpinn}"

THERMAL_HDF5="${THERMAL_HDF5:-data/raw/ambench/2022_single_track/AMB2022-03/mds2-2716/Thermography/AMB2022-03-718-AMMT-StaringCamera_Signal.h5}"
SCAN_ROOT="${SCAN_ROOT:-data/raw/ambench/2022_single_track/AMB2022-03/mds2-2716/ScanStrategy}"
SCAN_STRATEGY="${SCAN_STRATEGY:-${SCAN_ROOT}/AMB2022-03-AMMT-718-Pad_XYPT.h5}"

PAD_MAX_FRAMES="${PAD_MAX_FRAMES:-120}"
PAD_ROW_STEP="${PAD_ROW_STEP:-20}"
PAD_MAX_ROWS="${PAD_MAX_ROWS:-32}"
PAD_COL_STEP="${PAD_COL_STEP:-10}"
PAD_MAX_COLS="${PAD_MAX_COLS:-31}"
PAD_MAX_POINTS_PER_FRAME="${PAD_MAX_POINTS_PER_FRAME:-64}"
PAD_MIN_SIGNAL="${PAD_MIN_SIGNAL:-101}"
X_PAD_FRAME_STEP="${X_PAD_FRAME_STEP:-333}"
Y_PAD_FRAME_STEP="${Y_PAD_FRAME_STEP:-84}"

cd "$PROJECT_ROOT"
mkdir -p data/interim/ambench/2022_single_track/AMB2022-03 \
  outputs/data_audits outputs/data_splits outputs/reports logs

run_python() {
  PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
    "$CONDA" run -n "$ENV_NAME" python -X utf8 "$@"
}

run_python scripts/server/phase53_source_path_data_pivot_gate.py \
  --thermal-hdf5 "$THERMAL_HDF5" \
  --scan-root "$SCAN_ROOT" \
  --json-output outputs/reports/phase53_source_path_data_pivot_summary.json

convert_and_probe_pad() {
  local pad_name="$1"
  local frame_step="$2"
  local tag
  tag="$(printf '%s' "$pad_name" | tr '[:upper:]' '[:lower:]')"
  local table="data/interim/ambench/2022_single_track/AMB2022-03/${tag}_temperature_phase53_diag.csv"
  local manifest="outputs/data_audits/phase53_${tag}_temperature_diag_manifest.json"
  local split="outputs/data_splits/phase53_${tag}_temperature_diag_split.json"
  local summary="outputs/reports/phase53_${tag}_registered_source_path_rescale_diagnostic_summary.json"

  run_python -m gnnpinn.data.loaders.ambench_hdf5 \
    --thermal-hdf5 "$THERMAL_HDF5" \
    --dataset "ThermalData/${pad_name}/Signal" \
    --sample-id "amb2022_03_${tag}_phase53_registered_path_diag" \
    --output "$table" \
    --manifest "$manifest" \
    --split-manifest "$split" \
    --frame-start 0 \
    --frame-step "$frame_step" \
    --max-frames "$PAD_MAX_FRAMES" \
    --row-start 0 \
    --row-step "$PAD_ROW_STEP" \
    --max-rows "$PAD_MAX_ROWS" \
    --col-start 0 \
    --col-step "$PAD_COL_STEP" \
    --max-cols "$PAD_MAX_COLS" \
    --calibrate-temperature \
    --min-signal "$PAD_MIN_SIGNAL" \
    --sampling-mode balanced_hot_gradient \
    --background-fraction 0.25 \
    --max-points-per-frame "$PAD_MAX_POINTS_PER_FRAME" \
    --split-strategy frame \
    > "logs/phase53_${tag}_convert.log" 2>&1

  run_python scripts/server/phase52_registered_source_path_probe.py \
    --scan-strategy "$SCAN_STRATEGY" \
    --table "$table" \
    --target temperature_C \
    --split-manifest "$split" \
    --allow-independent-rescale \
    --json-output "$summary" \
    > "logs/phase53_${tag}_registered_source_path_probe.log" 2>&1
}

convert_and_probe_pad X_pad1 "$X_PAD_FRAME_STEP"
convert_and_probe_pad Y_pad1 "$Y_PAD_FRAME_STEP"

ROLLUP_SCRIPT="$(mktemp /tmp/phase53_rollup_XXXXXX.py)"
cat > "$ROLLUP_SCRIPT" <<'PY'
import json
from pathlib import Path

reports = Path("outputs/reports")
inventory = json.loads((reports / "phase53_source_path_data_pivot_summary.json").read_text(encoding="utf-8"))
payload = {
    "inventory_decision": inventory["decision"],
    "pad_diagnostics": {},
}
for tag in ["x_pad1", "y_pad1"]:
    path = reports / f"phase53_{tag}_registered_source_path_rescale_diagnostic_summary.json"
    item = json.loads(path.read_text(encoding="utf-8"))
    payload["pad_diagnostics"][tag] = {
        "decision": item["decision"],
        "compatibility": item["compatibility"],
        "table_rows": item["table_summary"]["n_rows"],
        "summary": item["summary"],
    }
out = reports / "phase53_source_path_pivot_gate_rollup.json"
out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote: {out}")
PY
run_python "$ROLLUP_SCRIPT"
rm -f "$ROLLUP_SCRIPT"
