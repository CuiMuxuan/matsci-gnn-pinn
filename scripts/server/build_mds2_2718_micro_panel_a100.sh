#!/usr/bin/env bash
set -euo pipefail

cd /root/matsci-gnn-pinn
mkdir -p logs outputs/data_audits data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
DATA_ROOT="${DATA_ROOT:-data/raw/ambench/2022_single_track/AMB2022-03/mds2-2718}"
AUDIT_ROOT="${AUDIT_ROOT:-outputs/data_audits/mds2_2718_micro_panel}"
PROCESSED_ROOT="${PROCESSED_ROOT:-data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-3}"
DOWNLOAD_TIMEOUT_SECONDS="${DOWNLOAD_TIMEOUT_SECONDS:-300}"
DOWNLOAD_BACKEND="${DOWNLOAD_BACKEND:-curl}"

mkdir -p "$AUDIT_ROOT"

"$CONDA_BIN" run -n "$CONDA_ENV" python -m gnnpinn.data.ambench_downloads \
  --dataset-id mds2-2718 \
  --root "$DATA_ROOT" \
  --download \
  --include-optional \
  --verify-sha256 \
  --retries "$DOWNLOAD_RETRIES" \
  --timeout-seconds "$DOWNLOAD_TIMEOUT_SECONDS" \
  --resume-partial \
  --download-backend "$DOWNLOAD_BACKEND" \
  --output outputs/data_audits/ambench_mds2_2718_micro_panel_download_report.json

inspect_one() {
  local sample_id="$1"
  local relative_path="$2"
  local output="$AUDIT_ROOT/${sample_id}_inspection.json"

  "$CONDA_BIN" run -n "$CONDA_ENV" python -m gnnpinn.data.loaders.ambench_microstructure \
    --image "$DATA_ROOT/$relative_path" \
    --sample-id "$sample_id" \
    --threshold-quantile 0.9 \
    --grid-rows 8 \
    --grid-cols 8 \
    --graph-k 4 \
    --output "$output"
}

inspect_one \
  "AMB2022-718-SH1-BP1-P2-L2.1-3_m" \
  "Single_Track_Cross_Sections/AMB2022-718-SH1-BP1-P2-L2.1-3_m.tif"
inspect_one \
  "AMB2022-718-SH1-BP1-P2-L2.1-3" \
  "Single_Track_Cross_Sections/AMB2022-718-SH1-BP1-P2-L2.1-3.tif"
inspect_one \
  "AMB2022-718-SH1-BP1-P1-L3.1-3_m" \
  "Single_Track_Cross_Sections/AMB2022-718-SH1-BP1-P1-L3.1-3_m.tif"
inspect_one \
  "AMB2022-718-SH1-BP1-P3-L0-2_m" \
  "Single_Track_Cross_Sections/AMB2022-718-SH1-BP1-P3-L0-2_m.tif"
inspect_one \
  "AMB2022-718-SH1-BP1-P4-L0-2_m" \
  "Single_Track_Cross_Sections/AMB2022-718-SH1-BP1-P4-L0-2_m.tif"
inspect_one \
  "AMB2022-718-SH1-BP1-P4-L0-2" \
  "Single_Track_Cross_Sections/AMB2022-718-SH1-BP1-P4-L0-2.tif"

"$CONDA_BIN" run -n "$CONDA_ENV" python -m gnnpinn.data.loaders.ambench_microstructure \
  --mode aggregate \
  --inspection "$AUDIT_ROOT/AMB2022-718-SH1-BP1-P2-L2.1-3_m_inspection.json" \
  --inspection "$AUDIT_ROOT/AMB2022-718-SH1-BP1-P2-L2.1-3_inspection.json" \
  --inspection "$AUDIT_ROOT/AMB2022-718-SH1-BP1-P1-L3.1-3_m_inspection.json" \
  --inspection "$AUDIT_ROOT/AMB2022-718-SH1-BP1-P3-L0-2_m_inspection.json" \
  --inspection "$AUDIT_ROOT/AMB2022-718-SH1-BP1-P4-L0-2_m_inspection.json" \
  --inspection "$AUDIT_ROOT/AMB2022-718-SH1-BP1-P4-L0-2_inspection.json" \
  --jsonl-output "$PROCESSED_ROOT/micro_graph_features_panel.jsonl" \
  --csv-output "$PROCESSED_ROOT/micro_graph_features_panel.csv" \
  --output outputs/data_audits/ambench_mds2_2718_micro_panel_feature_table_manifest.json
