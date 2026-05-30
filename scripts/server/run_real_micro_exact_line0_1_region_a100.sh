#!/usr/bin/env bash
set -euo pipefail

cd /root/matsci-gnn-pinn

if [[ "${REBUILD_REGION_FEATURES:-1}" == "1" ]]; then
  FEATURE_BASENAME="${FEATURE_BASENAME:-micro_graph_features_line0_1_panel_v2}" \
  MANIFEST_OUTPUT="${MANIFEST_OUTPUT:-outputs/data_audits/ambench_mds2_2718_line0_1_micro_panel_region_feature_table_manifest.json}" \
  bash scripts/server/build_mds2_2718_line0_1_micro_panel_feature_v2_a100.sh
fi

MICRO_FEATURES="${MICRO_FEATURES:-data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features_line0_1_panel_v2.jsonl}" \
TAG_PREFIX="${TAG_PREFIX:-exactline_region_v1}" \
CLOSURE_GRAPH_MODE=real_micro_region \
CLOSURE_GRAPH_NORMALIZE="${CLOSURE_GRAPH_NORMALIZE:-0}" \
bash scripts/server/run_real_micro_exact_line0_1_a100.sh
