#!/usr/bin/env bash
set -euo pipefail

cd /root/matsci-gnn-pinn

AUDIT_ROOT="${AUDIT_ROOT:-outputs/data_audits/mds2_2718_line0_1_micro_panel_region_embedding_v1}" \
FEATURE_BASENAME="${FEATURE_BASENAME:-micro_graph_features_line0_1_panel_region_embedding_v1}" \
MANIFEST_OUTPUT="${MANIFEST_OUTPUT:-outputs/data_audits/ambench_mds2_2718_line0_1_micro_panel_region_embedding_v1_manifest.json}" \
REGION_EMBEDDING_DIM="${REGION_EMBEDDING_DIM:-8}" \
REGION_EMBEDDING_NORMALIZE="${REGION_EMBEDDING_NORMALIZE:-1}" \
bash scripts/server/build_mds2_2718_line0_1_micro_panel_a100.sh
