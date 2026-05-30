#!/usr/bin/env bash
set -euo pipefail

cd /root/matsci-gnn-pinn

AUDIT_ROOT="${AUDIT_ROOT:-outputs/data_audits/mds2_2718_line0_1_micro_panel_v2}" \
FEATURE_BASENAME="${FEATURE_BASENAME:-micro_graph_features_line0_1_panel_v2}" \
MANIFEST_OUTPUT="${MANIFEST_OUTPUT:-outputs/data_audits/ambench_mds2_2718_line0_1_micro_panel_feature_v2_manifest.json}" \
bash scripts/server/build_mds2_2718_line0_1_micro_panel_a100.sh
