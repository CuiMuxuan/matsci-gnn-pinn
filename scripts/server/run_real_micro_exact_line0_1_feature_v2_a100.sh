#!/usr/bin/env bash
set -euo pipefail

cd /root/matsci-gnn-pinn

MICRO_FEATURES="${MICRO_FEATURES:-data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features_line0_1_panel_v2.jsonl}" \
TAG_PREFIX="${TAG_PREFIX:-exactline_v2}" \
bash scripts/server/run_real_micro_exact_line0_1_a100.sh
