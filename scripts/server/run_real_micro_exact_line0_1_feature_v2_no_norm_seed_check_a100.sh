#!/usr/bin/env bash
set -euo pipefail

cd /root/matsci-gnn-pinn

MICRO_FEATURES="${MICRO_FEATURES:-data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features_line0_1_panel_v2.jsonl}" \
MICRO_SAMPLE_ID="${MICRO_SAMPLE_ID:-AMB2022-718-SH1-BP1-P3-L0-1_m}" \
RUN_TAG_SUFFIX="${RUN_TAG_SUFFIX:-exactline_v2_nonorm_seedcheck_AMB2022_718_SH1_BP1_P3_L0_1_m}" \
CLOSURE_GRAPH_NORMALIZE=0 \
RUN_G4=1 \
RUN_G8=0 \
SEEDS="${SEEDS:-1 2}" \
bash scripts/server/run_real_micro_exact_line0_1_seed_check_a100.sh
