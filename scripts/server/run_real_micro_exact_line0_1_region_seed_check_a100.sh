#!/usr/bin/env bash
set -euo pipefail

cd /root/matsci-gnn-pinn

MICRO_FEATURES="${MICRO_FEATURES:-data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features_line0_1_panel_v2.jsonl}"
MICRO_SAMPLE_ID="${MICRO_SAMPLE_ID:-AMB2022-718-SH1-BP1-P4-L0-1}"
RUN_TAG_SUFFIX="${RUN_TAG_SUFFIX:-exactline_region_seedcheck_AMB2022_718_SH1_BP1_P4_L0_1}"
SEEDS="${SEEDS:-1 2}"

for seed in $SEEDS
do
  MICRO_AGGREGATE=0 \
  MICRO_FEATURES="$MICRO_FEATURES" \
  MICRO_SAMPLE_ID="$MICRO_SAMPLE_ID" \
  RUN_TAG_SUFFIX="$RUN_TAG_SUFFIX" \
  CLOSURE_GRAPH_MODE=real_micro_region \
  CLOSURE_GRAPH_NORMALIZE=0 \
  RUN_G4=0 \
  RUN_G8=1 \
  SEED="$seed" \
  bash scripts/server/run_real_micro_graph_conditioned_closure_a100.sh
done
