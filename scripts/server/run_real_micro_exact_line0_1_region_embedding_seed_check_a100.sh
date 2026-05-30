#!/usr/bin/env bash
set -euo pipefail

cd /root/matsci-gnn-pinn

MICRO_FEATURES="${MICRO_FEATURES:-data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features_line0_1_panel_region_embedding_v1.jsonl}"
MICRO_SAMPLE_ID="${MICRO_SAMPLE_ID:-AMB2022-718-SH1-BP1-P4-L0-1_m}"
SEEDS="${SEEDS:-1 2}"
STEPS="${STEPS:-2000}"

for seed in $SEEDS
do
  MICRO_AGGREGATE=0 \
  MICRO_FEATURES="$MICRO_FEATURES" \
  MICRO_SAMPLE_ID="$MICRO_SAMPLE_ID" \
  RUN_TAG_SUFFIX="exactline_region_embedding_seedcheck_AMB2022_718_SH1_BP1_P4_L0_1_m" \
  CLOSURE_GRAPH_MODE=real_micro_region_embedding \
  CLOSURE_GRAPH_NORMALIZE=0 \
  CLOSURE_GRAPH_REGION_ROW_SOURCE=y \
  CLOSURE_GRAPH_REGION_COL_SOURCE=x \
  CLOSURE_GRAPH_REGION_FLIP_ROW=0 \
  CLOSURE_GRAPH_REGION_FLIP_COL=1 \
  CLOSURE_GRAPH_REGION_SELECTION=nearest \
  RUN_G4=1 \
  RUN_G8=0 \
  SEED="$seed" \
  STEPS="$STEPS" \
  bash scripts/server/run_real_micro_graph_conditioned_closure_a100.sh
done
