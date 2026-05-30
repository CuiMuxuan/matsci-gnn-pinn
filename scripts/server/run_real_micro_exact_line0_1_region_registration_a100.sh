#!/usr/bin/env bash
set -euo pipefail

cd /root/matsci-gnn-pinn

MICRO_FEATURES="${MICRO_FEATURES:-data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features_line0_1_panel_v2.jsonl}"
MICRO_SAMPLE_ID="${MICRO_SAMPLE_ID:-AMB2022-718-SH1-BP1-P4-L0-1}"
SEED="${SEED:-0}"
STEPS="${STEPS:-2000}"

run_variant() {
  local variant="$1"
  local row_source="$2"
  local col_source="$3"
  local flip_row="$4"
  local flip_col="$5"
  local selection="$6"

  MICRO_AGGREGATE=0 \
  MICRO_FEATURES="$MICRO_FEATURES" \
  MICRO_SAMPLE_ID="$MICRO_SAMPLE_ID" \
  RUN_TAG_SUFFIX="exactline_region_registration_${variant}_AMB2022_718_SH1_BP1_P4_L0_1" \
  CLOSURE_GRAPH_MODE=real_micro_region \
  CLOSURE_GRAPH_NORMALIZE=0 \
  CLOSURE_GRAPH_REGION_ROW_SOURCE="$row_source" \
  CLOSURE_GRAPH_REGION_COL_SOURCE="$col_source" \
  CLOSURE_GRAPH_REGION_FLIP_ROW="$flip_row" \
  CLOSURE_GRAPH_REGION_FLIP_COL="$flip_col" \
  CLOSURE_GRAPH_REGION_SELECTION="$selection" \
  RUN_G4=0 \
  RUN_G8=1 \
  SEED="$SEED" \
  STEPS="$STEPS" \
  bash scripts/server/run_real_micro_graph_conditioned_closure_a100.sh
}

run_variant rowcol_swap x y 0 0 nearest
run_variant row_flip y x 1 0 nearest
run_variant col_flip y x 0 1 nearest
run_variant row_col_flip y x 1 1 nearest
run_variant inverse_distance y x 0 0 inverse_distance
