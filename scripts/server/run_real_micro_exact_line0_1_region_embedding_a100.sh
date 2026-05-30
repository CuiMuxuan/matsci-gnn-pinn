#!/usr/bin/env bash
set -euo pipefail

cd /root/matsci-gnn-pinn

if [[ "${REBUILD_REGION_EMBEDDINGS:-1}" == "1" ]]; then
  bash scripts/server/build_mds2_2718_line0_1_micro_panel_region_embedding_a100.sh
fi

MICRO_FEATURES="${MICRO_FEATURES:-data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features_line0_1_panel_region_embedding_v1.jsonl}" \
TAG_PREFIX="${TAG_PREFIX:-exactline_region_embedding_v1}" \
CLOSURE_GRAPH_MODE=real_micro_region_embedding \
CLOSURE_GRAPH_NORMALIZE="${CLOSURE_GRAPH_NORMALIZE:-0}" \
CLOSURE_GRAPH_REGION_ROW_SOURCE="${CLOSURE_GRAPH_REGION_ROW_SOURCE:-y}" \
CLOSURE_GRAPH_REGION_COL_SOURCE="${CLOSURE_GRAPH_REGION_COL_SOURCE:-x}" \
CLOSURE_GRAPH_REGION_FLIP_ROW="${CLOSURE_GRAPH_REGION_FLIP_ROW:-0}" \
CLOSURE_GRAPH_REGION_FLIP_COL="${CLOSURE_GRAPH_REGION_FLIP_COL:-1}" \
CLOSURE_GRAPH_REGION_SELECTION="${CLOSURE_GRAPH_REGION_SELECTION:-nearest}" \
bash scripts/server/run_real_micro_exact_line0_1_a100.sh
