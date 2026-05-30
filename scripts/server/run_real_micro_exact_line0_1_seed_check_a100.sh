#!/usr/bin/env bash
set -euo pipefail

cd /root/matsci-gnn-pinn
mkdir -p logs outputs/runs

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

ACTIVE_ID="${ACTIVE_ID:-ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1}"
ACTIVE_TABLE="${ACTIVE_TABLE:-data/interim/ambench/2022_single_track/AMB2022-03/${ACTIVE_ID}.csv}"
ACTIVE_SPLIT="${ACTIVE_SPLIT:-outputs/data_splits/${ACTIVE_ID}_split.json}"
MICRO_FEATURES="${MICRO_FEATURES:-data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features_line0_1_panel.jsonl}"
MICRO_SAMPLE_ID="${MICRO_SAMPLE_ID:-AMB2022-718-SH1-BP1-P4-L0-1_m}"
RUN_TAG_SUFFIX="${RUN_TAG_SUFFIX:-exactline_seedcheck_AMB2022_718_SH1_BP1_P4_L0_1_m}"
SEEDS="${SEEDS:-1 2}"

for seed in $SEEDS
do
  MICRO_AGGREGATE=0 \
  MICRO_FEATURES="$MICRO_FEATURES" \
  MICRO_SAMPLE_ID="$MICRO_SAMPLE_ID" \
  RUN_TAG_SUFFIX="$RUN_TAG_SUFFIX" \
  RUN_G4=1 \
  RUN_G8=0 \
  SEED="$seed" \
  ACTIVE_ID="$ACTIVE_ID" \
  ACTIVE_TABLE="$ACTIVE_TABLE" \
  ACTIVE_SPLIT="$ACTIVE_SPLIT" \
  bash scripts/server/run_real_micro_graph_conditioned_closure_a100.sh
done
