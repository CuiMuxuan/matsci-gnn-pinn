#!/usr/bin/env bash
set -euo pipefail

cd /root/matsci-gnn-pinn
mkdir -p logs outputs/runs

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

ACTIVE_ID="${ACTIVE_ID:-ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1}"
ACTIVE_TABLE="${ACTIVE_TABLE:-data/interim/ambench/2022_single_track/AMB2022-03/${ACTIVE_ID}.csv}"
ACTIVE_SPLIT="${ACTIVE_SPLIT:-outputs/data_splits/${ACTIVE_ID}_split.json}"
MICRO_FEATURES="${MICRO_FEATURES:-data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features_panel.jsonl}"

# The active thermal table is Line_0_1 with 285 W, 960 mm/s, 67 um.
# The mds2-2718 measurement XLSX maps Line 0_* at the same process condition
# to BP1/P3 and BP1/P4 single-track cross-sections.
for sample_id in \
  "AMB2022-718-SH1-BP1-P3-L0-2_m" \
  "AMB2022-718-SH1-BP1-P4-L0-2_m" \
  "AMB2022-718-SH1-BP1-P4-L0-2"
do
  safe_sample_id="${sample_id//[^A-Za-z0-9_]/_}"
  MICRO_AGGREGATE=0 \
  MICRO_FEATURES="$MICRO_FEATURES" \
  MICRO_SAMPLE_ID="$sample_id" \
  RUN_TAG_SUFFIX="sameproc_${safe_sample_id}" \
  ACTIVE_ID="$ACTIVE_ID" \
  ACTIVE_TABLE="$ACTIVE_TABLE" \
  ACTIVE_SPLIT="$ACTIVE_SPLIT" \
  bash scripts/server/run_real_micro_graph_conditioned_closure_a100.sh
done
