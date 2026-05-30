#!/usr/bin/env bash
set -euo pipefail

cd /root/matsci-gnn-pinn
mkdir -p logs outputs/runs outputs/data_audits data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
STEPS="${STEPS:-2000}"
DEVICE="${DEVICE:-cuda}"
RUN_TAG_SUFFIX="${RUN_TAG_SUFFIX:-}"
SEED="${SEED:-0}"
RUN_G4="${RUN_G4:-1}"
RUN_G8="${RUN_G8:-1}"

ACTIVE_ID="${ACTIVE_ID:-ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1}"
ACTIVE_TABLE="${ACTIVE_TABLE:-data/interim/ambench/2022_single_track/AMB2022-03/${ACTIVE_ID}.csv}"
ACTIVE_SPLIT="${ACTIVE_SPLIT:-outputs/data_splits/${ACTIVE_ID}_split.json}"
MICRO_INSPECTION="${MICRO_INSPECTION:-outputs/data_audits/ambench_mds2_2718_micrograph_inspection.json}"
MICRO_FEATURES="${MICRO_FEATURES:-data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features.jsonl}"
MICRO_FEATURES_CSV="${MICRO_FEATURES_CSV:-data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features.csv}"
MICRO_SAMPLE_ID="${MICRO_SAMPLE_ID:-AMB2022-718-SH1-BP1-P2-L2.1-3_m}"
MICRO_SAMPLE_ID_COLUMN="${MICRO_SAMPLE_ID_COLUMN:-}"
MICRO_AGGREGATE="${MICRO_AGGREGATE:-1}"

if [[ "$MICRO_AGGREGATE" == "1" ]]; then
  "$CONDA_BIN" run -n "$CONDA_ENV" python -m gnnpinn.data.loaders.ambench_microstructure \
    --mode aggregate \
    --inspection "$MICRO_INSPECTION" \
    --jsonl-output "$MICRO_FEATURES" \
    --csv-output "$MICRO_FEATURES_CSV" \
    --output outputs/data_audits/ambench_mds2_2718_micrograph_feature_table_manifest.json
fi

run_one() {
  local embedding_dim="$1"
  local gate="$2"
  local graph_l1="$3"
  local tag="$4"
  if [[ -n "$RUN_TAG_SUFFIX" ]]; then
    tag="${tag}_${RUN_TAG_SUFFIX}"
  fi
  local run_id="${ACTIVE_ID}_macro_pinn_real_micro_sparse_closure_h256_l4_lr1e_3_clr1e_5_staged1500_random4096_${tag}_s${SEED}_v1"
  local graph_selection_args=(--closure-graph-features "$MICRO_FEATURES")
  if [[ -n "$MICRO_SAMPLE_ID_COLUMN" ]]; then
    graph_selection_args+=(--closure-graph-sample-id-column "$MICRO_SAMPLE_ID_COLUMN")
  else
    graph_selection_args+=(--closure-graph-sample-id "$MICRO_SAMPLE_ID")
  fi

  "$CONDA_BIN" run -n "$CONDA_ENV" python -m gnnpinn.train.macro_pinn \
    --table "$ACTIVE_TABLE" \
    --target temperature_C \
    --split-manifest "$ACTIVE_SPLIT" \
    --output-dir "outputs/runs/${run_id}" \
    --steps "$STEPS" \
    --hidden-dim 256 \
    --layers 4 \
    --lr 1e-3 \
    --closure-lr 1e-5 \
    --seed "$SEED" \
    --device "$DEVICE" \
    --input-normalization minmax \
    --pde-weight 1e-6 \
    --pde-field normalized \
    --closure-start-step 1500 \
    --residual-sample-size 4096 \
    --residual-sampling-mode random \
    --residual-sampling-seed 2026 \
    --rho-cp 1.0 \
    --conductivity 1.0 \
    --closure-mode sparse_linear \
    --closure-feature T \
    --closure-feature x \
    --closure-feature y \
    --closure-feature t \
    --closure-polynomial-order 1 \
    --closure-l1-weight 1e-5 \
    --closure-threshold 1e-6 \
    --closure-graph-mode real_micro \
    "${graph_selection_args[@]}" \
    --closure-graph-embedding-dim "$embedding_dim" \
    --closure-graph-gate "$gate" \
    --closure-graph-l1-weight "$graph_l1" \
    --hot-quantile 0.9 \
    --gradient-quantile 0.9 \
    --log-every 100
}

if [[ "$RUN_G4" == "1" ]]; then
  run_one 4 0.25 1e-4 g4_gate0_25_gl1e_4
fi

if [[ "$RUN_G8" == "1" ]]; then
  run_one 8 0.25 1e-4 g8_gate0_25_gl1e_4
fi
