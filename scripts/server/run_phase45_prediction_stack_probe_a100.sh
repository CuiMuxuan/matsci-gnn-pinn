#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/matsci-gnn-pinn}"
CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-500}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
LAYERS="${LAYERS:-4}"
LR="${LR:-1e-3}"
SEED="${SEED:-7}"
DATASET_LIMITS="${DATASET_LIMITS:-12 21}"
DATASET_ORDER="${DATASET_ORDER:-process_round_robin}"
PROFILE_SPLITS="${PROFILE_SPLITS:-laser_power}"
N_ESTIMATORS="${N_ESTIMATORS:-80}"
WEIGHT_STEP="${WEIGHT_STEP:-0.1}"

cd "$REPO_ROOT"
mkdir -p logs outputs/reports outputs/predictions/phase45

run_variant() {
  local dataset_limit="$1"
  local split_strategy="$2"
  local run_tag="$3"
  local profile="$4"
  local process_columns="$5"
  local derived_mode="$6"
  local conditioning_mode="$7"
  local feature_norm="$8"
  local run_id="ambench_multiline_process_temperature_broad${dataset_limit}_${DATASET_ORDER}_${split_strategy}_${run_tag}_smoke_a100_sxm4_40gb_v1"
  local prediction_dir="outputs/predictions/phase45/${run_id}"

  echo "=== Phase45 variant ${run_id} ==="
  RUN_ID="$run_id" \
  SPLIT_STRATEGY="$split_strategy" \
  TRAIN_FRACTION="0.34" \
  VAL_FRACTION="0.33" \
  TEST_FRACTION="0.33" \
  RUN_NO_PROCESS="1" \
  PROCESS_FEATURE_TAG="$run_tag" \
  PROCESS_CONDITIONING_MODE="$conditioning_mode" \
  PROCESS_FEATURE_NORMALIZATION="$feature_norm" \
  PROCESS_CONDITIONING_PROFILE="$profile" \
  PROCESS_FEATURE_COLUMNS="$process_columns" \
  PROCESS_DERIVED_FEATURE_MODE="$derived_mode" \
  DATASET_SELECTION="all_single_track" \
  DATASET_LIMIT="$dataset_limit" \
  DATASET_ORDER="$DATASET_ORDER" \
  FRAME_STEP="${FRAME_STEP:-20}" \
  MAX_FRAMES="${MAX_FRAMES:-30}" \
  ROW_STEP="${ROW_STEP:-8}" \
  MAX_ROWS="${MAX_ROWS:-80}" \
  COL_STEP="${COL_STEP:-8}" \
  MAX_COLS="${MAX_COLS:-38}" \
  MAX_POINTS_PER_FRAME="${MAX_POINTS_PER_FRAME:-96}" \
  N_ESTIMATORS="$N_ESTIMATORS" \
  PREDICTION_OUTPUT_DIR="$prediction_dir" \
  CONDA_BIN="$CONDA_BIN" \
  CONDA_ENV="$CONDA_ENV" \
  DEVICE="$DEVICE" \
  STEPS="$STEPS" \
  HIDDEN_DIM="$HIDDEN_DIM" \
  LAYERS="$LAYERS" \
  LR="$LR" \
  SEED="$SEED" \
    bash scripts/server/run_multiline_process_conditioned_thermal_a100.sh
}

run_stack_probe() {
  local dataset_limit="$1"
  local split_strategy="$2"
  local guard_tag="phase45_guard"
  local derived_tag="phase45_derived_only"
  local guard_run_id="ambench_multiline_process_temperature_broad${dataset_limit}_${DATASET_ORDER}_${split_strategy}_${guard_tag}_smoke_a100_sxm4_40gb_v1"
  local derived_run_id="ambench_multiline_process_temperature_broad${dataset_limit}_${DATASET_ORDER}_${split_strategy}_${derived_tag}_smoke_a100_sxm4_40gb_v1"
  local table="data/interim/ambench/2022_single_track/AMB2022-03/${guard_run_id}.csv"
  local split="outputs/data_splits/${guard_run_id}_split.json"
  local guard_pred_dir="outputs/predictions/phase45/${guard_run_id}"
  local derived_pred_dir="outputs/predictions/phase45/${derived_run_id}"
  local output="outputs/reports/phase45_broad${dataset_limit}_${split_strategy}_prediction_stack_probe_summary.json"

  echo "=== Phase45 stack probe broad${dataset_limit} ${split_strategy} ==="
  "$CONDA_BIN" run -n "$CONDA_ENV" python scripts/server/phase45_prediction_stack_probe.py \
    --table "$table" \
    --target temperature_C \
    --split-manifest "$split" \
    --prediction "$guard_pred_dir/${guard_run_id}_mean_constant_predictions.csv" \
    --prediction "$guard_pred_dir/${guard_run_id}_knn_process_predictions.csv" \
    --prediction "$guard_pred_dir/${guard_run_id}_extra_trees_process_predictions.csv" \
    --prediction "$guard_pred_dir/${guard_run_id}_macro_pinn_minmax_no_process_v1_predictions.csv" \
    --prediction "$guard_pred_dir/${guard_run_id}_macro_pinn_minmax_${guard_tag}_v1_predictions.csv" \
    --prediction "$derived_pred_dir/${derived_run_id}_macro_pinn_minmax_${derived_tag}_v1_predictions.csv" \
    --label mean \
    --label knn_process \
    --label extra_trees_process \
    --label no_process_macro_pinn \
    --label broad_process_v1 \
    --label derived_only_am_energy_v1 \
    --weight-step "$WEIGHT_STEP" \
    --hot-quantile 0.9 \
    --gradient-quantile 0.9 \
    --json-output "$output"
}

for split_strategy in $PROFILE_SPLITS; do
  if [[ "$split_strategy" != "laser_power" ]]; then
    echo "Phase45 runner currently supports laser_power only; got ${split_strategy}" >&2
    exit 2
  fi
  for dataset_limit in $DATASET_LIMITS; do
    run_variant "$dataset_limit" "$split_strategy" "phase45_guard" "broad_process_v1" \
      "laser_power_W scan_speed_mm_s spot_size_um" "none" "concat" "same"
    run_variant "$dataset_limit" "$split_strategy" "phase45_derived_only" "none" \
      "" "am_energy_v1" "concat" "same"
    run_stack_probe "$dataset_limit" "$split_strategy"
  done
done
