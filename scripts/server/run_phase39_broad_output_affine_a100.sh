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
DATASET_LIMIT="${DATASET_LIMIT:-12}"
DATASET_ORDER="${DATASET_ORDER:-process_round_robin}"
PROFILE_SPLITS="${PROFILE_SPLITS:-spot_size laser_power}"
OUTPUT_AFFINE_TAG="${OUTPUT_AFFINE_TAG:-out_affine}"
OUTPUT_AFFINE_SCALE="${OUTPUT_AFFINE_SCALE:-0.5}"
OUTPUT_AFFINE_LR="${OUTPUT_AFFINE_LR:-}"

cd "$REPO_ROOT"
mkdir -p logs

run_output_affine_holdout() {
  local split_strategy="$1"
  local train_fraction="$2"
  local val_fraction="$3"
  local test_fraction="$4"
  local run_id="ambench_multiline_process_temperature_broad${DATASET_LIMIT}_${DATASET_ORDER}_${split_strategy}_${OUTPUT_AFFINE_TAG}_smoke_a100_sxm4_40gb_v1"

  echo "=== ${run_id} ==="
  RUN_ID="$run_id" \
  SPLIT_STRATEGY="$split_strategy" \
  TRAIN_FRACTION="$train_fraction" \
  VAL_FRACTION="$val_fraction" \
  TEST_FRACTION="$test_fraction" \
  RUN_NO_PROCESS="0" \
  PROCESS_FEATURE_TAG="$OUTPUT_AFFINE_TAG" \
  PROCESS_CONDITIONING_MODE="concat" \
  PROCESS_FEATURE_NORMALIZATION="same" \
  PROCESS_CONDITIONING_PROFILE="broad_process_v1" \
  OUTPUT_AFFINE_MODE="linear" \
  OUTPUT_AFFINE_SCALE="$OUTPUT_AFFINE_SCALE" \
  OUTPUT_AFFINE_LR="$OUTPUT_AFFINE_LR" \
  DATASET_SELECTION="all_single_track" \
  DATASET_LIMIT="$DATASET_LIMIT" \
  DATASET_ORDER="$DATASET_ORDER" \
  FRAME_STEP="${FRAME_STEP:-20}" \
  MAX_FRAMES="${MAX_FRAMES:-30}" \
  ROW_STEP="${ROW_STEP:-8}" \
  MAX_ROWS="${MAX_ROWS:-80}" \
  COL_STEP="${COL_STEP:-8}" \
  MAX_COLS="${MAX_COLS:-38}" \
  MAX_POINTS_PER_FRAME="${MAX_POINTS_PER_FRAME:-96}" \
  N_ESTIMATORS="${N_ESTIMATORS:-80}" \
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

for split_strategy in $PROFILE_SPLITS; do
  case "$split_strategy" in
    laser_power)
      run_output_affine_holdout laser_power 0.34 0.33 0.33
      ;;
    spot_size)
      run_output_affine_holdout spot_size 0.34 0.33 0.33
      ;;
    *)
      echo "Phase 39 output affine runner currently supports feature-active broad_process_v1 splits only: spot_size laser_power" >&2
      echo "Unsupported PROFILE_SPLITS entry: ${split_strategy}" >&2
      exit 2
      ;;
  esac
done
