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
PROCESS_DERIVED_FEATURE_MODE="${PROCESS_DERIVED_FEATURE_MODE:-am_energy_v1}"
PROCESS_ENCODER_MODE="${PROCESS_ENCODER_MODE:-linear}"
PROCESS_ENCODER_DIM="${PROCESS_ENCODER_DIM:-3}"
PROCESS_FEATURE_TAG="${PROCESS_FEATURE_TAG:-proc_enc}"

cd "$REPO_ROOT"
mkdir -p logs

run_process_encoder_holdout() {
  local dataset_limit="$1"
  local split_strategy="$2"
  local train_fraction="$3"
  local val_fraction="$4"
  local test_fraction="$5"
  local run_id="ambench_multiline_process_temperature_broad${dataset_limit}_${DATASET_ORDER}_${split_strategy}_${PROCESS_FEATURE_TAG}_smoke_a100_sxm4_40gb_v1"

  echo "=== ${run_id} process_encoder=${PROCESS_ENCODER_MODE}@${PROCESS_ENCODER_DIM} ==="
  RUN_ID="$run_id" \
  SPLIT_STRATEGY="$split_strategy" \
  TRAIN_FRACTION="$train_fraction" \
  VAL_FRACTION="$val_fraction" \
  TEST_FRACTION="$test_fraction" \
  RUN_NO_PROCESS="0" \
  PROCESS_FEATURE_TAG="$PROCESS_FEATURE_TAG" \
  PROCESS_CONDITIONING_MODE="concat" \
  PROCESS_FEATURE_NORMALIZATION="same" \
  PROCESS_CONDITIONING_PROFILE="broad_process_v1" \
  PROCESS_DERIVED_FEATURE_MODE="$PROCESS_DERIVED_FEATURE_MODE" \
  PROCESS_ENCODER_MODE="$PROCESS_ENCODER_MODE" \
  PROCESS_ENCODER_DIM="$PROCESS_ENCODER_DIM" \
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

for dataset_limit in $DATASET_LIMITS; do
  for split_strategy in $PROFILE_SPLITS; do
    case "$split_strategy" in
      laser_power)
        run_process_encoder_holdout "$dataset_limit" laser_power 0.34 0.33 0.33
        ;;
      spot_size)
        run_process_encoder_holdout "$dataset_limit" spot_size 0.34 0.33 0.33
        ;;
      scan_speed)
        run_process_encoder_holdout "$dataset_limit" scan_speed 0.34 0.33 0.33
        ;;
      line)
        run_process_encoder_holdout "$dataset_limit" line 0.6 0.2 0.2
        ;;
      process)
        run_process_encoder_holdout "$dataset_limit" process 0.6 0.2 0.2
        ;;
      *)
        echo "Unsupported PROFILE_SPLITS entry: ${split_strategy}" >&2
        exit 2
        ;;
    esac
  done
done
