#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/matsci-gnn-pinn}"
CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-2000}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
LAYERS="${LAYERS:-4}"
LR="${LR:-1e-3}"
SEED="${SEED:-7}"
PROFILE_SPLITS="${PROFILE_SPLITS:-line laser_power scan_speed spot_size process}"

cd "$REPO_ROOT"
mkdir -p logs

run_profile_holdout() {
  local split_strategy="$1"
  local train_fraction="$2"
  local val_fraction="$3"
  local test_fraction="$4"
  local run_id="ambench_multiline_process_temperature_${split_strategy}_process_axis_profile_a100_sxm4_40gb_v1"

  echo "=== ${run_id} ==="
  RUN_ID="$run_id" \
  SPLIT_STRATEGY="$split_strategy" \
  TRAIN_FRACTION="$train_fraction" \
  VAL_FRACTION="$val_fraction" \
  TEST_FRACTION="$test_fraction" \
  PROCESS_FEATURE_TAG="process_axis_profile" \
  PROCESS_CONDITIONING_MODE="concat" \
  PROCESS_FEATURE_NORMALIZATION="same" \
  PROCESS_CONDITIONING_PROFILE="process_axis_v1" \
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
    line)
      run_profile_holdout line 0.6 0.2 0.2
      ;;
    laser_power)
      run_profile_holdout laser_power 0.34 0.33 0.33
      ;;
    scan_speed)
      run_profile_holdout scan_speed 0.34 0.33 0.33
      ;;
    spot_size)
      run_profile_holdout spot_size 0.34 0.33 0.33
      ;;
    process)
      run_profile_holdout process 0.6 0.2 0.2
      ;;
    *)
      echo "Unsupported PROFILE_SPLITS entry: ${split_strategy}" >&2
      exit 2
      ;;
  esac
done
