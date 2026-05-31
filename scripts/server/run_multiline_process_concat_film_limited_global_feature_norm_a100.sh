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
FILM_STRENGTH="${FILM_STRENGTH:-0.25}"
FILM_STRENGTH_TAG="${FILM_STRENGTH_TAG:-strength0_25}"

cd "$REPO_ROOT"
mkdir -p logs

run_limited_concat_film_holdout() {
  local split_strategy="$1"
  local train_fraction="$2"
  local val_fraction="$3"
  local test_fraction="$4"
  local run_id="ambench_multiline_process_temperature_${split_strategy}_concat_film_${FILM_STRENGTH_TAG}_global_standard_a100_sxm4_40gb_v1"

  echo "=== ${run_id} ==="
  RUN_ID="$run_id" \
  SPLIT_STRATEGY="$split_strategy" \
  TRAIN_FRACTION="$train_fraction" \
  VAL_FRACTION="$val_fraction" \
  TEST_FRACTION="$test_fraction" \
  PROCESS_FEATURE_TAG="process_concat_film_${FILM_STRENGTH_TAG}_global_standard" \
  PROCESS_CONDITIONING_MODE="concat_film" \
  PROCESS_FEATURE_NORMALIZATION="global_standard" \
  PROCESS_FILM_STRENGTH="$FILM_STRENGTH" \
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

run_limited_concat_film_holdout line 0.6 0.2 0.2
run_limited_concat_film_holdout scan_speed 0.34 0.33 0.33
run_limited_concat_film_holdout spot_size 0.34 0.33 0.33
