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
SEEDS="${SEEDS:-1 2}"
DATASET_LIMIT="${DATASET_LIMIT:-12}"
DATASET_ORDER="${DATASET_ORDER:-process_round_robin}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-laser_power}"
OUTPUT_AFFINE_SCALE="${OUTPUT_AFFINE_SCALE:-0.5}"
OUTPUT_AFFINE_LR="${OUTPUT_AFFINE_LR:-}"

cd "$REPO_ROOT"
mkdir -p logs

run_one() {
  local seed="$1"
  local tag="$2"
  local output_affine_mode="$3"
  local run_id="ambench_multiline_process_temperature_broad${DATASET_LIMIT}_${DATASET_ORDER}_${SPLIT_STRATEGY}_${tag}_smoke_a100_sxm4_40gb_v1"

  echo "=== ${run_id} seed=${seed} output_affine=${output_affine_mode} ==="
  RUN_ID="$run_id" \
  SPLIT_STRATEGY="$SPLIT_STRATEGY" \
  TRAIN_FRACTION="0.34" \
  VAL_FRACTION="0.33" \
  TEST_FRACTION="0.33" \
  RUN_NO_PROCESS="0" \
  PROCESS_FEATURE_TAG="$tag" \
  PROCESS_CONDITIONING_MODE="concat" \
  PROCESS_FEATURE_NORMALIZATION="same" \
  PROCESS_CONDITIONING_PROFILE="broad_process_v1" \
  OUTPUT_AFFINE_MODE="$output_affine_mode" \
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
  SEED="$seed" \
    bash scripts/server/run_multiline_process_conditioned_thermal_a100.sh
}

for seed in $SEEDS; do
  run_one "$seed" "bpv1_s${seed}" "none"
  run_one "$seed" "oa_s${seed}" "linear"
done
