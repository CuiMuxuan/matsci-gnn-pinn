#!/usr/bin/env bash
# Export row-level predictions and summarize residual anatomy for Phase 59.
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
N_ESTIMATORS="${N_ESTIMATORS:-80}"
DATASET_LIMIT="${DATASET_LIMIT:-21}"
DATASET_ORDER="${DATASET_ORDER:-process_round_robin}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-spot_size}"
PROFILE_TAG="${PROFILE_TAG:-phase59_density_anatomy}"
PREDICTION_DIR="${PREDICTION_DIR:-outputs/predictions/phase59/${PROFILE_TAG}_broad${DATASET_LIMIT}_${SPLIT_STRATEGY}}"

cd "$REPO_ROOT"
mkdir -p logs outputs/reports "$PREDICTION_DIR"

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

echo "[phase59] export predictions broad${DATASET_LIMIT} ${SPLIT_STRATEGY} ${PROFILE_TAG}"
env \
  PROFILE_SPLITS="$SPLIT_STRATEGY" \
  DATASET_LIMIT="$DATASET_LIMIT" \
  DATASET_ORDER="$DATASET_ORDER" \
  PROCESS_PROFILE_RUN_TAG="$PROFILE_TAG" \
  PROCESS_FEATURE_TAG="$PROFILE_TAG" \
  PROCESS_CONDITIONING_PROFILE="broad_process_v1" \
  FRAME_STEP="${FRAME_STEP:-15}" \
  MAX_FRAMES="${MAX_FRAMES:-40}" \
  ROW_STEP="${ROW_STEP:-6}" \
  MAX_ROWS="${MAX_ROWS:-120}" \
  COL_STEP="${COL_STEP:-6}" \
  MAX_COLS="${MAX_COLS:-50}" \
  MAX_POINTS_PER_FRAME="${MAX_POINTS_PER_FRAME:-128}" \
  STEPS="$STEPS" \
  HIDDEN_DIM="$HIDDEN_DIM" \
  LAYERS="$LAYERS" \
  LR="$LR" \
  SEED="$SEED" \
  N_ESTIMATORS="$N_ESTIMATORS" \
  DEVICE="$DEVICE" \
  PREDICTION_OUTPUT_DIR="$PREDICTION_DIR" \
  CONDA_BIN="$CONDA_BIN" \
  CONDA_ENV="$CONDA_ENV" \
  bash scripts/server/run_phase30_broad_process_selector_smoke_a100.sh

RUN_ID="ambench_multiline_process_temperature_broad${DATASET_LIMIT}_${DATASET_ORDER}_${SPLIT_STRATEGY}_${PROFILE_TAG}_smoke_a100_sxm4_40gb_v1"

"$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 scripts/server/summarize_phase59_residual_anatomy.py \
  --prediction "$PREDICTION_DIR/${RUN_ID}_mean_constant_predictions.csv" \
  --prediction "$PREDICTION_DIR/${RUN_ID}_macro_pinn_minmax_no_process_v1_predictions.csv" \
  --prediction "$PREDICTION_DIR/${RUN_ID}_macro_pinn_minmax_${PROFILE_TAG}_v1_predictions.csv" \
  --label mean \
  --label no_process \
  --label broad_process_v1 \
  --target temperature_C \
  --candidate broad_process_v1 \
  --reference mean \
  --secondary-reference no_process \
  --analysis-split test \
  --min-group-n "${MIN_GROUP_N:-20}" \
  --top-n "${TOP_N:-30}" \
  --json-output "outputs/reports/phase59_broad21_density_residual_anatomy.json" \
  --markdown-output "outputs/reports/phase59_broad21_density_residual_anatomy.md"

"$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 scripts/server/summarize_phase59_residual_upper_bound.py \
  --prediction "$PREDICTION_DIR/${RUN_ID}_mean_constant_predictions.csv" \
  --prediction "$PREDICTION_DIR/${RUN_ID}_macro_pinn_minmax_no_process_v1_predictions.csv" \
  --prediction "$PREDICTION_DIR/${RUN_ID}_macro_pinn_minmax_${PROFILE_TAG}_v1_predictions.csv" \
  --label mean \
  --label no_process \
  --label broad_process_v1 \
  --target temperature_C \
  --candidate broad_process_v1 \
  --reference mean \
  --secondary-reference no_process \
  --fit-split train \
  --selection-split val \
  --analysis-split test \
  --min-fit-n "${MIN_FIT_N:-20}" \
  --shrinkage "${SHRINKAGE:-20}" \
  --json-output "outputs/reports/phase59_broad21_density_residual_upper_bound.json" \
  --markdown-output "outputs/reports/phase59_broad21_density_residual_upper_bound.md"
