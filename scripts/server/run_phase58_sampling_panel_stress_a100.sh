#!/usr/bin/env bash
# Run Phase 58 alternate-density and auxiliary-panel stress tests.
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
DATASET_ORDER="${DATASET_ORDER:-process_round_robin}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-spot_size}"
DENSITY_DATASET_LIMITS="${DENSITY_DATASET_LIMITS:-12 21}"
PANEL_DATASET_LIMITS="${PANEL_DATASET_LIMITS:-15}"
DENSITY_PROFILE_TAG="${DENSITY_PROFILE_TAG:-phase58_density_profile}"
PANEL_PROFILE_TAG="${PANEL_PROFILE_TAG:-phase58_panel_profile}"

cd "$REPO_ROOT"
mkdir -p logs outputs/reports

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

run_profile_smoke() {
  local dataset_limit="$1"
  local profile_tag="$2"
  shift 2
  echo "[phase58] ${profile_tag} broad${dataset_limit} ${SPLIT_STRATEGY}"
  env \
    PROFILE_SPLITS="$SPLIT_STRATEGY" \
    DATASET_LIMIT="$dataset_limit" \
    DATASET_ORDER="$DATASET_ORDER" \
    PROCESS_PROFILE_RUN_TAG="$profile_tag" \
    PROCESS_FEATURE_TAG="$profile_tag" \
    PROCESS_CONDITIONING_PROFILE="broad_process_v1" \
    STEPS="$STEPS" \
    HIDDEN_DIM="$HIDDEN_DIM" \
    LAYERS="$LAYERS" \
    LR="$LR" \
    SEED="$SEED" \
    N_ESTIMATORS="$N_ESTIMATORS" \
    DEVICE="$DEVICE" \
    CONDA_BIN="$CONDA_BIN" \
    CONDA_ENV="$CONDA_ENV" \
    "$@" \
    bash scripts/server/run_phase30_broad_process_selector_smoke_a100.sh
}

summarize_profile() {
  local profile_tag="$1"
  local output_stem="$2"
  shift 2
  local summary_args=()
  for dataset_limit in "$@"; do
    summary_args+=(--dataset-limit "$dataset_limit")
  done
  "$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 scripts/server/summarize_phase55_spot_size_seed_check.py \
    "${summary_args[@]}" \
    --dataset-order "$DATASET_ORDER" \
    --split "$SPLIT_STRATEGY" \
    --seed "$SEED" \
    --profile-tag "$profile_tag" \
    --json-output "outputs/reports/${output_stem}.json" \
    --markdown-output "outputs/reports/${output_stem}.md" \
    --require-complete
}

for dataset_limit in $DENSITY_DATASET_LIMITS; do
  run_profile_smoke "$dataset_limit" "$DENSITY_PROFILE_TAG" \
    FRAME_STEP="${DENSITY_FRAME_STEP:-15}" \
    MAX_FRAMES="${DENSITY_MAX_FRAMES:-40}" \
    ROW_STEP="${DENSITY_ROW_STEP:-6}" \
    MAX_ROWS="${DENSITY_MAX_ROWS:-120}" \
    COL_STEP="${DENSITY_COL_STEP:-6}" \
    MAX_COLS="${DENSITY_MAX_COLS:-50}" \
    MAX_POINTS_PER_FRAME="${DENSITY_MAX_POINTS_PER_FRAME:-128}"
done
summarize_profile "$DENSITY_PROFILE_TAG" \
  phase58_sampling_density_stress_summary \
  $DENSITY_DATASET_LIMITS

for dataset_limit in $PANEL_DATASET_LIMITS; do
  run_profile_smoke "$dataset_limit" "$PANEL_PROFILE_TAG" \
    FRAME_STEP="${PANEL_FRAME_STEP:-20}" \
    MAX_FRAMES="${PANEL_MAX_FRAMES:-30}" \
    ROW_STEP="${PANEL_ROW_STEP:-8}" \
    MAX_ROWS="${PANEL_MAX_ROWS:-80}" \
    COL_STEP="${PANEL_COL_STEP:-8}" \
    MAX_COLS="${PANEL_MAX_COLS:-38}" \
    MAX_POINTS_PER_FRAME="${PANEL_MAX_POINTS_PER_FRAME:-96}"
done
summarize_profile "$PANEL_PROFILE_TAG" \
  phase58_process_panel_stress_summary \
  $PANEL_DATASET_LIMITS
