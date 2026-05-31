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

cd "$REPO_ROOT"
mkdir -p logs

run_routed_holdout() {
  local split_strategy="$1"
  local train_fraction="$2"
  local val_fraction="$3"
  local test_fraction="$4"
  local film_prior="$5"
  local route_tag="$6"
  local run_id="ambench_multiline_process_temperature_${split_strategy}_routed_${route_tag}_global_standard_a100_sxm4_40gb_v1"

  echo "=== ${run_id} ==="
  RUN_ID="$run_id" \
  SPLIT_STRATEGY="$split_strategy" \
  TRAIN_FRACTION="$train_fraction" \
  VAL_FRACTION="$val_fraction" \
  TEST_FRACTION="$test_fraction" \
  PROCESS_FEATURE_TAG="process_routed_${route_tag}_global_standard" \
  PROCESS_CONDITIONING_MODE="routed" \
  PROCESS_FEATURE_NORMALIZATION="global_standard" \
  PROCESS_ROUTE_FILM_PRIOR="$film_prior" \
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

# Phase 25/26 evidence says line and scan-speed prefer the concat expert, while
# spot-size prefers FiLM. The routed model starts from that axis prior and can
# still train the process-feature gate unless FREEZE_PROCESS_ROUTE=1 is set in
# run_multiline_process_conditioned_thermal_a100.sh.
run_routed_holdout line 0.6 0.2 0.2 0.2 concat_prior
run_routed_holdout scan_speed 0.34 0.33 0.33 0.2 concat_prior
run_routed_holdout spot_size 0.34 0.33 0.33 0.8 film_prior
