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
RUN_ID="${RUN_ID:-ambench_multiline_process_temperature_a100_sxm4_40gb_v1}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-line}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.6}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
TEST_FRACTION="${TEST_FRACTION:-0.2}"
PROCESS_FEATURE_TAG="${PROCESS_FEATURE_TAG:-process_features}"
PROCESS_CONDITIONING_MODE="${PROCESS_CONDITIONING_MODE:-concat}"

cd "$REPO_ROOT"
mkdir -p logs outputs/baselines outputs/data_audits outputs/data_splits outputs/runs

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

TABLE="data/interim/ambench/2022_single_track/AMB2022-03/${RUN_ID}.csv"
MANIFEST="outputs/data_audits/${RUN_ID}_manifest.json"
SPLIT="outputs/data_splits/${RUN_ID}_split.json"

DATASET_ARGS=(
  --dataset ThermalData/Line_0_1/Signal
  --dataset ThermalData/Line_1_1_1/Signal
  --dataset ThermalData/Line_1_2_1/Signal
  --dataset ThermalData/Line_2_1_1/Signal
  --dataset ThermalData/Line_2_2_1/Signal
  --dataset ThermalData/Line_3_1_1/Signal
  --dataset ThermalData/Line_3_2_1/Signal
)

"$CONDA_BIN" run -n "$CONDA_ENV" python -m gnnpinn.data.loaders.ambench_hdf5 \
  --sample-id "$RUN_ID" \
  "${DATASET_ARGS[@]}" \
  --output "$TABLE" \
  --manifest "$MANIFEST" \
  --split-manifest "$SPLIT" \
  --calibrate-temperature \
  --split-strategy "$SPLIT_STRATEGY" \
  --train-fraction "$TRAIN_FRACTION" \
  --val-fraction "$VAL_FRACTION" \
  --test-fraction "$TEST_FRACTION" \
  --seed "$SEED" \
  --min-signal 100 \
  --frame-step "${FRAME_STEP:-10}" \
  --max-frames "${MAX_FRAMES:-60}" \
  --row-step "${ROW_STEP:-4}" \
  --max-rows "${MAX_ROWS:-160}" \
  --col-step "${COL_STEP:-4}" \
  --max-cols "${MAX_COLS:-76}" \
  --sampling-mode "${SAMPLING_MODE:-balanced_hot_gradient}" \
  --hot-quantile "${HOT_QUANTILE:-0.9}" \
  --gradient-quantile "${GRADIENT_QUANTILE:-0.9}" \
  --background-fraction "${BACKGROUND_FRACTION:-0.15}" \
  --max-points-per-frame "${MAX_POINTS_PER_FRAME:-256}"

run_baseline() {
  local strategy="$1"
  local tag="$2"
  shift 2
  "$CONDA_BIN" run -n "$CONDA_ENV" python -m gnnpinn.eval.field_baseline \
    --table "$TABLE" \
    --target temperature_C \
    --strategy "$strategy" \
    --split-manifest "$SPLIT" \
    "$@" \
    --n-neighbors "${N_NEIGHBORS:-8}" \
    --n-estimators "${N_ESTIMATORS:-200}" \
    --random-state "$SEED" \
    --hot-quantile 0.9 \
    --gradient-quantile 0.9 \
    --output "outputs/baselines/${RUN_ID}_${strategy}_${tag}_regions_q90.json"
}

run_baseline mean constant
run_baseline knn coords \
  --feature-column x --feature-column y --feature-column t
run_baseline knn process \
  --feature-column x --feature-column y --feature-column t \
  --feature-column laser_power_W --feature-column scan_speed_mm_s --feature-column spot_size_um
run_baseline extra_trees coords \
  --feature-column x --feature-column y --feature-column t
run_baseline extra_trees process \
  --feature-column x --feature-column y --feature-column t \
  --feature-column laser_power_W --feature-column scan_speed_mm_s --feature-column spot_size_um

run_macro_pinn() {
  local tag="$1"
  shift
  "$CONDA_BIN" run -n "$CONDA_ENV" python -m gnnpinn.train.macro_pinn \
    --table "$TABLE" \
    --target temperature_C \
    --split-manifest "$SPLIT" \
    --output-dir "outputs/runs/${RUN_ID}_macro_pinn_minmax_${tag}_v1" \
    --steps "$STEPS" \
    --hidden-dim "$HIDDEN_DIM" \
    --layers "$LAYERS" \
    --lr "$LR" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --input-normalization minmax \
    "$@" \
    --hot-quantile 0.9 \
    --gradient-quantile 0.9 \
    --log-every 100
}

run_macro_pinn no_process
run_macro_pinn "$PROCESS_FEATURE_TAG" \
  --input-conditioning-mode "$PROCESS_CONDITIONING_MODE" \
  --input-feature-column laser_power_W \
  --input-feature-column scan_speed_mm_s \
  --input-feature-column spot_size_um
