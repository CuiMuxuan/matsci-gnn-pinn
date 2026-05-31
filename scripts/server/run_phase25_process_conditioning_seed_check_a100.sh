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
SEEDS="${SEEDS:-1 2}"

cd "$REPO_ROOT"
mkdir -p logs outputs/runs

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

run_macro_pinn() {
  local run_id="$1"
  local seed="$2"
  local tag="$3"
  shift 3

  local table="data/interim/ambench/2022_single_track/AMB2022-03/${run_id}.csv"
  local split="outputs/data_splits/${run_id}_split.json"
  local out_dir="outputs/runs/${run_id}_seed${seed}_macro_pinn_minmax_${tag}_v1"

  echo "=== ${out_dir} ==="
  "$CONDA_BIN" run -n "$CONDA_ENV" python -m gnnpinn.train.macro_pinn \
    --table "$table" \
    --target temperature_C \
    --split-manifest "$split" \
    --output-dir "$out_dir" \
    --steps "$STEPS" \
    --hidden-dim "$HIDDEN_DIM" \
    --layers "$LAYERS" \
    --lr "$LR" \
    --seed "$seed" \
    --device "$DEVICE" \
    --input-normalization minmax \
    "$@" \
    --hot-quantile 0.9 \
    --gradient-quantile 0.9 \
    --log-every 100
}

for seed in $SEEDS; do
  scan_run_id="ambench_multiline_process_temperature_scan_speed_concat_global_standard_a100_sxm4_40gb_v1"
  run_macro_pinn "$scan_run_id" "$seed" no_process
  run_macro_pinn "$scan_run_id" "$seed" process_concat_global_standard \
    --input-conditioning-mode concat \
    --input-feature-normalization global_standard \
    --input-feature-column laser_power_W \
    --input-feature-column scan_speed_mm_s \
    --input-feature-column spot_size_um

  spot_run_id="ambench_multiline_process_temperature_spot_size_film_global_standard_a100_sxm4_40gb_v1"
  run_macro_pinn "$spot_run_id" "$seed" no_process
  run_macro_pinn "$spot_run_id" "$seed" process_film_global_standard \
    --input-conditioning-mode film \
    --input-feature-normalization global_standard \
    --input-feature-column laser_power_W \
    --input-feature-column scan_speed_mm_s \
    --input-feature-column spot_size_um
done
