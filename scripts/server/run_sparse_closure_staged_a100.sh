#!/usr/bin/env bash
set -euo pipefail

cd /root/matsci-gnn-pinn
mkdir -p logs outputs/runs

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
STEPS="${STEPS:-2000}"
DEVICE="${DEVICE:-cuda}"

ACTIVE_ID="ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1"
ACTIVE_TABLE="data/interim/ambench/2022_single_track/AMB2022-03/ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1.csv"
ACTIVE_SPLIT="outputs/data_splits/ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1_split.json"

run_one() {
  local start_step="$1"
  local sample_size="$2"
  local mode="$3"
  local run_id="${ACTIVE_ID}_macro_pinn_sparse_closure_h256_l4_lr1e_3_pde_1e_6_staged_${start_step}_${mode}_${sample_size}_v1"

  "$CONDA_BIN" run -n "$CONDA_ENV" python -m gnnpinn.train.macro_pinn \
    --table "$ACTIVE_TABLE" \
    --target temperature_C \
    --split-manifest "$ACTIVE_SPLIT" \
    --output-dir "outputs/runs/${run_id}" \
    --steps "$STEPS" \
    --hidden-dim 256 \
    --layers 4 \
    --lr 1e-3 \
    --seed 0 \
    --device "$DEVICE" \
    --input-normalization minmax \
    --pde-weight 1e-6 \
    --pde-field normalized \
    --closure-start-step "$start_step" \
    --residual-sample-size "$sample_size" \
    --residual-sampling-mode "$mode" \
    --residual-sampling-seed 2026 \
    --residual-hot-quantile 0.9 \
    --residual-gradient-quantile 0.9 \
    --rho-cp 1.0 \
    --conductivity 1.0 \
    --closure-mode sparse_linear \
    --closure-feature T \
    --closure-feature x \
    --closure-feature y \
    --closure-feature t \
    --closure-polynomial-order 1 \
    --closure-l1-weight 1e-5 \
    --closure-threshold 1e-6 \
    --hot-quantile 0.9 \
    --gradient-quantile 0.9 \
    --log-every 100
}

for start_step in 1000 1500; do
  run_one "$start_step" 2048 random
  run_one "$start_step" 4096 random
done
