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
  local closure_lr="$1"
  local freeze_flag="$2"
  local tag="$3"
  local run_id="${ACTIVE_ID}_macro_pinn_sparse_closure_h256_l4_lr1e_3_clr${closure_lr}_staged1500_random4096_${tag}_v1"

  local freeze_args=()
  if [[ "$freeze_flag" == "freeze" ]]; then
    freeze_args=(--freeze-backbone-after-closure-start)
  fi

  "$CONDA_BIN" run -n "$CONDA_ENV" python -m gnnpinn.train.macro_pinn \
    --table "$ACTIVE_TABLE" \
    --target temperature_C \
    --split-manifest "$ACTIVE_SPLIT" \
    --output-dir "outputs/runs/${run_id}" \
    --steps "$STEPS" \
    --hidden-dim 256 \
    --layers 4 \
    --lr 1e-3 \
    --closure-lr "$closure_lr" \
    --seed 0 \
    --device "$DEVICE" \
    --input-normalization minmax \
    --pde-weight 1e-6 \
    --pde-field normalized \
    --closure-start-step 1500 \
    --residual-sample-size 4096 \
    --residual-sampling-mode random \
    --residual-sampling-seed 2026 \
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
    --log-every 100 \
    "${freeze_args[@]}"
}

run_one 1e-4 trainable clr1e_4_trainable
run_one 1e-5 trainable clr1e_5_trainable
run_one 1e-4 freeze clr1e_4_freeze
run_one 1e-5 freeze clr1e_5_freeze
