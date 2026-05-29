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

run_one() {
  local dataset_id="$1"
  local table="$2"
  local split="$3"
  local hidden="$4"
  local layers="$5"
  local lr="$6"
  local seed="$7"
  local tag="$8"
  local lr_tag="${lr//./p}"
  lr_tag="${lr_tag//-/_}"
  local run_id="${dataset_id}_macro_pinn_minmax_h${hidden}_l${layers}_lr${lr_tag}_s${seed}_${tag}_v1"

  if [[ -f "outputs/runs/${run_id}/metrics.json" ]]; then
    echo "Skipping existing run: ${run_id}"
    return
  fi

  "$CONDA_BIN" run -n "$CONDA_ENV" python -m gnnpinn.train.macro_pinn \
    --table "$table" \
    --target temperature_C \
    --split-manifest "$split" \
    --output-dir "outputs/runs/${run_id}" \
    --steps "$STEPS" \
    --hidden-dim "$hidden" \
    --layers "$layers" \
    --lr "$lr" \
    --seed "$seed" \
    --device "$DEVICE" \
    --input-normalization minmax \
    --hot-quantile 0.9 \
    --gradient-quantile 0.9 \
    --log-every 100
}

UNIFORM_ID="ambench_line_0_1_temperature_dense_a100_sxm4_40gb_v1"
UNIFORM_TABLE="data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_dense_a100_sxm4_40gb_v1.csv"
UNIFORM_SPLIT="outputs/data_splits/ambench_line_0_1_temperature_dense_a100_sxm4_40gb_v1_split.json"

ACTIVE_ID="ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1"
ACTIVE_TABLE="data/interim/ambench/2022_single_track/AMB2022-03/ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1.csv"
ACTIVE_SPLIT="outputs/data_splits/ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1_split.json"

for seed in 1 2; do
  run_one "$UNIFORM_ID" "$UNIFORM_TABLE" "$UNIFORM_SPLIT" 128 4 1e-3 "$seed" "seedcheck"
  run_one "$ACTIVE_ID" "$ACTIVE_TABLE" "$ACTIVE_SPLIT" 128 4 3e-4 "$seed" "seedcheck"
  run_one "$ACTIVE_ID" "$ACTIVE_TABLE" "$ACTIVE_SPLIT" 256 4 1e-3 "$seed" "seedcheck"
done
