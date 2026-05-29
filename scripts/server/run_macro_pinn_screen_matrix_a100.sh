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
  local lr_tag="${lr//./p}"
  lr_tag="${lr_tag//-/_}"
  local run_id="${dataset_id}_macro_pinn_minmax_h${hidden}_l${layers}_lr${lr_tag}_s${seed}_screen_v1"

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

for dataset in uniform active; do
  if [[ "$dataset" == "uniform" ]]; then
    dataset_id="$UNIFORM_ID"
    table="$UNIFORM_TABLE"
    split="$UNIFORM_SPLIT"
  else
    dataset_id="$ACTIVE_ID"
    table="$ACTIVE_TABLE"
    split="$ACTIVE_SPLIT"
  fi

  run_one "$dataset_id" "$table" "$split" 64 3 1e-3 0
  run_one "$dataset_id" "$table" "$split" 128 4 1e-3 0
  run_one "$dataset_id" "$table" "$split" 128 4 3e-4 0
  run_one "$dataset_id" "$table" "$split" 256 4 1e-3 0
done
