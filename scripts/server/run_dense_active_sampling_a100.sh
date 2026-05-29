#!/usr/bin/env bash
set -euo pipefail

cd /root/matsci-gnn-pinn
mkdir -p logs outputs/baselines outputs/data_audits outputs/data_splits outputs/runs

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
RUN_ID="ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1"
TABLE="data/interim/ambench/2022_single_track/AMB2022-03/${RUN_ID}.csv"
MANIFEST="outputs/data_audits/${RUN_ID}_manifest.json"
SPLIT="outputs/data_splits/${RUN_ID}_split.json"

"$CONDA_BIN" run -n "$CONDA_ENV" python -m gnnpinn.data.loaders.ambench_hdf5 \
  --sample-id "amb2022_03_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1" \
  --output "$TABLE" \
  --manifest "$MANIFEST" \
  --split-manifest "$SPLIT" \
  --calibrate-temperature \
  --split-strategy frame \
  --min-signal 100 \
  --frame-step 5 \
  --max-frames 120 \
  --row-step 2 \
  --max-rows 320 \
  --col-step 2 \
  --max-cols 152 \
  --sampling-mode balanced_hot_gradient \
  --hot-quantile 0.9 \
  --gradient-quantile 0.9 \
  --background-fraction 0.15

for strategy in mean knn random_forest extra_trees; do
  "$CONDA_BIN" run -n "$CONDA_ENV" python -m gnnpinn.eval.field_baseline \
    --table "$TABLE" \
    --target temperature_C \
    --strategy "$strategy" \
    --split-manifest "$SPLIT" \
    --feature-column x \
    --feature-column y \
    --feature-column t \
    --feature-column laser_power_W \
    --feature-column scan_speed_mm_s \
    --feature-column spot_size_um \
    --n-neighbors 8 \
    --n-estimators 200 \
    --random-state 7 \
    --hot-quantile 0.9 \
    --gradient-quantile 0.9 \
    --output "outputs/baselines/${RUN_ID}_${strategy}_regions_q90.json"
done

"$CONDA_BIN" run -n "$CONDA_ENV" python -m gnnpinn.train.macro_pinn \
  --table "$TABLE" \
  --target temperature_C \
  --split-manifest "$SPLIT" \
  --output-dir "outputs/runs/${RUN_ID}_macro_pinn_minmax_v1" \
  --steps 2000 \
  --hidden-dim 128 \
  --layers 4 \
  --device cuda \
  --input-normalization minmax \
  --hot-quantile 0.9 \
  --gradient-quantile 0.9 \
  --log-every 100
