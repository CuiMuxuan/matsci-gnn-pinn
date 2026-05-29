#!/usr/bin/env bash
set -euo pipefail
cd /root/matsci-gnn-pinn
mkdir -p logs
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
/home/vipuser/miniconda3/bin/conda run -n gnnpinn python -m gnnpinn.train.macro_pinn \
  --table data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_dense_a100_sxm4_40gb_v1.csv \
  --target temperature_C \
  --split-manifest outputs/data_splits/ambench_line_0_1_temperature_dense_a100_sxm4_40gb_v1_split.json \
  --output-dir outputs/runs/ambench_line_0_1_temperature_dense_a100_sxm4_40gb_macro_pinn_data_only_v1 \
  --steps 2000 \
  --hidden-dim 128 \
  --layers 4 \
  --device cuda \
  --log-every 100
