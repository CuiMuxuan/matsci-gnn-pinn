#!/usr/bin/env bash
set -euo pipefail
cd /root/matsci-gnn-pinn
mkdir -p logs
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
TABLE=data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_dense_a100_sxm4_40gb_v1.csv
SPLIT=outputs/data_splits/ambench_line_0_1_temperature_dense_a100_sxm4_40gb_v1_split.json
for weight in 0 1e-8 1e-7 1e-6 1e-5; do
  safe_weight=${weight//-/_neg_}
  safe_weight=${safe_weight//./p}
  /home/vipuser/miniconda3/bin/conda run -n gnnpinn python -m gnnpinn.train.macro_pinn \
    --table "$TABLE" \
    --target temperature_C \
    --split-manifest "$SPLIT" \
    --output-dir "outputs/runs/ambench_line_0_1_temperature_dense_a100_sxm4_40gb_macro_pinn_minmax_pde_${safe_weight}_v1" \
    --steps 2000 \
    --hidden-dim 128 \
    --layers 4 \
    --device cuda \
    --input-normalization minmax \
    --pde-weight "$weight" \
    --conductivity 1.0 \
    --rho-cp 1.0 \
    --log-every 100
 done
