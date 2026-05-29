# AMB2022-03 Line_0_1 Signal Split Smoke Results

## 数据与任务

- 数据集：AM-Bench 2022 / `mds2-2716`
- HDF5 文件：`Thermography/AMB2022-03-718-AMMT-StaringCamera_Signal.h5`
- HDF5 数据集：`ThermalData/Line_0_1/Signal`
- 原始 shape：`(700, 640, 304)`，即 `(frame, y_pixel, x_pixel)`
- 字段表：`data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_signal_subset.csv`
- 样本数：320
- 目标：`signal` raw digital level
- 切分：`outputs/data_splits/ambench_line_0_1_signal_subset_split.json`

当前结果是 smoke 级别真实数据闭环，不应被解读为最终论文性能。目标列仍是 raw digital level，温度校准需在后续阶段验证 NIST calibration equation 后再加入。

## 复现命令

转换 HDF5 小样本：

```powershell
conda run -n gnnpinn-cu130 python -m gnnpinn.data.loaders.ambench_hdf5 `
  --output data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_signal_subset.csv `
  --manifest outputs/data_audits/ambench_line_0_1_hdf5_conversion_manifest.json `
  --split-manifest outputs/data_splits/ambench_line_0_1_signal_subset_split.json
```

split-aware mean baseline：

```powershell
conda run -n gnnpinn-cu130 python -m gnnpinn.eval.field_baseline `
  --table data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_signal_subset.csv `
  --target signal `
  --strategy mean `
  --split-manifest outputs/data_splits/ambench_line_0_1_signal_subset_split.json `
  --output outputs/baselines/ambench_line_0_1_signal_split_mean_baseline.json
```

split-aware Macro PINN smoke：

```powershell
conda run -n gnnpinn-cu130 python -m gnnpinn.train.macro_pinn `
  --table data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_signal_subset.csv `
  --target signal `
  --split-manifest outputs/data_splits/ambench_line_0_1_signal_subset_split.json `
  --output-dir outputs/runs/ambench_line_0_1_signal_macro_pinn_split_smoke `
  --steps 100 `
  --hidden-dim 32 `
  --layers 2 `
  --log-every 25
```

## 指标

| Method | Split | N | RMSE | MAE | Relative L2 |
|---|---:|---:|---:|---:|---:|
| Mean baseline, fit=train | train | 224 | 274.4019 | 44.4605 | 0.9965 |
| Mean baseline, fit=train | val | 48 | 511.8639 | 97.5491 | 0.9943 |
| Mean baseline, fit=train | test | 48 | 29.8650 | 26.0699 | 1.2315 |
| Macro PINN smoke | train | 224 | 241.7156 | 37.9653 | 0.8778 |
| Macro PINN smoke | val | 48 | 524.0058 | 98.7187 | 1.0179 |
| Macro PINN smoke | test | 48 | 104.1901 | 29.5328 | 4.2963 |

## 解释与下一步

- Macro PINN 在 train split 上优于 mean baseline，说明训练链路有效。
- val/test 指标不稳定，主要因为当前 split 是随机行切分，320 点采样又很稀疏，test split 的目标幅值分布与 train/val 不一致。
- `relative_l2` 对低幅值 test split 很敏感，因此 test relative L2 暂不适合作为论文结论。
- 下一步本地优先做时间/帧分组切分或工况分组切分，再扩大采样密度。
- 当采样规模扩大到数万至数十万点、或引入 PINN PDE residual/GNN coupling 后，再切换到 A800/A100 服务器。
