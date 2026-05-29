# AMB2022-03 Line_0_1 Calibrated Temperature Frame-Split Probe

## 数据与转换

- 数据集：AM-Bench 2022 / `mds2-2716`
- HDF5 文件：`Thermography/AMB2022-03-718-AMMT-StaringCamera_Signal.h5`
- HDF5 数据集：`ThermalData/Line_0_1/Signal`
- 目标：`temperature_C`
- 校准来源：`Calibration/ThermalCal` 属性
- 校准系数：`a=0.9655`, `b=197.2`, `c=43920000.0`
- 有效点过滤：`signal >= 100`
- 中等密度 probe 字段表：`data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_medium_probe.csv`
- 样本数：1044
- split 策略：按 frame 顺序切分，避免随机像素行切分造成的信息泄漏。

## 复现命令

```powershell
conda run -n gnnpinn-cu130 python -m gnnpinn.data.loaders.ambench_hdf5 `
  --sample-id amb2022_03_line_0_1_temperature_medium_probe `
  --output data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_medium_probe.csv `
  --manifest outputs/data_audits/ambench_line_0_1_temperature_medium_probe_manifest.json `
  --split-manifest outputs/data_splits/ambench_line_0_1_temperature_medium_probe_split.json `
  --calibrate-temperature `
  --split-strategy frame `
  --min-signal 100 `
  --frame-step 25 `
  --max-frames 20 `
  --row-step 8 `
  --max-rows 80 `
  --col-step 4 `
  --max-cols 76
```

```powershell
conda run -n gnnpinn-cu130 python -m gnnpinn.eval.field_baseline `
  --table data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_medium_probe.csv `
  --target temperature_C `
  --strategy mean `
  --split-manifest outputs/data_splits/ambench_line_0_1_temperature_medium_probe_split.json `
  --output outputs/baselines/ambench_line_0_1_temperature_medium_probe_mean_baseline.json
```

```powershell
conda run -n gnnpinn-cu130 python -m gnnpinn.train.macro_pinn `
  --table data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_medium_probe.csv `
  --target temperature_C `
  --split-manifest outputs/data_splits/ambench_line_0_1_temperature_medium_probe_split.json `
  --output-dir outputs/runs/ambench_line_0_1_temperature_medium_probe_macro_pinn `
  --steps 200 `
  --hidden-dim 64 `
  --layers 3 `
  --log-every 50
```

## 指标

| Method | Split | N | RMSE | MAE | Relative L2 |
|---|---:|---:|---:|---:|---:|
| Mean baseline, fit=train | train | 804 | 108.1227 | 83.5354 | 0.0927 |
| Mean baseline, fit=train | val | 145 | 114.9470 | 85.8904 | 0.0968 |
| Mean baseline, fit=train | test | 95 | 108.7099 | 81.8058 | 0.0914 |
| Macro PINN probe | train | 804 | 102.8402 | 76.2336 | 0.0881 |
| Macro PINN probe | val | 145 | 153.8624 | 110.5560 | 0.1296 |
| Macro PINN probe | test | 95 | 157.9750 | 116.8676 | 0.1329 |

## 结论

- 本地已完成从 NIST raw digital level 到校准温度字段表的完整闭环。
- frame-based split 已能复现 train/val/test 指标，比随机行切分更接近论文实验。
- 中等密度 probe 仍可在本机快速完成，因此本地验证阶段已经基本打通。
- 下一步若要形成论文主结果，应扩大到密集采样、多 frame/多 line、多 seed，并启用 PDE residual 或 GNN/coupled 模块。这一阶段会显著增加训练时间和显存需求，建议切换到云 GPU。
