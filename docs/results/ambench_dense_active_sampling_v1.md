# AM-Bench Dense Active Sampling v1

## Context

- Server: Ubuntu 22, NVIDIA A100-SXM4-40GB
- Code commit: `600ce84`
- Dataset: NIST AM-Bench 2022 / AMB2022-03 / `mds2-2716`
- Source HDF5: `Thermography/AMB2022-03-718-AMMT-StaringCamera_Signal.h5`
- Target: calibrated `temperature_C`
- Split: frame-based split manifest

## Data Conversion

Command entry:

```bash
bash scripts/server/run_dense_active_sampling_a100.sh > logs/ambench_dense_active_sampling_a100_v1.log 2>&1
```

Sampling configuration:

| Item | Value |
| --- | ---: |
| Sampling mode | `balanced_hot_gradient` |
| Hot quantile | 0.9 |
| Gradient quantile | 0.9 |
| Background fraction | 0.15 |
| Source sampled grid valid points | 41054 |
| Written points | 14518 |
| Frames sampled | 120 |
| Frames with written rows | 71 |
| Hot selected points before union | 4152 |
| Gradient selected points before union | 4257 |
| Background anchor points | 6193 |

Split counts:

| Split | Points |
| --- | ---: |
| train | 11087 |
| val | 2527 |
| test | 904 |

## Test Split Metrics

| Method | Test RMSE | Test MAE | Test Relative L2 | Hot q90 RMSE | Gradient q90 RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mean baseline | 141.218029 | 122.968151 | 0.115541 | 236.901227 | 130.965911 |
| kNN | 186.993044 | 130.440398 | 0.152993 | 356.360517 | 226.885395 |
| RandomForest | 166.246187 | 114.724438 | 0.136018 | 321.843457 | 200.815695 |
| ExtraTrees | 174.812168 | 121.577421 | 0.143027 | 336.043613 | 202.553757 |
| Macro PINN minmax | 65.892559 | 44.769622 | 0.053912 | 30.868055 | 66.005554 |

Region sizes on the test split:

| Region | Points |
| --- | ---: |
| Hot q90 | 154 |
| Gradient q90 | 91-94 |

## Comparison With Uniform Dense Sampling

Compared with the previous uniform dense table (`41054` points), active sampling reduces the table to `14518` points while keeping the Macro PINN clearly ahead of the mean and model baselines. Global test RMSE is worse than the previous uniform minmax Macro PINN result (`51.655371` -> `65.892559`), but hot-zone q90 RMSE improves substantially (`51.937943` -> `30.868055`). Gradient q90 RMSE stays roughly unchanged (`66.029932` -> `66.005554`).

This means the current active sampling is useful for melt-pool/hot-zone accuracy, but it should not replace the uniform dense table as the global-field baseline. The two data views should be carried forward together: uniform dense for global reconstruction, balanced hot/gradient for melt-pool-focused learning and diagnostics.

## Artifacts

- Data table: `data/interim/ambench/2022_single_track/AMB2022-03/ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1.csv`
- Data manifest: `outputs/data_audits/ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1_manifest.json`
- Split manifest: `outputs/data_splits/ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1_split.json`
- Baselines: `outputs/baselines/ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1_*_regions_q90.json`
- Macro PINN run: `outputs/runs/ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1_macro_pinn_minmax_v1/`
- Environment freeze files saved in the run directory: `conda_from_history.yml`, `pip_freeze.txt`, `env_report.txt`, `nvidia_smi.txt`

## Interpretation

Active sampling validates the A3 hypothesis only partially. It improves the hottest test-region error and reduces data volume, but the global held-out-frame metric becomes weaker than uniform dense sampling. This is still a useful result for the paper route because it separates two objectives that should be reported independently: global thermal-field reconstruction and melt-pool-relevant hot-zone prediction.

Next, keep `--input-normalization minmax` and move to a small Macro PINN training matrix on both uniform dense and active sampled tables. The immediate priority is width/layer/lr/seed robustness before reintroducing PDE residual or sparse closure.
