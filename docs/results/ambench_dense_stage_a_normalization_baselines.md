# AM-Bench Dense Stage A: Normalization and Strong Baselines
## Context
- Server: Ubuntu 22, NVIDIA A100-SXM4-40GB
- Repository commit for code change: `ce7a14d`
- Dataset: AM-Bench 2022 / AMB2022-03 / `mds2-2716`
- Table: `data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_dense_a100_sxm4_40gb_v1.csv`
- Target: `temperature_C`; split: frame-based split manifest

## Strong Baselines

| Method | Test RMSE | Test MAE | Test Relative L2 |
| --- | ---: | ---: | ---: |
| constant:mean:fit=train | 108.387552 | 85.212557 | 0.091644 |
| model:knn:fit=train | 149.362700 | 103.822629 | 0.126289 |
| model:random_forest:fit=train | 149.529723 | 102.558852 | 0.126430 |
| model:extra_trees:fit=train | 153.067908 | 107.324742 | 0.129422 |

## Macro PINN Input Normalization Ablation

| Input normalization | Test RMSE | Test MAE | Test Relative L2 | All-point RMSE |
| --- | ---: | ---: | ---: | ---: |
| none | 148.265776 | 102.813809 | 0.125361 | 117.397597 |
| minmax | 51.655371 | 34.551693 | 0.043676 | 29.399075 |
| standard | 68.373154 | 47.269227 | 0.057811 | 29.452124 |

## Minmax Macro PINN PDE Weight Scan

| PDE weight | Test RMSE | Test MAE | Test Relative L2 | All-point RMSE |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 51.655371 | 34.551693 | 0.043676 | 29.399075 |
| 1e-08 | 92.565169 | 73.173061 | 0.078266 | 87.081595 |
| 1e-07 | 101.501261 | 80.084237 | 0.085821 | 102.993691 |
| 1e-06 | 104.028080 | 79.226447 | 0.087958 | 105.150920 |
| 1e-05 | 103.825584 | 81.683459 | 0.087786 | 105.814977 |

## Findings

- Input scaling is the dominant failure mode in the previous dense Macro PINN run. Minmax input normalization reduces test RMSE from 148.265776 to 51.655371.
- The model baselines using coordinate/time/process features under the current frame split are worse than the train-fitted mean baseline, which suggests poor frame extrapolation for local nonparametric/tree methods.
- The current constant-property PDE residual is not yet physically scaled. Any positive PDE weight in this scan worsens the minmax data-only model, so the next step should be PDE nondimensionalization or closure/source-term modeling rather than simply increasing PDE weight.

## Next Actions

1. Keep `--input-normalization minmax` as the default server-stage Macro PINN setting for this AM-Bench dense table.
2. Add hot-zone and gradient-band metrics/sampling so that improvements are evaluated on melt-pool-relevant regions, not only global frame split RMSE.
3. Rework PDE residual scaling and heat-source treatment before using PDE loss as a paper claim.
4. Run 3-seed minmax data-only and selected PDE/closure variants after hot-zone metrics are implemented.
