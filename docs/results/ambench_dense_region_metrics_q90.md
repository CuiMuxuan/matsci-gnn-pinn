# AM-Bench Dense Region Metrics q90

## Context

- Server: Ubuntu 22, NVIDIA A100-SXM4-40GB
- Code commit: `af09db5`
- Dataset: AM-Bench 2022 / AMB2022-03 / `mds2-2716`
- Target: `temperature_C`; split: frame-based split manifest
- Region metrics: `hot_q90` is the top 10% target-temperature region within each split; `gradient_q90` is the top 10% spatial-gradient-score region within each split.

## Test Split Metrics

| Method | Global RMSE | Hot q90 RMSE | Gradient q90 RMSE | Global MAE | Hot q90 MAE | Gradient q90 MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| constant:mean:fit=train | 108.387552 | 255.162035 | 160.089312 | 85.212557 | 253.169547 | 132.894164 |
| model:knn:fit=train | 149.362700 | 355.701648 | 239.949462 | 103.822629 | 354.272046 | 208.673322 |
| model:random_forest:fit=train | 149.529723 | 358.569639 | 237.524896 | 102.558852 | 357.073989 | 205.342959 |
| model:extra_trees:fit=train | 153.067908 | 361.533497 | 242.842196 | 107.324742 | 360.126836 | 212.214804 |
| macro_pinn:minmax | 51.655371 | 51.937943 | 66.029932 | 34.551693 | 46.654718 | 51.072380 |

## Findings

- The minmax Macro PINN remains the strongest method on global test RMSE and also improves the hot and high-gradient q90 regions.
- Region metrics expose that the high-gradient subset is substantially harder than the global field for all methods, so future sampling should explicitly target this band.
- These metrics provide a better bridge toward melt-pool-relevant evaluation than global frame-split RMSE alone.

## Next Actions

1. Add a data conversion/sampling mode that oversamples hot and gradient-band points while keeping background anchors.
2. Run minmax Macro PINN on balanced hot/background and gradient-band datasets.
3. Use q90 region metrics as mandatory columns in subsequent closure and PDE residual scans.
