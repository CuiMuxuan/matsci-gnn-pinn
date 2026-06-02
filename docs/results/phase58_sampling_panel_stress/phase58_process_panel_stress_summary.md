# Phase 55 Spot-Size Route Seed Validation

Transfer gate: `seed_robust_transfer_positive`

| dataset | route | aggregate gate | paired seed gate |
|---|---|---|---|
| broad15 | film/global_standard | yes | yes |

## Aggregate Metrics

| dataset | method | n | test RMSE mean +/- std | hot q90 mean +/- std | gradient q90 mean +/- std |
|---|---|---:|---:|---:|---:|
| broad15 | no_process | 1 | 206.100512 +/- 0.000000 | 363.828210 +/- 0.000000 | 330.201687 +/- 0.000000 |
| broad15 | broad_process_v1 | 1 | 138.855456 +/- 0.000000 | 158.622677 +/- 0.000000 | 165.869192 +/- 0.000000 |

## Strong-Baseline Deltas

| dataset | metric | broad mean | best strong baseline | no-process mean | delta vs strong | delta vs no-process |
|---|---|---:|---:|---:|---:|---:|
| broad15 | rmse | 138.855456 | 151.850578 (mean) | 206.100512 | -12.995122 | -67.245057 |
| broad15 | hot_q90_rmse | 158.622677 | 252.554440 (mean) | 363.828210 | -93.931762 | -205.205533 |
| broad15 | gradient_q90_rmse | 165.869192 | 233.732337 (mean) | 330.201687 | -67.863145 | -164.332495 |

## Per-Seed Metrics

| dataset | method | seed | test RMSE | hot q90 RMSE | gradient q90 RMSE |
|---|---|---:|---:|---:|---:|
| broad15 | no_process | 7 | 206.100512 | 363.828210 | 330.201687 |
| broad15 | broad_process_v1 | 7 | 138.855456 | 158.622677 | 165.869192 |
