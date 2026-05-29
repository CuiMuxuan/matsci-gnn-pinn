# AM-Bench Macro PINN Screen Matrix v1

## Context

- Server: Ubuntu 22, NVIDIA A100-SXM4-40GB
- Code commits: `6a1079c` for screen matrix, `a9d8c91` for seed-check script
- Target: calibrated `temperature_C`
- Model family: data-only Macro PINN, `--input-normalization minmax`, `pde_weight=0`
- Region metrics: hot q90 and spatial-gradient q90 on each test split

## Commands

Screen matrix:

```bash
bash scripts/server/run_macro_pinn_screen_matrix_a100.sh > logs/ambench_macro_pinn_screen_matrix_a100_v1.log 2>&1
```

Seed check:

```bash
bash scripts/server/run_macro_pinn_seed_check_a100.sh > logs/ambench_macro_pinn_seed_check_a100_v1.log 2>&1
```

## Single-Seed Screen Results

| Dataset | Hidden | Layers | LR | Seed | Test RMSE | Test MAE | Hot q90 RMSE | Gradient q90 RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform dense | 64 | 3 | 1e-3 | 0 | 65.663493 | 48.922885 | 95.342541 | 98.786883 |
| uniform dense | 128 | 4 | 1e-3 | 0 | 57.379604 | 35.247212 | 34.422982 | 58.368662 |
| uniform dense | 128 | 4 | 3e-4 | 0 | 98.771551 | 75.770600 | 225.734297 | 224.678919 |
| uniform dense | 256 | 4 | 1e-3 | 0 | 69.241612 | 44.551552 | 33.400658 | 60.445608 |
| active hot/gradient | 64 | 3 | 1e-3 | 0 | 68.760134 | 48.324435 | 59.688458 | 73.659626 |
| active hot/gradient | 128 | 4 | 1e-3 | 0 | 61.608843 | 37.778385 | 22.303900 | 63.986217 |
| active hot/gradient | 128 | 4 | 3e-4 | 0 | 58.792020 | 39.950708 | 36.956233 | 64.390129 |
| active hot/gradient | 256 | 4 | 1e-3 | 0 | 64.171156 | 37.869696 | 18.303149 | 65.799432 |

## Three-Seed Candidate Summary

| Candidate | Test RMSE mean | Test RMSE std | Test MAE mean | Hot q90 RMSE mean | Hot q90 RMSE std | Gradient q90 RMSE mean | Gradient q90 RMSE std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform dense h128/l4/lr1e-3 | 61.222230 | 3.409577 | 38.299423 | 34.542146 | 0.378095 | 57.713301 | 0.652700 |
| active h128/l4/lr3e-4 | 67.074483 | 10.182834 | 46.341987 | 49.974661 | 26.291582 | 70.300371 | 9.931036 |
| active h256/l4/lr1e-3 | 60.143582 | 3.050260 | 35.973779 | 17.033393 | 2.084327 | 63.622568 | 1.540029 |

## Interpretation

The screen matrix strengthens the Stage A conclusion: `minmax` Macro PINN is not a one-off success. A moderate `h128/l4/lr1e-3` model is stable on the uniform dense table, with low variance across seeds and consistent hot/gradient q90 behavior.

The active hot/gradient table gives the most useful melt-pool-focused result with `h256/l4/lr1e-3`: its three-seed hot q90 RMSE is `17.033393 +/- 2.084327`, much lower than the uniform dense candidate's `34.542146 +/- 0.378095`. Its global test RMSE is also competitive with the uniform candidate. The `h128/l4/lr3e-4` active candidate is less stable and should not be used as the main active-sampling configuration.

The current best data-only baselines are now:

| Use | Dataset | Configuration |
| --- | --- | --- |
| Global reconstruction | uniform dense | `hidden_dim=128`, `layers=4`, `lr=1e-3`, `input_normalization=minmax` |
| Hot-zone focused model | active hot/gradient | `hidden_dim=256`, `layers=4`, `lr=1e-3`, `input_normalization=minmax` |

## Next Decision

The project can now move beyond pure data-only tuning. The next research step should not be the old constant-property PDE residual, because earlier scans showed it degrades performance. The better next branch is a scaled heat-source or sparse closure MVP:

1. Keep the two data-only configurations above as fixed baselines.
2. Implement a closure/source-term branch that learns a correction term instead of forcing the current constant-property residual.
3. Evaluate closure variants on global test RMSE, hot q90 RMSE, gradient q90 RMSE, residual scale, and closure sparsity.

## Artifacts

- Screen script: `scripts/server/run_macro_pinn_screen_matrix_a100.sh`
- Seed-check script: `scripts/server/run_macro_pinn_seed_check_a100.sh`
- Screen log: `logs/ambench_macro_pinn_screen_matrix_a100_v1.log`
- Seed-check log: `logs/ambench_macro_pinn_seed_check_a100_v1.log`
- Run directories: `outputs/runs/*screen_v1/` and `outputs/runs/*seedcheck_v1/`
- Environment freeze files were saved for representative best runs:
  - `outputs/runs/ambench_line_0_1_temperature_dense_a100_sxm4_40gb_v1_macro_pinn_minmax_h128_l4_lr1e_3_s0_screen_v1/`
  - `outputs/runs/ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1_macro_pinn_minmax_h256_l4_lr1e_3_s0_screen_v1/`
