# AM-Bench Sparse Closure MVP v1

## Context

- Server: Ubuntu 22, NVIDIA A100-SXM4-40GB
- Code commit: `4cffcda`
- Target: calibrated `temperature_C`
- Model: Macro PINN with sparse linear source closure
- PDE residual field: normalized model output
- Closure features: `T, x, y, t`
- Closure library: bias + linear terms
- Closure L1 weight: `1e-5`

## Commands

Initial scan:

```bash
bash scripts/server/run_sparse_closure_mvp_a100.sh > logs/ambench_sparse_closure_mvp_a100_v1.log 2>&1
```

Low-weight correction scan:

```bash
bash scripts/server/run_sparse_closure_low_weight_a100.sh > logs/ambench_sparse_closure_low_weight_a100_v1.log 2>&1
```

The low-weight scan was added after `pde_weight=1e-4` and `1e-3` clearly over-regularized the model.

## Results

| Dataset | PDE weight | Test RMSE | Test MAE | Hot q90 RMSE | Gradient q90 RMSE | Final PDE loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform dense | 1e-3 | 102.146403 | 76.132038 | 243.109519 | 240.910745 | 3.458655 |
| uniform dense | 1e-4 | 96.448626 | 68.360822 | 240.442474 | 238.199377 | 95.164452 |
| uniform dense | 1e-5 | 81.632505 | 64.421231 | 182.028648 | 181.774461 | 2324.112305 |
| uniform dense | 1e-6 | 64.528347 | 49.548184 | 92.136607 | 98.680265 | 16810.667969 |
| active hot/gradient | 1e-3 | 132.260569 | 107.479709 | 219.700385 | 211.673021 | 1.599893 |
| active hot/gradient | 1e-4 | 129.083732 | 110.789971 | 194.033981 | 189.388606 | 29.145166 |
| active hot/gradient | 1e-5 | 125.281492 | 98.340917 | 215.098875 | 207.265134 | 2112.061279 |
| active hot/gradient | 1e-6 | 80.462084 | 66.089187 | 90.508746 | 100.498295 | 13240.869141 |

## Baseline Comparison

| Baseline | Test RMSE mean | Hot q90 RMSE mean | Gradient q90 RMSE mean |
| --- | ---: | ---: | ---: |
| uniform data-only h128/l4/lr1e-3, 3 seed | 61.222230 | 34.542146 | 57.713301 |
| active data-only h256/l4/lr1e-3, 3 seed | 60.143582 | 17.033393 | 63.622568 |
| best uniform sparse closure, seed 0 | 64.528347 | 92.136607 | 98.680265 |
| best active sparse closure, seed 0 | 80.462084 | 90.508746 | 100.498295 |

## Learned Closure Expressions

Best uniform low-weight run:

```text
0.325855702161789*T + 0.325855702161789*t + 0.533704340457916*x + 0.280943661928177*y + 0.568598866462708
```

Best active low-weight run:

```text
0.714747667312622*T + 0.714747667312622*t + 0.762038111686707*x + 0.694019973278046*y + 0.78198254108429
```

## Interpretation

This MVP is a useful negative/diagnostic result. Sparse source closure now runs end to end, saves coefficients, exports expressions, and produces stable artifacts. However, with the current all-point residual formulation, the closure branch does not beat the fixed data-only baselines.

The main signal is scale sensitivity. `pde_weight=1e-3` and `1e-4` over-regularize both datasets. Reducing to `1e-6` restores reasonable global metrics, but hot-zone and gradient metrics are still worse than data-only. This suggests that the current residual is acting as a broad smoothness constraint rather than a physically helpful heat-source model.

## Next Action

Do not run 3 seed sparse closure yet. First revise the residual training design:

1. Add residual/collocation sampling so PDE loss is evaluated on a smaller subset instead of every train point.
2. Allow residual sampling to focus on hot/gradient regions.
3. Add an explicit heat-source feature basis such as `laser_power_W`, `scan_speed_mm_s`, and local time/position relative to the scan path once scan strategy alignment is available.
4. Then rerun `pde_weight=1e-6` and lower values before committing to seed studies.

This keeps direction one alive as a closure-discovery path, but it shows that the next engineering task is residual sampling/scaling rather than a larger sparse-library sweep.

## Artifacts

- Script: `scripts/server/run_sparse_closure_mvp_a100.sh`
- Low-weight script: `scripts/server/run_sparse_closure_low_weight_a100.sh`
- Logs:
  - `logs/ambench_sparse_closure_mvp_a100_v1.log`
  - `logs/ambench_sparse_closure_low_weight_a100_v1.log`
- Run directories:
  - `outputs/runs/*macro_pinn_sparse_closure*_v1/`
  - `outputs/runs/*macro_pinn_sparse_closure*_low_v1/`
