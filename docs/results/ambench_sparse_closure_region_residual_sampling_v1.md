# AM-Bench Sparse Closure Region Residual Sampling v1

## Context

- Server: Ubuntu 22, NVIDIA A100-SXM4-40GB
- Code commit: `09da8fe`
- Dataset: active hot/gradient AM-Bench table
- Model: Macro PINN with sparse linear source closure
- PDE field: normalized model output
- PDE weight: `1e-6`
- Residual modes: `hot`, `gradient`, `hot_gradient`
- Closure features: `T, x, y, t`

## Command

```bash
bash scripts/server/run_sparse_closure_region_residual_sampling_a100.sh > logs/ambench_sparse_closure_region_residual_sampling_a100_v1.log 2>&1
```

## Results

| Residual mode | Sample size | Candidate points | Test RMSE | Test MAE | Hot q90 RMSE | Gradient q90 RMSE | Final PDE loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hot | 2048 | 2128 | 114.540389 | 93.645868 | 134.434827 | 135.683952 | 4908.314453 |
| hot | 4096 | 2128 | 100.499553 | 78.280240 | 182.293145 | 175.352863 | 71999.820312 |
| gradient | 2048 | 1186 | 135.230256 | 120.642979 | 212.966660 | 206.773733 | 11354.845703 |
| gradient | 4096 | 1186 | 135.230256 | 120.642979 | 212.966660 | 206.773733 | 11354.845703 |
| hot_gradient | 2048 | 2189 | 102.598827 | 81.939466 | 111.501009 | 127.103513 | 13717.685547 |
| hot_gradient | 4096 | 2189 | 97.442111 | 80.835405 | 122.969667 | 129.951066 | 38879.632812 |

## Comparison

| Method | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE |
| --- | ---: | ---: | ---: |
| active data-only h256/l4/lr1e-3, 3 seed mean | 60.143582 | 17.033393 | 63.622568 |
| random residual sampling, 2048 | 71.072510 | 65.723667 | 80.805241 |
| random residual sampling, 4096 | 81.110114 | 52.467783 | 76.992195 |
| best region residual sampling | 97.442111 | 111.501009 | 127.103513 |

## Interpretation

Region-aware residual sampling is worse than random residual sampling in the current sparse closure formulation. Concentrating the residual loss on hot or high-gradient candidate points appears to over-constrain the local structure and pushes the model away from the data-only optimum. The gradient candidate pool is only `1186` points, so `sample_size=2048` and `4096` collapse to the same effective residual set.

This is a useful stopping signal for the current closure branch. The problem is no longer just “where to sample residual points”; the residual itself is still too blunt. The learned source expressions become large positive linear combinations of `T,x,y,t`, which is not yet a physically meaningful heat-source representation.

## Next Action

Do not run 3 seed or polynomial order 2 on this branch yet. The next implementation should add staged training or warm-started residual fine-tuning:

1. Train data-only for most steps.
2. Turn on a very small residual/closure weight only in the final fine-tuning stage.
3. Optionally freeze or use a lower learning rate for the Macro PINN backbone while learning closure coefficients.

This tests whether closure can act as a small correction rather than dominating the temperature fit.

## Artifacts

- Script: `scripts/server/run_sparse_closure_region_residual_sampling_a100.sh`
- Log: `logs/ambench_sparse_closure_region_residual_sampling_a100_v1.log`
- Runs: `outputs/runs/*residual_hot*_v1/` and `outputs/runs/*residual_gradient*_v1/`
