# AM-Bench Sparse Closure Residual Sampling v1

## Context

- Server: Ubuntu 22, NVIDIA A100-SXM4-40GB
- Code commit: `62b7394`
- Target: calibrated `temperature_C`
- Model: Macro PINN with sparse linear source closure
- PDE residual field: normalized model output
- PDE weight: `1e-6`
- Closure features: `T, x, y, t`
- Residual sampling: random train-point subset per step

## Command

```bash
bash scripts/server/run_sparse_closure_residual_sampling_a100.sh > logs/ambench_sparse_closure_residual_sampling_a100_v1.log 2>&1
```

## Results

| Dataset | Residual points/step | Test RMSE | Test MAE | Hot q90 RMSE | Gradient q90 RMSE | Final PDE loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform dense | 2048 | 66.496204 | 52.048810 | 93.675988 | 100.043368 | 12643.827148 |
| uniform dense | 4096 | 70.341116 | 56.046876 | 114.163806 | 118.281023 | 101874.500000 |
| active hot/gradient | 2048 | 71.072510 | 54.806314 | 65.723667 | 80.805241 | 8993.016602 |
| active hot/gradient | 4096 | 81.110114 | 61.977610 | 52.467783 | 76.992195 | 7630.437500 |

## Comparison

| Method | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE |
| --- | ---: | ---: | ---: |
| uniform data-only h128/l4/lr1e-3, 3 seed mean | 61.222230 | 34.542146 | 57.713301 |
| active data-only h256/l4/lr1e-3, 3 seed mean | 60.143582 | 17.033393 | 63.622568 |
| best all-point sparse closure uniform | 64.528347 | 92.136607 | 98.680265 |
| best residual-sampled sparse closure uniform | 66.496204 | 93.675988 | 100.043368 |
| best all-point sparse closure active | 80.462084 | 90.508746 | 100.498295 |
| best residual-sampled sparse closure active | 71.072510 | 65.723667 | 80.805241 |

## Interpretation

Residual sampling improves the active hot/gradient closure branch compared with all-point sparse closure, especially on global test RMSE and gradient q90. However, it still does not outperform the data-only active baseline, and uniform dense closure remains worse than data-only.

The useful signal is that residual sampling changes the failure mode: active closure no longer collapses as badly as all-point residual training. This suggests the next improvement should not be a larger sparse library yet. The next experiment should sample residual points by hot/gradient priority instead of random train subsets.

## Next Action

Implement residual sampling modes:

```text
--residual-sampling-mode random|hot|gradient|hot_gradient
```

Use active hot/gradient data first, with:

```text
pde_weight=1e-6
residual_sample_size=2048 or 4096
closure_features=T,x,y,t
```

Only if hot/gradient residual sampling approaches the data-only hot q90 baseline should this branch move to 3 seed or polynomial order 2.

## Artifacts

- Script: `scripts/server/run_sparse_closure_residual_sampling_a100.sh`
- Log: `logs/ambench_sparse_closure_residual_sampling_a100_v1.log`
- Runs: `outputs/runs/*pde_1e_6_residual_*_v1/`
