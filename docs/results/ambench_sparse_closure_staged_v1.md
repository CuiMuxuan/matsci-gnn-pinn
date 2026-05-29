# AM-Bench Sparse Closure Staged Training v1

## Context

- Server: Ubuntu 22, NVIDIA A100-SXM4-40GB
- Code commit: `e26e6f6`
- Dataset: active hot/gradient AM-Bench table
- Model: Macro PINN with sparse linear source closure
- PDE field: normalized model output
- PDE weight: `1e-6`
- Residual sampling: random train subset
- Closure start steps: `1000`, `1500`

## Command

```bash
bash scripts/server/run_sparse_closure_staged_a100.sh > logs/ambench_sparse_closure_staged_a100_v1.log 2>&1
```

## Results

| Closure start step | Residual points/step | Test RMSE | Test MAE | Hot q90 RMSE | Gradient q90 RMSE | Final PDE loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 2048 | 99.215480 | 84.410754 | 76.866176 | 94.009690 | 779016.125000 |
| 1000 | 4096 | 115.802120 | 96.300133 | 171.779769 | 170.243559 | 30282.173828 |
| 1500 | 2048 | 70.642025 | 56.204066 | 61.976157 | 75.630230 | 31314.554688 |
| 1500 | 4096 | 71.341922 | 55.468233 | 56.310533 | 73.296876 | 11943.918945 |

## Comparison

| Method | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE |
| --- | ---: | ---: | ---: |
| active data-only h256/l4/lr1e-3, 3 seed mean | 60.143582 | 17.033393 | 63.622568 |
| random residual sampling, 2048 | 71.072510 | 65.723667 | 80.805241 |
| random residual sampling, 4096 | 81.110114 | 52.467783 | 76.992195 |
| staged closure, start 1500, 2048 | 70.642025 | 61.976157 | 75.630230 |
| staged closure, start 1500, 4096 | 71.341922 | 56.310533 | 73.296876 |

## Interpretation

Staged training helps compared with early or region-concentrated residual training. Starting closure at step `1500` is substantially better than step `1000`, and it recovers most of the global performance lost by the all-point closure runs.

However, staged closure still does not beat the data-only baseline. It also does not recover the active data-only hot-zone advantage. The learned closure is still a broad positive linear source in `T,x,y,t`, so the closure branch remains too entangled with the macro field fit.

## Next Action

The next change should separate optimizer behavior between the Macro PINN backbone and closure coefficients:

1. Add optional `--closure-lr` so sparse coefficients can train more slowly than the backbone.
2. Add optional `--freeze-backbone-after-closure-start` to test pure closure fine-tuning after data-only warmup.
3. Re-run only active table, `closure_start_step=1500`, `residual_sample_size=4096`, `pde_weight=1e-6`.

This is a better next step than increasing sparse polynomial order because the current linear closure already has enough freedom to harm the field fit.

## Artifacts

- Script: `scripts/server/run_sparse_closure_staged_a100.sh`
- Log: `logs/ambench_sparse_closure_staged_a100_v1.log`
- Runs: `outputs/runs/*staged_*_v1/`
