# AM-Bench Sparse Closure Optimizer Ablation v1

## Context

- Server: Ubuntu 22, NVIDIA A100-SXM4-40GB
- Code commit: `b171253`
- Dataset: active hot/gradient AM-Bench table
- Model: Macro PINN with sparse linear source closure
- PDE field: normalized model output
- PDE weight: `1e-6`
- Closure start step: `1500`
- Residual sampling: random train subset, `4096` points/step
- Compared controls: closure coefficient learning rate and optional backbone freezing

## Command

```bash
bash scripts/server/run_sparse_closure_optimizer_ablation_a100.sh \
  > logs/ambench_sparse_closure_optimizer_ablation_a100_v1.log 2>&1
```

## Results

| Closure LR | Backbone after closure start | Test RMSE | Test MAE | Hot q90 RMSE | Gradient q90 RMSE | Final PDE loss |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `1e-4` | trainable | 109.950813 | 97.167877 | 128.164674 | 133.914912 | 41117.039062 |
| `1e-5` | trainable | 70.494433 | 50.953729 | 31.542155 | 64.558069 | 6499.495117 |
| `1e-4` | frozen | 75.487093 | 60.751385 | 66.471083 | 81.179762 | 77996.406250 |
| `1e-5` | frozen | 82.785000 | 65.369827 | 54.034580 | 78.664636 | 18191.269531 |

## Comparison

| Method | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE |
| --- | ---: | ---: | ---: |
| active data-only h256/l4/lr1e-3, 3 seed mean | 60.143582 | 17.033393 | 63.622568 |
| staged closure, start 1500, 4096 | 71.341922 | 56.310533 | 73.296876 |
| optimizer ablation, closure lr `1e-5`, trainable | 70.494433 | 31.542155 | 64.558069 |
| optimizer ablation, closure lr `1e-5`, frozen | 82.785000 | 54.034580 | 78.664636 |

## Learned Closure Expressions

Best run, `closure_lr=1e-5`, trainable backbone:

```text
0.00517352391034365*T + 0.00517352391034365*t + 0.00579203246161342*x + 0.00484903994947672*y + 0.00602354714646935
```

Frozen backbone, `closure_lr=1e-5`:

```text
0.00568518182262778*T + 0.00568518182262778*t + 0.00618002656847239*x + 0.00539807137101889*y + 0.00654948223382235
```

## Interpretation

Lowering the closure coefficient learning rate is the first change that substantially improves the staged sparse closure branch. The best run recovers most of the gradient q90 gap versus data-only and improves hot q90 from the staged baseline.

Freezing the Macro PINN backbone after warmup does not help in this configuration. It prevents severe coefficient growth but also limits field adaptation, so the closure branch remains weaker than the trainable-backbone low-lr run.

The closure branch still does not beat the active data-only baseline, especially on hot q90. The sparse closure path is now technically credible and interpretable, but it is not yet a positive paper result by itself.

## Next Action

Do not expand polynomial order yet. The next project-facing step should enter D1:

1. Add a GNN-conditioned closure interface using toy/synthetic graph embeddings.
2. Keep the best C1 setting as the closure baseline: `closure_lr=1e-5`, trainable backbone, `closure_start_step=1500`, `residual_sample_size=4096`.
3. Compare sparse closure against GNN-conditioned closure on the same active AM-Bench table.

This keeps direction one as a reproducible entry point while moving toward the direction three coupling claim.

## Artifacts

- Script: `scripts/server/run_sparse_closure_optimizer_ablation_a100.sh`
- Log: `logs/ambench_sparse_closure_optimizer_ablation_a100_v1.log`
- Runs: `outputs/runs/*random4096_clr*_v1/`
