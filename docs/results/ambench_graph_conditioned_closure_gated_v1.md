# AM-Bench Gated Graph-Conditioned Closure v1

## Context

- Server: Ubuntu 22, NVIDIA A100-SXM4-40GB
- Code commit: `64aa23c`
- Dataset: active hot/gradient AM-Bench table
- Model: Macro PINN with sparse linear closure and gated coordinate RBF graph features
- Graph mode: `coordinate_rbf`
- Graph features: `embedding_dim=6`, `length_scale=0.25`
- PDE field: normalized model output
- PDE weight: `1e-6`
- Closure LR: `1e-5`
- Closure start step: `1500`
- Residual sampling: random train subset, `4096` points/step
- Graph coefficient L1 weight: `1e-4`

## Command

```bash
bash scripts/server/run_graph_conditioned_closure_gated_a100.sh \
  > logs/ambench_graph_conditioned_closure_gated_a100_v1.log 2>&1
```

## Results

| Graph gate | Graph L1 | Test RMSE | Test MAE | Hot q90 RMSE | Gradient q90 RMSE | Final PDE loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | `1e-4` | 71.359550 | 55.668336 | 52.698404 | 71.601495 | 4683.980469 |
| 0.25 | `1e-4` | 72.840576 | 52.910100 | 31.919197 | 61.623894 | 5723.561523 |

## Comparison

| Method | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE |
| --- | ---: | ---: | ---: |
| active data-only h256/l4/lr1e-3, 3 seed mean | 60.143582 | 17.033393 | 63.622568 |
| sparse closure best, `closure_lr=1e-5`, trainable backbone | 70.494433 | 31.542155 | 64.558069 |
| coordinate RBF graph closure, `g6_ls0_25` | 68.237717 | 50.900600 | 71.879264 |
| gated graph closure, gate 0.25 | 72.840576 | 31.919197 | 61.623894 |

## Expressions

Gate 0.10:

```text
0.00529340840876102*T - 1.98369434656342e-6*g2 - 1.04156117686216e-6*g3 - 2.31819512919174e-6*g5 + 0.00529340840876102*t + 0.00557777471840382*x + 0.0050269216299057*y + 0.00577192567288876
```

Gate 0.25:

```text
0.00505439005792141*T + 0.00505439005792141*t + 0.00585786253213882*x + 0.00478038005530834*y + 0.0060384594835341
```

## Interpretation

Gating is useful for controlling graph-induced degradation. The gate `0.25` run nearly matches the best sparse closure hot q90 metric and improves gradient q90 beyond both sparse closure and the active data-only mean.

However, the graph coefficients are almost fully suppressed by `graph_l1_weight=1e-4`; the gate 0.25 expression retains no graph term above the `1e-6` threshold. This means the current result is best interpreted as a stability control and a useful ablation, not yet as strong evidence that graph features add physical signal.

The global test RMSE remains worse than sparse closure best and active data-only. The next step should reduce graph L1 or use a bounded learnable gate while keeping the acceptance criterion focused on not degrading hot q90.

## Next Action

Run a minimal graph-L1 sensitivity scan:

1. Keep `gate=0.25`, `embedding_dim=6`, `length_scale=0.25`.
2. Compare `graph_l1_weight=1e-5` and `1e-6`.
3. Accept the branch only if hot q90 remains near sparse closure best and graph terms survive thresholding.

If graph terms still vanish or hurt hot q90, move to manuscript-ready negative-result framing for synthetic graph conditioning and start planning real/semireal microstructure alignment.

## Artifacts

- Script: `scripts/server/run_graph_conditioned_closure_gated_a100.sh`
- Log: `logs/ambench_graph_conditioned_closure_gated_a100_v1.log`
- Runs: `outputs/runs/*graph_gated_sparse_closure*gate*_v1/`
