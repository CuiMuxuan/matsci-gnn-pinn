# AM-Bench Coordinate RBF Graph-Conditioned Closure v1

## Context

- Server: Ubuntu 22, NVIDIA A100-SXM4-40GB
- Code commit: `a032f44`
- Dataset: active hot/gradient AM-Bench table
- Model: Macro PINN with sparse linear closure and per-point coordinate RBF graph features
- Graph mode: `coordinate_rbf`
- PDE field: normalized model output
- PDE weight: `1e-6`
- Closure LR: `1e-5`
- Closure start step: `1500`
- Residual sampling: random train subset, `4096` points/step

## Command

```bash
bash scripts/server/run_graph_conditioned_closure_coordinate_rbf_a100.sh \
  > logs/ambench_graph_conditioned_closure_coordinate_rbf_a100_v2.log 2>&1
```

The first attempt failed because the server table has three coordinate columns plus time. The CLI now infers `state_dim = coord_dim + time_dim`, and the second run completed.

## Results

| Run | Embedding dim | Length scale | Test RMSE | Test MAE | Hot q90 RMSE | Gradient q90 RMSE | Final PDE loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `g4_ls0_35` | 4 | 0.35 | 109.067742 | 98.257430 | 150.254087 | 153.408553 | 46314.546875 |
| `g6_ls0_25` | 6 | 0.25 | 68.237717 | 53.094058 | 50.900600 | 71.879264 | 3316.148926 |

## Comparison

| Method | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE |
| --- | ---: | ---: | ---: |
| active data-only h256/l4/lr1e-3, 3 seed mean | 60.143582 | 17.033393 | 63.622568 |
| sparse closure best, `closure_lr=1e-5`, trainable backbone | 70.494433 | 31.542155 | 64.558069 |
| toy/static graph-conditioned closure | 70.057147 | 76.367170 | 87.591751 |
| coordinate RBF graph-conditioned closure, `g6_ls0_25` | 68.237717 | 50.900600 | 71.879264 |

## Best Learned Expression

Best coordinate RBF run, `g6_ls0_25`:

```text
0.00503194658085704*T + 0.000362917955499142*g0 + 0.00523830158635974*g3 + 0.000838401610963047*g4 + 0.00503194658085704*t + 0.00563566712662578*x + 0.0046725464053452*y + 0.00577792525291443
```

Graph feature metadata:

```text
state_dim = 4
features = g0,g1,g2,g3,g4,g5
length_scale = 0.25
```

## Interpretation

Per-point coordinate RBF graph features are better than static global graph features. The `g6_ls0_25` run improves global test RMSE relative to the best sparse closure and substantially improves over the toy/static graph-conditioned run.

However, it still does not beat the sparse closure baseline on hot q90 or gradient q90, and it remains weaker than active data-only. The region metrics show that adding coordinate graph features can help global reconstruction while still harming the regions most important for melt-pool and gradient behavior.

The `g4_ls0_35` run is a clear negative result. Larger length scale and fewer anchors made the graph-conditioned source too coarse and over-constrained the thermal field.

## Next Action

Do not expand embedding dimension blindly. The next graph-conditioned closure step should protect the data-only/hot-zone fit:

1. Add a gated graph residual term, e.g. `q = q_sparse + alpha * q_graph` with small or learnable bounded `alpha`.
2. Add an option to detach graph features from direct source magnitude or penalize graph coefficients separately.
3. Re-run against the best sparse closure and active data-only baselines.

This is a better path than static embeddings or wider RBF libraries because the current failure mode is local-region degradation, not lack of graph feature availability.

## Artifacts

- Script: `scripts/server/run_graph_conditioned_closure_coordinate_rbf_a100.sh`
- Log: `logs/ambench_graph_conditioned_closure_coordinate_rbf_a100_v2.log`
- Runs: `outputs/runs/*coordinate_rbf_g*_v1/`
