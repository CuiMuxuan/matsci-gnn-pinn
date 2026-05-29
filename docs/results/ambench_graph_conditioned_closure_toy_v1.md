# AM-Bench Toy Graph-Conditioned Closure v1

## Context

- Server: Ubuntu 22, NVIDIA A100-SXM4-40GB
- Code commit: `859e804`
- Dataset: active hot/gradient AM-Bench table
- Model: Macro PINN with sparse linear closure and toy graph-conditioned features
- Graph mode: `toy_static`
- PDE field: normalized model output
- PDE weight: `1e-6`
- Closure LR: `1e-5`
- Closure start step: `1500`
- Residual sampling: random train subset, `4096` points/step

## Command

```bash
bash scripts/server/run_graph_conditioned_closure_toy_a100.sh \
  > logs/ambench_graph_conditioned_closure_toy_a100_v1.log 2>&1
```

## Results

| Method | Test RMSE | Test MAE | Hot q90 RMSE | Gradient q90 RMSE | Final PDE loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph-conditioned toy/static closure | 70.057147 | 56.005710 | 76.367170 | 87.591751 | 17527.181641 |

## Comparison

| Method | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE |
| --- | ---: | ---: | ---: |
| active data-only h256/l4/lr1e-3, 3 seed mean | 60.143582 | 17.033393 | 63.622568 |
| sparse closure best, `closure_lr=1e-5`, trainable backbone | 70.494433 | 31.542155 | 64.558069 |
| toy graph-conditioned closure | 70.057147 | 76.367170 | 87.591751 |

## Learned Expression

```text
0.00726301642134786*T - 0.00167107989545912*g0 + 0.00410409178584814*g1 + 0.00726301642134786*t + 0.00750003103166819*x + 0.00685312133282423*y + 0.00765356840565801
```

Graph-conditioned terms:

```text
g0, g1
```

## Interpretation

This run validates the direction-three training interface: a graph encoder can feed closure features, the Macro PINN can train with those features, and metrics/checkpoint artifacts preserve graph-conditioning metadata.

Scientifically, the static toy graph does not improve the closure. It slightly improves global test RMSE relative to the best sparse closure, but it substantially worsens hot q90 and gradient q90. The likely reason is that `g0/g1` are global constants for all residual points. In a first-order sparse library, they behave mostly like extra intercept terms rather than spatially meaningful microstructure information.

The next direction-three step should not be another static global embedding. It should make graph-conditioned features depend on spatial region, process frame, or local neighborhood so that graph information can vary across residual samples.

## Next Action

Implement a region-aware graph-conditioning interface:

1. Generate deterministic local graph features from normalized coordinates and time, such as nearest-anchor/RBF graph embeddings.
2. Expose those features as `g0/g1/...` per residual point rather than as one global vector.
3. Re-run the same active AM-Bench split against the sparse closure best run.

This keeps the project on the direction-three path while avoiding premature dependence on hard-to-align real microstructure images.

## Artifacts

- Script: `scripts/server/run_graph_conditioned_closure_toy_a100.sh`
- Log: `logs/ambench_graph_conditioned_closure_toy_a100_v1.log`
- Run: `outputs/runs/ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1_macro_pinn_graph_conditioned_sparse_closure_h256_l4_lr1e_3_clr1e_5_staged1500_random4096_toy_static_v1/`
