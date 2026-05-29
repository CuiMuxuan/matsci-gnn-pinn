# AM-Bench Graph Closure L1 Sensitivity v1

## Context

- Server: Ubuntu 22, NVIDIA A100-SXM4-40GB
- Code commit: `d4e226f`
- Dataset: active hot/gradient AM-Bench table
- Model: Macro PINN with gated coordinate RBF graph-conditioned sparse closure
- Graph mode: `coordinate_rbf`
- Graph features: `embedding_dim=6`, `length_scale=0.25`
- Graph gate: `0.25`
- PDE field: normalized model output
- PDE weight: `1e-6`
- Closure LR: `1e-5`
- Closure start step: `1500`
- Residual sampling: random train subset, `4096` points/step

## Command

```bash
bash scripts/server/run_graph_conditioned_closure_graph_l1_sensitivity_a100.sh \
  > logs/ambench_graph_conditioned_closure_graph_l1_sensitivity_a100_v1.log 2>&1
```

## Results

| Graph L1 | Test RMSE | Test MAE | Hot q90 RMSE | Gradient q90 RMSE | Final PDE loss | Graph terms survive threshold? |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `1e-4` | 72.840576 | 52.910100 | 31.919197 | 61.623894 | 5723.561523 | no |
| `1e-5` | 83.349281 | 70.001036 | 96.364302 | 106.171976 | 21580.593750 | yes |
| `1e-6` | 89.998395 | 74.430809 | 126.042897 | 129.963512 | 53066.171875 | yes |

## Comparison

| Method | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE |
| --- | ---: | ---: | ---: |
| active data-only h256/l4/lr1e-3, 3 seed mean | 60.143582 | 17.033393 | 63.622568 |
| sparse closure best, `closure_lr=1e-5`, trainable backbone | 70.494433 | 31.542155 | 64.558069 |
| gated graph closure, graph L1 `1e-4` | 72.840576 | 31.919197 | 61.623894 |
| gated graph closure, graph L1 `1e-5` | 83.349281 | 96.364302 | 106.171976 |
| gated graph closure, graph L1 `1e-6` | 89.998395 | 126.042897 | 129.963512 |

## Interpretation

The sensitivity scan resolves the synthetic graph-conditioning question for this stage. When graph coefficients are strongly penalized (`1e-4`), local-region metrics remain stable, but graph terms vanish from the symbolic expression. When the penalty is weakened enough for graph terms to survive thresholding, hot q90 and gradient q90 degrade sharply.

This means the current synthetic coordinate/RBF graph features do not provide a positive direction-three result. They are useful as an interface and negative-control branch, but not as the main scientific claim.

## Decision

Stop expanding synthetic graph-conditioned closure at this stage. Preserve it as:

1. a tested direction-three interface;
2. a negative-control experiment showing that arbitrary coordinate-derived graph features can harm melt-pool regions;
3. motivation for real or semi-real microstructure conditioning.

The next project step should move toward D2: real/semireal microstructure alignment, ExaCA-generated microstructure features, or a manuscript-facing negative-result section for synthetic graph conditioning.

## Artifacts

- Script: `scripts/server/run_graph_conditioned_closure_graph_l1_sensitivity_a100.sh`
- Log: `logs/ambench_graph_conditioned_closure_graph_l1_sensitivity_a100_v1.log`
- Runs: `outputs/runs/*gate0_25_gl1e_[56]_v1/`
