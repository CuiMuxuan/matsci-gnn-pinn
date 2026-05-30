# AM-Bench Exact Line 0_1 Region-Level Real Micro Closure v1

## Context

Phase 20 moves the `mds2-2718` optical microscopy signal from a sample-level vector to local patch features selected per residual point. The aggregate feature table now preserves the 8x8 inspection grid as `region_features`, and `--closure-graph-mode real_micro_region` maps normalized residual coordinates to the nearest micrograph patch.

- Code commits: `a2a0bce`, seed-check wrapper `d59f808`
- Server: A100-SXM4-40GB, `/root/matsci-gnn-pinn`
- Thermal table: `data/interim/ambench/2022_single_track/AMB2022-03/ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1.csv`
- Split: `outputs/data_splits/ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1_split.json`
- Feature table: `data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features_line0_1_panel_v2.jsonl`
- Main log: `logs/ambench_real_micro_exact_line0_1_region_a100_v1.log`
- Focused seed-check log: `logs/ambench_real_micro_exact_line0_1_region_seedcheck_a100_v1.log`

## Commands

```bash
bash scripts/server/run_real_micro_exact_line0_1_region_a100.sh \
  > logs/ambench_real_micro_exact_line0_1_region_a100_v1.log 2>&1

bash scripts/server/run_real_micro_exact_line0_1_region_seed_check_a100.sh \
  > logs/ambench_real_micro_exact_line0_1_region_seedcheck_a100_v1.log 2>&1
```

The run uses the same conservative sparse-closure settings as Phase 18/19: `closure_lr=1e-5`, `closure_start_step=1500`, `residual_sample_size=4096`, `pde_weight=1e-6`, graph gate `0.25`, and graph L1 `1e-4`. Region feature normalization is disabled for this first route.

## Seed-0 Sweep

Primary numbers below are test split metrics.

| Micro sample | Embedding | Test RMSE | Test MAE | Test Relative L2 | Hot q90 RMSE | Gradient q90 RMSE | All-point RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `P4-L0-1` | `g8` | 74.072293 | 52.445422 | 0.060604 | 21.377514 | 62.469599 | 48.399709 |
| `P4-L0-1_m` | `g8` | 80.344687 | 61.187101 | 0.065736 | 41.005244 | 69.578652 | 60.724807 |
| `P4-L0-1` | `g4` | 89.043133 | 68.824502 | 0.072853 | 39.950484 | 72.829579 | 67.833418 |
| `P3-L0-1` | `g4` | 92.567427 | 74.545024 | 0.075736 | 118.764746 | 124.941821 | 86.192642 |
| `P3-L0-1` | `g8` | 93.496917 | 76.029773 | 0.076497 | 92.487022 | 107.764132 | 81.949443 |
| `P4-L0-1_m` | `g4` | 97.275719 | 76.849572 | 0.079589 | 92.404289 | 112.141976 | 84.405114 |
| `P3-L0-1_m` | `g8` | 107.870679 | 95.145227 | 0.088257 | 143.693292 | 145.937481 | 103.863402 |
| `P3-L0-1_m` | `g4` | 121.326545 | 99.498663 | 0.099266 | 190.859918 | 186.220071 | 118.595724 |

The best seed-0 test split result is `P4-L0-1/g8`: test RMSE `74.072293`, hot q90 RMSE `21.377514`, gradient q90 RMSE `62.469599`. Its global test RMSE does not beat the Phase 19 no-normalization best (`65.136753`), but its hot q90 is the strongest real-micro local metric so far, so it warranted a focused seed check.

## Focused Seed Check

Focused seed check target: `P4-L0-1/g8`.

| Seed | Test RMSE | Test MAE | Test Relative L2 | Hot q90 RMSE | Gradient q90 RMSE | All-point RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 74.072293 | 52.445422 | 0.060604 | 21.377514 | 62.469599 | 48.399709 |
| 1 | 71.154449 | 56.026936 | 0.058217 | 59.825116 | 78.404052 | 53.333157 |
| 2 | 108.765909 | 91.561695 | 0.088990 | 150.220178 | 153.419387 | 101.679946 |
| mean | 84.664217 | - | - | 77.140936 | 98.097679 | 67.804271 |
| std | 20.923602 | - | - | 66.143679 | 48.567944 | 29.440716 |

## Interpretation

The deterministic region-level provider gives a real weak-positive signal, especially in seed 0: local patch features reduce hot q90 RMSE to `21.377514` and gradient q90 RMSE to `62.469599`. That is better than the Phase 19 no-normalization seed-0 local candidate on hot q90 and slightly better on gradient q90.

However, the branch is not stable. Seed 2 collapses on all reported metrics, and the 3-seed mean (`84.664217` test RMSE, `77.140936` hot q90 RMSE, `98.097679` gradient q90 RMSE) is not strong enough for a model-innovation claim. This is a better diagnostic than global sample-level scalar features, but not yet a reliable performance result.

Likely remaining issues:

- The current coordinate mapping is only a deterministic convention: `coords[:, 1] -> row`, `coords[:, 0] -> col`.
- Micrograph orientation, cropping, and physical line-to-image registration are not yet calibrated.
- Nearest-patch selection is discontinuous and may amplify seed sensitivity.
- Five local scalar patch features may still be too shallow to capture microstructure state.

## Decision

Do not request an A100-SXM4-80GB server for this route. The current A100-SXM4-40GB is sufficient for deterministic region features and small mapping ablations.

Next work should either:

1. run a small coordinate-registration ablation for row/col swap, flips, and smoother patch interpolation, or
2. pivot to fixed learned patch embeddings only after the coordinate-registration question is bounded.

The result should be treated as weak-positive evidence that local microstructure conditioning is a better direction than more global hand-crafted image scalars, but not yet as a stable result.
