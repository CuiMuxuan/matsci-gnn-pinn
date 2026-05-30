# AM-Bench Exact Line 0_1 Real Micro Closure v1

## Context

This experiment tests the physically closest `mds2-2718` optical microscopy samples for the active AMB2022-03 thermal table. Earlier same-process runs used `P3/P4-L0-2`; the measurement workbook indicates that exact `Line_0_1` microscopy should use `P3/P4-L0-1`.

- Run commit: `d675ed7`
- Server: A100-SXM4-40GB, `/root/matsci-gnn-pinn`
- Thermal source table: `data/interim/ambench/2022_single_track/AMB2022-03/ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1.csv`
- Split: `outputs/data_splits/ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1_split.json`
- Micro features: `data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features_line0_1_panel.jsonl`
- Build log: `logs/ambench_mds2_2718_line0_1_micro_panel_build_a100_v1.log`
- Main run log: `logs/ambench_real_micro_exact_line0_1_a100_v1.log`
- Focused seed-check log: `logs/ambench_real_micro_exact_line0_1_p4_masked_g4_seedcheck_a100_v1.log`

## Data Preparation

The exact-line panel contains four TIFF-derived feature records:

| Sample | View |
| --- | --- |
| `AMB2022-718-SH1-BP1-P3-L0-1_m` | masked |
| `AMB2022-718-SH1-BP1-P3-L0-1` | unmasked |
| `AMB2022-718-SH1-BP1-P4-L0-1_m` | masked |
| `AMB2022-718-SH1-BP1-P4-L0-1` | unmasked |

Server build command:

```bash
bash scripts/server/build_mds2_2718_line0_1_micro_panel_a100.sh \
  > logs/ambench_mds2_2718_line0_1_micro_panel_build_a100_v1.log 2>&1
```

The build passed and the feature table manifest reports `n_records=4`.

## Training Commands

Seed-0 exact-line sweep:

```bash
bash scripts/server/run_real_micro_exact_line0_1_a100.sh \
  > logs/ambench_real_micro_exact_line0_1_a100_v1.log 2>&1
```

Focused seed check for the best seed-0 candidate:

```bash
bash scripts/server/run_real_micro_exact_line0_1_seed_check_a100.sh \
  > logs/ambench_real_micro_exact_line0_1_p4_masked_g4_seedcheck_a100_v1.log 2>&1
```

Both scripts reuse the current conservative real-micro sparse closure settings:

- `hidden_dim=256`, `layers=4`, `lr=1e-3`
- `closure_lr=1e-5`
- `closure_start_step=1500`
- `residual_sample_size=4096`
- `pde_weight=1e-6`
- `closure_graph_gate=0.25`
- `closure_graph_l1_weight=1e-4`

## Seed-0 Sweep

Primary numbers below are test split metrics.

| Micro sample | Embedding | Test RMSE | Test MAE | Test Relative L2 | Hot q90 RMSE | Gradient q90 RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `P4-L0-1_m` | `g4` | 67.891122 | 49.264025 | 0.055547 | 42.174310 | 68.729519 |
| `P4-L0-1_m` | `g8` | 76.681939 | 62.396528 | 0.062739 | 75.046355 | 89.454802 |
| `P3-L0-1` | `g4` | 85.156947 | 72.819639 | 0.069673 | 96.724997 | 105.846416 |
| `P4-L0-1` | `g4` | 112.544885 | 92.946727 | 0.092081 | 179.484801 | 174.613298 |
| `P3-L0-1_m` | `g4` | 114.791823 | 96.460592 | 0.093920 | 194.641643 | 188.388302 |
| `P3-L0-1_m` | `g8` | 123.559360 | 98.290870 | 0.101093 | 216.896634 | 208.131633 |
| `P3-L0-1` | `g8` | 126.649365 | 108.291681 | 0.103621 | 198.872222 | 193.440091 |
| `P4-L0-1` | `g8` | 157.409398 | 114.090414 | 0.128788 | 304.870078 | 288.825873 |

The best seed-0 candidate is `P4-L0-1_m/g4`. It improves global test RMSE over the sparse closure best (`70.494433`) but does not improve the local-region metrics (`hot q90=31.542155`, `gradient q90=64.558069` for the sparse closure best).

## Focused Seed Check

Focused seed check for `P4-L0-1_m/g4`:

| Seed | Test RMSE | Test MAE | Test Relative L2 | Hot q90 RMSE | Gradient q90 RMSE | All-point RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 67.891122 | 49.264025 | 0.055547 | 42.174310 | 68.729519 | 47.366621 |
| 1 | 88.886707 | 71.522495 | 0.072725 | 70.279494 | 90.460804 | 74.422964 |
| 2 | 75.288781 | 60.130537 | 0.061599 | 67.100988 | 82.855641 | 62.275348 |
| mean | 77.355537 | - | - | 59.851597 | 80.681988 | 61.354978 |
| std | 10.649284 | - | - | 15.391250 | 11.027500 | 13.551632 |

## Interpretation

Exact-line alignment is better than the prior same-process `L0-2` check, but it is still not stable enough for a model-innovation claim. The best seed-0 run is encouraging, yet the 3-seed mean falls behind the sparse closure best on test, hot-zone, and gradient-zone metrics.

This result narrows the diagnosis:

- Nonphysical frame-cycle assignment was a clear negative control.
- Same-process but wrong line replicate (`L0-2`) was unstable.
- Exact-line `P4-L0-1_m` is the best real-micro candidate so far, but the current sample-level hand-crafted feature vector is too weak or too noisy.

## Next Action

Do not scale this exact-line setting as a headline claim. The next useful branch should improve the microstructure representation or alignment granularity before another large sweep:

1. Add richer image-derived features, such as melt-pool geometry, mask morphology, texture descriptors, or learned image embeddings.
2. Compare masked vs unmasked feature construction directly for `P4-L0-1`.
3. If `mds2-2718` remains sample-level only, consider `mds2-2775` or ExaCA/PFHub for more controlled microstructure-to-thermal conditioning.
