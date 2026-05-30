# AM-Bench Exact Line 0_1 Real Micro Feature v2 No-Normalization Closure v1

## Context

Phase 19 first tested richer hand-crafted `mds2-2718` micrograph features with the default per-feature min-max normalization inside the `real_micro` provider. That run did not improve the exact-line branch. This ablation disables the `real_micro` feature-vector normalization to check whether absolute geometry or texture magnitudes were being washed out.

- Code commit: `4c01d3a`
- Server: A100-SXM4-40GB, `/root/matsci-gnn-pinn`
- Thermal table: `data/interim/ambench/2022_single_track/AMB2022-03/ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1.csv`
- Split: `outputs/data_splits/ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1_split.json`
- Feature table: `data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features_line0_1_panel_v2.jsonl`
- Main run log: `logs/ambench_real_micro_exact_line0_1_feature_v2_no_norm_a100_v1.log`
- Focused seed-check log: `logs/ambench_real_micro_exact_line0_1_feature_v2_no_norm_p3_masked_g4_seedcheck_a100_v1.log`

## Commands

```bash
bash scripts/server/run_real_micro_exact_line0_1_feature_v2_no_norm_a100.sh \
  > logs/ambench_real_micro_exact_line0_1_feature_v2_no_norm_a100_v1.log 2>&1

bash scripts/server/run_real_micro_exact_line0_1_feature_v2_no_norm_seed_check_a100.sh \
  > logs/ambench_real_micro_exact_line0_1_feature_v2_no_norm_p3_masked_g4_seedcheck_a100_v1.log 2>&1
```

The run reuses the Phase 18/19 conservative real-micro sparse-closure settings: `closure_lr=1e-5`, `closure_start_step=1500`, `residual_sample_size=4096`, `pde_weight=1e-6`, graph gate `0.25`, and graph L1 `1e-4`.

## Seed-0 Sweep

Primary numbers below are test split metrics.

| Micro sample | Embedding | Test RMSE | Test MAE | Test Relative L2 | Hot q90 RMSE | Gradient q90 RMSE | All-point RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `P3-L0-1_m` | `g8` | 65.136753 | 49.376165 | 0.053293 | 58.455151 | 75.643381 | 48.823001 |
| `P3-L0-1_m` | `g4` | 67.843165 | 50.496229 | 0.055508 | 40.724303 | 66.187221 | 50.033655 |
| `P4-L0-1` | `g4` | 68.836281 | 53.969796 | 0.056320 | 64.658443 | 78.533089 | 57.046411 |
| `P4-L0-1_m` | `g4` | 80.900726 | 66.602846 | 0.066191 | 96.975709 | 106.951168 | 70.493774 |
| `P3-L0-1` | `g4` | 86.397181 | 72.219210 | 0.070688 | 135.289319 | 135.929342 | 80.755367 |
| `P4-L0-1_m` | `g8` | 92.177614 | 80.402863 | 0.075417 | 122.473526 | 128.898752 | 81.844654 |
| `P4-L0-1` | `g8` | 102.406957 | 82.837321 | 0.083787 | 169.934755 | 165.787871 | 100.331998 |
| `P3-L0-1` | `g8` | 130.228852 | 110.801070 | 0.106550 | 208.492610 | 202.134655 | 137.715617 |

The best seed-0 global test RMSE is `P3-L0-1_m/g8` at `65.136753`. The best seed-0 local-region candidate is `P3-L0-1_m/g4`, with hot q90 RMSE `40.724303` and gradient q90 RMSE `66.187221`.

## Focused Seed Check

The focused seed check was run for `P3-L0-1_m/g4` because it had the strongest hot/gradient seed-0 behavior.

| Seed | Test RMSE | Test MAE | Test Relative L2 | Hot q90 RMSE | Gradient q90 RMSE | All-point RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 67.843165 | 50.496229 | 0.055508 | 40.724303 | 66.187221 | 50.033655 |
| 1 | 68.898278 | 50.953496 | 0.056371 | 44.502756 | 67.149202 | 50.573151 |
| 2 | 126.476105 | 107.644663 | 0.103480 | 196.949166 | 192.117945 | 128.012086 |
| mean | 87.739183 | - | - | 94.058742 | 108.484789 | 76.206297 |
| std | 33.551307 | - | - | 89.125747 | 72.430034 | 44.865940 |

## Interpretation

No-normalization gives one useful seed-0 diagnostic but not a stable branch. The `P3-L0-1_m/g4` run nearly matches the Phase 18 seed-0 local metrics and slightly improves the sparse-closure gradient q90, but the focused seed check fails: seed 2 collapses to high global, hot-zone, and gradient-zone error.

The current evidence is now consistent across three hand-crafted sample-level attempts:

- v1 exact-line features produced a promising but unstable `P4-L0-1_m/g4` seed-0 run.
- v2 richer global image descriptors did not improve the branch.
- v2 no-normalization recovered one good seed-0 run but did not survive a 3-seed check.

Therefore the project should stop expanding global hand-crafted scalar micrograph descriptors for `mds2-2718` as a performance branch. The next aligned model-development step is to move the microstructure signal closer to the residual point: patch-level, region-level, or learned image embeddings, with the current scalar `real_micro` runs preserved as controlled negative or weak-positive evidence.

## Next Action

Start Phase 20 with a local implementation and A100 smoke path for region-level or learned microstructure representation. The first low-risk route is a deterministic patch-grid micro feature provider:

```text
thermal point (x, y, t) -> normalized spatial bin -> patch/region micro feature vector -> graph-conditioned sparse closure
```

This keeps training small enough for the current A100-SXM4-40GB server. Only request an A100-SXM4-80GB server if the branch moves to dense learned image encoders, larger image backbones, or multi-process/multi-line training that does not fit or finish on the current 40GB card.
