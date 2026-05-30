# AM-Bench Exact Line 0_1 Real Micro Feature v2 Closure v1

## Context

Phase 18 showed that exact `Line_0_1` microstructure alignment is more credible than frame cycling or same-process `L0-2`, but the v1 sample-level feature vector was not stable enough for a model claim. This experiment tests a richer hand-crafted micrograph feature schema before moving to learned image embeddings or larger microstructure datasets.

- Code commit: `3eb5f9c`
- Server: A100-SXM4-40GB, `/root/matsci-gnn-pinn`
- Thermal table: `data/interim/ambench/2022_single_track/AMB2022-03/ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1.csv`
- Split: `outputs/data_splits/ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1_split.json`
- Feature table: `data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features_line0_1_panel_v2.jsonl`
- Build log: `logs/ambench_mds2_2718_line0_1_micro_panel_feature_v2_build_a100_v1.log`
- Run log: `logs/ambench_real_micro_exact_line0_1_feature_v2_a100_v1.log`

## Feature Schema

The v2 inspection path keeps the coarse grid graph summary and adds:

- intensity distribution: quantiles, normalized IQR/P90-P10 range, entropy
- threshold-mask geometry: centroid, span, bounding-box area, fill fraction, perimeter fraction, half-plane fractions, anisotropy
- texture descriptors: normalized gradient magnitude, laplacian magnitude, mask-boundary gradient

The server build completed with `n_records=4` and `56` feature columns. The closure metadata confirms that `g0..g7` selected:

```text
image_mask_fraction
mask_centroid_row_norm
mask_centroid_col_norm
mask_bbox_area_fraction
mask_span_row_norm
mask_span_col_norm
mask_perimeter_fraction
gradient_magnitude_q90_norm
```

## Commands

```bash
bash scripts/server/build_mds2_2718_line0_1_micro_panel_feature_v2_a100.sh \
  > logs/ambench_mds2_2718_line0_1_micro_panel_feature_v2_build_a100_v1.log 2>&1

bash scripts/server/run_real_micro_exact_line0_1_feature_v2_a100.sh \
  > logs/ambench_real_micro_exact_line0_1_feature_v2_a100_v1.log 2>&1
```

The sweep reuses the conservative real-micro sparse-closure settings from Phase 18: `closure_lr=1e-5`, `closure_start_step=1500`, `residual_sample_size=4096`, `pde_weight=1e-6`, graph gate `0.25`, and graph L1 `1e-4`.

## Results

Primary numbers below are test split metrics.

| Micro sample | Embedding | Test RMSE | Test MAE | Test Relative L2 | Hot q90 RMSE | Gradient q90 RMSE | All-point RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `P4-L0-1` | `g8` | 71.676666 | 56.861267 | 0.058644 | 79.314583 | 90.549052 | 59.995252 |
| `P4-L0-1_m` | `g8` | 74.720795 | 61.097196 | 0.061135 | 78.144498 | 89.445586 | 62.982481 |
| `P4-L0-1_m` | `g4` | 84.943401 | 67.488454 | 0.069499 | 149.449791 | 147.396941 | 81.990784 |
| `P3-L0-1_m` | `g8` | 107.489089 | 90.592221 | 0.087945 | 156.200424 | 157.355465 | 104.448043 |
| `P3-L0-1` | `g4` | 115.283853 | 100.467470 | 0.094322 | 165.705627 | 165.751027 | 109.879911 |
| `P4-L0-1` | `g4` | 116.268983 | 96.941639 | 0.095128 | 66.850586 | 96.677347 | 100.050711 |
| `P3-L0-1` | `g8` | 129.397894 | 104.944204 | 0.105870 | 212.996759 | 205.263415 | 135.045238 |
| `P3-L0-1_m` | `g4` | 129.749261 | 110.929329 | 0.106158 | 204.921083 | 198.963974 | 135.102870 |

## Interpretation

The richer v2 hand-crafted feature schema does not improve the exact-line branch. Its best seed-0 result, `P4-L0-1/g8`, is close to the sparse closure global test RMSE but has much worse hot and gradient q90 metrics. It also does not beat the Phase 18 best seed-0 candidate `P4-L0-1_m/g4`.

This is useful negative evidence: simply adding more global image statistics is not enough. A follow-up no-normalization diagnostic was run to test whether the per-sample vector min-max normalization was washing out absolute geometry and texture magnitudes.

## Next Action

The no-normalization ablation is documented in [ambench_real_micro_exact_line0_1_feature_v2_no_norm_closure_v1.md](ambench_real_micro_exact_line0_1_feature_v2_no_norm_closure_v1.md). It recovered one good seed-0 diagnostic but failed the focused 3-seed check, so the next branch should move to region-level or learned microstructure representations rather than adding more global scalar descriptors.
