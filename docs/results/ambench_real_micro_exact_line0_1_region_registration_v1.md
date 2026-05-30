# AM-Bench Exact Line 0_1 Region Registration Ablation v1

## Context

Phase 21 tests whether the weak-positive Phase 20 region-level real-micro signal was limited by the assumed thermal-to-micrograph coordinate mapping. The provider now supports row/column source selection, row/column flips, and nearest-patch versus inverse-distance patch selection for `real_micro_region`.

- Code commit: `11c49d1`
- Seed-check script commit: `77a4bd9`
- Server: A100-SXM4-40GB, `/root/matsci-gnn-pinn`
- Candidate: `AMB2022-718-SH1-BP1-P4-L0-1`, `g8`
- Main log: `logs/ambench_real_micro_exact_line0_1_region_registration_a100_v1.log`
- Focused seed-check log: `logs/ambench_real_micro_exact_line0_1_region_registration_seedcheck_a100_v1.log`

## Commands

```bash
bash scripts/server/run_real_micro_exact_line0_1_region_registration_a100.sh \
  > logs/ambench_real_micro_exact_line0_1_region_registration_a100_v1.log 2>&1

bash scripts/server/run_real_micro_exact_line0_1_region_registration_seed_check_a100.sh \
  > logs/ambench_real_micro_exact_line0_1_region_registration_seedcheck_a100_v1.log 2>&1
```

The matrix keeps the Phase 20 sparse-closure settings fixed and only changes `real_micro_region` coordinate registration. The baseline Phase 20 mapping is `row=y`, `col=x`, no flips, nearest patch.

## Seed-0 Registration Matrix

Primary numbers below are test split metrics.

| Variant | Mapping | Selection | Test RMSE | Test MAE | Test Relative L2 | Hot q90 RMSE | Gradient q90 RMSE | All-point RMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `col_flip` | `row=y`, `col=1-x` | nearest | 66.288087 | 49.428813 | 0.054235 | 49.074617 | 72.040545 | 48.696416 |
| `rowcol_swap` | `row=x`, `col=y` | nearest | 73.446353 | 55.406608 | 0.060092 | 41.113203 | 65.508254 | 57.902442 |
| `row_flip` | `row=1-y`, `col=x` | nearest | 87.814185 | 73.866673 | 0.071847 | 95.190762 | 106.661656 | 78.389601 |
| `inverse_distance` | `row=y`, `col=x` | inverse-distance | 112.249266 | 95.904606 | 0.091839 | 148.561955 | 152.118443 | 110.342015 |
| `row_col_flip` | `row=1-y`, `col=1-x` | nearest | 112.710742 | 94.308093 | 0.092217 | 160.435534 | 157.185039 | 120.940640 |

Compared with the Phase 20 seed-0 default mapping (`P4-L0-1/g8`: test RMSE `74.072293`, hot q90 `21.377514`, gradient q90 `62.469599`), `col_flip` improves global test RMSE but weakens the seed-0 hot-zone metric. `rowcol_swap` is closest to preserving the Phase 20 local behavior but does not improve global test RMSE enough to settle the mapping issue.

## Focused Seed Check

Focused seed check target: `col_flip`, `P4-L0-1/g8`.

| Seed | Test RMSE | Test MAE | Test Relative L2 | Hot q90 RMSE | Gradient q90 RMSE | All-point RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 66.288087 | 49.428813 | 0.054235 | 49.074617 | 72.040545 | 48.696416 |
| 1 | 70.492631 | 53.230197 | 0.057675 | 43.685898 | 66.825943 | 52.802060 |
| 2 | 102.955156 | 87.482216 | 0.084235 | 152.086208 | 152.859314 | 94.457885 |
| mean | 79.911958 | - | - | 81.615574 | 97.241934 | 65.318787 |
| std | 20.066421 | - | - | 61.088806 | 48.236581 | 25.318558 |

## Interpretation

The coordinate registration ablation is useful but does not solve stability. Column flipping improves the seed-0 global test RMSE from `74.072293` to `66.288087`, close to the Phase 19 no-normalization global candidate (`65.136753`). It also gives seed 1 a similar global test RMSE (`70.492631`).

The branch still fails the 3-seed robustness check because seed 2 collapses across global, hot-zone, and gradient-zone metrics. In addition, the original mapping had the strongest seed-0 hot q90 (`21.377514`), while `col_flip` trades that local hot-zone behavior for better global test error.

Current evidence:

- More global scalar descriptors are still the wrong direction.
- Local region features are a better diagnostic than sample-level vectors.
- The simple deterministic coordinate registration and nearest-patch provider are not yet stable enough for a model-innovation claim.
- Inverse-distance smoothing over all patches is not helpful in this form.

## Decision

Phase 21 should close as a bounded negative or weak-positive diagnostic. Do not request A100-SXM4-80GB for this path.

Recommended next branch:

1. Keep the current A100-SXM4-40GB for small experiments.
2. Stop expanding hand-crafted scalar patch features.
3. Move to fixed learned patch embeddings or a stronger physically registered microstructure source.
4. If learned embeddings are attempted, start frozen and low-dimensional rather than training a dense image encoder end to end.
