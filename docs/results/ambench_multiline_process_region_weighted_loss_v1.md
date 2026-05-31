# AM-Bench Multi-Line Process Region-Weighted Data Loss v1

## Context

- Phase: 35.
- Branch: train-split region-weighted supervised data loss under `broad_process_v1`.
- Motivation: Phase 34 learned residual correction produced only a negligible global RMSE change while worsening hot-zone and gradient-band errors.
- Baseline route: broad12 `spot_size`, `broad_process_v1`.
- Key files:
  - `src/gnnpinn/train/macro_pinn.py`
  - `scripts/server/run_multiline_process_conditioned_thermal_a100.sh`
  - `scripts/server/run_phase35_broad_region_weighted_loss_a100.sh`
  - `scripts/server/summarize_phase30_broad_process_selector_smoke.py`

## Baseline

The comparison target remains the strongest current broad12 `spot_size` route:

| Method | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE |
|---|---:|---:|---:|
| `broad_process_v1` | 136.309183 | 165.228535 | 169.049295 |

## Implementation

The training CLI now supports optional train-split-only data-loss weighting:

```text
--data-loss-weighting none|hot|gradient|hot_gradient
--data-loss-hot-quantile
--data-loss-gradient-quantile
--data-loss-region-weight
```

The default is `none`, preserving existing behavior. When enabled, target/hot and gradient thresholds are fit only on the optimization split. The weighted supervised loss is normalized by the sum of weights:

```text
data_loss = sum(weight_i * squared_error_i) / sum(weight_i)
```

Metrics and checkpoints record `data_loss_weighting` metadata, including selected train points, quantiles, weight, weight sum, and selector thresholds.

## Focused A100 Probes

All focused probes use broad12 `spot_size`, `DATASET_LIMIT=12`, `DATASET_ORDER=process_round_robin`, `STEPS=500`, `N_ESTIMATORS=80`, and `hot_gradient` train-split-only weighting. Each summary passed the existing manifest/split comparability gate.

```bash
PROFILE_SPLITS=spot_size DATASET_LIMIT=12 DATASET_ORDER=process_round_robin \
STEPS=500 N_ESTIMATORS=80 \
  bash scripts/server/run_phase35_broad_region_weighted_loss_a100.sh \
  > logs/phase35_broad12_spot_size_region_hotgrad_w2_a100_v1.log 2>&1
```

Summary:

```bash
python scripts/server/summarize_phase30_broad_process_selector_smoke.py \
  --dataset-limit 12 \
  --dataset-order process_round_robin \
  --split spot_size \
  --include-broad-region-weighted \
  --json-output outputs/reports/phase35_broad12_spot_size_region_hotgrad_w2_summary.json \
  --require-comparable
```

Single-seed focused results:

| Method | Region weight | Test RMSE | Delta vs base | Hot q90 RMSE | Delta vs base | Gradient q90 RMSE | Delta vs base |
|---|---:|---:|---:|---:|---:|---:|---:|
| `broad_process_v1` |  | 136.309183 |  | 165.228535 |  | 169.049295 |  |
| `rw125` | 1.25 | 139.773470 | +3.464287 | 160.431178 | -4.797357 | 168.187786 | -0.861509 |
| `rw135` | 1.35 | 140.881887 | +4.572704 | 199.804301 | +34.575766 | 192.678431 | +23.629136 |
| `rw15` | 1.50 | 143.462665 | +7.153482 | 128.090811 | -37.137724 | 150.283022 | -18.766273 |
| `rw2` | 2.00 | 153.645414 | +17.336232 | 132.990923 | -32.237612 | 154.642037 | -14.407259 |

Artifacts:

| Tag | Summary artifact | Log |
|---|---|---|
| `rw2` | `outputs/reports/phase35_broad12_spot_size_region_hotgrad_w2_summary.json` | `logs/phase35_broad12_spot_size_region_hotgrad_w2_a100_v1.log` |
| `rw15` | `outputs/reports/phase35_broad12_spot_size_region_hotgrad_w15_summary.json` | `logs/phase35_broad12_spot_size_region_hotgrad_w15_a100_v1.log` |
| `rw125` | `outputs/reports/phase35_broad12_spot_size_region_hotgrad_w125_summary.json` | `logs/phase35_broad12_spot_size_region_hotgrad_w125_a100_v1.log` |
| `rw135` | `outputs/reports/phase35_broad12_spot_size_region_hotgrad_w135_summary.json` | `logs/phase35_broad12_spot_size_region_hotgrad_w135_a100_v1.log` |

Interpretation:

- `rw15` is the most informative candidate: it gives the largest hot-zone improvement and a meaningful gradient-band improvement, at the cost of a moderate global RMSE increase.
- `rw125` is the conservative candidate: it keeps global RMSE closer to baseline and gives only small region improvements.
- `rw135` is a negative diagnostic because both hot and gradient regions worsen.
- `rw2` confirms that stronger weighting can help regions but over-trades against global RMSE.

## Seed Check

The paired model-seed check used the same broad12 `spot_size` table/split as `broad_process_v1`. It compares unweighted `broad_process_v1` against the two candidates worth testing further, `rw15` and `rw125`.

```bash
SEEDS="1 2" REGION_WEIGHTED_TAGS="rw15 rw125" STEPS=500 \
  bash scripts/server/run_phase35_region_weighted_seed_check_a100.sh \
  > logs/phase35_broad12_spot_size_region_weighted_seed_check_a100_v1.log 2>&1
```

After completion:

```bash
python scripts/server/summarize_phase35_region_weighted_seed_check.py \
  --json-output outputs/reports/phase35_broad12_spot_size_region_weighted_seed_check_summary.json \
  --require-complete
```

Seed-check results:

| Method | Seed | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE |
|---|---:|---:|---:|---:|
| `broad_process_v1` | 7 | 136.309183 | 165.228535 | 169.049295 |
| `broad_process_v1` | 1 | 135.853737 | 155.194459 | 157.829090 |
| `broad_process_v1` | 2 | 136.991427 | 165.953018 | 168.968161 |
| `rw15` | 7 | 143.462665 | 128.090811 | 150.283022 |
| `rw15` | 1 | 140.137121 | 123.422556 | 141.892456 |
| `rw15` | 2 | 144.590582 | 238.047082 | 219.506452 |
| `rw125` | 7 | 139.773470 | 160.431178 | 168.187786 |
| `rw125` | 1 | 135.948038 | 142.894250 | 151.472441 |
| `rw125` | 2 | 146.194012 | 230.327132 | 213.789004 |

Three-seed aggregate:

| Method | Test RMSE mean +/- std | Hot q90 mean +/- std | Gradient q90 mean +/- std |
|---|---:|---:|---:|
| `broad_process_v1` | 136.384782 +/- 0.467526 | 162.125337 +/- 4.909788 | 165.282182 +/- 5.270236 |
| `rw15` | 142.730123 +/- 1.890466 | 163.186816 +/- 52.968498 | 170.560643 +/- 34.779012 |
| `rw125` | 140.638507 +/- 4.227388 | 177.884187 +/- 37.767560 | 177.816410 +/- 26.335924 |

The single-seed region gains did not survive the paired seed check. Both weighted variants increase global RMSE on average; `rw15` loses its hot/gradient advantage because seed 2 degrades sharply, and `rw125` is worse than the unweighted baseline on both region aggregates.

## Decision

Close Phase 35 as a negative diagnostic. Keep train-split data-loss weighting as a reproducible analysis option, but do not expand it to other splits or larger datasets. The next branch should move away from scalar loss reweighting and toward a more structured representation of process/microstructure context.

This branch is small and should fit the current A100-SXM4-40GB server. Do not request A100-SXM4-80GB unless later scaling demonstrably exceeds the current GPU.
