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

## First A100 Probe

Default focused probe:

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

If weight `2.0` improves hot q90 or gradient q90 without unacceptable global degradation, run a stronger focused variant:

```bash
PROFILE_SPLITS=spot_size DATASET_LIMIT=12 DATASET_ORDER=process_round_robin \
STEPS=500 N_ESTIMATORS=80 REGION_WEIGHTED_RUN_TAG=rw4 \
DATA_LOSS_REGION_WEIGHT=4.0 \
  bash scripts/server/run_phase35_broad_region_weighted_loss_a100.sh \
  > logs/phase35_broad12_spot_size_region_hotgrad_w4_a100_v1.log 2>&1
```

Then summarize with:

```bash
python scripts/server/summarize_phase30_broad_process_selector_smoke.py \
  --dataset-limit 12 \
  --dataset-order process_round_robin \
  --split spot_size \
  --include-broad-region-weighted \
  --broad-region-weighted-tag rw4 \
  --json-output outputs/reports/phase35_broad12_spot_size_region_hotgrad_w4_summary.json \
  --require-comparable
```

## Decision Gate

Continue this branch only if at least one focused weighted-loss run improves hot q90 and/or gradient q90 while preserving global RMSE close to `broad_process_v1`. If it merely trades severe global degradation for small region gains, close the branch and move to a more structured model/data representation change.

This branch is small and should fit the current A100-SXM4-40GB server. Do not request A100-SXM4-80GB unless later scaling demonstrably exceeds the current GPU.
