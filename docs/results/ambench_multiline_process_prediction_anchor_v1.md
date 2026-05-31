# AM-Bench Multi-Line Prediction Anchor v1

## Context

- Phase: 42.
- Branch: baseline-facing training objective under the `broad_process_v1` route guard.
- Motivation: validation metrics did not reliably select raw process scalars versus derived-only `am_energy_v1`. The next low-cost objective is to regularize Macro PINN predictions toward the train-target mean in normalized target space, testing whether a mild baseline-facing anchor can control global RMSE while retaining hot/gradient improvements.

## Implementation

The training CLI now supports:

```text
--prediction-anchor-weight <float>
```

Default `0.0` preserves existing behavior. When the weight is positive, training adds:

```text
prediction_anchor_weight * mean(prediction ** 2)
```

to the supervised/PDE/closure objective. With the default target normalization, `prediction == 0` corresponds to the train-split target mean or residual-target mean. Metrics and checkpoints record `prediction_anchor` metadata and per-log-step `prediction_anchor_loss`.

The generic A100 runner accepts:

```text
PREDICTION_ANCHOR_WEIGHT
```

and the Phase 42 wrapper is:

```bash
bash scripts/server/run_phase42_broad_prediction_anchor_a100.sh
```

## Commands

Focused broad12 + broad21 `laser_power` validation:

```bash
PROFILE_SPLITS=laser_power DATASET_LIMITS="12 21" DATASET_ORDER=process_round_robin \
PREDICTION_ANCHOR_WEIGHT=0.05 PREDICTION_ANCHOR_TAG=pred_anchor \
STEPS=500 N_ESTIMATORS=80 \
  bash scripts/server/run_phase42_broad_prediction_anchor_a100.sh \
  > logs/phase42_laser_power_prediction_anchor_a100_v1.log 2>&1
```

Summaries:

```bash
python scripts/server/summarize_phase30_broad_process_selector_smoke.py \
  --dataset-limit 12 \
  --dataset-order process_round_robin \
  --split laser_power \
  --include-broad-prediction-anchor \
  --json-output outputs/reports/phase42_broad12_laser_power_prediction_anchor_summary.json \
  --require-comparable

python scripts/server/summarize_phase30_broad_process_selector_smoke.py \
  --dataset-limit 21 \
  --dataset-order process_round_robin \
  --split laser_power \
  --include-broad-prediction-anchor \
  --json-output outputs/reports/phase42_broad21_laser_power_prediction_anchor_summary.json \
  --require-comparable
```

## Decision Gate

Compare `broad_prediction_anchor` against mean, kNN, ExtraTrees, no-process Macro PINN, `process_axis_v1`, and `broad_process_v1` on matched broad12/broad21 `laser_power` manifests/splits. Continue only if it improves or preserves global RMSE while improving hot q90 and gradient q90 on both broad12 and broad21. If the result is split-local or region-only, close it as a diagnostic and pivot again.
