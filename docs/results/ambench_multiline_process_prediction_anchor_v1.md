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

## Results

Focused broad12/broad21 `laser_power` validation:

| Dataset | Weight | Method | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Signal |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| broad12 | n/a | `broad_process_v1` | 140.753534 | 254.473291 | 215.411533 | route guard |
| broad12 | 0.05 | `broad_prediction_anchor` | 138.096020 | 241.730006 | 208.947734 | positive vs route guard |
| broad12 | 0.01 | `broad_prediction_anchor` | 137.328485 | 245.484120 | 210.119530 | positive vs route guard |
| broad12 | n/a | `mean` | 132.965887 | 242.427068 | 208.105836 | strongest global baseline |
| broad21 | n/a | `broad_process_v1` | 178.040331 | 296.909567 | 254.954359 | route guard |
| broad21 | 0.05 | `broad_prediction_anchor` | 192.755869 | 320.927695 | 275.299993 | negative |
| broad21 | 0.01 | `broad_prediction_anchor` | 200.097570 | 292.510967 | 267.746425 | global/gradient negative |
| broad21 | n/a | `mean` | 131.741364 | 237.730958 | 205.133029 | strongest global baseline |

The anchor behaves like useful shrinkage on broad12 but does not transfer to broad21. The smaller `0.01` weight slightly improves broad21 hot q90 against `broad_process_v1`, but worsens global RMSE and gradient q90. Lower weights would mainly converge back to the route guard and are unlikely to create a meaningful all-metric improvement.

## Decision

Close prediction anchoring as a split-local diagnostic. It is reproducible and remains available through `--prediction-anchor-weight`, but it should not be seed-expanded or used as a paper-facing model claim. Phase 42 confirms that neither simple raw/derived validation selection nor scalar prediction shrinkage solves the broad12/broad21 transfer split.

The next branch should pivot away from scalar output shrinkage and toward a stronger process representation or architecture that can explicitly handle broad21 `laser_power` without sacrificing broad12.
