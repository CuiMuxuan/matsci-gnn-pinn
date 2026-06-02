# AM-Bench Multiline Process Group-Balanced Objective v1

## Status

Closed as a negative objective diagnostic. Phase 44 started after `process_encoder_v1` improved broad21 `laser_power` but failed broad12. This branch kept `broad_process_v1` as the route guard and changed the supervised objective before adding another process encoder.

## Hypothesis

The broad12/broad21 transfer split may be partly caused by uneven process-condition contribution inside the training split. A train-split inverse-frequency data-loss balance over full process conditions may reduce overfitting to dominant process groups while preserving the known `broad_process_v1` route decisions.

## Implementation

- `--data-loss-group-balance-column <column>` enables group-balanced data-loss weights.
- `process_condition` is a synthetic group key from `laser_power_W`, `scan_speed_mm_s`, and `spot_size_um`.
- `--data-loss-group-balance-strength` blends uniform weights with full inverse-frequency group balance.
- Region weighting remains separate; the actual product weights are recorded in `data_loss_objective`.
- Server runner: `scripts/server/run_phase44_broad_group_balance_a100.sh`.

## Focused Validation

```bash
PROFILE_SPLITS=laser_power DATASET_LIMITS="12 21" DATASET_ORDER=process_round_robin \
PROCESS_FEATURE_TAG=group_bal PROCESS_DERIVED_FEATURE_MODE=am_energy_v1 \
DATA_LOSS_GROUP_BALANCE_COLUMN=process_condition DATA_LOSS_GROUP_BALANCE_STRENGTH=1.0 \
STEPS=500 N_ESTIMATORS=80 \
  bash scripts/server/run_phase44_broad_group_balance_a100.sh \
  > logs/phase44_laser_power_group_balance_a100_v1.log 2>&1
```

Summarize with:

```bash
python scripts/server/summarize_phase30_broad_process_selector_smoke.py \
  --split laser_power \
  --dataset-limit 12 \
  --dataset-order process_round_robin \
  --include-broad-process-group-balance

python scripts/server/summarize_phase30_broad_process_selector_smoke.py \
  --split laser_power \
  --dataset-limit 21 \
  --dataset-order process_round_robin \
  --include-broad-process-group-balance
```

Server artifacts:

- Log: `logs/phase44_broad12_broad21_laser_power_group_balance_a100_v1.log`
- broad12 summary: `outputs/reports/phase44_broad12_laser_power_group_balance_summary.json`
- broad21 summary: `outputs/reports/phase44_broad21_laser_power_group_balance_summary.json`

Both summaries passed `--require-comparable`.

## Results

Metric order is test RMSE / hot q90 RMSE / gradient q90 RMSE.

| Split | `broad_process_v1` | `broad_process_group_balance` | Decision |
|-------|--------------------|-------------------------------|----------|
| broad12 `laser_power` | `140.753534 / 254.473291 / 215.411533` | `189.364413 / 356.845339 / 289.133792` | Regresses all metrics |
| broad21 `laser_power` | `178.040331 / 296.909567 / 254.954359` | `212.704856 / 221.878476 / 238.794848` | Improves regions but regresses global RMSE |

The group-balanced run used `process_condition` with strength `1.0`, `am_energy_v1` derived inputs, and the existing `broad_process_v1` route guard. The broad12 train split had only one `process_condition` group for this holdout, so the group-balance component could not reshape broad12 train supervision; broad21 had one held-out train process group as well in the summarized artifact. This makes the branch a useful metadata/objective plumbing check, but not a credible transfer fix.

## Decision Gate

Continue only if `broad_process_group_balance` improves or preserves global RMSE while improving hot q90 and gradient q90 on both broad12 and broad21 `laser_power`. If the result is only broad12-local or broad21-local, close it as an objective diagnostic before any seed expansion.

The gate fails. Do not run seed expansion. Keep `broad_process_v1` as the broad-data route guard.
