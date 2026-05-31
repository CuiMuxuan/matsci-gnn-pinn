# AM-Bench Multiline Process Group-Balanced Objective v1

## Status

In progress. Phase 44 starts after `process_encoder_v1` improved broad21 `laser_power` but failed broad12. The next branch keeps `broad_process_v1` as the route guard and changes the supervised objective before adding another process encoder.

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

## Decision Gate

Continue only if `broad_process_group_balance` improves or preserves global RMSE while improving hot q90 and gradient q90 on both broad12 and broad21 `laser_power`. If the result is only broad12-local or broad21-local, close it as an objective diagnostic before any seed expansion.
