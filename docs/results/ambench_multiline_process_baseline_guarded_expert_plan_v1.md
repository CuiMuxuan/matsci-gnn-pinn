# AM-Bench Baseline-Guarded Process Expert Plan v1

## Status

Gate 1 infrastructure is implemented locally after Phase 44. A100 focused execution is pending.

## Why Phase 45 Changes Direction

Phases 33-44 tested fixed spacetime features, sparse/residual corrections, region loss weighting, process graph features, strong-baseline residualization, residual backbones, output affine calibration, derived process features, validation selection, process encoders, and process-group balancing. The repeated pattern is split-local improvement:

- broad12 improves while broad21 regresses;
- broad21 improves while broad12 regresses;
- hot-zone or gradient metrics improve while global RMSE regresses.

That means the next contribution should not be another small scalar/objective/module tweak. It must explicitly face both guards:

1. the conservative `broad_process_v1` route guard;
2. the strongest classical baseline for each split, especially train-mean on broad21 `laser_power`.

On broad21 `laser_power`, the known comparison is:

| Method | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE |
| --- | ---: | ---: | ---: |
| mean | 131.741364 | 237.730958 | 205.133029 |
| kNN process | 155.888818 | 311.251309 | 248.736635 |
| ExtraTrees process | 165.322860 | 326.226457 | 257.019931 |
| `broad_process_v1` | 178.040331 | 296.909567 | 254.954359 |
| derived-only `am_energy_v1` | 171.892969 | 211.624381 | 207.270255 |

The important signal is not that a Macro PINN variant already beats everything. It does not. The signal is that different experts own different error regions, and a paper-facing model must combine them without test leakage or unstable routing.

## Phase 45 Hypothesis

A transfer-stable model contribution is possible only if train/validation evidence can identify when to trust:

- a strong conservative baseline for global temperature level;
- `broad_process_v1` for known positive process axes;
- physically derived or process-specialized experts for hot-zone and gradient errors.

The candidate paper contribution is therefore:

```text
Baseline-Guarded Process Expert:
prediction = frozen_guard_prediction + bounded process-aware expert correction
```

The correction must be constrained enough that it cannot freely destroy global RMSE:

- nonnegative or simplex expert weights;
- bounded correction amplitude in normalized target space;
- optional residual correction around the frozen guard prediction;
- train/validation-only selection and metadata audit.

## Gate 1: Prediction-Level Upper Bound

Before implementing a new architecture, Phase 45 should first test whether existing aligned predictions contain a transferable improvement.

Required aligned predictions:

- mean baseline;
- kNN baseline;
- ExtraTrees baseline;
- no-process Macro PINN;
- `process_axis_v1`;
- `broad_process_v1`;
- Phase 41 derived-only `am_energy_v1`;
- Phase 43 process encoder;
- Phase 44 group balance;
- any other diagnostic already available on the same manifest/split.

Fit only on train/validation:

- convex weights over prediction columns;
- nonnegative least squares or simplex-constrained grid/optimizer;
- optional two-objective validation score: global RMSE plus hot/gradient q90 penalties;
- no test metric used for selecting weights or experts.

Continue only if the fitted stack improves or preserves global RMSE while improving hot q90 and gradient q90 on both broad12 and broad21 `laser_power`, and is competitive with the strongest classical baseline.

If this gate fails, do not build a trainable MoE. Report that existing expert predictions do not contain a validation-selectable transferable improvement and pivot to a data/split target where the model has a real advantage.

## Gate 2: Model Implementation

If Gate 1 passes, implement `baseline_guarded_expert_v1`.

Minimum implementation:

- add prediction export if current run artifacts do not persist row-aligned predictions;
- add a small expert combiner or guarded residual head;
- freeze the selected guard prediction path or treat it as a non-trainable anchor;
- constrain correction scale and expert weights;
- record metadata in metrics/checkpoints:
  - guard source;
  - expert sources/features;
  - constraint type;
  - selected weights;
  - validation score used for selection;
  - train/val/test split manifest hashes.

Do not replace `broad_process_v1`; compare against it.

## A100 Validation Order

1. broad12 + broad21 `laser_power`, seed 7 only, strict comparability.
2. If positive, paired seeds 1/2 for both broad12 and broad21 `laser_power`.
3. If still positive, expand to broad12/broad21 `spot_size`.
4. If still positive, run broad21 all-axis summary: `line`, `laser_power`, `scan_speed`, `spot_size`, `process`.

Success gate for seed expansion:

```text
global RMSE <= best of broad_process_v1 and strongest classical baseline, or no material regression
and hot q90 RMSE improves
and gradient q90 RMSE improves
on both broad12 and broad21 laser_power
```

Because the classical mean baseline can dominate global RMSE, the final paper table should report global, hot-zone, and gradient metrics separately rather than hiding the tradeoff in a single score.

## Hardware

Current A100-SXM4-40GB remains sufficient for Phase 45. Do not request A100-SXM4-80GB unless prediction export or the constrained expert implementation demonstrably exceeds current memory/runtime.

## Immediate Next Implementation Tasks

1. Done: `--prediction-output` is available for both baseline and Macro PINN evaluation paths.
2. Done: `scripts/server/phase45_prediction_stack_probe.py` consumes aligned prediction CSV files plus a split manifest.
3. Done: local tests cover no-test-leakage stack fitting and prediction export.
4. Done locally: `scripts/server/run_phase45_prediction_stack_probe_a100.sh` generates broad12/broad21 `laser_power` prediction CSVs and stack summaries without overwriting Phase 41-44 artifacts.
5. Pending: run the Phase 45 A100 script and summarize Gate 1 before any trainable model code is added.

Focused A100 command:

```bash
cd /root/matsci-gnn-pinn
PROFILE_SPLITS=laser_power DATASET_LIMITS="12 21" DATASET_ORDER=process_round_robin \
STEPS=500 N_ESTIMATORS=80 WEIGHT_STEP=0.1 \
  bash scripts/server/run_phase45_prediction_stack_probe_a100.sh \
  > logs/phase45_prediction_stack_probe_a100_v1.log 2>&1
```
