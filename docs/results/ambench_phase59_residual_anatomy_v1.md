# AM-Bench Phase 59 Residual Anatomy

## Purpose

Phase 59 diagnoses the Phase 58 alternate-density broad21 `spot_size` failure
before any new model branch is allowed to expand. The question is whether the
failure exposes a train/validation-visible residual structure that could justify
a new paper-facing model contribution, or whether it is only a route-boundary
case that should be documented.

## Artifacts

```text
docs/results/phase59_residual_anatomy/
```

Server runner:

```bash
bash scripts/server/run_phase59_broad21_density_residual_anatomy_a100.sh
```

The runner exports row-aligned prediction CSVs for the mean baseline,
no-process Macro PINN, and `broad_process_v1`, then builds residual-slice and
upper-bound summaries.

## Residual Anatomy Result

On the alternate-density broad21 `spot_size` test split:

| Method | RMSE | MAE | Bias |
|---|---:|---:|---:|
| mean | 139.725646 | 116.525079 | -35.568503 |
| no-process Macro PINN | 226.518793 | 174.460834 | -166.617306 |
| `broad_process_v1` | 153.259459 | 125.542693 | -28.582547 |

`broad_process_v1` still beats no-process, but it loses to the mean baseline.
The largest candidate-vs-mean residual deltas occur in:

| Slice | n | `broad_process_v1` RMSE | mean RMSE | Delta |
|---|---:|---:|---:|---:|
| `Line_1_1_1` | 368 | 165.425776 | 143.582200 | 21.843576 |
| hot q90 region | 111 | 272.542708 | 253.129723 | 19.412985 |
| low time/frame bin | 462 | 150.448143 | 131.865568 | 18.582575 |
| `Line_1_1_2` | 387 | 154.110788 | 136.787462 | 17.323326 |

All test rows come from the held-out process tuple
`laser_power_W=285 / scan_speed_mm_s=960 / spot_size_um=49`, so this is a
density-sensitive route-boundary case for the small-spot branch.

## Upper-Bound Probe

The no-test-leakage upper-bound probe fits corrections on train, selects on
validation, and reports test only after selection. It sets
`uses_test_for_selection=false`.

Validation selected:

```text
blend:broad_process_v1->mean:alpha=1
```

This is an explicit fallback to the mean baseline. Group residual/bias
corrections using line, process tuple, process axes, and time/frame bins ranked
below the mean fallback on validation and did not beat the mean reference on
test.

| Variant | Val RMSE | Test RMSE | Test hot q90 | Test gradient q90 |
|---|---:|---:|---:|---:|
| mean fallback | 138.500717 | 139.725646 | 253.129723 | 198.003809 |
| 75% mean blend | 140.133258 | 141.785617 | 257.031462 | 200.637306 |
| 50% mean blend | 144.869162 | 144.766469 | 261.258946 | 203.812865 |
| time-bin group bias | 162.132533 | 153.083447 | 269.653909 | 211.294892 |
| global bias correction | 162.142446 | 153.086051 | 269.693001 | 211.250815 |

## Decision

Phase 59 closes as a negative model-expansion gate. The broad21
alternate-density failure is real and structured, but the validation-visible
correction that wins is a route fallback to the mean baseline, not a learnable
residual correction that preserves the current model claim.

Do not use this failure to justify Candidate A/B model expansion. Keep it as a
claim-boundary result and proceed with manuscript packaging around the fixed
Phase 55 seed-robust `spot_size` floor unless a new validation-visible model
signal appears.
