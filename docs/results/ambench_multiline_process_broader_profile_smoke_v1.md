# Phase 29: Broader Process Dataset Profile Smoke

## Context

- Server: Ubuntu A100-SXM4-40GB environment recorded in `docs/server_environment_snapshot.md`.
- Starting synced commit: `a65f841 Validate process-axis profile routes`.
- Scripts:
  - `scripts/server/run_multiline_process_conditioned_thermal_a100.sh`
  - `scripts/server/run_phase29_broad_process_profile_smoke_a100.sh`
  - `scripts/server/summarize_phase29_broad_process_profile_smoke.py`
- Log:
  - `logs/ambench_phase29_broad12_rr_profile_smoke_a100_v1.log`
- Summary artifact:
  - `outputs/reports/phase29_broad12_process_round_robin_profile_smoke_summary.json`

Phase 28 validated `process_axis_v1` on the seven representative-line panel. Phase 29 tests whether the same profile still behaves well when the dataset is broadened beyond those seven lines.

## Dataset Selection

The first naive broad smoke used the first 9 sorted `ThermalData/Line_*/Signal` datasets. That smoke was not process-balanced: it only covered `285 W / 960 mm/s`, so it was useful as a chain test but not as process-generalization evidence.

Phase 29 adds `--dataset-order process_round_robin`, which groups candidate line datasets by `(laser_power, scan_speed, spot_size)` and alternates process groups before applying `--dataset-limit`. The broad12 smoke uses:

```bash
DATASET_LIMIT=12 DATASET_ORDER=process_round_robin \
  bash scripts/server/run_phase29_broad_process_profile_smoke_a100.sh \
  > logs/ambench_phase29_broad12_rr_profile_smoke_a100_v1.log 2>&1
```

Selected datasets:

```text
ThermalData/Line_3_2_1/Signal
ThermalData/Line_2_2_1/Signal
ThermalData/Line_1_1_1/Signal
ThermalData/Line_0_1/Signal
ThermalData/Line_1_2_1/Signal
ThermalData/Line_2_1_1/Signal
ThermalData/Line_3_1_1/Signal
ThermalData/Line_3_2_2/Signal
ThermalData/Line_2_2_2/Signal
ThermalData/Line_1_1_2/Signal
ThermalData/Line_0_2/Signal
ThermalData/Line_1_2_2/Signal
```

The converted table has `3451` sampled calibrated-temperature rows and covers:

| Axis | Groups |
| --- | --- |
| `laser_power_W` | `245`, `285`, `325` |
| `scan_speed_mm_s` | `800`, `960`, `1200` |
| `spot_size_um` | `49`, `67`, `82` |
| full process condition | 7 process tuples |

## Metrics

Lower is better. Values are test split metrics from `scripts/server/summarize_phase29_broad_process_profile_smoke.py`.

| Split | Method | Selected route | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE |
| --- | --- | --- | ---: | ---: | ---: |
| `line` | mean baseline | none | 134.042138 | 239.584080 | 214.370905 |
| `line` | no-process Macro PINN | none | 126.308616 | 217.257126 | 195.314294 |
| `line` | `process_axis_v1` | concat / same | 149.638162 | 225.303067 | 199.018753 |
| `line` | ExtraTrees process | none | 157.177933 | 304.939730 | 247.877010 |
| `laser_power` | mean baseline | none | 132.965887 | 242.427068 | 208.105836 |
| `laser_power` | no-process Macro PINN | none | 167.614004 | 331.709484 | 258.291492 |
| `laser_power` | `process_axis_v1` | concat / global-standard | 140.753534 | 254.473291 | 215.411533 |
| `laser_power` | ExtraTrees process | none | 172.115853 | 339.998010 | 265.481391 |
| `scan_speed` | mean baseline | none | 145.115776 | 250.659348 | 209.791354 |
| `scan_speed` | no-process Macro PINN | none | 186.173938 | 345.736994 | 266.380605 |
| `scan_speed` | `process_axis_v1` | concat / global-standard | 226.454041 | 389.782976 | 299.192165 |
| `scan_speed` | ExtraTrees process | none | 200.334538 | 366.818427 | 281.975889 |
| `spot_size` | mean baseline | none | 151.850578 | 252.554440 | 233.119660 |
| `spot_size` | no-process Macro PINN | none | 206.100512 | 363.828210 | 329.278428 |
| `spot_size` | `process_axis_v1` | FiLM / global-standard | 136.309183 | 165.228535 | 169.049295 |
| `spot_size` | ExtraTrees process | none | 207.831140 | 358.040079 | 326.972947 |
| `process` | mean baseline | none | 147.381589 | 251.032500 | 213.464819 |
| `process` | no-process Macro PINN | none | 181.091525 | 325.205379 | 266.149257 |
| `process` | `process_axis_v1` | concat / same | 220.019735 | 380.347408 | 300.630821 |
| `process` | ExtraTrees process | none | 185.376281 | 342.337265 | 272.920994 |

## Interpretation

The balanced broad12 result changes the Phase 28 story in an important way. The seven-line profile was not a final universal route; it was partly tied to the representative panel. On broad12:

- `spot_size` remains the strongest positive axis. FiLM/global-standard beats the mean baseline and sharply improves hot/gradient q90 regions.
- `laser_power` still improves no-process Macro PINN, but it no longer beats the train-mean baseline.
- `line` favors the no-process Macro PINN on this smoke table; forcing the line-like process profile hurts global RMSE.
- `scan_speed` and full `process` degrade under the old profile and should not be expanded without revision.
- kNN and ExtraTrees process baselines are not competitive on this smoke, so the train-mean baseline remains the main baseline-facing gate.

This is a useful negative/diagnostic result, not a failure of the project. It prevents overclaiming from the seven-line panel and points to a clearer next node: the profile needs an explicit ability to fall back to no-process modeling or a broad-data route, rather than always forcing process features.

## Decision

Close Phase 29 as a broader-data validation node:

- the code now supports regex-selected broad line panels, dataset limits, and process-balanced ordering;
- the server smoke verifies that broad12 covers multiple power, speed, spot-size, and full-process groups;
- `process_axis_v1` transfers only partially to the broader panel;
- the next phase should implement a revised broad-process selector/profile that can choose no-process, concat, or FiLM routes per split and should validate that route on broad12 before scaling to all 21 single-track lines.

The current A100-SXM4-40GB remains sufficient.

## Artifacts

```text
outputs/reports/phase29_broad12_process_round_robin_profile_smoke_summary.json
outputs/data_audits/ambench_multiline_process_temperature_broad12_process_round_robin_*_process_axis_profile_smoke_a100_sxm4_40gb_v1_manifest.json
outputs/data_splits/ambench_multiline_process_temperature_broad12_process_round_robin_*_process_axis_profile_smoke_a100_sxm4_40gb_v1_split.json
outputs/baselines/ambench_multiline_process_temperature_broad12_process_round_robin_*_process_axis_profile_smoke_a100_sxm4_40gb_v1_*_regions_q90.json
outputs/runs/ambench_multiline_process_temperature_broad12_process_round_robin_*_process_axis_profile_smoke_a100_sxm4_40gb_v1_macro_pinn_minmax_*_v1/
```
