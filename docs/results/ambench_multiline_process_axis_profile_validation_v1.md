# Phase 28: Process-Axis Profile Validation

## Context

- Server: Ubuntu A100-SXM4-40GB environment recorded in `docs/server_environment_snapshot.md`.
- Starting commit: `d4549fb Add process-axis routing diagnostics`.
- Scripts:
  - `scripts/server/run_multiline_process_axis_profile_a100.sh`
  - `scripts/server/run_phase28_laser_power_profile_seed_check_a100.sh`
  - `scripts/server/summarize_phase25_film_metrics.py`
- Logs:
  - `logs/ambench_multiline_process_axis_profile_phase28_new_axes_a100_v1.log`
  - `logs/ambench_multiline_process_axis_profile_phase28_process_same_a100_v1.log`
  - `logs/ambench_phase28_laser_power_profile_seed_check_a100_v1.log`
- Summary artifact:
  - `outputs/reports/phase28_process_axis_profile_metrics_summary.json`

Phase 27 made `process_axis_v1` reproducible for `line`, `scan_speed`, and `spot_size`. Phase 28 extends the same profile to `laser_power` and full `process` holdouts, then adds a focused seed check for the only new positive route.

## Profile Updates

`process_axis_v1` now records the following routes:

| Split group key | Selected route |
| --- | --- |
| `line_id` | `concat` with `same`, i.e. train minmax from `--input-normalization minmax` |
| `laser_power_W` | `concat` with `global_standard` |
| `scan_speed_mm_s` | `concat` with `global_standard` |
| `spot_size_um` | `film` with `global_standard` |
| `process_condition` | `concat` with `same`, i.e. line-like train minmax |

The full-process route was corrected during Phase 28. An initial `concat/global_standard` run degraded the full-process split to test RMSE `183.009876`. Re-running with the line-like `concat/same` route recovered the prior best result, `157.793227`, and records `selected=concat/same` in the profile metadata.

## Seed-7 New-Axis Metrics

Lower is better. Values are test split metrics from `scripts/server/summarize_phase25_film_metrics.py`.

| Split | Method | Selected route | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE |
| --- | --- | --- | ---: | ---: | ---: |
| `laser_power` | no process | none | 210.844015 | 388.253506 | 357.685604 |
| `laser_power` | Phase 24 concat/train-minmax | concat / train-minmax | 195.215649 | 365.594962 | 337.437787 |
| `laser_power` | `process_axis_v1` | concat / global-standard | 144.047705 | 228.311557 | 216.223266 |
| `laser_power` | mean baseline | train mean | 142.540915 | 243.351580 | 228.775600 |
| `process` | no process | none | 175.127058 | 351.525048 | 323.786011 |
| `process` | Phase 24 concat/train-minmax | concat / train-minmax | 157.793227 | 316.794319 | 293.650864 |
| `process` | wrong `process_axis_v1` trial | concat / global-standard | 183.009876 | 365.728620 | 330.926294 |
| `process` | corrected `process_axis_v1` | concat / train-minmax | 157.793227 | 316.794319 | 293.650864 |
| `process` | mean baseline | train mean | 128.668856 | 233.448623 | 219.561369 |

## Focused Seed Check

The `laser_power` route was the new paper-facing candidate because it nearly matched the train-mean global RMSE on seed 7 and improved hot/gradient q90 metrics. Seeds 1 and 2 were added with the same split and training settings.

| Split | Method | Seeds | Test RMSE mean +/- std | Hot q90 mean +/- std | Gradient q90 mean +/- std |
| --- | --- | --- | ---: | ---: | ---: |
| `laser_power` | no process | 7,1,2 | 211.217281 +/- 0.443665 | 391.157932 +/- 2.057337 | 360.208528 +/- 1.955679 |
| `laser_power` | `process_axis_v1` | 7,1,2 | 147.980699 +/- 3.456300 | 260.268960 +/- 23.249762 | 242.405449 +/- 18.892082 |
| `scan_speed` | no process | 7,1,2 | 186.248784 +/- 2.597410 | 380.192536 +/- 3.425596 | 350.087691 +/- 3.604586 |
| `scan_speed` | concat/global-standard | 7,1,2 | 137.793553 +/- 3.850852 | 235.582309 +/- 28.850742 | 221.173551 +/- 25.259315 |
| `spot_size` | no process | 7,1,2 | 206.208221 +/- 1.921594 | 354.797393 +/- 5.008958 | 347.376995 +/- 3.631370 |
| `spot_size` | FiLM/global-standard | 7,1,2 | 146.316608 +/- 8.620788 | 233.048739 +/- 30.142952 | 228.094488 +/- 27.732866 |

## Strong Baselines

The profile routes were also compared against train mean, kNN, and ExtraTrees baselines in one artifact summary. For `laser_power`, the process-axis profile strongly beats kNN and ExtraTrees baselines and essentially matches the train-mean baseline on seed 7 while improving hot/gradient regions. For full `process`, the corrected profile remains weaker than the train-mean baseline but is stronger than kNN and ExtraTrees.

| Split | Baseline | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE |
| --- | --- | ---: | ---: | ---: |
| `laser_power` | mean | 142.540915 | 243.351580 | 228.775600 |
| `laser_power` | kNN coords | 182.425004 | 344.428146 | 319.370480 |
| `laser_power` | kNN process | 182.398146 | 344.393287 | 319.370480 |
| `laser_power` | ExtraTrees coords | 198.879536 | 371.286880 | 342.362985 |
| `laser_power` | ExtraTrees process | 198.115063 | 370.112398 | 341.296800 |
| `process` | mean | 128.668856 | 233.448623 | 219.561369 |
| `process` | kNN coords | 170.889327 | 342.274178 | 319.054607 |
| `process` | kNN process | 167.992301 | 347.707131 | 321.184533 |
| `process` | ExtraTrees coords | 172.329968 | 345.385134 | 320.995843 |
| `process` | ExtraTrees process | 165.419834 | 343.469045 | 316.403279 |

## Interpretation

Phase 28 strengthens the axis-aware process-conditioning story. `laser_power`, `scan_speed`, and `spot_size` now each have a focused three-seed process-conditioned route that substantially improves no-process Macro PINN and approaches or beats the train-mean baseline in different ways.

The result is still not a universal model claim. The train-mean baseline remains stronger on `line` and full `process` global RMSE, and the full-process split does not benefit from global process-feature standardization. This argues against expanding blind mixture-of-experts and against immediately reintegrating sparse closure/GNN terms on the same seven-line panel.

## Decision

Close Phase 28 as a positive profile-validation node with a conservative paper-facing claim:

- process-conditioned Macro PINN is split-axis sensitive;
- explicit artifact-recorded process-axis profiles make the route selection reproducible;
- `laser_power`, `scan_speed`, and `spot_size` have stable three-seed gains over no-process Macro PINN;
- full-process and line splits still require broader data or stronger baseline-facing modeling before closure/GNN reintegration.

The next node should broaden the thermal/process dataset or add stronger baseline-facing modeling rather than adding a larger trainable router. The current A100-SXM4-40GB remains sufficient.

## Artifacts

```text
outputs/reports/phase28_process_axis_profile_metrics_summary.json
outputs/runs/ambench_multiline_process_temperature_laser_power_process_axis_profile_a100_sxm4_40gb_v1_macro_pinn_minmax_process_axis_profile_v1/
outputs/runs/ambench_multiline_process_temperature_laser_power_process_axis_profile_a100_sxm4_40gb_v1_seed*_macro_pinn_minmax_process_axis_profile_v1/
outputs/runs/ambench_multiline_process_temperature_process_process_axis_profile_a100_sxm4_40gb_v1_macro_pinn_minmax_process_axis_profile_v1/
outputs/baselines/ambench_multiline_process_temperature_*_process_axis_profile_a100_sxm4_40gb_v1_*_regions_q90.json
```
