# Phase 24: Process-Axis Holdout Generalization

## Context

- Server: Ubuntu A100-SXM4-40GB environment recorded in `docs/server_environment_snapshot.md`.
- Commit: `3d56a11 Add process-axis holdout splits`.
- Main log: `logs/ambench_multiline_process_holdout_splits_a100_v1.log`.
- Script: `scripts/server/run_multiline_process_holdout_splits_a100.sh`.

Command:

```bash
bash scripts/server/run_multiline_process_holdout_splits_a100.sh \
  > logs/ambench_multiline_process_holdout_splits_a100_v1.log 2>&1
```

The run reuses the seven-line calibrated `mds2-2716` thermography panel from Phase 23 and generates five grouped splits: `line`, `laser_power`, `scan_speed`, `spot_size`, and full `process` condition. Each split compares a coordinate-only Macro PINN with a Macro PINN that appends normalized `laser_power_W`, `scan_speed_mm_s`, and `spot_size_um` as input features.

## Split Groups

| Split strategy | Train groups | Val groups | Test groups | Train/Val/Test points |
| --- | --- | --- | --- | --- |
| `line` | `Line_0_1`, `Line_2_2_1`, `Line_3_1_1`, `Line_3_2_1` | `Line_2_1_1` | `Line_1_1_1`, `Line_1_2_1` | 8087 / 1884 / 4871 |
| `laser_power` | 245 W | 285 W | 325 W | 1983 / 10762 / 2097 |
| `scan_speed` | 800 mm/s | 960 mm/s | 1200 mm/s | 2201 / 10757 / 1884 |
| `spot_size` | 82 um | 67 um | 49 um | 3679 / 9971 / 1192 |
| `process` | `(245,960,67)`, `(285,800,67)`, `(285,960,67)`, `(325,960,67)` | `(285,1200,67)` | `(285,960,49)`, `(285,960,82)` | 8087 / 1884 / 4871 |

## Test Metrics

Lower is better. `Delta` is process-feature Macro PINN minus coordinate-only Macro PINN.

| Split strategy | Mean baseline RMSE | Macro PINN no process RMSE | Macro PINN process RMSE | Delta RMSE | Delta % | Delta hot q90 RMSE | Delta gradient q90 RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `line` | 128.668856 | 175.127058 | 157.793227 | -17.333832 | -9.90% | -34.730729 | -30.135147 |
| `laser_power` | 142.540915 | 210.844015 | 195.215649 | -15.628366 | -7.41% | -22.658544 | -20.247817 |
| `scan_speed` | 134.626834 | 186.921887 | 140.459979 | -46.461908 | -24.86% | -84.995201 | -76.709827 |
| `spot_size` | 148.611388 | 208.741300 | 227.573411 | +18.832110 | +9.02% | +22.389507 | +21.260380 |
| `process` | 128.668856 | 175.127058 | 157.793227 | -17.333832 | -9.90% | -34.730729 | -30.135147 |

## Interpretation

Process metadata improves the Macro PINN in four of five grouped holdouts, including the strongest improvement on scan-speed extrapolation. It also improves the hot-zone and high-gradient q90 subsets in those same four splits. This supports the Phase 23 decision to keep multi-line process-conditioned thermal modeling as the main branch.

The result is not yet a final paper result. The train-mean baseline remains stronger than both Macro PINN variants on global test RMSE in every split. The spot-size holdout is also a clear failure case: process features degrade global, hot q90, and gradient q90 metrics.

One important technical caveat is that single-axis holdouts leave only one scalar group in the train split for `laser_power`, `scan_speed`, or `spot_size`. With train-fitted min-max normalization, the held-out process feature can become a constant train signal and a one-sided extrapolation input at test time. A plain concatenated MLP can use this weakly, but the failure on spot size suggests that process conditioning should be made more structured rather than only appended as raw input columns.

## Decision

Phase 24 closes as a positive evidence gate for process-aware modeling, not as a completed model claim.

Next branch: implement a process-conditioned FiLM Macro PINN. FiLM should modulate hidden coordinate/time layers from process parameters, while preserving the existing concatenation path as the default backward-compatible baseline. The first A100 comparison should include `line`, `scan_speed`, and `spot_size` splits, because these cover the positive baseline case, the strongest positive process-axis case, and the current failure case.

## Artifacts

```text
outputs/data_audits/ambench_multiline_process_temperature_*_holdout_a100_sxm4_40gb_v1_manifest.json
outputs/data_splits/ambench_multiline_process_temperature_*_holdout_a100_sxm4_40gb_v1_split.json
outputs/baselines/ambench_multiline_process_temperature_*_holdout_a100_sxm4_40gb_v1_mean_constant_regions_q90.json
outputs/runs/ambench_multiline_process_temperature_*_holdout_a100_sxm4_40gb_v1_macro_pinn_minmax_no_process_v1/
outputs/runs/ambench_multiline_process_temperature_*_holdout_a100_sxm4_40gb_v1_macro_pinn_minmax_process_features_v1/
```
