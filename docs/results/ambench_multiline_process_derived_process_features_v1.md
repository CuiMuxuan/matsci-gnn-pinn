# AM-Bench Multi-Line Derived Process Features v1

## Context

- Phase: 40 to 41.
- Branch: transfer-safe process representation under the `broad_process_v1` route guard.
- Motivation: Phase 39 output affine calibration improved broad12 `laser_power` across seeds, but failed broad21 transfer. Phase 40 checked whether smaller calibration scales fix the transfer failure before adding a new architecture.
- Decision: smaller output-affine scales did not recover broad21. Phase 41 pivots from output calibration to deterministic AM process-derived input features.

## Phase 40 Scale Sweep

Broad21 `laser_power` transfer comparison:

| Method | Output affine scale | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Signal |
| --- | ---: | ---: | ---: | ---: | --- |
| `mean` | n/a | 131.741364 | 237.730958 | 205.133029 | strong baseline |
| `knn_process` | n/a | 155.888818 | 311.251309 | 248.736635 | baseline |
| `extra_trees_process` | n/a | 165.322860 | 326.226457 | 257.019931 | baseline |
| `broad_process_v1` | n/a | 178.040331 | 296.909567 | 254.954359 | route guard |
| `broad_output_affine` | 0.50 | 210.939830 | 407.352779 | 326.056523 | negative |
| `broad_output_affine` | 0.25 | 195.281223 | 383.819303 | 305.364963 | negative |
| `broad_output_affine` | 0.10 | 224.145034 | 412.785926 | 334.457714 | negative |

Conclusion: the transfer problem is not only an over-large correction scale. The branch should not continue by tuning output-affine scale.

## Implementation

The training CLI now supports:

```text
--input-derived-process-features none|am_energy_v1
```

`am_energy_v1` appends four deterministic features computed from existing row metadata:

| Feature | Formula |
| --- | --- |
| `line_energy_J_per_mm` | `laser_power_W / scan_speed_mm_s` |
| `energy_density_proxy_J_per_mm_um` | `laser_power_W / (scan_speed_mm_s * spot_size_um)` |
| `energy_density_area_proxy_J_per_mm_um2` | `laser_power_W / (scan_speed_mm_s * spot_size_um * spot_size_um)` |
| `dwell_time_ms` | `spot_size_um / scan_speed_mm_s` |

Default behavior is unchanged with `none`. Metrics and checkpoints record `derived_process_features` metadata and include these names in `input_features.effective_columns`.

The server runner also accepts `PROCESS_FEATURE_COLUMNS`. By default it uses the raw process scalars:

```text
laser_power_W scan_speed_mm_s spot_size_um
```

Set `PROCESS_FEATURE_COLUMNS=""` to test a derived-only process representation without the raw scalar columns.

## Commands

Focused broad21 A100 validation:

```bash
PROFILE_SPLITS=laser_power DATASET_LIMIT=21 DATASET_ORDER=process_round_robin \
STEPS=500 N_ESTIMATORS=80 \
  bash scripts/server/run_phase41_broad_derived_process_features_a100.sh \
  > logs/phase41_broad21_laser_power_derived_process_a100_v1.log 2>&1
```

Summary:

```bash
python scripts/server/summarize_phase30_broad_process_selector_smoke.py \
  --dataset-limit 21 \
  --dataset-order process_round_robin \
  --split laser_power \
  --include-broad-derived-process \
  --json-output outputs/reports/phase41_broad21_laser_power_derived_process_summary.json \
  --require-comparable
```

Derived-only diagnostic if raw plus derived features over-trade global RMSE:

```bash
PROFILE_SPLITS=laser_power DATASET_LIMIT=21 DATASET_ORDER=process_round_robin \
STEPS=500 N_ESTIMATORS=80 PROCESS_FEATURE_COLUMNS="" PROCESS_FEATURE_TAG=phys_only \
  bash scripts/server/run_phase41_broad_derived_process_features_a100.sh \
  > logs/phase41_broad21_laser_power_derived_only_a100_v1.log 2>&1
```

## Decision Gate

Compare `broad_derived_process` against mean, kNN, ExtraTrees, no-process Macro PINN, `process_axis_v1`, and `broad_process_v1` on the same broad21 `laser_power` manifest/split. If it improves broad21 global, hot q90, and gradient q90 without falling behind the strongest baselines, run a paired broad12/broad21 check. If it is negative, close the branch and pivot again.

The branch remains lightweight and should fit the current A100-SXM4-40GB server. Do not request A100-SXM4-80GB unless a later learned encoder or coupled graph branch exceeds the current GPU envelope.

## Results

Broad21 `laser_power` focused validation:

| Dataset/Split | Method | Feature set | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Signal |
| --- | --- | --- | ---: | ---: | ---: | --- |
| broad21 `laser_power` | `broad_process_v1` | raw process scalars | 178.040331 | 296.909567 | 254.954359 | route guard |
| broad21 `laser_power` | `broad_derived_process` | raw + `am_energy_v1` | 212.704856 | 221.878476 | 238.794848 | global negative, region positive |
| broad21 `laser_power` | `broad_derived_process` | derived-only `am_energy_v1` | 171.892969 | 211.624381 | 207.270255 | positive vs route guard |
| broad21 `laser_power` | `mean` | train mean | 131.741364 | 237.730958 | 205.133029 | strongest global baseline |
| broad21 `laser_power` | `knn_process` | coords + raw process | 155.888818 | 311.251309 | 248.736635 | strong global baseline |

The derived-only representation improves `broad_process_v1` on all three Macro PINN decision metrics, but it still trails the mean/kNN baselines on global RMSE and trails the mean baseline slightly on gradient q90.

Broad12 transfer check:

| Dataset/Split | Method | Feature set | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Signal |
| --- | --- | --- | ---: | ---: | ---: | --- |
| broad12 `laser_power` | `broad_process_v1` | raw process scalars | 140.753534 | 254.473291 | 215.411533 | route guard |
| broad12 `laser_power` | `broad_derived_process` | derived-only `am_energy_v1` | 162.766699 | 303.019663 | 254.346542 | negative |
| broad12 `laser_power` | `mean` | train mean | 132.965887 | 242.427068 | 208.105836 | strongest baseline |

## Decision

Phase 41 is a useful diagnostic but not a paper-facing model claim. Derived-only `am_energy_v1` gives the first broad21 `laser_power` Macro PINN improvement over `broad_process_v1` after the output-affine failures, especially in hot-zone and gradient-band metrics. However, the same representation fails on broad12 and still trails the strongest broad21 global baselines. Do not run seed expansion yet.

The next branch should not simply add more derived scalars. It should either learn or validate-select the process representation/profile from train/validation evidence, or pivot to a stronger baseline-facing architecture that can preserve the broad21 hot/gradient gain without broad12 regression.
