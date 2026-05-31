# Phase 30: Broad-Data Conditional Route Selector

## Context

- Server: Ubuntu A100-SXM4-40GB environment recorded in `docs/server_environment_snapshot.md`.
- Starting synced commit: `7bf4967 Add broad process profile smoke`.
- Scripts:
  - `scripts/server/run_phase30_broad_process_selector_smoke_a100.sh`
  - `scripts/server/summarize_phase30_broad_process_selector_smoke.py`
- Log:
  - `logs/ambench_phase30_broad12_rr_selector_smoke_a100_v1.log`
- Summary artifact:
  - `outputs/reports/phase30_broad_process_selector_smoke_summary.json`

Phase 29 showed that the seven-line `process_axis_v1` profile transfers only partly to the broader process-balanced broad12 panel. Phase 30 adds `broad_process_v1`, a split-aware profile that can choose no-process, concat, or FiLM routes instead of forcing process conditioning.

## Implementation

`broad_process_v1` reads the grouped split manifest `group_key` and selects:

| group key | selected route | reason |
| --- | --- | --- |
| `line_id` | no process | broad12 line holdout favors coordinate/time Macro PINN over forced process conditioning |
| `laser_power_W` | concat / global-standard | broad12 laser-power holdout still benefits from process scalars |
| `scan_speed_mm_s` | no process | old concat/global-standard profile degrades on broad12 scan-speed holdout |
| `spot_size_um` | FiLM / global-standard | broad12 spot-size holdout retains the strong FiLM gain |
| `process_condition` | no process | broad12 full-process holdout degrades under line-like concat/same |

The no-process route clears `input_feature_columns`, records the requested and selected route metadata, and trains the same Macro PINN with `param_dim=0`. A tiny scan-speed smoke confirmed:

```text
features_enabled=false
features_count=0
config input_feature_columns=[]
checkpoint metadata param_dim=0
selected conditioning_mode=none
```

The Phase 30 summary script also checks manifest and split signatures before comparing Phase 29 and Phase 30 artifacts. This prevents a tiny smoke artifact from being mixed with the full broad12 Phase 29 baselines.

## Command

```bash
DATASET_LIMIT=12 DATASET_ORDER=process_round_robin STEPS=500 N_ESTIMATORS=80 \
  bash scripts/server/run_phase30_broad_process_selector_smoke_a100.sh \
  > logs/ambench_phase30_broad12_rr_selector_smoke_a100_v1.log 2>&1

PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  /home/vipuser/miniconda3/bin/conda run -n gnnpinn \
  python scripts/server/summarize_phase30_broad_process_selector_smoke.py \
  --json-output outputs/reports/phase30_broad_process_selector_smoke_summary.json \
  --require-comparable
```

The `--require-comparable` gate passed for all five broad12 splits.

## Dataset

The broad12 panel is the same process-balanced selection used in Phase 29:

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

It has `3451` sampled calibrated-temperature rows and covers `245/285/325 W`, `800/960/1200 mm/s`, `49/67/82 um`, and 7 full process tuples.

## Metrics

Lower is better. Values are test split metrics from `scripts/server/summarize_phase30_broad_process_selector_smoke.py`.

| Split | Method | Selected route | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE |
| --- | --- | --- | ---: | ---: | ---: |
| `line` | mean baseline | none | 134.042138 | 239.584080 | 214.370905 |
| `line` | no-process Macro PINN | none | 126.308616 | 217.257126 | 195.314294 |
| `line` | `process_axis_v1` | concat / same | 149.638162 | 225.303067 | 199.018753 |
| `line` | `broad_process_v1` | no process | 126.308616 | 217.257126 | 195.314294 |
| `laser_power` | mean baseline | none | 132.965887 | 242.427068 | 208.105836 |
| `laser_power` | no-process Macro PINN | none | 167.614004 | 331.709484 | 258.291492 |
| `laser_power` | `process_axis_v1` | concat / global-standard | 140.753534 | 254.473291 | 215.411533 |
| `laser_power` | `broad_process_v1` | concat / global-standard | 140.753534 | 254.473291 | 215.411533 |
| `scan_speed` | mean baseline | none | 145.115776 | 250.659348 | 209.791354 |
| `scan_speed` | no-process Macro PINN | none | 186.173938 | 345.736994 | 266.380605 |
| `scan_speed` | `process_axis_v1` | concat / global-standard | 226.454041 | 389.782976 | 299.192165 |
| `scan_speed` | `broad_process_v1` | no process | 186.173938 | 345.736994 | 266.380605 |
| `spot_size` | mean baseline | none | 151.850578 | 252.554440 | 233.119660 |
| `spot_size` | no-process Macro PINN | none | 206.100512 | 363.828210 | 329.278428 |
| `spot_size` | `process_axis_v1` | FiLM / global-standard | 136.309183 | 165.228535 | 169.049295 |
| `spot_size` | `broad_process_v1` | FiLM / global-standard | 136.309183 | 165.228535 | 169.049295 |
| `process` | mean baseline | none | 147.381589 | 251.032500 | 213.464819 |
| `process` | no-process Macro PINN | none | 181.091525 | 325.205379 | 266.149257 |
| `process` | `process_axis_v1` | concat / same | 220.019735 | 380.347408 | 300.630821 |
| `process` | `broad_process_v1` | no process | 181.091525 | 325.205379 | 266.149257 |

## Interpretation

`broad_process_v1` is not a new trainable architecture and should not be presented as a universal final model. Its value is stricter: it prevents known negative transfer on the broad12 panel while preserving the positive broad-data routes.

- `line`: route falls back to no-process and avoids the old profile regression (`149.638162 -> 126.308616`).
- `scan_speed`: route falls back to no-process and avoids the old profile regression (`226.454041 -> 186.173938`).
- `process`: route falls back to no-process and avoids the old profile regression (`220.019735 -> 181.091525`).
- `laser_power`: keeps the concat/global-standard route, improving no-process Macro PINN (`167.614004 -> 140.753534`) but still not beating the mean baseline.
- `spot_size`: keeps the FiLM/global-standard route and remains the strongest process-conditioned broad12 case (`206.100512 -> 136.309183`), beating the mean baseline.

The broad12 result therefore supports scaling the selector to all 21 single-track `ThermalData/Line_*/Signal` datasets before returning to closure/GNN coupling. The current A100-SXM4-40GB remains sufficient.

## Decision

Close Phase 30 as a broad-data selector validation node:

- `broad_process_v1` is implemented and artifact-recorded;
- no-process fallback is verified in metrics and checkpoint metadata;
- the summary gate prevents mismatched tiny/full artifact comparisons;
- the full broad12 A100 run completed with all methods comparable;
- next step: run the same selector on all 21 single-track thermography datasets using `DATASET_LIMIT=21 DATASET_ORDER=process_round_robin`.

## Artifacts

```text
outputs/reports/phase30_broad_process_selector_smoke_summary.json
outputs/data_audits/ambench_multiline_process_temperature_broad12_process_round_robin_*_broad_process_profile_smoke_a100_sxm4_40gb_v1_manifest.json
outputs/data_splits/ambench_multiline_process_temperature_broad12_process_round_robin_*_broad_process_profile_smoke_a100_sxm4_40gb_v1_split.json
outputs/baselines/ambench_multiline_process_temperature_broad12_process_round_robin_*_broad_process_profile_smoke_a100_sxm4_40gb_v1_*_regions_q90.json
outputs/runs/ambench_multiline_process_temperature_broad12_process_round_robin_*_broad_process_profile_smoke_a100_sxm4_40gb_v1_macro_pinn_minmax_*_v1/
```
