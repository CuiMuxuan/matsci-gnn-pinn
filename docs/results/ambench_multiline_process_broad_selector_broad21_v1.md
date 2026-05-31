# Phase 31: Broad21 All Single-Track Selector Scaling

## Context

- Server: Ubuntu A100-SXM4-40GB environment recorded in `docs/server_environment_snapshot.md`.
- Starting synced commit: `7bf4967 Add broad process profile smoke`.
- Scripts:
  - `scripts/server/run_phase29_broad_process_profile_smoke_a100.sh`
  - `scripts/server/run_phase30_broad_process_selector_smoke_a100.sh`
  - `scripts/server/summarize_phase30_broad_process_selector_smoke.py`
- Logs:
  - `logs/ambench_phase31_broad21_rr_process_axis_profile_a100_v1.log`
  - `logs/ambench_phase31_broad21_rr_selector_smoke_a100_v1.log`
- Summary artifact:
  - `outputs/reports/phase31_broad21_process_selector_summary.json`

Phase 30 validated the selector on broad12. Phase 31 scales the same process-balanced order to all 21 single-track `ThermalData/Line_*/Signal` datasets and compares the old `process_axis_v1` profile against the new `broad_process_v1` selector.

## Command

```bash
DATASET_LIMIT=21 DATASET_ORDER=process_round_robin STEPS=500 N_ESTIMATORS=80 \
  bash scripts/server/run_phase29_broad_process_profile_smoke_a100.sh \
  > logs/ambench_phase31_broad21_rr_process_axis_profile_a100_v1.log 2>&1

DATASET_LIMIT=21 DATASET_ORDER=process_round_robin STEPS=500 N_ESTIMATORS=80 \
  bash scripts/server/run_phase30_broad_process_selector_smoke_a100.sh \
  > logs/ambench_phase31_broad21_rr_selector_smoke_a100_v1.log 2>&1

PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  /home/vipuser/miniconda3/bin/conda run -n gnnpinn \
  python scripts/server/summarize_phase30_broad_process_selector_smoke.py \
  --dataset-limit 21 \
  --dataset-order process_round_robin \
  --json-output outputs/reports/phase31_broad21_process_selector_summary.json \
  --require-comparable
```

The summary gate passed for all five broad21 splits.

## Dataset

The 21-line process-balanced panel keeps the same `process_round_robin` ordering as Phase 29/30, but expands to all single-track thermography lines.

## Metrics

Lower is better. Values are test split metrics from `scripts/server/summarize_phase30_broad_process_selector_smoke.py`.

| Split | Method | Selected route | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE |
| --- | --- | --- | ---: | ---: | ---: |
| `line` | mean baseline | none | 131.161929 | 243.188033 | 214.217962 |
| `line` | no-process Macro PINN | none | 126.194921 | 234.351122 | 205.642173 |
| `line` | `process_axis_v1` | concat / same | 125.449323 | 227.220909 | 195.528670 |
| `line` | `broad_process_v1` | no process | 126.194921 | 234.351122 | 205.642173 |
| `laser_power` | mean baseline | none | 131.741364 | 237.730958 | 205.133029 |
| `laser_power` | no-process Macro PINN | none | 192.833317 | 356.915918 | 286.944447 |
| `laser_power` | `process_axis_v1` | concat / global-standard | 178.040331 | 296.909567 | 254.954359 |
| `laser_power` | `broad_process_v1` | concat / global-standard | 178.040331 | 296.909567 | 254.954359 |
| `scan_speed` | mean baseline | none | 144.014351 | 251.407139 | 212.910489 |
| `scan_speed` | no-process Macro PINN | none | 227.128663 | 392.018079 | 304.940054 |
| `scan_speed` | `process_axis_v1` | concat / global-standard | 469.347549 | 249.841501 | 408.809014 |
| `scan_speed` | `broad_process_v1` | no process | 227.128663 | 392.018079 | 304.940054 |
| `spot_size` | mean baseline | none | 149.185412 |  | 231.072566 |
| `spot_size` | no-process Macro PINN | none | 210.423419 |  | 337.723910 |
| `spot_size` | `process_axis_v1` | FiLM / global-standard | 147.389475 |  | 177.908136 |
| `spot_size` | `broad_process_v1` | FiLM / global-standard | 147.389475 |  | 177.908136 |
| `process` | mean baseline | none | 145.350346 | 248.754243 | 216.442403 |
| `process` | no-process Macro PINN | none | 166.231596 | 308.389105 | 251.049837 |
| `process` | `process_axis_v1` | concat / same | 229.613547 | 393.910906 | 317.989269 |
| `process` | `broad_process_v1` | no process | 166.231596 | 308.389105 | 251.049837 |

## Interpretation

Broad21 confirms that the selector is not just a broad12 smoke artifact.

- `line`: the old `process_axis_v1` route is now slightly better than no-process, but `broad_process_v1` remains conservative and falls back to no-process.
- `laser_power`: both profile variants keep the positive concat/global-standard route.
- `scan_speed`: the old profile becomes catastrophically bad (`469.347549` RMSE), while `broad_process_v1` safely falls back to no-process.
- `spot_size`: both variants keep the positive FiLM/global-standard route and remain below the mean baseline.
- `process`: the old profile is much worse than no-process; the selector avoids that regression.

This result says the broad selector is directionally correct, but it is still only a conservative route guard, not a universal best-route oracle. The next branch can either:

1. refine the broad selector with a broad21-specific line route, or
2. leave the selector conservative and pivot to a stronger model/data branch now that the route fallback is verified at all 21 single-track lines.

The current A100-SXM4-40GB remains sufficient for either path.

## Decision

Close Phase 31 as broad21 selector scaling:

- broad21 all-21 artifacts were generated with matching dataset settings and the summary gate passed;
- `broad_process_v1` preserved positive `laser_power` and `spot_size` routes while avoiding `scan_speed` and `process` regressions;
- the repo now has an end-to-end broad12 and broad21 selector story.

## Artifacts

```text
outputs/reports/phase31_broad21_process_selector_summary.json
outputs/data_audits/ambench_multiline_process_temperature_broad21_process_round_robin_*_process_axis_profile_smoke_a100_sxm4_40gb_v1_manifest.json
outputs/data_splits/ambench_multiline_process_temperature_broad21_process_round_robin_*_process_axis_profile_smoke_a100_sxm4_40gb_v1_split.json
outputs/data_audits/ambench_multiline_process_temperature_broad21_process_round_robin_*_broad_process_profile_smoke_a100_sxm4_40gb_v1_manifest.json
outputs/data_splits/ambench_multiline_process_temperature_broad21_process_round_robin_*_broad_process_profile_smoke_a100_sxm4_40gb_v1_split.json
outputs/baselines/ambench_multiline_process_temperature_broad21_process_round_robin_*_*_regions_q90.json
outputs/runs/ambench_multiline_process_temperature_broad21_process_round_robin_*_macro_pinn_minmax_*_v1/
```
