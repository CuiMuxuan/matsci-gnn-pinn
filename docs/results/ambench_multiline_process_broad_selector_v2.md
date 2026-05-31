# Phase 32: Broad-Data Selector v2 Diagnostic

## Artifacts

- Code:
  - `src/gnnpinn/train/macro_pinn.py`
  - `scripts/server/run_phase30_broad_process_selector_smoke_a100.sh`
  - `scripts/server/summarize_phase30_broad_process_selector_smoke.py`
- Server logs:
  - `logs/ambench_phase32_broad12_rr_selector_v2_a100_v1.log`
  - `logs/ambench_phase32_broad21_rr_selector_v2_a100_v1.log`
- Server summaries:
  - `outputs/reports/phase32_broad12_process_selector_v2_summary.json`
  - `outputs/reports/phase32_broad21_process_selector_v2_summary.json`

## Motivation

Phase 31 showed a narrow refinement opportunity: on all 21 single-track lines, the old `process_axis_v1` `line_id -> concat/same` route was slightly better than the conservative `broad_process_v1` no-process fallback (`125.449323` vs `126.194921` test RMSE). Phase 32 therefore added `broad_process_v2` as a diagnostic profile:

- `line_id -> concat/same`
- `laser_power_W -> concat/global_standard`
- `spot_size_um -> film/global_standard`
- `scan_speed_mm_s -> none`
- `process_condition -> none`

The runner now accepts profile/tag overrides so `broad_process_v2` writes `broad_process_profile_v2` artifacts and does not overwrite Phase 30/31 `broad_process_v1` outputs.

## Commands

```bash
DATASET_LIMIT=12 DATASET_ORDER=process_round_robin STEPS=500 N_ESTIMATORS=80 \
PROCESS_CONDITIONING_PROFILE=broad_process_v2 \
PROCESS_FEATURE_TAG=broad_process_profile_v2 \
PROCESS_PROFILE_RUN_TAG=broad_process_profile_v2 \
  bash scripts/server/run_phase30_broad_process_selector_smoke_a100.sh \
  > logs/ambench_phase32_broad12_rr_selector_v2_a100_v1.log 2>&1

DATASET_LIMIT=21 DATASET_ORDER=process_round_robin STEPS=500 N_ESTIMATORS=80 \
PROCESS_CONDITIONING_PROFILE=broad_process_v2 \
PROCESS_FEATURE_TAG=broad_process_profile_v2 \
PROCESS_PROFILE_RUN_TAG=broad_process_profile_v2 \
  bash scripts/server/run_phase30_broad_process_selector_smoke_a100.sh \
  > logs/ambench_phase32_broad21_rr_selector_v2_a100_v1.log 2>&1

python scripts/server/summarize_phase30_broad_process_selector_smoke.py \
  --include-broad-process-v2 \
  --json-output outputs/reports/phase32_broad12_process_selector_v2_summary.json \
  --require-comparable

python scripts/server/summarize_phase30_broad_process_selector_smoke.py \
  --dataset-limit 21 \
  --dataset-order process_round_robin \
  --include-broad-process-v2 \
  --json-output outputs/reports/phase32_broad21_process_selector_v2_summary.json \
  --require-comparable
```

Both summaries passed the manifest/split comparability gate.

## Results

Lower is better. Values are test split RMSE.

| Dataset | Split | no-process | process_axis_v1 | broad_process_v1 | broad_process_v2 | Decision |
|---|---|---:|---:|---:|---:|---|
| broad12 | `line` | 126.308616 | 149.638162 | 126.308616 | 149.638162 | v2 harms broad12 line generalization. |
| broad12 | `laser_power` | 167.614004 | 140.753534 | 140.753534 | 140.753534 | v1/v2 tied. |
| broad12 | `scan_speed` | 186.173938 | 226.454041 | 186.173938 | 186.173938 | v1/v2 fallback is necessary. |
| broad12 | `spot_size` | 206.100512 | 136.309183 | 136.309183 | 136.309183 | v1/v2 tied. |
| broad12 | `process` | 181.091525 | 220.019735 | 181.091525 | 181.091525 | v1/v2 fallback is necessary. |
| broad21 | `line` | 126.194921 | 125.449323 | 126.194921 | 125.449323 | v2 recovers the small broad21 line gain. |
| broad21 | `laser_power` | 192.833317 | 178.040331 | 178.040331 | 178.040331 | v1/v2 tied. |
| broad21 | `scan_speed` | 227.128663 | 469.347549 | 227.128663 | 227.128663 | v1/v2 fallback is necessary. |
| broad21 | `spot_size` | 210.423419 | 147.389475 | 147.389475 | 147.389475 | v1/v2 tied. |
| broad21 | `process` | 166.231596 | 229.613547 | 166.231596 | 166.231596 | v1/v2 fallback is necessary. |

## Decision

`broad_process_v2` should remain a diagnostic profile, not the default broad-data selector:

- It recovers the small broad21 `line` improvement from `126.194921` to `125.449323`.
- It exactly reproduces the broad12 old-profile line regression from `126.308616` to `149.638162`.
- It does not change the useful `laser_power`, `spot_size`, `scan_speed`, or `process` decisions relative to `broad_process_v1`.

The conservative `broad_process_v1` route guard remains the safer default. Phase 33 should move to stronger broad-data neural representation or closure/GNN reintegration rather than more hand-tuned selector variants. The current A100-SXM4-40GB server remained sufficient.
