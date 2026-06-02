# Phase 55: Spot-Size Transferable Route Seed Validation

## Status

Phase 55 is complete.

The current paper-facing model contribution is now a seed-validated broad-data route guard with a clean process-conditioned `spot_size` positive branch. The claim is not that process conditioning wins on every holdout axis; the claim is that the explicit `broad_process_v1` route guard prevents known negative-transfer routes, and its `spot_size -> FiLM/global-standard` route is stable across broad12 and broad21 against strong baselines.

## Implementation

New runner:

```text
scripts/server/run_phase55_spot_size_route_seed_check_a100.sh
```

New summary:

```text
scripts/server/summarize_phase55_spot_size_seed_check.py
```

New test:

```text
tests/test_phase55_spot_size_seed_summary.py
```

The runner reuses the existing Phase 30 `spot_size` table and split artifacts, then trains only seed-specific no-process and `broad_process_v1` Macro PINN artifacts for model seeds 1 and 2. Seed 7 is read from the existing Phase 30 run. This keeps the holdout split fixed and avoids mixing split changes with model-seed robustness.

The summary compares seeds 7/1/2 against:

- no-process Macro PINN
- mean baseline
- kNN coords/process
- ExtraTrees coords/process

Required metrics are test RMSE, hot q90 RMSE, and gradient q90 RMSE.

## A100 Command

```bash
DATASET_LIMITS="12 21" \
SEEDS="1 2" \
SUMMARY_SEEDS="7 1 2" \
STEPS=500 \
bash scripts/server/run_phase55_spot_size_route_seed_check_a100.sh \
  > logs/phase55_spot_size_route_seed_check_a100.log 2>&1
```

Verification summary:

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
/home/vipuser/miniconda3/bin/conda run -n gnnpinn \
python -X utf8 scripts/server/summarize_phase55_spot_size_seed_check.py \
  --dataset-limit 12 \
  --dataset-limit 21 \
  --seed 7 \
  --seed 1 \
  --seed 2 \
  --require-complete \
  --require-pass \
  --json-output outputs/reports/phase55_spot_size_route_seed_check_summary_verify.json
```

## Result

Transfer gate: `seed_robust_transfer_positive`.

| Dataset | Method | N | Test RMSE mean +/- std | Hot q90 RMSE mean +/- std | Gradient q90 RMSE mean +/- std |
| --- | --- | ---: | ---: | ---: | ---: |
| broad12 | no-process | 3 | `238.093690 +/- 29.836097` | `424.409003 +/- 53.733799` | `382.799174 +/- 48.263668` |
| broad12 | `broad_process_v1` | 3 | `136.384782 +/- 0.467526` | `162.125337 +/- 4.909788` | `165.282182 +/- 5.270236` |
| broad21 | no-process | 3 | `217.922642 +/- 5.308273` | `401.488520 +/- 15.153059` | `360.868300 +/- 17.032414` |
| broad21 | `broad_process_v1` | 3 | `146.002303 +/- 1.118699` | `164.313888 +/- 3.548500` | `174.735839 +/- 2.301005` |

Against the best strong baseline:

| Dataset | Metric | `broad_process_v1` mean | Best strong baseline | Delta |
| --- | --- | ---: | ---: | ---: |
| broad12 | test RMSE | `136.384782` | mean `151.850578` | `-15.465796` |
| broad12 | hot q90 RMSE | `162.125337` | mean `252.554440` | `-90.429103` |
| broad12 | gradient q90 RMSE | `165.282182` | mean `233.119660` | `-67.837478` |
| broad21 | test RMSE | `146.002303` | mean `149.185412` | `-3.183109` |
| broad21 | hot q90 RMSE | `164.313888` | mean `251.976794` | `-87.662906` |
| broad21 | gradient q90 RMSE | `174.735839` | mean `231.072566` | `-56.336727` |

Per-seed `broad_process_v1` also beats the best strong baseline and no-process on all required metrics for both broad12 and broad21.

## Interpretation

This is the first current-work result that satisfies all three paper-facing constraints together:

- transferable across broad12 and broad21,
- seed-robust across model seeds 7/1/2,
- stronger than both no-process Macro PINN and the best classical baseline on global, hot-zone, and gradient-band RMSE.

The paper-facing model contribution should be framed as an explicit process-route guard with a stable `spot_size` FiLM/global-standard branch. `line` remains route-guard/no-process fallback evidence. `laser_power`, `scan_speed`, and full `process` remain route-guard-only rather than strong-baseline claims.

## Artifacts

```text
outputs/reports/phase55_spot_size_route_seed_check_summary.json
outputs/reports/phase55_spot_size_route_seed_check_summary.md
outputs/reports/phase55_spot_size_route_seed_check_summary_verify.json
docs/results/ambench_multiline_process_spot_size_seed_validation_v1.md
```
