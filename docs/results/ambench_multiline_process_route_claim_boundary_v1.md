# Phase 54: Broad-Data Process Route Claim Boundary

## Status

Phase 54 is complete.

The source-inversion / Bayesian inverse-closure / registered source-path work from Phases 46-53 is frozen as diagnostic for the current AMB2022-03 bundle. The current paper-facing model contribution should be framed around the conservative `broad_process_v1` route guard and its explicit claim boundary, not around broad12/broad21 source-inversion expansion.

## Implementation

New summary script:

```text
scripts/server/summarize_phase54_process_route_claim_boundary.py
```

New test:

```text
tests/test_phase54_claim_boundary_summary.py
```

The script reads existing Phase 30-style summary JSON files. It does not retrain models and does not overwrite training artifacts. It separates:

- `paper_claim_positive`: `broad_process_v1` beats the best strong baseline and is non-worse than no-process on all required metrics.
- `route_guard_positive`: `broad_process_v1` improves or preserves a Macro PINN route but does not beat the best strong baseline on all required metrics.
- `incomplete_metric`: at least one required metric is missing.
- `diagnostic_negative`: no strong-baseline or route-guard case.
- `incomparable`: manifest/split comparability failed.

Strong baselines are `mean`, `knn_coords`, `knn_process`, `extra_trees_coords`, and `extra_trees_process`. Neural references are `no_process` and `process_axis_v1`.

## Command

Server:

```text
root@223.109.239.30 -p 22036
GPU host: NVIDIA A100-SXM4-40GB
```

Input summaries:

```bash
PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  /home/vipuser/miniconda3/bin/conda run -n gnnpinn \
  python -X utf8 scripts/server/summarize_phase30_broad_process_selector_smoke.py \
  --dataset-limit 12 \
  --dataset-order process_round_robin \
  --json-output outputs/reports/phase54_broad12_claim_boundary_input_summary.json \
  --require-comparable

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  /home/vipuser/miniconda3/bin/conda run -n gnnpinn \
  python -X utf8 scripts/server/summarize_phase30_broad_process_selector_smoke.py \
  --dataset-limit 21 \
  --dataset-order process_round_robin \
  --json-output outputs/reports/phase54_broad21_claim_boundary_input_summary.json \
  --require-comparable
```

Claim-boundary summary:

```bash
PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  /home/vipuser/miniconda3/bin/conda run -n gnnpinn \
  python -X utf8 scripts/server/summarize_phase54_process_route_claim_boundary.py \
  --input outputs/reports/phase54_broad12_claim_boundary_input_summary.json \
  --input outputs/reports/phase54_broad21_claim_boundary_input_summary.json \
  --json-output outputs/reports/phase54_process_route_claim_boundary_summary.json \
  --markdown-output outputs/reports/phase54_process_route_claim_boundary_summary.md \
  --require-comparable
```

## Verification

Local:

```text
python -X utf8 -m py_compile scripts/server/summarize_phase54_process_route_claim_boundary.py tests/test_phase54_claim_boundary_summary.py
python -X utf8 -m pytest -q tests/test_phase54_claim_boundary_summary.py --basetemp .pytest_phase54_claim_boundary
python -X utf8 -m pytest -q tests/test_phase30_summary.py tests/test_phase54_claim_boundary_summary.py --basetemp C:\pt54
git diff --check
```

Results:

```text
4 passed
18 passed
git diff --check passed except the existing README CRLF warning
```

Server:

```text
tests/test_phase54_claim_boundary_summary.py: 4 passed
```

## Claim-Boundary Summary

| Classification | Splits |
| --- | --- |
| `paper_claim_positive` | `broad12:line`, `broad12:spot_size`, `broad21:line`, `broad21:spot_size` |
| `route_guard_positive` | `broad12:laser_power`, `broad12:process`, `broad12:scan_speed`, `broad21:laser_power`, `broad21:process`, `broad21:scan_speed` |
| `incomplete_metric` | none |
| `diagnostic_negative` | none |
| `incomparable` | none |

Detailed table:

| Dataset | Split | Class | Route | Test RMSE vs best strong baseline | Hot q90 RMSE vs best strong baseline | Gradient q90 RMSE vs best strong baseline |
| --- | --- | --- | --- | ---: | ---: | ---: |
| broad12 | `laser_power` | route guard | concat / global-standard | `140.753534` vs mean `132.965887` | `254.473291` vs mean `242.427068` | `215.411533` vs mean `208.105836` |
| broad12 | `line` | paper claim, no-process fallback | none | `126.308616` vs mean `134.042138` | `217.257126` vs kNN-coords `228.525979` | `195.314294` vs kNN-coords `210.675696` |
| broad12 | `process` | route guard | none | `181.091525` vs mean `147.381589` | `325.205379` vs mean `251.032500` | `266.149257` vs mean `213.464819` |
| broad12 | `scan_speed` | route guard | none | `186.173938` vs mean `145.115776` | `345.736994` vs mean `250.659348` | `266.380605` vs mean `209.791354` |
| broad12 | `spot_size` | paper claim | FiLM / global-standard | `136.309183` vs mean `151.850578` | `165.228535` vs mean `252.554440` | `169.049295` vs mean `233.119660` |
| broad21 | `laser_power` | route guard | concat / global-standard | `178.040331` vs mean `131.741364` | `296.909567` vs mean `237.730958` | `254.954359` vs mean `205.133029` |
| broad21 | `line` | paper claim, no-process fallback | none | `126.194921` vs mean `131.161929` | `234.351122` vs mean `243.188033` | `205.642173` vs mean `214.217962` |
| broad21 | `process` | route guard | none | `166.231596` vs mean `145.350346` | `308.389105` vs mean `248.754243` | `251.049837` vs mean `216.442403` |
| broad21 | `scan_speed` | route guard | none | `227.128663` vs mean `144.014351` | `392.018079` vs mean `251.407139` | `304.940054` vs mean `212.910489` |
| broad21 | `spot_size` | paper claim | FiLM / global-standard | `147.389475` vs mean `149.185412` | `163.081706` vs mean `251.976794` | `177.908136` vs mean `231.072566` |

## Interpretation

The clean process-conditioned paper-facing positives are broad12 and broad21 `spot_size`: FiLM/global-standard under `broad_process_v1` beats the best strong baseline and no-process Macro PINN on global, hot-zone, and gradient-band RMSE.

The `line` positives are no-process fallback results. They can support the route-guard contribution because the selector avoids harmful process conditioning, but they should not be written as evidence that process conditioning improves line holdout.

The `laser_power`, `process`, and `scan_speed` rows are route-guard evidence only. They are useful for showing that `broad_process_v1` records and avoids negative-transfer routes, but they do not beat the strongest classical baseline boundary.

The broad21 `spot_size` hot-q90 value was regenerated under the same manifest/split comparability gate after fixing a floating-point quantile boundary issue in `region_metric_tables`. The original region selector produced `n_points=0` because the interpolated q90 threshold exceeded the observed maximum by a tiny floating-point amount. The helper now clamps quantile thresholds to the observed min/max before selecting region points.

## Decision

Close Phase 54 as paper-facing claim-boundary consolidation.

Do not run source-inversion broad12/broad21 validation under the current AMB2022-03 source-path data. Do not seed-expand route-guard-only axes. The next step is manuscript-facing table and figure planning around `broad_process_v1`, with broad12/broad21 `spot_size` as the process-conditioned strong-baseline-positive evidence.

## Artifacts

```text
outputs/reports/phase54_broad12_claim_boundary_input_summary.json
outputs/reports/phase54_broad21_claim_boundary_input_summary.json
outputs/reports/phase54_broad21_spot_size_hot_q90_input_summary.json
outputs/reports/phase54_process_route_claim_boundary_summary.json
outputs/reports/phase54_process_route_claim_boundary_summary.md
docs/results/ambench_multiline_process_route_claim_boundary_v1.md
```
