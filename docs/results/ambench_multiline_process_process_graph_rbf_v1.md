# AM-Bench Multi-Line Process Graph RBF v1

## Context

- Phase: 36.
- Branch: structured process-neighborhood RBF graph features under `broad_process_v1`.
- Motivation: Phase 35 scalar region-weighted loss did not survive paired seed checks. The next branch should encode process relationships structurally rather than tuning scalar loss weights.
- Baseline route guard: `broad_process_v1`.
- Default behavior: unchanged. Process graph features are disabled unless `--process-graph-feature-mode rbf` is passed.

## Implementation

The training CLI now supports optional process-neighborhood graph features:

```text
--process-graph-feature-mode none|rbf
--process-graph-feature-column
--process-graph-feature-count
--process-graph-length-scale
--process-graph-fit-scope train|global
```

When enabled, the branch builds standardized process vectors from row metadata such as `laser_power_W`, `scan_speed_mm_s`, and `spot_size_um`. It selects unique process anchors from the requested fit scope and appends RBF similarities as `process_graph_rbf_*` input features. These features can be used together with ordinary process scalars, or as graph-only features after a broad profile falls back to no-process scalars.

Metrics and checkpoints record:

- selected process columns;
- fit scope;
- requested and actual anchor count;
- RBF length scale;
- final effective input feature names;
- checkpoint `param_dim`.

## Commands

Focused broad12 A100 validation:

```bash
PROFILE_SPLITS="spot_size laser_power" DATASET_LIMIT=12 DATASET_ORDER=process_round_robin \
STEPS=500 N_ESTIMATORS=80 \
  bash scripts/server/run_phase36_broad_process_graph_rbf_a100.sh \
  > logs/phase36_broad12_process_graph_rbf_a100_v1.log 2>&1
```

Summary:

```bash
python scripts/server/summarize_phase30_broad_process_selector_smoke.py \
  --dataset-limit 12 \
  --dataset-order process_round_robin \
  --split spot_size \
  --split laser_power \
  --include-broad-process-graph-rbf \
  --json-output outputs/reports/phase36_broad12_process_graph_rbf_summary.json \
  --require-comparable
```

Validation before full A100 run:

```bash
python -X utf8 -m pytest -q tests/test_macro_pinn_train.py tests/test_phase30_summary.py --basetemp .pytest_tmp
python -X utf8 -m py_compile scripts/server/summarize_phase30_broad_process_selector_smoke.py
bash -n scripts/server/run_multiline_process_conditioned_thermal_a100.sh
bash -n scripts/server/run_phase36_broad_process_graph_rbf_a100.sh
```

## Results

Focused broad12 paired summary:

| Split | Method | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `spot_size` | `broad_process_v1` | 136.384782 | 162.125337 | 165.282182 | baseline |
| `spot_size` | `pg_rbf_global` | 148.632815 | 255.706330 | 236.036198 | negative |
| `laser_power` | `broad_process_v1` | 143.639451 | 266.975170 | 225.572273 | baseline |
| `laser_power` | `pg_rbf_global` | 140.689962 | 245.732430 | 211.178946 | promising on broad12 |

The broader broad21 `laser_power` seed check did not preserve the broad12 signal:

| Dataset | Method | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| broad21 `laser_power` | `broad_process_v1` | 168.816296 | 263.145682 | 229.087591 | baseline |
| broad21 `laser_power` | `pg_rbf_global` | 271.516427 | 257.153407 | 291.054335 | unstable |

Seed 7 looked useful for broad21 `laser_power`, but seeds 1 and 2 degraded global RMSE badly. This closes the process-neighborhood RBF branch as a diagnostic rather than a model claim.

## Decision

Compare `broad_process_graph_rbf` against:

- mean, kNN, and ExtraTrees baselines;
- no-process Macro PINN;
- `process_axis_v1`;
- `broad_process_v1`;
- Phase 35 diagnostics only as negative context.

Phase 36 is closed as a negative/unstable diagnostic. The implementation remains useful as a reproducible optional feature path, but the evidence does not support further tuning of process-RBF anchors or length scales. The next branch should test whether a strong train-split baseline can be residualized before Macro PINN training, so the neural model learns structure left unexplained by ExtraTrees/kNN/mean baselines.

Current implementation remains small and should fit the A100-SXM4-40GB server. Do not request A100-SXM4-80GB unless later learned encoders or larger coupled graph branches exceed the current GPU.
