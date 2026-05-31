# AM-Bench Multi-Line Target Residual Baseline v1

## Context

- Phase: 37.
- Branch: strong-baseline residualized Macro PINN.
- Motivation: Phase 36 process-neighborhood RBF graph features produced one broad12 `laser_power` signal but failed the broader broad21 seed check. Recent branches repeatedly show single-seed or split-local improvements that do not survive stability checks.
- Baseline route guard: `broad_process_v1`.
- Default behavior: unchanged. Target residual training is disabled unless `--target-residual-baseline` is set.

## Implementation

The training CLI now supports fitting a train-split baseline, training Macro PINN on residual targets, and adding the baseline back for metrics:

```text
--target-residual-baseline none|mean|knn|extra_trees
--target-residual-baseline-feature-column
--target-residual-baseline-n-neighbors
--target-residual-baseline-n-estimators
--target-residual-baseline-random-state
```

When enabled:

1. A baseline is fit only on the optimization train split.
2. Macro PINN trains on `target - baseline_prediction`.
3. Target normalization is fit on the residual target.
4. Predictions add the baseline back before split and region metrics are computed.

Metrics and checkpoints record:

- baseline strategy;
- feature columns;
- fit split and fit point count;
- baseline hyperparameters;
- train residual RMSE;
- `target_normalization.target_space = residual`.

The current implementation is data-only by design: `--target-residual-baseline` rejects positive `--pde-weight`, because differentiating through a non-differentiable tree/kNN baseline would make PDE residual semantics unclear.

## Commands

Focused broad12 A100 validation:

```bash
PROFILE_SPLITS="spot_size laser_power" DATASET_LIMIT=12 DATASET_ORDER=process_round_robin \
STEPS=500 N_ESTIMATORS=80 \
  bash scripts/server/run_phase37_broad_target_residual_a100.sh \
  > logs/phase37_broad12_target_residual_et_a100_v1.log 2>&1
```

Summary:

```bash
python scripts/server/summarize_phase30_broad_process_selector_smoke.py \
  --dataset-limit 12 \
  --dataset-order process_round_robin \
  --split spot_size \
  --split laser_power \
  --include-broad-target-residual \
  --json-output outputs/reports/phase37_broad12_target_residual_summary.json \
  --require-comparable
```

Validation before full A100 run:

```bash
python -X utf8 -m pytest -q tests/test_macro_pinn_train.py tests/test_phase30_summary.py tests/test_phase36_seed_summary.py --basetemp .pytest_tmp
python -X utf8 -m py_compile src/gnnpinn/train/macro_pinn.py scripts/server/summarize_phase30_broad_process_selector_smoke.py scripts/server/summarize_phase36_process_graph_seed_check.py
bash -n scripts/server/run_multiline_process_conditioned_thermal_a100.sh
bash -n scripts/server/run_phase37_broad_target_residual_a100.sh
```

## Results

Focused broad12 summary:

| Split | Method | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `spot_size` | mean | 151.850578 | 252.554440 | 233.119660 | baseline |
| `spot_size` | `broad_process_v1` | 136.309183 | 165.228535 | 169.049295 | current route guard |
| `spot_size` | ExtraTrees process | 207.831140 | 358.040079 | 326.972947 | strong baseline failed on this split |
| `spot_size` | `target_resid_et` | 207.575682 | 357.674336 | 326.647214 | mirrors weak ExtraTrees baseline |
| `laser_power` | mean | 132.965887 | 242.427068 | 208.105836 | strongest global baseline |
| `laser_power` | `broad_process_v1` | 140.753534 | 254.473291 | 215.411533 | current route guard |
| `laser_power` | ExtraTrees process | 172.115853 | 339.998010 | 265.481391 | train residual RMSE was 0 |
| `laser_power` | `target_resid_et` | 172.134324 | 340.026811 | 265.502092 | adds no useful residual structure |

The residualized Macro PINN did not meet the seed-check gate. On `spot_size`, it stayed close to the weak ExtraTrees baseline and far behind `broad_process_v1`. On `laser_power`, the ExtraTrees train residual RMSE was `0.000000`, indicating that the baseline overfit the train split and left no smooth residual structure for Macro PINN to learn.

## Decision

Compare `broad_target_residual` against:

- mean, kNN, and ExtraTrees baselines;
- no-process Macro PINN;
- `process_axis_v1`;
- `broad_process_v1`;
- Phase 36 process graph RBF only as negative context.

Phase 37 is closed as a negative diagnostic. The implementation remains useful as a reproducible control, but ExtraTrees residualization should not be expanded to paired seed checks or broad21. The next branch should adjust Macro PINN backbone structure directly or revisit physically aligned microstructure/data representation rather than subtracting an overfit non-differentiable baseline.

Current implementation remains small and should fit the A100-SXM4-40GB server. Do not request A100-SXM4-80GB unless later learned encoders or larger coupled graph branches exceed the current GPU.
