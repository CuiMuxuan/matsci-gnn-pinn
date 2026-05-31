# AM-Bench Multi-Line Residual Backbone v1

## Context

- Phase: 38.
- Branch: residual Macro PINN backbone under the `broad_process_v1` route guard.
- Motivation: Phase 37 strong-baseline residualization mostly mirrored an overfit ExtraTrees baseline and did not expose useful smooth residual structure. The next low-risk architecture change should alter the Macro PINN backbone itself rather than add another output head or subtract a non-differentiable baseline.
- Default behavior: unchanged. The original MLP backbone remains active unless `--backbone-mode residual` is passed.

## Implementation

The training CLI now supports:

```text
--backbone-mode mlp|residual
--backbone-residual-scale
```

In residual mode:

- concat and routed-concat experts use a same-width hidden residual MLP;
- FiLM and concat-FiLM routes keep the existing input projection and add residual hidden transitions after the first hidden layer, where tensor widths match;
- metrics and checkpoints record `backbone.mode`, residual scale, hidden width, layer count, and model parameter count.

The branch intentionally preserves the `broad_process_v1` route guard. This means broad12 `spot_size` still uses the FiLM/global-standard route, while `laser_power` uses concat/global-standard.

## Commands

Focused broad12 A100 validation:

```bash
PROFILE_SPLITS="spot_size laser_power" DATASET_LIMIT=12 DATASET_ORDER=process_round_robin \
STEPS=500 N_ESTIMATORS=80 BACKBONE_RESIDUAL_SCALE=0.5 \
  bash scripts/server/run_phase38_broad_residual_backbone_a100.sh \
  > logs/phase38_broad12_residual_backbone_a100_v1.log 2>&1
```

Summary:

```bash
python scripts/server/summarize_phase30_broad_process_selector_smoke.py \
  --dataset-limit 12 \
  --dataset-order process_round_robin \
  --split spot_size \
  --split laser_power \
  --include-broad-residual-backbone \
  --json-output outputs/reports/phase38_broad12_residual_backbone_summary.json \
  --require-comparable
```

Validation before full A100 run:

```bash
python -X utf8 -m pytest -q tests/test_macro_pinn_train.py tests/test_phase30_summary.py tests/test_phase36_seed_summary.py --basetemp .pytest_tmp
python -X utf8 -m py_compile src/gnnpinn/models/pinn/coordinate_networks.py src/gnnpinn/models/pinn/macro_pinn.py src/gnnpinn/train/macro_pinn.py scripts/server/summarize_phase30_broad_process_selector_smoke.py
bash -n scripts/server/run_multiline_process_conditioned_thermal_a100.sh
bash -n scripts/server/run_phase38_broad_residual_backbone_a100.sh
```

## Decision Gate

Compare `broad_residual_backbone` against:

- mean, kNN, and ExtraTrees baselines;
- no-process Macro PINN;
- `process_axis_v1`;
- `broad_process_v1`;
- Phase 36/37 diagnostics only as negative context.

If broad12 `spot_size` or `laser_power` improves global, hot q90, and gradient q90 metrics without obvious regression, run a paired model-seed check before any broad21 expansion. If it is negative, close as a backbone diagnostic and pivot toward physically aligned microstructure/data representation or a more expressive but still train-split-safe process-conditioned architecture.

Current implementation remains small and should fit the A100-SXM4-40GB server. Do not request A100-SXM4-80GB unless a later learned encoder or coupled graph branch exceeds current GPU memory/runtime.

## Results

Focused broad12 summary:

| Split | Method | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `spot_size` | mean | 151.850578 | 252.554440 | 233.119660 | baseline |
| `spot_size` | `broad_process_v1` | 136.309183 | 165.228535 | 169.049295 | current route guard |
| `spot_size` | `res_backbone` | 136.025906 | 205.891992 | 197.838013 | tiny global gain, region regression |
| `laser_power` | mean | 132.965887 | 242.427068 | 208.105836 | strongest global baseline |
| `laser_power` | `broad_process_v1` | 140.753534 | 254.473291 | 215.411533 | current route guard |
| `laser_power` | `res_backbone` | 159.276166 | 239.054654 | 214.876025 | region gain, global regression |

The comparability gate passed and the A100-SXM4-40GB was sufficient.

## Decision

Phase 38 is closed as a negative diagnostic. The residual backbone produced a small `spot_size` global RMSE improvement but clearly worsened hot-zone and gradient-band metrics. On `laser_power`, it slightly improved region metrics but lost too much global RMSE relative to both mean and `broad_process_v1`.

Do not seed-check `res_backbone` and do not expand it to broad21. The next branch should target a process-conditioned output calibration mechanism that can adjust target scale/offset directly while preserving the current route guard and train-split discipline.
