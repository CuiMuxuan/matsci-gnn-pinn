# AM-Bench Multi-Line Output Affine v1

## Context

- Phase: 39.
- Branch: process-conditioned output affine calibration under the `broad_process_v1` route guard.
- Motivation: Phase 38 residual backbone gave split-local tradeoffs but no stable improvement. Prior phases show process features are useful on `spot_size` and `laser_power`, but hidden-network changes often move global and hot/gradient metrics in opposite directions. This branch tests a smaller hypothesis: use process features to calibrate the Macro PINN output scale and offset directly.
- Default behavior: unchanged. Output affine calibration is disabled unless `--output-affine-mode linear` is passed.

## Implementation

The training CLI now supports:

```text
--output-affine-mode none|linear
--output-affine-scale
--output-affine-lr
```

When enabled, a zero-initialized linear head maps the active normalized process/input features to `(gamma, beta)` and applies the correction in the normalized training target space:

```text
prediction <- (1 + scale * gamma) * prediction + scale * beta
```

Because the head is zero-initialized, the branch starts as identity and can be compared directly against `broad_process_v1`. Metrics and checkpoints record mode, input dimension, scale, learning rate, parameter count, and identity initialization.

## Commands

Focused broad12 A100 validation:

```bash
PROFILE_SPLITS="spot_size laser_power" DATASET_LIMIT=12 DATASET_ORDER=process_round_robin \
STEPS=500 N_ESTIMATORS=80 OUTPUT_AFFINE_SCALE=0.5 \
  bash scripts/server/run_phase39_broad_output_affine_a100.sh \
  > logs/phase39_broad12_output_affine_a100_v1.log 2>&1
```

Summary:

```bash
python scripts/server/summarize_phase30_broad_process_selector_smoke.py \
  --dataset-limit 12 \
  --dataset-order process_round_robin \
  --split spot_size \
  --split laser_power \
  --include-broad-output-affine \
  --json-output outputs/reports/phase39_broad12_output_affine_summary.json \
  --require-comparable
```

Validation before full A100 run:

```bash
python -X utf8 -m pytest -q tests/test_macro_pinn_train.py tests/test_phase30_summary.py tests/test_phase36_seed_summary.py --basetemp .pytest_tmp
python -X utf8 -m py_compile src/gnnpinn/train/macro_pinn.py scripts/server/summarize_phase30_broad_process_selector_smoke.py
bash -n scripts/server/run_multiline_process_conditioned_thermal_a100.sh
bash -n scripts/server/run_phase39_broad_output_affine_a100.sh
```

## Decision Gate

Compare `broad_output_affine` against:

- mean, kNN, and ExtraTrees baselines;
- no-process Macro PINN;
- `process_axis_v1`;
- `broad_process_v1`;
- Phase 38 residual backbone only as negative context.

If broad12 `spot_size` or `laser_power` improves global, hot q90, and gradient q90 metrics without an obvious regression, run a paired model-seed check before any broad21 expansion. If it is negative, close as an output-calibration diagnostic and pivot toward a stronger data representation or process-conditioned architecture.

Current implementation remains small and should fit the A100-SXM4-40GB server. Do not request A100-SXM4-80GB unless a later learned encoder or coupled graph branch exceeds current GPU memory/runtime.

## Results

Focused broad12 validation:

| Dataset/Split | Method | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Signal |
| --- | --- | ---: | ---: | ---: | --- |
| broad12 `spot_size` | `broad_process_v1` | 136.309183 | 165.228535 | 169.049295 | baseline |
| broad12 `spot_size` | `broad_output_affine` | 137.814723 | 170.105606 | 173.564283 | negative |
| broad12 `laser_power` | `broad_process_v1` | 140.753534 | 254.473291 | 215.411533 | baseline |
| broad12 `laser_power` | `broad_output_affine` | 139.161435 | 238.174812 | 207.673483 | positive |

The `laser_power` result passed the focused gate, so a paired seed check was run for seeds 7, 1, and 2:

| Method | n | Test RMSE mean +/- std | Hot q90 mean +/- std | Gradient q90 mean +/- std |
| --- | ---: | ---: | ---: | ---: |
| `broad_process_v1` | 3 | 148.172067 +/- 13.324303 | 296.570673 +/- 40.916567 | 242.768775 +/- 27.486302 |
| `broad_output_affine` | 3 | 136.412542 +/- 19.762516 | 267.616397 +/- 52.851692 | 224.432279 +/- 33.740885 |

Delta `broad_output_affine - broad_process_v1`: test `-11.759525`, hot q90 `-28.954276`, gradient q90 `-18.336496`.

Broad21 transfer check on `laser_power`:

| Dataset/Split | Method | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Signal |
| --- | --- | ---: | ---: | ---: | --- |
| broad21 `laser_power` | `broad_process_v1` | 178.040331 | 296.909567 | 254.954359 | baseline |
| broad21 `laser_power` | `broad_output_affine` | 210.939830 | 407.352779 | 326.056523 | negative |

## Decision

Phase 39 closes as a local-positive but non-transferable output-calibration diagnostic. The zero-initialized output affine head produced a stable broad12 `laser_power` improvement across three seeds, but the same setting failed the broader all-21-line transfer check. Do not claim output-affine calibration as a paper-facing model innovation yet, and do not expand to broad21 seed checks.

The next branch should either regularize/bound the calibration more strongly for transfer, or pivot to a different process-conditioned architecture/data representation that improves broad21 rather than only broad12.
