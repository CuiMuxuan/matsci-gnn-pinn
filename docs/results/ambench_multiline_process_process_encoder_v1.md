# AM-Bench Multi-Line Process Encoder v1

## Context

- Phase: 43.
- Branch: transfer-stable process representation or architecture pivot.
- Motivation: Phase 42 closed simple validation selection and scalar prediction anchoring as split-local diagnostics. The next branch should change how process information enters the Macro PINN rather than adding another output shrinkage term.
- Hypothesis: a low-rank trainable process encoder can start from the stable raw-process route guard while learning controlled corrections from physics-derived `am_energy_v1` features.

## Implementation

The Macro PINN now supports an optional process encoder:

```text
--input-process-encoder-mode none|linear
--input-process-encoder-dim <int>
```

Default behavior is unchanged with `none`. The first enabled mode, `linear`, is initialized to copy the leading input features into the output latent and zero all other weights. For the Phase 43 focused run, the active feature input is:

```text
laser_power_W scan_speed_mm_s spot_size_um + am_energy_v1
```

and the encoder compresses the 7 normalized features to a 3-dimensional latent before the existing `broad_process_v1` conditioning route. Metrics and checkpoints record `process_encoder` metadata, including input dimension, output dimension, initialization, and parameter count.

## Commands

Focused broad12 + broad21 `laser_power` validation:

```bash
PROFILE_SPLITS=laser_power DATASET_LIMITS="12 21" DATASET_ORDER=process_round_robin \
PROCESS_DERIVED_FEATURE_MODE=am_energy_v1 PROCESS_ENCODER_MODE=linear PROCESS_ENCODER_DIM=3 \
PROCESS_FEATURE_TAG=proc_enc STEPS=500 N_ESTIMATORS=80 \
  bash scripts/server/run_phase43_broad_process_encoder_a100.sh \
  > logs/phase43_laser_power_process_encoder_a100_v1.log 2>&1
```

Summaries:

```bash
python scripts/server/summarize_phase30_broad_process_selector_smoke.py \
  --dataset-limit 12 \
  --dataset-order process_round_robin \
  --split laser_power \
  --include-broad-process-encoder \
  --json-output outputs/reports/phase43_broad12_laser_power_process_encoder_summary.json \
  --require-comparable

python scripts/server/summarize_phase30_broad_process_selector_smoke.py \
  --dataset-limit 21 \
  --dataset-order process_round_robin \
  --split laser_power \
  --include-broad-process-encoder \
  --json-output outputs/reports/phase43_broad21_laser_power_process_encoder_summary.json \
  --require-comparable
```

## Decision Gate

Compare `broad_process_encoder` against mean, kNN, ExtraTrees, no-process Macro PINN, `process_axis_v1`, `broad_process_v1`, derived-only `am_energy_v1`, and Phase 42 prediction-anchor diagnostics. Continue only if it improves or preserves global RMSE while improving hot q90 and gradient q90 on both broad12 and broad21 `laser_power`. If it is only broad12-local or broad21-local, close it as another representation diagnostic before any seed expansion.

## Results

Focused broad12/broad21 `laser_power` validation:

| Dataset | Method | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Signal |
| --- | --- | ---: | ---: | ---: | --- |
| broad12 | `mean` | 132.965887 | 242.427068 | 208.105836 | strongest baseline |
| broad12 | `broad_process_v1` | 140.753534 | 254.473291 | 215.411533 | route guard |
| broad12 | `broad_process_encoder` | 189.137331 | 369.311362 | 293.900869 | negative |
| broad21 | `mean` | 131.741364 | 237.730958 | 205.133029 | strongest baseline |
| broad21 | `broad_process_v1` | 178.040331 | 296.909567 | 254.954359 | route guard |
| broad21 | `broad_process_encoder` | 172.459317 | 264.292100 | 237.096411 | positive vs route guard |
| broad21 | derived-only `am_energy_v1` | 171.892969 | 211.624381 | 207.270255 | stronger broad21 diagnostic |

The encoder is broad21-positive but broad12-negative. It improves broad21 `laser_power` global/hot/gradient metrics against `broad_process_v1`, but it is still weaker than the prior derived-only broad21 diagnostic and fails badly on broad12.

## Decision

Close Phase 43 `process_encoder_v1` as a representation diagnostic, not a paper-facing claim. The branch confirms that trainable raw+derived process fusion can move broad21 in the right direction, but it does not solve transfer stability across broad12 and broad21.

Do not seed-expand this branch. The next branch should target the data/objective side of the transfer split, especially process-group balance or train-split weighting by process condition, before adding more process encoders.
