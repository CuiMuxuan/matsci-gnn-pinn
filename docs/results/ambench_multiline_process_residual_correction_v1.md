# Phase 34: Learned Residual Correction Diagnostic

## Artifacts

- Code:
  - `src/gnnpinn/train/macro_pinn.py`
  - `scripts/server/run_multiline_process_conditioned_thermal_a100.sh`
  - `scripts/server/run_phase34_broad_residual_correction_a100.sh`
  - `scripts/server/summarize_phase30_broad_process_selector_smoke.py`
- Server logs:
  - `logs/phase34_broad12_spot_size_residual_mlp_a100_v1.log`
  - `logs/phase34_broad12_spot_size_residual_s003_a100_v1.log`
- Server summaries:
  - `outputs/reports/phase34_broad12_spot_size_residual_mlp_summary.json`

## Motivation

Phase 33 showed that fixed Fourier spacetime features degraded all broad12 splits under the conservative `broad_process_v1` route guard. The first Phase 34 sparse-closure probes also harmed the strongest current broad12 `spot_size` route:

- `broad_process_v1` spot-size baseline: test RMSE `136.309183`, hot q90 `165.228535`, gradient q90 `169.049295`.
- sparse closure probe: test RMSE `151.152357`, hot q90 `197.625638`, gradient q90 `191.064629`.
- sparse lite probe: test RMSE `158.792203`, hot q90 `258.922260`, gradient q90 `237.740336`.

The next structural branch is therefore a weak learned residual correction, not a stronger PDE/closure penalty. The branch keeps raw coordinate/time inputs and `broad_process_v1` routing, then adds a small zero-initialized MLP correction:

```text
prediction = MacroPINN(coords, time, process?) + scale * ResidualMLP([coords, time, process?])
```

Default behavior remains unchanged because `--residual-correction-mode none` is the default.

## Implementation

New training CLI options:

```text
--residual-correction-mode none|mlp
--residual-correction-hidden-dim
--residual-correction-layers
--residual-correction-scale
--residual-correction-lr
--residual-correction-start-step
```

The residual MLP consumes the same normalized coordinate/time tensors used by the Macro PINN plus any normalized process-feature columns. Its last linear layer is zero-initialized so the run starts at the base Macro PINN prediction and learns only a small correction. Metrics and checkpoints record a `residual_correction` payload, including mode, input dimension, scale, start step, learning rate, and parameter count.

## Commands

Implementation validation:

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
/home/vipuser/miniconda3/bin/conda run -n gnnpinn \
  python -m pytest -q tests/test_macro_pinn_train.py tests/test_phase30_summary.py --basetemp .pytest_tmp

python -m compileall -q src scripts/server
bash -n scripts/server/run_multiline_process_conditioned_thermal_a100.sh
bash -n scripts/server/run_phase34_broad_residual_correction_a100.sh
```

Focused broad12 `spot_size` diagnostic:

```bash
PROFILE_SPLITS=spot_size DATASET_LIMIT=12 DATASET_ORDER=process_round_robin \
STEPS=500 N_ESTIMATORS=80 \
  bash scripts/server/run_phase34_broad_residual_correction_a100.sh \
  > logs/phase34_broad12_spot_size_residual_mlp_a100_v1.log 2>&1

python scripts/server/summarize_phase30_broad_process_selector_smoke.py \
  --dataset-limit 12 \
  --dataset-order process_round_robin \
  --split spot_size \
  --include-broad-process-residual \
  --json-output outputs/reports/phase34_broad12_spot_size_residual_mlp_summary.json \
  --require-comparable
```

## Acceptance

The branch should be compared against the same manifest/split as Phase 30:

- mean, kNN, and ExtraTrees baselines;
- no-process Macro PINN;
- `broad_process_v1`;
- Phase 33 Fourier only as diagnostic context.

If the residual MLP improves `spot_size` without sacrificing hot q90 and gradient q90, scale it to `laser_power` and then all broad12 splits. If it fails on `spot_size`, close it as a negative branch and pivot to weak closure/GNN coupling or a sampling/alignment change. Current implementation does not require A100-SXM4-80GB.

## Results

The focused broad12 `spot_size` run passed the manifest/split comparability gate. Lower is better.

| Method | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Residual config | Decision |
|---|---:|---:|---:|---|---|
| `broad_process_v1` | 136.309183 | 165.228535 | 169.049295 | none | Current strongest route. |
| `broad_residual_mlp` | 136.294049 | 192.514042 | 187.270890 | scale `0.1`, lr `5e-4`, start `100` | Global RMSE improves by only `0.015134`, but hot/gradient regions degrade sharply. |
| `broad_residual_s003` | 136.327054 | 165.248033 | 169.062741 | scale `0.03`, lr `1e-4`, start `300` | Nearly preserves region metrics but gives no useful improvement. |

The first residual MLP behaves like a small global fitting correction rather than a useful thermal-region correction: it slightly reduces test RMSE while making the high-temperature and high-gradient bands worse. The weaker/later residual setting removes most of that harm but also removes the already tiny gain.

## Decision

Close learned residual correction as a negative Phase 34 branch:

- Do not scale `broad_residual_mlp` to all broad12 splits or broad21.
- Keep `--residual-correction-mode mlp` as an explicit diagnostic option because the implementation is reproducible and artifact-recorded.
- Keep `broad_process_v1` as the default broad-data route guard.
- The next branch should target the metric that actually matters here: hot-zone and gradient-band accuracy. A low-risk next step is train-split region-weighted data loss under `broad_process_v1`, rather than another unconstrained residual head or stronger sparse closure.
