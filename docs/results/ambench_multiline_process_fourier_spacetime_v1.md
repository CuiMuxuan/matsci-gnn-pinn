# Phase 33: Fourier Spacetime Representation Diagnostic

## Artifacts

- Code:
  - `src/gnnpinn/models/pinn/macro_pinn.py`
  - `src/gnnpinn/train/macro_pinn.py`
  - `scripts/server/run_multiline_process_conditioned_thermal_a100.sh`
  - `scripts/server/run_phase33_broad_fourier_selector_smoke_a100.sh`
  - `scripts/server/summarize_phase30_broad_process_selector_smoke.py`
- Server logs:
  - `logs/phase33_fourier_tiny_spot_size_smoke.log`
  - `logs/phase33_broad12_fourier_selector_a100_v1.log`
  - `logs/phase33_broad12_fourier_b1_spot_size_a100_v1.log`
- Server summaries:
  - `outputs/reports/phase33_broad12_fourier_selector_summary.json`
  - `outputs/reports/phase33_tiny_spot_size_fourier_summary.json`

## Motivation

Phase 32 closed manual selector refinement: `broad_process_v1` remains the safest broad-data route guard, while `broad_process_v2` is only a broad21 line diagnostic. Phase 33 therefore tested a representation change rather than another hand-tuned route: fixed multi-scale Fourier features over the coordinate/time input.

The new mode keeps `broad_process_v1` routing unchanged and replaces only the neural spacetime basis:

```text
[x, y, z?, t] -> [raw, sin(pi * 2^k * raw), cos(pi * 2^k * raw)]
```

Default behavior remains `spacetime_encoding=raw`; Fourier features are enabled only by explicit CLI/server flags.

## Commands

Implementation validation:

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
/home/vipuser/miniconda3/bin/conda run -n gnnpinn \
  python -m pytest -q tests/test_macro_pinn_train.py tests/test_phase30_summary.py --basetemp .pytest_tmp
```

Tiny link check:

```bash
PROFILE_SPLITS=spot_size DATASET_LIMIT=12 DATASET_ORDER=process_round_robin \
STEPS=5 N_ESTIMATORS=5 FRAME_STEP=200 MAX_FRAMES=2 ROW_STEP=80 MAX_ROWS=8 \
COL_STEP=38 MAX_COLS=8 MAX_POINTS_PER_FRAME=32 \
  bash scripts/server/run_phase33_broad_fourier_selector_smoke_a100.sh \
  > logs/phase33_fourier_tiny_spot_size_smoke.log 2>&1
```

Full broad12 diagnostic:

```bash
DATASET_LIMIT=12 DATASET_ORDER=process_round_robin STEPS=500 N_ESTIMATORS=80 \
  bash scripts/server/run_phase33_broad_fourier_selector_smoke_a100.sh \
  > logs/phase33_broad12_fourier_selector_a100_v1.log 2>&1

python scripts/server/summarize_phase30_broad_process_selector_smoke.py \
  --dataset-limit 12 \
  --dataset-order process_round_robin \
  --include-broad-process-fourier \
  --json-output outputs/reports/phase33_broad12_fourier_selector_summary.json \
  --require-comparable
```

Small band-count check:

```bash
PROFILE_SPLITS=spot_size DATASET_LIMIT=12 DATASET_ORDER=process_round_robin \
STEPS=500 N_ESTIMATORS=80 SPACETIME_FOURIER_BANDS=1 FOURIER_RUN_TAG=broad_process_fourier_b1 \
  bash scripts/server/run_phase33_broad_fourier_selector_smoke_a100.sh \
  > logs/phase33_broad12_fourier_b1_spot_size_a100_v1.log 2>&1
```

The full broad12 summary passed the manifest/split comparability gate.

## Results

Lower is better. Values are test split RMSE from the comparable broad12 summary.

| Split | Mean | no-process | broad_process_v1 | Fourier/4 + broad_process_v1 | Decision |
|---|---:|---:|---:|---:|---|
| `line` | 134.042138 | 126.308616 | 126.308616 | 168.277643 | Fourier worsens the no-process fallback. |
| `laser_power` | 132.965887 | 167.614004 | 140.753534 | 189.900869 | Fourier loses the concat/global-standard gain. |
| `scan_speed` | 145.115776 | 186.173938 | 186.173938 | 199.328381 | Fourier worsens the fallback. |
| `spot_size` | 151.850578 | 206.100512 | 136.309183 | 213.024115 | Fourier destroys the strongest FiLM/global-standard route. |
| `process` | 147.381589 | 181.091525 | 181.091525 | 243.410025 | Fourier worsens the fallback. |

The band-1 spot-size check was less severe but still not competitive:

| Variant | Split | Test RMSE | Note |
|---|---|---:|---|
| Fourier/1 + broad_process_v1 | `spot_size` | 153.271041 | Better than its paired no-process run (`160.384107`) but still weaker than `broad_process_v1` (`136.309183`) and slightly worse than mean (`151.850578`). |

## Decision

Close Phase 33 as a negative representation diagnostic:

- Fourier spacetime features are implemented, tested, and artifact-recorded, but should not replace the raw coordinate/time basis.
- The broad12 all-split Fourier/4 run is worse than `broad_process_v1` on every split.
- A small band-1 spot-size probe reduces the damage but still fails to match the existing FiLM/global-standard route.
- There is no evidence to justify broad21 scaling for this branch.

The next research step should not be another fixed basis expansion. Move to a more structural branch: closure/GNN reintegration, learned residual correction under the broad selector, or a data representation that changes sampling/alignment rather than only the coordinate basis. The current A100-SXM4-40GB remained sufficient.
