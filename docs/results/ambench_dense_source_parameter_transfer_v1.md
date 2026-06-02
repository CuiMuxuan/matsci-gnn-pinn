# AM-Bench Dense-to-Sparse Source-Parameter Transfer v1

## Status

Phase 51 is closed as a synthetic-positive but AM-Bench-dense-negative diagnostic.

The branch tested whether Phase 50 failed because sparse observations could not identify moving-source parameters. The answer is no for the current parameterization: denser AM-Bench fitting improves global RMSE and coverage, but it still trades against hot-zone error, so it does not create a stable paper-facing path.

## Implementation

New script:

```text
scripts/server/phase51_dense_source_parameter_transfer_probe.py
```

The probe compares three paths with the same validation/test split:

| Path | Meaning |
| --- | --- |
| `sparse_search` | search source parameters and fit coefficients on sparse train observations |
| `dense_params_sparse_theta` | identify source parameters with denser train observations, then refit only coefficients on sparse train observations |
| `dense_upper_bound` | identify source parameters and fit coefficients with denser train observations |

The gate is deliberately stricter than Phase 50. Coverage alone is not enough; the dense parameter route must avoid degrading global RMSE, hot q90 RMSE, and gradient q90 RMSE together.

## A100 Validation

Server:

```text
root@223.109.239.30 -p 22036
GPU: NVIDIA A100-SXM4-40GB
```

Remote preflight:

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 /home/vipuser/miniconda3/bin/conda run -n gnnpinn \
  python -X utf8 -m py_compile \
  scripts/server/phase46_bayesian_inverse_closure_probe.py \
  scripts/server/phase50_moving_source_inversion_probe.py \
  scripts/server/phase51_dense_source_parameter_transfer_probe.py \
  tests/test_phase51_dense_source_parameter_transfer_probe.py

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 /home/vipuser/miniconda3/bin/conda run -n gnnpinn \
  python -X utf8 -m pytest -q tests/test_phase51_dense_source_parameter_transfer_probe.py --basetemp /tmp/p51pytest
```

Result:

```text
2 passed
```

Dense `Line_0_1` calibrated table generated on the server:

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 /home/vipuser/miniconda3/bin/conda run -n gnnpinn \
  python -m gnnpinn.data.loaders.ambench_hdf5 \
  --sample-id amb2022_03_line_0_1_temperature_phase51_dense_local \
  --output data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_phase51_dense.csv \
  --manifest outputs/data_audits/phase51_line0_1_temperature_dense_manifest.json \
  --split-manifest outputs/data_splits/phase51_line0_1_temperature_dense_split.json \
  --calibrate-temperature \
  --split-strategy frame \
  --min-signal 100 \
  --frame-step 10 \
  --max-frames 60 \
  --row-step 4 \
  --max-rows 160 \
  --col-step 2 \
  --max-cols 152
```

Generated table:

| Item | Value |
| --- | ---: |
| Rows | 10,205 |
| Train rows | 7,786 |
| Validation rows | 1,575 |
| Test rows | 844 |
| Frame step | 10 |
| Max frames | 60 |
| Row step | 4 |
| Col step | 2 |

Phase 51 A100 commands:

```bash
PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 /home/vipuser/miniconda3/bin/conda run -n gnnpinn \
  python -X utf8 scripts/server/phase51_dense_source_parameter_transfer_probe.py \
  --mode synthetic \
  --grid-mode fast \
  --synthetic-grid 16 \
  --synthetic-frames 8 \
  --synthetic-noise-std 8.0 \
  --sparse-fit-size 192 \
  --dense-fit-size 512 \
  --repeats 5 \
  --json-output outputs/reports/phase51_synthetic_dense_source_parameter_transfer_summary.json

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 /home/vipuser/miniconda3/bin/conda run -n gnnpinn \
  python -X utf8 scripts/server/phase51_dense_source_parameter_transfer_probe.py \
  --mode table \
  --table data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_phase51_dense.csv \
  --target temperature_C \
  --split-manifest outputs/data_splits/phase51_line0_1_temperature_dense_split.json \
  --grid-mode fast \
  --sparse-fit-size 512 \
  --dense-fit-size 4096 \
  --repeats 5 \
  --json-output outputs/reports/phase51_line0_1_dense_source_parameter_transfer_summary.json
```

## Results

### Synthetic

| Path | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Coverage |
| --- | ---: | ---: | ---: | ---: |
| `sparse_search` | 7.719156 | 8.411110 | 8.336583 | 0.937897 |
| `dense_params_sparse_theta` | 7.719156 | 8.411110 | 8.336583 | 0.937897 |
| `dense_upper_bound` | 7.612343 | 7.882416 | 7.818963 | 0.942298 |

Synthetic interpretation:

```text
Dense source-parameter fitting is well behaved on the controlled generator.
The dense upper bound improves global, hot, gradient, and coverage metrics.
```

### AM-Bench `Line_0_1` Dense Table

| Path | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Coverage |
| --- | ---: | ---: | ---: | ---: |
| `sparse_search` | 132.593587 | 181.224995 | 129.999543 | 0.832464 |
| `dense_params_sparse_theta` | 132.221241 | 182.115709 | 130.153119 | 0.843128 |
| `dense_upper_bound` | 128.049763 | 185.851461 | 128.169027 | 0.876540 |

Gains versus `sparse_search`:

| Path | Global RMSE gain | Hot q90 gain | Gradient q90 gain | Coverage gain |
| --- | ---: | ---: | ---: | ---: |
| `dense_params_sparse_theta` | 0.372346 | -0.890714 | -0.153575 | 0.010664 |
| `dense_upper_bound` | 4.543824 | -4.626466 | 1.830517 | 0.044076 |

AM-Bench interpretation:

```text
Dense fitting improves global RMSE and coverage, but hot-zone error worsens.
The dense-identified parameters also do not transfer cleanly to sparse coefficient refits.
```

The formal decision payload is negative:

```text
transfer_region_ok = false
dense_upper_bound_region_ok = false
coverage_ok = true
```

## Decision

Do not expand the current normalized moving-source parameterization to broad12/broad21 or seed expansion.

Phase 51 rules out the simplest explanation that Phase 50 failed only because of sparse observations. Even with 10,205 calibrated rows and dense parameter fitting on the A100 server, the current source family still moves error into the hot region.

The next useful branch should change the physical source representation itself. The lowest-risk next gate is to align source features to the AM-Bench scan-strategy XYPT file or measured process trajectory, then test whether physically registered source-path features preserve global/hot/gradient metrics on the same dense `Line_0_1` table before any broad12/broad21 expansion.
