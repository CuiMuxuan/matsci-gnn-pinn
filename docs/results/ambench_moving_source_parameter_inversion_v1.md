# AM-Bench Moving-Source Parameter Inversion v1

## Status

Phase 50 is closed as a synthetic-positive but AM-Bench-local-negative diagnostic.

The branch tested a stronger target formulation after Phase 49: instead of adding more linear heat-kernel proxy features, explicitly invert low-dimensional moving-source parameters.

## Implementation

New script:

```text
scripts/server/phase50_moving_source_inversion_probe.py
```

The model searches a small nonlinear grid over interpretable source parameters:

| Parameter | Meaning |
| --- | --- |
| `start_x` | initial normalized source x-position |
| `span_x` | normalized x travel over time |
| `center_y` | mean normalized source y-position |
| `sine_y_amp` | sinusoidal y wobble amplitude |
| `core_width` | narrow source width |
| `tail_width` | broad trailing source width |
| `tail_decay` | temporal decay of trailing source |

For each nonlinear candidate, the script fits linear amplitude/background coefficients on selected train observations, selects the candidate by validation objective, and evaluates only after selection on test points.

## Verification

```bash
python -X utf8 -m py_compile scripts/server/phase50_moving_source_inversion_probe.py tests/test_phase50_moving_source_inversion_probe.py
PYTHONPATH=src python -X utf8 -m pytest -q tests/test_phase50_moving_source_inversion_probe.py --basetemp C:/p50pytest
```

Result:

```text
2 passed
```

## Artifacts

| Artifact | Purpose |
| --- | --- |
| `outputs/reports/phase50_synthetic_moving_source_inversion_summary.json` | synthetic moving-source inversion gate |
| `outputs/reports/phase50_line0_1_moving_source_inversion_summary.json` | local AM-Bench `Line_0_1` moving-source inversion gate |

## Results

### Synthetic

| Method | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Coverage | Parameter recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase 49 base / random | 8.002149 | 7.835315 | 8.655339 | 0.913936 | n/a |
| Phase 49 base / uncertainty_source | 8.065546 | 7.722530 | 8.527884 | 0.912469 | n/a |
| Phase 50 moving-source inversion | 8.175431 | 8.012787 | 8.291478 | 0.887042 | 1.000000 |

The nonlinear inversion recovers the synthetic moving-source parameters reliably. The selected parameters match the synthetic generator in all repeats for the main trajectory terms:

```text
start_x=0.25, span_x=0.50, center_y=0.52, sine_y_amp=0.05, core_width=0.11
```

This is a real identifiability signal, but it does not improve the synthetic predictive metrics over the simpler linear source proxy.

### Local AM-Bench `Line_0_1`

| Method | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Coverage |
| --- | ---: | ---: | ---: | ---: |
| Phase 49 base / random | 171.880449 | 99.651322 | 128.046258 | 0.688421 |
| Phase 49 base / uncertainty_source | 116.038931 | 181.367100 | 148.307919 | 0.703158 |
| Phase 49 heat_kernel / random | 163.602019 | 121.990856 | 138.126989 | 0.840000 |
| Phase 49 heat_kernel / uncertainty_source | 124.831560 | 162.742524 | 140.899931 | 0.682105 |
| Phase 50 moving-source inversion | 149.222337 | 133.745276 | 126.979221 | 0.745263 |

Local interpretation:

```text
Phase 50 improves global RMSE versus random and improves gradient q90 slightly,
but it worsens hot q90 and remains weaker than uncertainty_source on global RMSE.
Coverage is just below the acceptance band.
```

The selected local AM-Bench parameters vary across repeats, which means the current sparse `Line_0_1` table does not identify a stable moving-source parameter set under this simple grid.

## Decision

Do not expand Phase 50 to A100 broad12/broad21.

The explicit moving-source formulation is scientifically better than generic attention or linear proxy expansion because it can recover the known synthetic source parameters. However, the local AM-Bench sparse proxy still does not support a stable paper-facing claim.

The next step should pivot away from this `Line_0_1` sparse proxy as the sole gate. Better options:

- define a synthetic-to-real bridge where source parameters are validated on calibrated dense subsets before sparse holdout tests;
- use denser local AM-Bench frames for source-parameter fitting, then test sparse downsampling;
- move the paper-facing target from full-field reconstruction to parameter-identifiability plus calibrated uncertainty on a benchmark with known or externally measurable source geometry.
