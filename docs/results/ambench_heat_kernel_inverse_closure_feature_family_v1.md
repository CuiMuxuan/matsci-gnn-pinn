# AM-Bench Heat-Kernel Inverse-Closure Feature Family v1

## Status

Phase 49 is closed as a synthetic-positive but AM-Bench-local-negative diagnostic.

The branch tested whether the Phase 48 bottleneck could be solved by changing the physical source/closure feature family rather than acquisition alone.

## Implementation

Updated script:

```text
scripts/server/phase46_bayesian_inverse_closure_probe.py
```

Added feature mode:

```bash
--feature-mode heat_kernel
```

The `heat_kernel` mode appends moving heat-source diffusion-kernel proxy features:

```text
heat_kernel_d{diffusion}_tau{decay}
source_hot_x_gradient
```

The basis approximates a Green's-function-like moving heat source with multiple diffusion scales, temporal decays, and lagged source positions. Coefficients remain linear and Bayesian; the Macro PINN training path is unchanged.

## Verification

```bash
python -X utf8 -m py_compile scripts/server/phase46_bayesian_inverse_closure_probe.py tests/test_phase46_bayesian_inverse_closure_probe.py
PYTHONPATH=src python -X utf8 -m pytest -q tests/test_phase46_bayesian_inverse_closure_probe.py --basetemp C:/p49pytest
```

Result:

```text
5 passed
```

## Artifacts

| Artifact | Purpose |
| --- | --- |
| `outputs/reports/phase49_synthetic_base_inverse_closure_summary.json` | synthetic base comparison |
| `outputs/reports/phase49_synthetic_heat_kernel_inverse_closure_summary.json` | synthetic heat-kernel comparison |
| `outputs/reports/phase49_line0_1_base_inverse_closure_summary.json` | local AM-Bench `Line_0_1` base comparison |
| `outputs/reports/phase49_line0_1_heat_kernel_inverse_closure_summary.json` | local AM-Bench `Line_0_1` heat-kernel comparison |
| `outputs/reports/phase49_synthetic_heat_kernel_region_inverse_closure_summary.json` | synthetic heat-kernel + region-aware acquisition gate |
| `outputs/reports/phase49_line0_1_heat_kernel_region_inverse_closure_summary.json` | local AM-Bench `Line_0_1` heat-kernel + region-aware acquisition gate |

## Base vs Heat-Kernel Results

### Synthetic

| Feature mode | Strategy | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Coverage | Source recovery |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| base | random | 8.002149 | 7.835315 | 8.655339 | 0.913936 | 1.000000 |
| base | uncertainty_source | 8.065546 | 7.722530 | 8.527884 | 0.912469 | 1.000000 |
| heat_kernel | random | 8.232006 | 8.841237 | 9.441283 | 0.907090 | 0.600000 |
| heat_kernel | uncertainty_source | 8.208595 | 7.927758 | 8.521494 | 0.924205 | 0.800000 |

### Local AM-Bench `Line_0_1`

| Feature mode | Strategy | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Coverage |
| --- | --- | ---: | ---: | ---: | ---: |
| base | random | 171.880449 | 99.651322 | 128.046258 | 0.688421 |
| base | uncertainty_source | 116.038931 | 181.367100 | 148.307919 | 0.703158 |
| heat_kernel | random | 163.602019 | 121.990856 | 138.126989 | 0.840000 |
| heat_kernel | uncertainty_source | 124.831560 | 162.742524 | 140.899931 | 0.682105 |

## Heat-Kernel + Region-Aware Acquisition

### Synthetic

| Strategy | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Coverage | Source recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 8.232006 | 8.841237 | 9.441283 | 0.907579 | 0.600000 |
| uncertainty_source | 8.208595 | 7.927758 | 8.521494 | 0.924205 | 0.800000 |
| region_quota_uncertainty | 8.269710 | 8.048870 | 8.580146 | 0.919804 | 0.600000 |
| pareto_source_gradient | 8.275942 | 8.076088 | 8.728812 | 0.919804 | 0.800000 |
| validation_selected_region_policy | 8.231446 | 8.066628 | 8.625961 | 0.921760 | 0.800000 |

The synthetic heat-kernel + validation-selected region policy passes its internal gate relative to heat-kernel random sampling. It slightly preserves global RMSE while improving hot and gradient metrics.

### Local AM-Bench `Line_0_1`

| Strategy | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Coverage |
| --- | ---: | ---: | ---: | ---: |
| random | 163.602019 | 121.990856 | 138.126989 | 0.915789 |
| uncertainty_source | 124.831560 | 162.742524 | 140.899931 | 0.951579 |
| region_quota_uncertainty | 224.519267 | 56.428929 | 139.401738 | 0.766316 |
| pareto_source_gradient | 209.328045 | 70.239626 | 137.989828 | 0.816842 |
| validation_selected_region_policy | 198.389759 | 74.617728 | 132.642762 | 0.825263 |

The AM-Bench local gate remains negative. Heat-kernel features expose a real region signal, but the strategies still trade global RMSE against hot/gradient metrics:

- `uncertainty_source` improves global RMSE but hurts hot/gradient.
- `region_quota_uncertainty` and `validation_selected_region_policy` improve hot/gradient but lose too much global RMSE.
- No candidate preserves global RMSE, hot q90, gradient q90, and coverage together.

## Decision

Do not expand Phase 49 to A100 broad12/broad21 yet.

The physical feature-family direction is more promising than generic attention because it creates a meaningful synthetic signal and improves some AM-Bench region metrics. However, it is not stable enough for a paper-facing model claim under the current local gate.

The next step should pivot from linear proxy features to a stronger target formulation:

- infer explicit moving-source parameters such as center offset, width, decay, and amplitude;
- fit those parameters with a bounded nonlinear inverse problem before Bayesian linearization;
- only then reintroduce uncertainty/acquisition around physically identifiable parameters.
