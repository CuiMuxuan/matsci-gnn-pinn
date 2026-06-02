# AM-Bench Region-Preserving Inverse-Closure Acquisition v1

## Status

Phase 48 is closed as a negative local diagnostic.

The branch tested whether the Phase 46/47 failure mode can be fixed at the sparse-acquisition layer:

```text
Problem: global RMSE can improve while hot-zone and gradient-band errors collapse.
Hypothesis: source/gradient-aware acquisition plus validation conformal calibration can preserve regions.
```

The implementation stayed local and lightweight. It did not change the Macro PINN training path and did not introduce CNN, GCN, Transformer attention, or meta-learning.

## Implementation

Updated script:

```text
scripts/server/phase46_bayesian_inverse_closure_probe.py
```

Added acquisition strategies:

| Strategy | Meaning |
| --- | --- |
| `region_quota_uncertainty` | Reserves sparse samples for source-prior hot and source-prior-gradient regions, then fills the rest by uncertainty. |
| `pareto_source_gradient` | Scores candidates by posterior uncertainty, source prior, and source-prior gradient. |
| `validation_selected_region_policy` | Chooses among `uncertainty_source`, `region_quota_uncertainty`, and `pareto_source_gradient` using validation metrics only. |

Added calibration:

| Option | Meaning |
| --- | --- |
| `--calibration-mode conformal90` | Uses validation residual nonconformity to scale predictive intervals without test-label leakage. |

Added gate controls:

```text
--active-strategy validation_selected_region_policy
--require-region-preservation
```

## Local Verification

```bash
python -X utf8 -m py_compile scripts/server/phase46_bayesian_inverse_closure_probe.py tests/test_phase46_bayesian_inverse_closure_probe.py
PYTHONPATH=src python -X utf8 -m pytest -q tests/test_phase46_bayesian_inverse_closure_probe.py --basetemp C:/p48pytest
```

Result:

```text
4 passed
```

## Artifacts

| Artifact | Purpose |
| --- | --- |
| `outputs/reports/phase48_synthetic_region_preserving_inverse_closure_summary.json` | synthetic region-preserving acquisition gate |
| `outputs/reports/phase48_line0_1_region_preserving_inverse_closure_summary.json` | local AM-Bench `Line_0_1` region-preserving acquisition gate |

## Results

### Synthetic

| Strategy | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Coverage | Source recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 8.002149 | 7.835315 | 8.655339 | 0.914425 | 1.000000 |
| uncertainty_source | 8.065546 | 7.722530 | 8.527884 | 0.912469 | 1.000000 |
| region_quota_uncertainty | 8.208480 | 7.763360 | 8.467253 | 0.905134 | 1.000000 |
| pareto_source_gradient | 8.165143 | 7.749178 | 8.477412 | 0.908068 | 1.000000 |
| validation_selected_region_policy | 8.113055 | 7.728250 | 8.458095 | 0.910513 | 1.000000 |

Synthetic interpretation:

```text
Region-aware acquisition improves hot/gradient metrics, but still gives worse global RMSE than random.
The gate remains negative because region preservation is not enough without global sparse reconstruction.
```

### Local AM-Bench `Line_0_1`

| Strategy | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Coverage |
| --- | ---: | ---: | ---: | ---: |
| random | 171.880449 | 99.651322 | 128.046258 | 0.772632 |
| uncertainty_source | 116.038931 | 181.367100 | 148.307919 | 0.922105 |
| region_quota_uncertainty | 205.832757 | 60.956245 | 129.545936 | 0.682105 |
| pareto_source_gradient | 208.200707 | 69.184290 | 139.286502 | 0.736842 |
| validation_selected_region_policy | 205.832757 | 60.956245 | 129.545936 | 0.682105 |

Local AM-Bench interpretation:

```text
uncertainty_source improves global RMSE and coverage but hurts hot/gradient.
region_quota_uncertainty improves hot q90 strongly but worsens global RMSE, gradient q90, and coverage.
validation selection chose region_quota_uncertainty in all repeats, so validation did not recover a balanced policy.
```

## Decision

Do not run A100 broad12/broad21 for Phase 48.

Do not expand this branch into CNN/GCN/Transformer attention or meta-learning yet.

The current evidence says the remaining bottleneck is not merely the acquisition policy. The source/closure feature family still does not align global reconstruction, hot-zone accuracy, gradient-band accuracy, and calibrated uncertainty in the same local gate.

## Next Implication

The next paper-facing attempt should change the physical feature family or target formulation before adding larger neural modules. Viable next directions:

- infer a more explicit moving heat-source parameterization, such as width/center/decay, instead of only linear amplitudes;
- add a deterministic thermal Green's-function or heat-kernel feature basis before Bayesian inference;
- use a local synthetic-to-AM-Bench bridge where the hidden parameters have known physical meaning and can be validated before any broad12/broad21 run.
