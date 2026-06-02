# AM-Bench Physics-Guided Closure Attention Probe v1

## Status

Phase 47 is closed as a negative local diagnostic.

The testable idea was the lightest useful version of PINN + attention:

```text
Bayesian low-dimensional inverse-closure proxy
+ deterministic physics-guided attention/gating over source-like features
```

This deliberately avoids a full CNN, GCN, Transformer, or meta-learning branch. The purpose was to check whether a simple physically motivated gate can help before spending A100 time or adding a larger architecture.

## Implementation

The probe reuses `scripts/server/phase46_bayesian_inverse_closure_probe.py`.

New option:

```bash
--feature-mode base|physics_attention
```

`physics_attention` appends gated source/closure features:

```text
attention = 0.5 * scaled(source_prior_score)
          + 0.5 * scaled(spatial_gradient(source_prior_score))
attn_feature = source_like_feature * attention
```

The gate uses source-prior information only. It does not use target temperature values to build features.

## Local Validation

Verification:

```bash
python -X utf8 -m py_compile scripts/server/phase46_bayesian_inverse_closure_probe.py tests/test_phase46_bayesian_inverse_closure_probe.py
PYTHONPATH=src python -X utf8 -m pytest -q tests/test_phase46_bayesian_inverse_closure_probe.py --basetemp C:/p47pytest
```

Result:

```text
3 passed
```

Artifacts:

| Artifact | Purpose |
| --- | --- |
| `outputs/reports/phase47_synthetic_base_closure_probe_summary.json` | synthetic base inverse-closure probe |
| `outputs/reports/phase47_synthetic_physics_attention_closure_probe_summary.json` | synthetic physics-attention probe |
| `outputs/reports/phase47_line0_1_base_closure_probe_summary.json` | local AM-Bench `Line_0_1` base sparse probe |
| `outputs/reports/phase47_line0_1_physics_attention_closure_probe_summary.json` | local AM-Bench `Line_0_1` physics-attention sparse probe |

## Results

### Synthetic

| Feature mode | Strategy | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Coverage | Source recovery |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| base | random | 8.002149 | 7.835315 | 8.655339 | 0.913936 | 1.000000 |
| base | uncertainty_source | 8.195683 | 7.749243 | 8.478088 | 0.907090 | 1.000000 |
| physics_attention | random | 8.029186 | 8.149420 | 8.846463 | 0.914425 | 1.000000 |
| physics_attention | uncertainty_source | 8.178254 | 7.893498 | 8.450195 | 0.911980 | 0.600000 |

Against the same strategy, attention gives no stable synthetic improvement. It slightly improves `uncertainty_source` global RMSE and gradient q90, but worsens hot q90 and weakens source recovery robustness.

### Local AM-Bench `Line_0_1`

| Feature mode | Strategy | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Coverage |
| --- | --- | ---: | ---: | ---: | ---: |
| base | random | 171.880449 | 99.651322 | 128.046258 | 0.688421 |
| base | uncertainty_source | 132.956146 | 146.360946 | 131.690693 | 0.629474 |
| physics_attention | random | 134.384901 | 155.581884 | 141.281181 | 0.852632 |
| physics_attention | uncertainty_source | 108.580442 | 221.871098 | 173.180299 | 0.783158 |

Attention improves global RMSE and coverage on the local table probe, but it sharply worsens the hot-zone and gradient-band metrics. This fails the current project gate because the branch cannot be paper-facing if it wins global RMSE by moving error into physically important regions.

## Decision

Do not expand this branch to CNN, GCN, Transformer attention, meta-learning, or A100 broad12/broad21 validation yet.

The easy attention/gating idea is useful as a diagnostic: it shows that source-prior gating can improve average sparse reconstruction, but the gate is not aligned with hot/gradient physical-error control. A larger attention module would be underjustified unless the feature/gate target is redesigned around region-preserving behavior first.

Next model contribution should not be another generic architecture add-on. It should first solve one of these local gates:

- multi-objective acquisition that does not sacrifice hot q90 or gradient q90;
- region-aware, no-test-leakage source/closure features;
- an inverse-discovery target where parameter recovery and calibrated uncertainty remain the main claim rather than global RMSE.
