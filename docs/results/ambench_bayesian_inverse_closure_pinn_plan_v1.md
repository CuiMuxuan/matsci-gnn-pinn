# AM-Bench Bayesian Inverse-Closure PINN Plan v1

## Status

Phase 46 is closed as a local feasibility validation phase. A lightweight probe was implemented and tested, but the result is negative for current AM-Bench/A100 expansion.

The next code-active branch should start only if the project accepts this reframing:

```text
from: global thermal-field RMSE competition
to: sparse-data inverse closure/source discovery with calibrated uncertainty
```

This is a necessary pivot after Phase 45. The prediction-stack gate showed that the current expert pool cannot produce a validation-selectable broad12 + broad21 `laser_power` improvement over the mean baseline and `broad_process_v1`. A trainable wrapper over the same experts is therefore underjustified.

## Evaluation of the Three Bayesian PINN Ideas

### 1. Bayesian inference plus adaptive sampling

Applicable, but only after narrowing the objective.

This idea should not enter as another generic Macro PINN training trick. Phases 33-45 already tested many small representation, objective, and routing changes; they repeatedly produced split-local improvements or global-vs-region tradeoffs.

The useful version is:

```text
use posterior uncertainty to select sparse observations / collocation points
in hot-zone and high-gradient regions
```

This can target the region metrics where physics and process sensitivity matter, while keeping a no-test-leakage selection protocol.

### 2. Lightweight Bayesian PINN for multiphysics or high-dimensional UQ

Not the immediate branch.

The current AM-Bench mainline is thermal/process holdout modeling. It does not yet define a high-dimensional random PDE, coupled multiphysics state, or credible prior over many uncertain material/process fields.

A broad Bayesian neural-weight framework would add inference infrastructure before the problem has identifiable uncertain variables. Keep it as a later extension after the inverse-closure branch defines physical parameters whose uncertainty is meaningful.

### 3. Bayesian PINN for interpretable scientific discovery

This is the strongest paper-facing pivot.

The project can make a clearer contribution by inferring hidden physical/source/closure parameters from sparse thermal observations, instead of trying to beat the mean baseline on every global full-field metric.

Candidate inferred quantities:

| Parameter group | Examples | Why it is useful |
| --- | --- | --- |
| Effective transport | effective thermal diffusivity or diffusion scale | Interpretable material/process response; can be validated synthetically first. |
| Heat-source scale | absorptivity-like amplitude, source width, source decay | Directly tied to laser power and melt-pool behavior. |
| Process correction | function of `laser_power_W`, `scan_speed_mm_s`, `spot_size_um` | Reuses the strongest process-conditioned evidence without forcing global route gains. |
| Sparse closure | low-dimensional source or residual coefficients | Compatible with the existing sparse closure/export path. |
| Observation noise | global or region-dependent noise scale | Enables uncertainty calibration and credible intervals. |

## Recommended Phase 47 Candidate

Name:

```text
bayesian_inverse_closure_v1
```

Core claim:

```text
A Bayesian inverse-closure PINN can recover low-dimensional hidden heat-source/closure
parameters from sparse AM-Bench thermal observations, report calibrated uncertainty,
and use posterior uncertainty to choose informative hot/gradient samples.
```

This is not a universal global-field predictor claim. Global RMSE remains reported, but the paper-facing novelty is:

- sparse-data efficiency;
- interpretable physical parameter recovery;
- calibrated posterior uncertainty;
- hot-zone / gradient-band predictive behavior.

## Minimum Model Design

Avoid full Bayesian neural weights in the first implementation. They are expensive and harder to identify.

Use:

```text
deterministic Macro PINN field model
+ Bayesian low-dimensional physical/closure parameters
+ Bayesian observation noise
+ optional uncertainty-guided sampling loop
```

First parameter vector:

```text
theta = [
  alpha_eff,          # effective diffusion scale
  q_amp,              # heat-source amplitude
  q_width,            # heat-source spatial scale
  process_gain_P,     # laser-power correction
  process_gain_v,     # scan-speed correction
  process_gain_d      # spot-size correction
]
```

Initial inference choices, from lightest to heavier:

| Method | Role |
| --- | --- |
| Laplace approximation around MAP | First local implementation; cheap enough for synthetic gate. |
| Ensemble MAP / deep ensemble over `theta` | Useful control when Laplace is unstable. |
| SVGD or HMC over `theta` only | Later, if the low-dimensional gate passes. |
| Full Bayesian neural weights | Out of scope for the first gate. |

## Gate 0: Synthetic Identifiability

Do this before spending A100 time on AM-Bench.

Synthetic task:

1. Generate a 2D heat/source field with known `theta_true`.
2. Sample sparse observations with controlled noise.
3. Fit deterministic inverse-PINN and Bayesian inverse-closure PINN.
4. Evaluate held-out field metrics and parameter posterior quality.

Pass criteria:

| Metric | Gate |
| --- | --- |
| Parameter recovery | 90% credible interval covers each identifiable `theta_true`, or posterior mean error is within a predeclared tolerance. |
| Calibration | empirical coverage is not materially worse than nominal coverage on synthetic held-out observations. |
| Sparse prediction | Bayesian method is at least competitive with deterministic inverse-PINN on global RMSE and improves uncertainty-aware region selection. |
| Stability | two additional seeds do not change the selected parameter family. |

Fail condition:

If `theta` is not identifiable synthetically, do not run AM-Bench. Reduce the parameter family or return to deterministic closure diagnostics.

## Gate 1: AM-Bench Sparse Inverse Validation

Only after Gate 0 passes.

Use broad12 and broad21 `laser_power`, matching the Phase 45 transfer pressure.

Observation protocol:

- train/validation/test split remains grouped and no-test-leakage;
- use sparse train observations, for example 5%, 10%, and 20% of training rows;
- compare random sparse sampling against uncertainty-guided sampling;
- compute metrics on the full held-out test split.

Required comparators:

| Comparator | Purpose |
| --- | --- |
| mean | strongest global baseline on broad21 `laser_power`. |
| kNN process | nonparametric process/spacetime baseline. |
| ExtraTrees process | strong tabular baseline. |
| no-process Macro PINN | neural field without process route. |
| `broad_process_v1` | current route guard. |
| deterministic inverse-PINN | separates Bayesian uncertainty benefit from inverse-closure benefit. |
| Bayesian inverse-closure PINN | candidate contribution. |

Required metrics:

```text
test RMSE
hot q90 RMSE
gradient q90 RMSE
negative log likelihood or calibration error
credible interval coverage
posterior width by region
parameter posterior mean/std
sampling efficiency curve vs observation budget
```

Pass criteria:

```text
1. Synthetic parameter recovery gate passed first.
2. AM-Bench sparse-data runs do not collapse global RMSE relative to deterministic inverse-PINN.
3. At least one sparse budget improves hot q90 and gradient q90 against random sampling on both broad12 and broad21.
4. Uncertainty coverage is credible enough to report; if coverage fails, the branch becomes a negative uncertainty-calibration result.
5. Any claim against mean / kNN / ExtraTrees / broad_process_v1 is stated by metric and budget, not generalized beyond the evidence.
```

## Gate 2: Seed Expansion

Run paired seeds only after Gate 1 passes on seed 7.

Seed expansion:

```text
seeds = 1, 2, 7
datasets = broad12, broad21
split = laser_power
sparse budgets = selected from Gate 1
```

Pass criteria:

- posterior parameter family remains stable;
- hot/gradient improvements survive seed averaging;
- uncertainty metrics remain interpretable;
- runtime remains within A100-SXM4-40GB unless inference method explicitly justifies larger hardware.

## Paper Positioning

Do not write the claim as:

```text
Bayesian PINN beats all baselines for full-field temperature prediction.
```

Use the narrower claim:

```text
Bayesian inverse-closure PINN provides interpretable low-dimensional source/closure
posteriors and improves sparse-data hot/gradient sampling efficiency under controlled
AM-Bench process holdouts.
```

This is stronger scientifically because it avoids hiding behind a mean-dominated global RMSE table, and it creates a contribution that can survive even when global field reconstruction is baseline-limited.

## Implementation Backlog

Recommended next implementation order:

1. Add a synthetic heat-source inverse benchmark generator with known `theta_true`.
2. Add deterministic inverse-closure MAP fitting over a low-dimensional `theta`.
3. Add Laplace or ensemble posterior over `theta`, not over all neural weights.
4. Add posterior predictive export and calibration metrics.
5. Add uncertainty-guided sparse sampler for hot/gradient candidate pools.
6. Add AM-Bench sparse-observation runner for broad12/broad21 `laser_power`.
7. Add summary script with synthetic recovery, sparse-budget curves, and baseline comparisons.

## References

- B-PINNs introduced Bayesian physics-informed neural networks for forward and inverse PDE problems with noisy data: https://arxiv.org/abs/2003.06097
- Bayesian physics-informed extreme learning machines target faster Bayesian inverse/forward PDE inference under noisy data: https://arxiv.org/abs/2205.06948
- Efficient Bayesian PINNs via Ensemble Kalman Inversion are relevant if MCMC/SVGD is too expensive for the first AM-Bench branch: https://www.sciencedirect.com/science/article/pii/S0021999124002559
- Adaptive/sensitivity-based PINN sampling supports the collocation-point side of the proposed uncertainty-guided sampling gate: https://www.sciencedirect.com/science/article/pii/S2405896324011108

## Local Validation Implementation

Implemented:

```text
scripts/server/phase46_bayesian_inverse_closure_probe.py
tests/test_phase46_bayesian_inverse_closure_probe.py
```

The probe deliberately avoids full Bayesian neural weights. It fits Bayesian linear posteriors over low-dimensional source/closure proxy features and compares sparse sampling strategies:

- `random`;
- `uncertainty_source`, which selects points using posterior predictive uncertainty weighted by source/hot-region prior score.

It supports two modes:

| Mode | Purpose |
| --- | --- |
| `synthetic` | Controlled heat-source parameter recovery with known source coefficients. |
| `table` | Local AM-Bench field-table proxy using low-dimensional source/closure features from coordinates and time. |

Validation commands:

```bash
PYTHONPATH=src python -X utf8 scripts/server/phase46_bayesian_inverse_closure_probe.py \
  --mode synthetic \
  --synthetic-grid 16 \
  --synthetic-frames 8 \
  --synthetic-noise-std 8.0 \
  --initial-size 48 \
  --acquisition-size 144 \
  --repeats 5 \
  --json-output outputs/reports/phase46_synthetic_bayesian_inverse_closure_probe_summary.json

PYTHONPATH=src python -X utf8 scripts/server/phase46_bayesian_inverse_closure_probe.py \
  --mode table \
  --table data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_medium_probe.csv \
  --target temperature_C \
  --split-manifest outputs/data_splits/ambench_line_0_1_temperature_medium_probe_split.json \
  --initial-size 64 \
  --acquisition-size 192 \
  --repeats 5 \
  --json-output outputs/reports/phase46_line0_1_temperature_medium_probe_bayesian_inverse_closure_summary.json
```

Targeted tests:

```bash
python -X utf8 -m pytest -q tests/test_phase46_bayesian_inverse_closure_probe.py --basetemp C:\p46pytest3
```

Result: `2 passed`.

## Local Gate Results

Metric order below follows the JSON summary fields.

### Synthetic Heat-Source Gate

Artifact:

```text
outputs/reports/phase46_synthetic_bayesian_inverse_closure_probe_summary.json
```

| Strategy | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | 90% coverage | Source recovery pass rate | Source MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 8.002149 | 7.835315 | 8.655339 | 0.913936 | 1.000000 | 1.880981 |
| uncertainty_source | 8.195683 | 7.749243 | 8.478088 | 0.907090 | 1.000000 | 1.781914 |

Interpretation:

- Positive: low-dimensional source parameters are identifiable in the controlled synthetic setting.
- Positive: uncertainty/source acquisition slightly improves hot q90 and gradient q90.
- Negative: global RMSE is slightly worse than random sampling.
- Gate status: negative for expansion, because the current success gate requires no global RMSE regression.

### Local AM-Bench `Line_0_1` Sparse Proxy

Artifact:

```text
outputs/reports/phase46_line0_1_temperature_medium_probe_bayesian_inverse_closure_summary.json
```

| Strategy | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | 90% coverage |
| --- | ---: | ---: | ---: | ---: |
| random | 171.880449 | 99.651322 | 128.046258 | 0.688421 |
| uncertainty_source | 132.956146 | 146.360946 | 131.690693 | 0.629474 |

Interpretation:

- Positive: uncertainty/source sampling substantially improves global test RMSE on this local sparse proxy.
- Negative: hot q90 RMSE worsens sharply.
- Negative: gradient q90 RMSE worsens slightly.
- Negative: predictive coverage is under-calibrated and below the declared acceptable range.
- Gate status: negative for current-work expansion.

## Final Phase 46 Decision

Phase 46 is complete, and its current validation result is a controlled negative diagnostic:

```text
Do not expand Bayesian inverse-closure PINN to A100 broad12/broad21 yet.
```

The idea is not invalid. The local result shows that low-dimensional source/closure parameters can be recovered synthetically, and uncertainty-guided sampling can move some metrics. However, under the current gate it does not jointly preserve:

- global RMSE;
- hot-zone RMSE;
- gradient-band RMSE;
- uncertainty calibration.

Therefore it is not yet a stable paper-facing contribution against the existing strong baseline/route-guard problem.

## Recommended Next Action

Do not request A100 time for Phase 46 as-is. If revisiting Bayesian PINN later, first tighten one of these pieces locally:

1. use a better heat-source feature family aligned to AM-Bench scan strategy rather than generic coordinate/time Gaussian proxies;
2. calibrate posterior noise or use conformal/validation calibration before judging coverage;
3. optimize acquisition as a multi-objective score that explicitly constrains hot q90 and gradient q90, not only uncertainty/source score;
4. compare against deterministic inverse-closure on the same sparse local proxy before moving to broad12/broad21.
