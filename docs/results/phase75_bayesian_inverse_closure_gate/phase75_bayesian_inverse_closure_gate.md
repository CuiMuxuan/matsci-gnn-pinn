# Phase 75 Bayesian Inverse-Closure Local Identifiability Gate

## Candidate

Candidate: `bayesian_inverse_closure_v1`.

A low-dimensional Bayesian inverse closure can provide uncertainty-aware adaptive sampling and interpretable source/closure coefficients, but it must preserve global, hot-zone, and gradient-band errors before broad-data training.

## Gate Decision

Status: `blocked_by_local_ambench_gate`.
Synthetic gate passed: `true`.
Local AM-Bench gate passed: `false`.
Phase 76 A100 seed-7 validation allowed: `false`.
A100-SXM4-80GB request now: `false`.

The candidate is identifiable on synthetic heat-source data, but the local AM-Bench gate still shifts error between global, hot q90, and gradient q90 metrics. Do not run broad12/broad21 A100 validation yet.

## Gate Rows

| Gate | Scope | Status | RMSE gain | Hot q90 gain | Gradient q90 gain | A100 allowed |
| --- | --- | --- | --- | --- | --- | --- |
| P75-SYNTH | synthetic_known_parameter_identifiability | positive | 0.111717 | 0.106836 | 0.294485 | false |
| P75-LOCAL | local_ambench_line0_1_region_preservation | negative | -50.403261 | 75.802315 | 19.402300 | false |

## Next Action

Close this candidate as a Phase 75 appendix diagnostic or redesign the local gate before any broad12/broad21 training.
