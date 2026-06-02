# Phase 95 PFHub-Style Physics Design Gate

## Gate Decision

Status: `local_design_ready_no_a100`.
Phase 96 local smoke allowed: `true`.
A100 training allowed now: `false`.
A100-SXM4-80GB request now: `false`.

Phase 95 is a design gate only. It prepares a local smoke protocol and cannot support paper claims by itself.

## Mechanism Matrix

| Mechanism | Name | Expected signal | Status |
| --- | --- | --- | --- |
| P95-MECH-001 | fixed_green_function_features | lower residual RMSE and top-gradient RMSE without global RMSE regression | eligible_for_phase96_local_smoke |
| P95-MECH-002 | bayesian_adaptive_collocation | same or better RMSE with fewer collocation points and calibrated coverage | eligible_for_phase96_local_smoke |
| P95-MECH-003 | small_meta_adaptation_probe | fewer adaptation steps to reach validation threshold | design_only_until_simpler_mechanisms_fail |

## Metric Contract

| Metric | Name | Pass rule | Guard |
| --- | --- | --- | --- |
| P95-MET-001 | global_rmse | non-worse than the strongest baseline | any global collapse closes the candidate |
| P95-MET-002 | pde_residual_rmse | improves over vanilla PINN/RBF comparator | must not hide worse prediction RMSE |
| P95-MET-003 | hot_or_top_gradient_region_rmse | improves while global RMSE is non-worse | same Phase 75 region-gain/global-collapse guard |
| P95-MET-004 | coverage_or_adaptation_efficiency | coverage within tolerance or fewer adaptation steps | does not override RMSE guards |

## Next Action

implement Phase 96 local smoke only; do not start A100 training
