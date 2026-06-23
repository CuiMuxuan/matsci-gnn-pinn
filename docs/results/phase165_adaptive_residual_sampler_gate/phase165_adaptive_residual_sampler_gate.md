# Phase 165 Adaptive Residual Sampler Gate

## Gate
- Status: `phase165_adaptive_residual_sampler_ready_low_budget_pinn_smoke_design`
- Selected sampler: `failure_informed_hot_gradient`
- Best control sampler: `jittered_stratified_control`
- Validation score gain vs best control: `0.2466220238`
- Test score gain vs best control: `0.2436607143`
- Phase 166 low-budget PINN smoke design allowed: `true`
- Phase 165 model training allowed: `false`
- A100 training allowed now: `false`
- A100-SXM4-80GB request now: `false`

## Interpretation
This package evaluates point sets only. A positive result means the adaptive sampler covers high-residual, hot, and high-gradient regions better than uniform controls at the same point budget on analytic heat fields. It does not train a PINN.

## Sampler Metrics
| sampler_id | sampler_family | scenario | seed_count | point_budget | high_residual_recall_mean | hot_recall_mean | gradient_recall_mean | boundary_fraction_mean | coverage_uniformity_mean | score_mean | score_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| failure_informed_hot_gradient | adaptive | validation_nominal | 3 | 256 | 0.3764880952 | 0.2760416667 | 0.34375 | 0.21875 | 1 | 0.4344791667 | 0.005553128 |
| failure_informed_hot_gradient | adaptive | test_shifted | 3 | 256 | 0.3630952381 | 0.3072916667 | 0.3385416667 | 0.2057291667 | 1 | 0.4345833333 | 0.0009049283 |
| jittered_stratified_control | control | validation_nominal | 3 | 256 | 0.0543154762 | 0.0580357143 | 0.0558035714 | 0.1783854167 | 1 | 0.1878571429 | 0.0039991296 |
| jittered_stratified_control | control | test_shifted | 3 | 256 | 0.056547619 | 0.0647321429 | 0.0587797619 | 0.1783854167 | 1 | 0.190922619 | 0.0017352747 |
| rad_rar_d_residual_gradient | adaptive | validation_nominal | 3 | 256 | 0.2946428571 | 0.1904761905 | 0.2842261905 | 0.1145833333 | 1 | 0.3617261905 | 0.0070103765 |
| rad_rar_d_residual_gradient | adaptive | test_shifted | 3 | 256 | 0.2894345238 | 0.224702381 | 0.2790178571 | 0.1158854167 | 1 | 0.3676587302 | 0.0110938429 |
| residual_density_adaptive | adaptive | validation_nominal | 3 | 256 | 0.2916666667 | 0.228422619 | 0.2819940476 | 0.1145833333 | 1 | 0.3683333333 | 0.0091617695 |
| residual_density_adaptive | adaptive | test_shifted | 3 | 256 | 0.3311011905 | 0.2470238095 | 0.2961309524 | 0.1184895833 | 1 | 0.3929662698 | 0.0045145702 |
| uniform_grid_control | control | validation_nominal | 3 | 256 | 0.0535714286 | 0.0535714286 | 0.0513392857 | 0.2265625 | 1 | 0.1855803571 | 0 |
| uniform_grid_control | control | test_shifted | 3 | 256 | 0.0535714286 | 0.0558035714 | 0.0535714286 | 0.2265625 | 1 | 0.1865625 | 0 |
