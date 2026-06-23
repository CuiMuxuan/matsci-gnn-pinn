# Phase 167 Low-Budget PINN Smoke

## Gate
- Status: `phase167_low_budget_pinn_smoke_closed_no_stable_model_gain`
- Selected variant: `uniform_grid_pinn`
- Best control variant: `uniform_grid_pinn`
- Validation relative gain vs best control: `0`
- Test reversal ratio vs best control: `1`
- Phase 168 focused review allowed: `false`
- Model claim allowed: `false`
- A100 training allowed now: `false`
- A100-SXM4-80GB request now: `false`

## Interpretation
This is a tiny synthetic training smoke. It can only support a later focused validation phase if the adaptive sampler PINN beats equal-budget controls. It does not support AM-Bench, Bayesian PINN, GCN, CNN/operator, or A100-SXM4-80GB claims.

## Summary Metrics
| variant_id | family | sampler_id | split | seed_count | temperature_rmse_mean | temperature_rmse_std | hot_q90_rmse_mean | gradient_q90_rmse_mean | residual_rmse_mean | selection_score_mean | selection_score_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| data_only_tiny_mlp_no_residual | control | none | val | 3 | 0.0530599302 | 0.0120796089 | 0.0829485639 | 0.0107289526 | 1.391260584 | 0.0962753229 | 0.0179532312 |
| data_only_tiny_mlp_no_residual | control | none | test | 3 | 0.0537460795 | 0.0118653373 | 0.0852090018 | 0.0117504604 | 1.40287734 | 0.0978540592 | 0.0173824472 |
| failure_informed_hot_gradient_pinn | adaptive_candidate | failure_informed_hot_gradient | val | 3 | 0.0623000557 | 0.019707404 | 0.0674921641 | 0.0132480029 | 1.127875615 | 0.0980784314 | 0.021271798 |
| failure_informed_hot_gradient_pinn | adaptive_candidate | failure_informed_hot_gradient | test | 3 | 0.0623379429 | 0.0193917964 | 0.0713232285 | 0.0135799553 | 1.125310668 | 0.0990854034 | 0.0220089735 |
| uniform_grid_pinn | control | uniform_grid_control | val | 3 | 0.0231991863 | 0.0049574202 | 0.0433924378 | 0.0076615097 | 0.5846984229 | 0.0439669985 | 0.0088339173 |
| uniform_grid_pinn | control | uniform_grid_control | test | 3 | 0.0233873529 | 0.0050067334 | 0.0438567775 | 0.0083206822 | 0.5830321114 | 0.0443451313 | 0.0089462295 |
| wrong_prior_failure_sampler_control | control | failure_informed_hot_gradient | val | 3 | 0.1253286199 | 0.0316746831 | 0.211526329 | 0.0068075883 | 3.033859682 | 0.2247392356 | 0.0336936192 |
| wrong_prior_failure_sampler_control | control | failure_informed_hot_gradient | test | 3 | 0.1245289645 | 0.0315200255 | 0.2117458553 | 0.0071352083 | 3.011773829 | 0.223712317 | 0.0347449469 |
