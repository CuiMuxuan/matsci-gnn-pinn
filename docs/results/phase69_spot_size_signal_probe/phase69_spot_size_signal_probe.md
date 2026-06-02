# Phase 69 Spot-Size Non-Training Signal Probe

## Purpose

Phase 69 implements the Phase 68 `P68-SPOT-SIGNAL` action. It decides whether Candidate A can enter A100 seed-7 training without adding any new model run.

## Candidate A Gate

Status: `paused_no_training_signal`.
Open for seed-7 A100 gate: `false`.
A100-SXM4-80GB request now: `false`.

fixed-sampling broad12/broad21 and broad15 support the current floor, but broad21 alternate-density is a strong-baseline boundary and Phase 59 validation-selected correction falls back to the mean

## Evidence Rows

| Source | Dataset | Metric | Candidate | Best strong | Delta | Status | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_sampling_phase55 | broad12 | rmse | 136.38478198713665 | 151.85057798740885 | -15.465796000272206 | pass | supports_current_floor |
| fixed_sampling_phase55 | broad12 | hot_q90_rmse | 162.12533706531732 | 252.5544395876147 | -90.42910252229737 | pass | supports_current_floor |
| fixed_sampling_phase55 | broad12 | gradient_q90_rmse | 165.2821822720331 | 233.11966000888705 | -67.83747773685394 | pass | supports_current_floor |
| alternate_density_phase58 | broad12 | rmse | 139.08521742957473 | 140.20136221472765 | -1.116144785152926 | pass | supports_current_floor |
| alternate_density_phase58 | broad12 | hot_q90_rmse | 235.69676787671267 | 253.374499322871 | -17.677731446158333 | pass | supports_current_floor |
| alternate_density_phase58 | broad12 | gradient_q90_rmse | 221.60422184594762 | 234.02889850502885 | -12.424676659081229 | pass | supports_current_floor |
| fixed_sampling_phase55 | broad21 | rmse | 146.00230263668138 | 149.18541210581552 | -3.1831094691341377 | pass | supports_current_floor |
| fixed_sampling_phase55 | broad21 | hot_q90_rmse | 164.31388799034224 | 251.9767944823301 | -87.66290649198785 | pass | supports_current_floor |
| fixed_sampling_phase55 | broad21 | gradient_q90_rmse | 174.73583873860284 | 231.07256605617917 | -56.33672731757633 | pass | supports_current_floor |
| alternate_density_phase58 | broad21 | rmse | 153.25945521573854 | 139.72564607665817 | 13.533809139080375 | boundary | beats_no_process_but_not_strong_baseline |
| alternate_density_phase58 | broad21 | hot_q90_rmse | 270.6289215088311 | 253.1297230306504 | 17.499198478180716 | boundary | beats_no_process_but_not_strong_baseline |
| alternate_density_phase58 | broad21 | gradient_q90_rmse | 250.5199346796704 | 231.78089371200775 | 18.739040967662646 | boundary | beats_no_process_but_not_strong_baseline |
| auxiliary_panel_phase58 | broad15 | rmse | 138.85545569147268 | 151.85057798740885 | -12.99512229593617 | pass | supports_current_floor |
| auxiliary_panel_phase58 | broad15 | hot_q90_rmse | 158.62267730730227 | 252.5544395876147 | -93.93176228031243 | pass | supports_current_floor |
| auxiliary_panel_phase58 | broad15 | gradient_q90_rmse | 165.86919189244728 | 233.7323367625887 | -67.86314487014141 | pass | supports_current_floor |

## Next Action

do not train Candidate A; continue manuscript v0 audit or non-training route/data probes
