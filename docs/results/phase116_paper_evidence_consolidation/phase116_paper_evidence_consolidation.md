# Phase 116 Paper Evidence Consolidation

## Gate Decision

Status: `phase116_paper_evidence_consolidation_ready_venue_unresolved`.
Evidence consolidated: `true`.
Submission ready: `false`.
Model training allowed: `false`.
A100 training allowed now: `false`.
A100-SXM4-80GB request now: `false`.

Phase 116 consolidates existing small artifacts only. It does not read raw data, run baselines, or train a model.

## Current Positive Floor

Main paper floor: `fixed-sampling broad12/broad21 spot_size under broad_process_v1, seeds 7/1/2`.

| Dataset | Metric | broad_process_v1 | Best strong baseline | Delta | Seeds |
| --- | --- | --- | --- | --- | --- |
| broad12 | Test RMSE | 136.384781987 | 151.850577987 | -15.4657960003 | 3 |
| broad12 | Hot q90 RMSE | 162.125337065 | 252.554439588 | -90.4291025223 | 3 |
| broad12 | Gradient q90 RMSE | 165.282182272 | 233.119660009 | -67.8374777369 | 3 |
| broad21 | Test RMSE | 146.002302637 | 149.185412106 | -3.18310946913 | 3 |
| broad21 | Hot q90 RMSE | 164.31388799 | 251.976794482 | -87.662906492 | 3 |
| broad21 | Gradient q90 RMSE | 174.735838739 | 231.072566056 | -56.3367273176 | 3 |

## NIST AMMT Addendum

| Branch | Blocked item | Reason | Use |
| --- | --- | --- | --- |
| melt_pool_camera_target | target_mp_mean_mean, target_mp_q90_mean, target_mp_max_mean, target_mp_max_range, target_mp_temporal_mean_range | Phase 113 focused review found validation/test reversal versus mean guard | appendix_nist_ammt_diagnostic |
| gcode_strategy_source | target_intensity_std:blocked_layer_time_strategy_shortcut; target_center_periphery_contrast:blocked_no_gain_over_xypt_guard; target_grid_mean_range:blocked_no_baseline_visible_gap; target_quadrant_contrast:blocked_no_baseline_visible_gap | Phase 114 found no candidate target after XYPT guard and shortcut checks | appendix_nist_ammt_diagnostic |
| nist_ammt_diagnostics | new NIST AMMT main-text model claim | Phase 111, Phase 113, and Phase 114 all close as diagnostic or negative branches | appendix_nist_ammt_diagnostic |
| compute | A100-SXM4-80GB escalation | no seed-positive model branch or 40GB memory/runtime blockage exists | appendix_nist_ammt_diagnostic |

## Submission Blockers

| Blocker | Priority | Category | Needed input |
| --- | --- | --- | --- |
| P92-MANUAL-001 | P0 | target venue or author guide | one named journal/conference with author instructions URL or local PDF |
| P92-MANUAL-002 | P0 | accepted target-near benchmark papers | at least 3 usable papers; 5 preferred; 10 maximum for this review package |
| P92-MANUAL-004 | P0 | No target venue, author guide, or accepted-paper benchmark set has been provided. | resolve Phase 89 manual verification queue |
| P92-MANUAL-005 | P0 | No target venue, author guide, or accepted-paper benchmark set has been provided. | resolve Phase 90 venue blocker queue |

## Next Action

use this package for manuscript/appendix drafting, or start a fresh baseline-first data-source intake; do not train from closed NIST AMMT branches
