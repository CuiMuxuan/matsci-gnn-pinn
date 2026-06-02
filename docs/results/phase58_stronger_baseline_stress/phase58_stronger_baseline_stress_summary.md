# Phase 58 Stronger-Baseline Stress Summary

Stress gate: `claim_survives_stronger_baselines`

| dataset | metric | frozen broad_process_v1 | best baseline after stress | delta | pass |
|---|---|---:|---:|---:|---|
| broad12 | rmse | 136.384782 | 151.850578 (mean) | -15.465796 | True |
| broad12 | hot_q90_rmse | 162.125337 | 252.554440 (mean) | -90.429103 | True |
| broad12 | gradient_q90_rmse | 165.282182 | 233.119660 (mean) | -67.837478 | True |
| broad21 | rmse | 146.002303 | 149.185412 (mean) | -3.183109 | True |
| broad21 | hot_q90_rmse | 164.313888 | 251.976794 (mean) | -87.662906 | True |
| broad21 | gradient_q90_rmse | 174.735839 | 231.072566 (mean) | -56.336727 | True |

## Missing Stress Baselines

- None
