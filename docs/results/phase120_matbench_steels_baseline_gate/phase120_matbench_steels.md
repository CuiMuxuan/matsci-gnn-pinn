# Phase 120 Matbench Steels Baseline Gate

- Status: `phase120_matbench_steels_gap_ready_focused_review`
- Rows: `312`
- Selected target: `yield_strength_mpa`
- Focused review allowed: `True`
- Model training allowed: `False`
- A100 training allowed now: `False`

## Review

| Target | Status | Best profile | Best method | Val RMSE | Test RMSE | Reason |
| --- | --- | --- | --- | --- | --- | --- |
| yield_strength_mpa | phase120_candidate_gap_ready_focused_review | all_element_fractions | extra_trees | 312.854 | 189.031 | composition profile beats mean and preserves test gain |
