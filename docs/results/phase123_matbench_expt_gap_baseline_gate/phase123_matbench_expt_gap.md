# Phase 123 Matbench Experimental Band-Gap Baseline Gate

- Status: `phase123_matbench_expt_gap_gap_ready_focused_review`
- Rows: `4604`
- Selected target: `gap_expt_ev`
- Focused review allowed: `True`
- Model training allowed: `False`
- A100 training allowed now: `False`

## Review

| Target | Status | Best profile | Best method | Val RMSE | Test RMSE | Shortcut blocks | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gap_expt_ev | phase123_candidate_gap_ready_focused_review | chemistry_descriptors | extra_trees | 0.716673 | 1.0531 | False | chemistry profile beats mean, preserves test gain, and is not shortcut-dominated |
