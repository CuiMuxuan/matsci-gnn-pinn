# Phase 141 Matbench Experimental Is-Metal Baseline Gate

- Status: `phase141_matbench_expt_is_metal_ready_focused_review`
- Rows: `4921`
- Selected target: `is_metal`
- Focused review allowed: `True`
- Model training allowed: `False`
- A100 training allowed now: `False`

## Review

| Target | Status | Best profile | Best method | Val BA | Test BA | Negative profile | Shortcut blocks | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| is_metal | ready_focused_review | composition_descriptors | extra_trees | 0.897955 | 0.883607 | dominant_element_shortcut | False | composition profile beats majority and shortcut controls |
