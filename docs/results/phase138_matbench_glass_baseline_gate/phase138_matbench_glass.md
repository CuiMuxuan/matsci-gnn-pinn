# Phase 138 Matbench Glass Baseline Gate

- Status: `phase138_matbench_glass_ready_focused_review`
- Rows: `5680`
- Selected target: `gfa`
- Focused review allowed: `True`
- Model training allowed: `False`
- A100 training allowed now: `False`

## Review

| Target | Status | Best profile | Best method | Val BA | Test BA | Negative profile | Shortcut blocks | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gfa | ready_focused_review | composition_descriptors | hist_gradient_boosting | 0.770347 | 0.719546 | dominant_element_shortcut | False | safe chemistry profile beats majority and shortcut controls |
