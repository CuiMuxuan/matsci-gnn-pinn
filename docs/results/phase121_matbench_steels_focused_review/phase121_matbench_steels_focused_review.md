# Phase 121 Matbench Steels Focused Review

- Status: `phase121_matbench_steels_focused_review_ready_low_capacity_mechanism_gate`
- Split pass rate: `0.8`
- Blocking audits: `none`
- Low-capacity mechanism design allowed: `True`
- Model training allowed: `False`

## Blocking Audits

_No rows._

## Split Reviews

| Split | Pass | Best profile | Val RMSE | Test RMSE | Shortcut dominates |
| --- | --- | --- | --- | --- | --- |
| phase120_registered_split | True | minor_elements_only | 252.86 | 225.715 | False |
| alloy_family_hash_3 | True | core_element_fractions | 128.365 | 326.296 | False |
| alloy_family_hash_4 | True | all_element_fractions | 186.971 | 162.533 | False |
| fe_ni_co_bins | False | composition_descriptors | 170.753 | 179.08 | False |
| cr_mo_ti_bins | True | all_element_fractions | 169.34 | 199.552 | False |
