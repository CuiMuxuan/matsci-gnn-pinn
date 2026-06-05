# Phase 124 Matbench Experimental Gap Focused Review

- Status: `phase124_matbench_expt_gap_focused_review_ready_low_capacity_mechanism_gate`
- Split pass rate: `1`
- Blocking audits: `none`
- Low-capacity mechanism design allowed: `True`
- Model training allowed: `False`

## Blocking Audits

_No rows._

## Split Reviews

| Split | Pass | Best profile | Val RMSE | Test RMSE | Shortcut | NN | Zero delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| phase123_registered_split | True | chemistry_descriptors | 0.716673 | 1.0531 | False | False | 0.0560942 |
| chemistry_family_hash_0 | True | chemistry_descriptors | 0.930396 | 0.885597 | False | False | 0.0894244 |
| chemistry_family_hash_1 | True | chemistry_descriptors | 0.811407 | 0.830588 | False | False | 0.0640176 |
| chemistry_family_hash_2 | True | chemistry_descriptors | 0.891366 | 0.864868 | False | False | 0.0468666 |
| chemistry_family_hash_3 | True | chemistry_descriptors | 0.838463 | 0.922821 | False | False | 0.0168401 |
| chemistry_family_hash_4 | True | chemistry_descriptors | 0.904688 | 0.910781 | False | False | 0.0343372 |
| dominant_element_hash | True | chemistry_descriptors | 0.779216 | 1.30416 | False | False | 0.163135 |
| anion_electronegativity_bins | True | chemistry_descriptors | 1.31515 | 0.608919 | False | False | 0.0960966 |
| family_fraction_bins | True | chemistry_descriptors | 0.870639 | 0.887686 | False | False | 0.202487 |
