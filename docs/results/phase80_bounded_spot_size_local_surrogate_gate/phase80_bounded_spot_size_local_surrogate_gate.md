# Phase 80 Bounded Spot-Size Local Surrogate Gate

## Gate Decision

Status: `blocked_by_local_surrogate_gate`.
Selected variant: `broad_process_v1:train_global_bias`.
Local surrogate passed: `false`.
A100 seed-7 allowed: `false`.
A100-SXM4-80GB request now: `false`.

validation gain over identity is below the pre-declared minimum

## Surrogate Rows

| Variant | Role | Val RMSE | Val gain vs identity | Test RMSE | Test gain vs reference | Status |
| --- | --- | --- | --- | --- | --- | --- |
| broad_process_v1:identity | identity | 162.391962 | 0.000000 | 153.259459 | -13.533813 | reference |
| mean | strong_reference | 138.500717 | 23.891245 | 139.725646 | 0.000000 | reference |
| broad_process_v1:train_global_bias | candidate | 162.142446 | 0.249517 | 153.086051 | -13.360405 | insufficient_validation_gain |
| broad_process_v1:train_group_bias:spot_size_um | candidate | 162.142446 | 0.249517 | 153.086051 | -13.360405 | insufficient_validation_gain |
| broad_process_v1:train_group_bias:process_tuple | candidate | 162.142446 | 0.249517 | 153.086051 | -13.360405 | insufficient_validation_gain |

## Next Action

do not run broad12/broad21 A100 training; close this surrogate or redesign with a stronger validation-visible signal
