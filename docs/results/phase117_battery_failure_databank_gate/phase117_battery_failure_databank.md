# Phase 117 Battery Failure Databank Gate

## Gate Decision

Status: `phase117_battery_failure_databank_gap_ready_focused_review`.
Focused review allowed: `true`.
Model training allowed: `false`.
A100 training allowed now: `false`.
A100-SXM4-80GB request now: `false`.

Phase 117 is a no-training external-data intake and strong-baseline review.

## Target Review

| Target | Best profile | Best method | Val gain | Test gain | Status |
| --- | --- | --- | --- | --- | --- |
| Corrected-Total-Energy-Yield-kJ | cell_pretest | hist_gradient_boosting | 15.639837084 | 15.601266489 | candidate_gap_ready_focused_review |
| Baseline-Plus-Heat-Loss-Total-Energy-Yield-kJ | cell_pretest | hist_gradient_boosting | 15.526171118 | 14.516058453 | candidate_gap_ready_focused_review |
| Baseline-Total-Energy-Yield-kJ | cell_pretest | hist_gradient_boosting | 14.570966641 | 14.761139929 | candidate_gap_ready_focused_review |
| Energy-Percent-Positive-Ejecta-% | cell_trigger_safe | knn | 4.069739720 | 6.647037240 | candidate_gap_ready_focused_review |
| Energy-Percent-Negative-Ejecta-% | trigger_numeric | knn | 6.548570225 | -0.861609346 | blocked_validation_test_reversal |
| Post-Test-Mass-Unrecovered-g | cell_trigger_safe | hist_gradient_boosting | 2.762002061 | 1.744400425 | candidate_gap_ready_focused_review |

## Next Action

enter focused leakage/target review before any model mechanism
