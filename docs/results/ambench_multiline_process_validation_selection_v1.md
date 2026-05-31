# AM-Bench Multi-Line Validation Selection v1

## Context

- Phase: 42.
- Motivation: Phase 41 derived-only `am_energy_v1` improved broad21 `laser_power` Macro PINN metrics but failed broad12. Phase 42 checks whether validation metrics can select raw process scalars versus derived-only process features without looking at test metrics.
- Scope: existing broad12/broad21 `laser_power` artifacts only; no new model training.

## Command

```bash
python scripts/server/summarize_phase42_validation_selection.py \
  --json-output outputs/reports/phase42_laser_power_validation_selection_summary.json
```

## Results

Key validation/test selection outcomes:

| Dataset | Metric | Val-selected | Test-best | Match |
| ---: | --- | --- | --- | --- |
| broad12 | RMSE | raw process | raw process | yes |
| broad12 | hot q90 RMSE | derived-only | raw process | no |
| broad12 | gradient q90 RMSE | raw process | raw process | yes |
| broad21 | RMSE | raw process | derived-only | no |
| broad21 | hot q90 RMSE | raw + derived | derived-only | no |
| broad21 | gradient q90 RMSE | derived-only | derived-only | yes |

Representative RMSE values:

| Dataset | Feature set | Val RMSE | Test RMSE |
| ---: | --- | ---: | ---: |
| broad12 | raw process | 131.448453 | 140.753534 |
| broad12 | derived-only `am_energy_v1` | 138.851861 | 162.766699 |
| broad21 | raw process | 129.419509 | 178.040331 |
| broad21 | raw + `am_energy_v1` | 185.690207 | 212.704856 |
| broad21 | derived-only `am_energy_v1` | 171.690109 | 171.892969 |

## Decision

Simple validation selection is not credible enough as the next paper-facing branch. It works for broad12 global/gradient and broad21 gradient, but fails for broad21 global and hot-zone selection. A hand-coded selector based on these validation metrics would be too brittle and would risk encoding split-specific artifacts.

The next branch should pivot to a stronger baseline-facing architecture or training objective that can preserve broad21 hot/gradient gains while controlling global RMSE, rather than selecting raw versus derived features with a single validation metric.
