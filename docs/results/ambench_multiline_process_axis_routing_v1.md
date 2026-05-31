# Phase 27: Split/Process-Aware Conditional Routing

## Context

- Server: Ubuntu A100-SXM4-40GB environment recorded in `docs/server_environment_snapshot.md`.
- Starting commit: `65d453a Add axis-sensitive process conditioning diagnostics`.
- Scripts:
  - `scripts/server/run_multiline_process_routed_global_feature_norm_a100.sh`
  - `scripts/server/run_multiline_process_axis_profile_a100.sh`
  - `scripts/server/summarize_phase25_film_metrics.py`
- Logs:
  - `logs/ambench_multiline_process_routed_global_feature_norm_a100_v1.log`
  - `logs/ambench_multiline_process_axis_profile_a100_v1.log`
- Summary artifact:
  - `outputs/reports/phase27_process_routing_metrics_summary.json`

Phase 25/26 showed that one universal conditioning mode is not stable across process axes: `scan_speed` favors concat + global process-feature standardization, `spot_size` favors FiLM + global process-feature standardization, and `line` remains strongest with the original concat/train-minmax route. Phase 27 tests two ways to make this axis sensitivity reproducible instead of selecting scripts by hand.

## Implemented Routes

### Trainable routed dual expert

`MacroPINN(conditioning_mode="routed")` trains two experts:

- concat expert: coordinates/time plus process scalars.
- FiLM expert: coordinates/time hidden layers modulated by process scalars.
- route gate: a process-feature gate outputs the FiLM expert weight.

The CLI records route prior, trainability, and gate summary in `metrics.json` and checkpoint metadata through:

```text
--input-conditioning-mode routed
--input-route-film-prior <float>
--freeze-input-route
```

### Explicit process-axis profile

`--input-conditioning-profile process_axis_v1` reads the grouped split manifest `group_key` and chooses the best-known Phase 25 route:

| Split group key | Selected route |
| --- | --- |
| `line_id` | `concat` with `same` feature normalization, i.e. train minmax from `--input-normalization minmax` |
| `scan_speed_mm_s` | `concat` with `global_standard` process-feature normalization |
| `spot_size_um` | `film` with `global_standard` process-feature normalization |
| `laser_power_W` | `concat` with `global_standard` process-feature normalization |
| `process_condition` | `concat` with `global_standard` process-feature normalization |

The profile is recorded under `input_features.conditioning_profile`, including the requested mode/norm and selected mode/norm.

## A100 Metrics

Lower is better. Values are seed `7` test split metrics.

| Split | Method | Selected route | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE |
| --- | --- | --- | ---: | ---: | ---: |
| `line` | no process | none | 175.127058 | 351.525048 | 323.786011 |
| `line` | best prior single route | concat / train-minmax | 157.793227 | 316.794319 | 293.650864 |
| `line` | routed dual expert | routed, concat prior 0.2 | 219.861259 | 414.621138 | 382.616352 |
| `line` | `process_axis_v1` | concat / train-minmax | 157.793227 | 316.794319 | 293.650864 |
| `line` | mean baseline | train mean | 128.668856 | 233.448623 | 219.561369 |
| `scan_speed` | no process | none | 186.921887 | 381.225257 | 351.412319 |
| `scan_speed` | best prior single route | concat / global-standard | 133.430469 | 205.400618 | 195.802712 |
| `scan_speed` | routed dual expert | routed, concat prior 0.2 | 150.293172 | 319.019144 | 293.844378 |
| `scan_speed` | `process_axis_v1` | concat / global-standard | 133.430469 | 205.400618 | 195.802712 |
| `scan_speed` | mean baseline | train mean | 134.626834 | 225.913026 | 214.310626 |
| `spot_size` | no process | none | 208.741300 | 360.475268 | 352.098361 |
| `spot_size` | best prior single route | FiLM / global-standard | 142.351582 | 222.607887 | 217.520433 |
| `spot_size` | routed dual expert | routed, FiLM prior 0.8 | 273.800445 | 440.170224 | 426.406790 |
| `spot_size` | `process_axis_v1` | FiLM / global-standard | 142.351582 | 222.607887 | 217.520433 |
| `spot_size` | mean baseline | train mean | 148.611388 | 255.039630 | 249.873261 |

## Interpretation

The trainable routed dual-expert architecture is a negative result. The route gate stayed close to its initialized prior (`line` FiLM gate mean `0.255271`, `scan_speed` `0.236963`, `spot_size` `0.806018`), so the failure is not mainly gate drift. The likely issue is optimization: training two fresh experts and mixing their outputs is harder than training the already-identified single route, especially with small grouped holdout tables.

The explicit `process_axis_v1` profile is not a new performance ceiling; it is a reproducibility and decision-control layer. It exactly recovers the best-known single-route behavior for the three tested axes and records the routing choice in artifacts. This prevents Phase 25's axis-sensitive conclusion from living only in shell-script naming.

The paper-facing claim should therefore be conservative:

- Positive: process-conditioned Macro PINN is axis-sensitive, and artifact-recorded process-axis routing makes the best route reproducible.
- Positive: `scan_speed` and `spot_size` each have a process-conditioned neural route that beats the train-mean baseline on seed `7`; `spot_size` was already supported by focused 3-seed checks in Phase 25.
- Negative/limited: trainable dual-expert routing is not stable enough; `line` still trails the mean baseline.

## Decision

Close Phase 27 as a useful routing/diagnostic node, not as a final universal model result.

The next aligned step is not larger blind routing. It should either:

1. expand `process_axis_v1` to `laser_power` and full `process` holdouts plus focused seed checks, or
2. broaden the thermal/process dataset and strong baselines before reintegrating closure/GNN terms.

No A100-SXM4-80GB server is needed yet. The current A100-SXM4-40GB ran all Phase 27 experiments comfortably.

## Artifacts

```text
outputs/reports/phase27_process_routing_metrics_summary.json
outputs/runs/ambench_multiline_process_temperature_*_routed_*_global_standard_a100_sxm4_40gb_v1_macro_pinn_minmax_*/
outputs/runs/ambench_multiline_process_temperature_*_process_axis_profile_a100_sxm4_40gb_v1_macro_pinn_minmax_*/
outputs/data_splits/ambench_multiline_process_temperature_*_process_axis_profile_a100_sxm4_40gb_v1_split.json
```
