# Phase 25: FiLM and Global Process-Feature Normalization

## Context

- Server: Ubuntu A100-SXM4-40GB environment recorded in `docs/server_environment_snapshot.md`.
- Starting code commit: `3b15057 Add process feature normalization controls`.
- Scripts:
  - `scripts/server/run_multiline_process_film_conditioned_a100.sh`
  - `scripts/server/run_multiline_process_film_global_feature_norm_a100.sh`
- `scripts/server/run_multiline_process_concat_global_feature_norm_a100.sh`
- `scripts/server/run_multiline_process_concat_film_global_feature_norm_a100.sh`
- `scripts/server/run_multiline_process_concat_film_limited_global_feature_norm_a100.sh`
- `scripts/server/run_phase25_process_conditioning_seed_check_a100.sh`
- `scripts/server/summarize_phase25_film_metrics.py`
- Logs:
  - `logs/ambench_multiline_process_film_conditioned_a100_v1.log`
  - `logs/ambench_multiline_process_film_global_feature_norm_a100_v1.log`
- `logs/ambench_multiline_process_concat_global_feature_norm_a100_v1.log`
- `logs/ambench_multiline_process_concat_film_global_feature_norm_a100_v1.log`
- `logs/ambench_multiline_process_concat_film_limited_global_feature_norm_a100_v1.log`
- `logs/ambench_phase25_process_conditioning_seed_check_a100_v1.log`

Phase 25 tests whether process-conditioned Macro PINN performance can be improved by structured hidden-layer modulation rather than appending process scalars only as input columns. It keeps the original concatenation path as the default and adds FiLM hidden-layer modulation through `--input-conditioning-mode film`. It also decouples process-feature normalization from coordinate/time normalization through:

```text
--input-feature-normalization same|none|minmax|standard|global_minmax|global_standard
```

The focused comparison uses three holdout splits from Phase 24:

- `line`: the original positive process-conditioned split.
- `scan_speed`: the strongest positive process-axis split for concat/minmax.
- `spot_size`: the failure case for concat/minmax.

## Single-Seed Test Metrics

Lower is better. Values below are test split metrics from seed `7`.

| Split | Method | Feature norm | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE |
| --- | --- | --- | ---: | ---: | ---: |
| `line` | no process | none | 175.127058 | 351.525048 | 323.786011 |
| `line` | concat | train minmax | 157.793227 | 316.794319 | 293.650864 |
| `line` | FiLM | train minmax | 190.264972 | 393.938252 | 359.946452 |
| `line` | FiLM | global standard | 199.636065 | 379.456343 | 349.086249 |
| `line` | concat | global standard | 183.009876 | 365.728620 | 330.926294 |
| `line` | concat + FiLM | global standard | 212.406614 | 431.198731 | 390.194323 |
| `line` | concat + FiLM, strength 0.25 | global standard | 191.129436 | 387.154130 | 349.554155 |
| `line` | mean baseline | train mean | 128.668856 | 233.448623 | 219.561369 |
| `scan_speed` | no process | none | 186.921887 | 381.225257 | 351.412319 |
| `scan_speed` | concat | train minmax | 140.459979 | 296.230056 | 274.702492 |
| `scan_speed` | FiLM | train minmax | 185.760979 | 380.490751 | 350.007010 |
| `scan_speed` | FiLM | global standard | 137.453479 | 265.516044 | 245.775762 |
| `scan_speed` | concat | global standard | 133.430469 | 205.400618 | 195.802712 |
| `scan_speed` | concat + FiLM | global standard | 140.235933 | 207.043398 | 199.639947 |
| `scan_speed` | concat + FiLM, strength 0.25 | global standard | 134.245947 | 241.842032 | 227.912462 |
| `scan_speed` | mean baseline | train mean | 134.626834 | 225.913026 | 214.310626 |
| `spot_size` | no process | none | 208.741300 | 360.475268 | 352.098361 |
| `spot_size` | concat | train minmax | 227.573411 | 382.864774 | 373.358740 |
| `spot_size` | FiLM | train minmax | 209.903434 | 360.677798 | 354.057569 |
| `spot_size` | FiLM | global standard | 142.351582 | 222.607887 | 217.520433 |
| `spot_size` | concat | global standard | 178.100236 | 316.044826 | 305.856529 |
| `spot_size` | concat + FiLM | global standard | 144.462442 | 240.660227 | 234.706128 |
| `spot_size` | concat + FiLM, strength 0.25 | global standard | 159.427222 | 278.083759 | 268.901702 |
| `spot_size` | mean baseline | train mean | 148.611388 | 255.039630 | 249.873261 |

## Focused Seed Check

The seed check reuses the generated tables and split manifests, and adds seeds `1` and `2` to the existing seed `7`. It focuses on the two strongest candidates:

- `scan_speed`: concat + global standard.
- `spot_size`: FiLM + global standard.

| Split | Method | Seeds | Test RMSE mean +/- std | Hot q90 mean +/- std | Gradient q90 mean +/- std |
| --- | --- | --- | ---: | ---: | ---: |
| `scan_speed` | no process | 7,1,2 | 186.248784 +/- 2.597410 | 380.192536 +/- 3.425596 | 350.087691 +/- 3.604586 |
| `scan_speed` | concat + global standard | 7,1,2 | 137.793553 +/- 3.850852 | 235.582309 +/- 28.850742 | 221.173551 +/- 25.259315 |
| `spot_size` | no process | 7,1,2 | 206.208221 +/- 1.921594 | 354.797393 +/- 5.008958 | 347.376995 +/- 3.631370 |
| `spot_size` | FiLM + global standard | 7,1,2 | 146.316608 +/- 8.620788 | 233.048739 +/- 30.142952 | 228.094488 +/- 27.732866 |

## Interpretation

FiLM v1 with train-fitted minmax normalization is a negative diagnostic. It does not retain the Phase 24 concat gains on `line` or `scan_speed`, and it does not improve `spot_size`. The likely issue is that single-axis grouped holdouts can make one process scalar nearly constant in the train split and extrapolating in test. This weakens FiLM generator learning when the process features are normalized only from the train split.

Global standardization is the important corrective control. It makes process-design scalars comparable across the full known experimental design and separates those scalars from coordinate/time minmax scaling.

The best conditioning mode is split dependent:

- `scan_speed`: concat + global standard is the strongest single-seed model and is stable across three seeds against no-process Macro PINN. It slightly beats the mean baseline on seed `7`, but the three-seed mean remains slightly above the mean baseline global RMSE.
- `spot_size`: FiLM + global standard is the first neural Macro PINN branch to beat the train-mean baseline on the spot-size holdout in the three-seed mean, including hot q90 and gradient q90 subsets.
- `line`: original concat + train-minmax remains the least bad process-conditioned neural model, but all neural variants trail the train-mean baseline.

The hybrid `concat_film` route is an informative negative result. Full-strength `concat_film` preserves much of the spot-size benefit but collapses on the line split. A limited FiLM correction with `--input-film-strength 0.25` reduces that collapse but still underperforms the best single-route methods and weakens the spot-size advantage. Simple additive stacking of concat and FiLM therefore is not the next best route.

## Decision

Phase 25/26 closes as a positive model-innovation gate, but not as a final universal architecture claim.

The strongest claim from this phase is axis-sensitive process conditioning: process-feature normalization is essential, and FiLM helps the spot-size failure case, while concat remains stronger for scan-speed and line splits. The failed hybrid tests suggest the next route should not blindly stack FiLM on top of concat. A better next target is split/process-aware routing or a lightweight axis-conditioned selector that chooses or gates the conditioning path based on the held-out process axis, with `global_standard` process-feature normalization retained.

No A100-SXM4-80GB server is needed yet. All Phase 25 runs fit comfortably on the current A100-SXM4-40GB.

## Artifacts

```text
outputs/reports/phase25_film_metrics_summary.json
outputs/reports/phase25_process_conditioning_seed_check_summary.json
outputs/reports/phase26_concat_film_metrics_summary.json
outputs/reports/phase26_limited_concat_film_metrics_summary.json
outputs/runs/ambench_multiline_process_temperature_*_film_a100_sxm4_40gb_v1_macro_pinn_minmax_*/
outputs/runs/ambench_multiline_process_temperature_*_film_global_standard_a100_sxm4_40gb_v1_macro_pinn_minmax_*/
outputs/runs/ambench_multiline_process_temperature_*_concat_global_standard_a100_sxm4_40gb_v1_macro_pinn_minmax_*/
outputs/runs/ambench_multiline_process_temperature_*_concat_film_global_standard_a100_sxm4_40gb_v1_macro_pinn_minmax_*/
outputs/runs/ambench_multiline_process_temperature_*_concat_film_strength0_25_global_standard_a100_sxm4_40gb_v1_macro_pinn_minmax_*/
outputs/runs/ambench_multiline_process_temperature_*_seed{1,2}_macro_pinn_minmax_*/
```
