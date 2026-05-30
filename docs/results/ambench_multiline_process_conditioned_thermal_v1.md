# Phase 23: Multi-Line Process-Conditioned Thermal Modeling

## Run Context

- Code commit used on the server: `7e22068`
- Server: Ubuntu 22.04.3, NVIDIA A100-SXM4-40GB
- Source data: AM-Bench 2022 / AMB2022-03 / `mds2-2716`
- HDF5 source: `data/raw/ambench/2022_single_track/AMB2022-03/mds2-2716/Thermography/AMB2022-03-718-AMMT-StaringCamera_Signal.h5`
- Main log: `logs/ambench_multiline_process_conditioned_thermal_a100_v1.log`
- Run id: `ambench_multiline_process_temperature_a100_sxm4_40gb_v1`

## Command

```bash
bash scripts/server/run_multiline_process_conditioned_thermal_a100.sh \
  > logs/ambench_multiline_process_conditioned_thermal_a100_v1.log 2>&1
```

The script builds a calibrated multi-line temperature field table, splits by `line_id`, runs coordinate-only and process-conditioned baselines, and compares Macro PINN training without and with process metadata input features.

## Dataset And Split

The run used seven representative single-track thermography datasets:

| Dataset | Process condition |
| --- | --- |
| `ThermalData/Line_0_1/Signal` | 285 W, 960 mm/s, 67 um |
| `ThermalData/Line_1_1_1/Signal` | 285 W, 960 mm/s, 49 um |
| `ThermalData/Line_1_2_1/Signal` | 285 W, 960 mm/s, 82 um |
| `ThermalData/Line_2_1_1/Signal` | 285 W, 1200 mm/s, 67 um |
| `ThermalData/Line_2_2_1/Signal` | 285 W, 800 mm/s, 67 um |
| `ThermalData/Line_3_1_1/Signal` | 325 W, 960 mm/s, 67 um |
| `ThermalData/Line_3_2_1/Signal` | 245 W, 960 mm/s, 67 um |

The converted table contains `14,842` calibrated `temperature_C` points. The line-held-out split has `8,087` train points, `1,884` validation points, and `4,871` test points.

Artifacts:

```text
outputs/data_audits/ambench_multiline_process_temperature_a100_sxm4_40gb_v1_manifest.json
outputs/data_splits/ambench_multiline_process_temperature_a100_sxm4_40gb_v1_split.json
```

## Baselines

Primary numbers below are split-level RMSE in degrees Celsius. Process-feature baselines add `laser_power_W`, `scan_speed_mm_s`, and `spot_size_um` to `x/y/t`.

| Method | Features | Train RMSE | Val RMSE | Test RMSE | Test hot q90 RMSE | Test gradient q90 RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Mean constant | target mean | 142.042269 | 133.503759 | 128.668856 | 233.448623 | 219.561369 |
| kNN | `x,y,t` | 61.535446 | 167.923210 | 170.889327 | 342.274178 | 319.054607 |
| kNN | `x,y,t` + process | 59.271461 | 172.729039 | 167.992301 | 347.707131 | 321.184533 |
| ExtraTrees | `x,y,t` | 12.867650 | 181.430089 | 172.329968 | 345.385134 | 320.995843 |
| ExtraTrees | `x,y,t` + process | 0.000000 | 176.232706 | 165.419834 | 343.469045 | 316.403279 |

The tree and kNN baselines improve only modestly when process metadata is appended. The mean baseline remains the best global held-out-line test RMSE in this first split, which shows the split is difficult and the current learned models still need refinement.

## Macro PINN Comparison

Both Macro PINN runs used min-max coordinate normalization. The process-conditioned run appends normalized `laser_power_W`, `scan_speed_mm_s`, and `spot_size_um` as model input features.

| Run | Input features | Train RMSE | Val RMSE | Test RMSE | Test MAE | Test Relative L2 | Test hot q90 RMSE | Test gradient q90 RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `macro_pinn_minmax_no_process_v1` | `x,y,t` | 38.949631 | 340.584340 | 175.127058 | 132.398589 | 0.144167 | 351.525048 | 323.786011 |
| `macro_pinn_minmax_process_features_v1` | `x,y,t,power,speed,spot` | 64.151252 | 177.859802 | 157.793227 | 126.442100 | 0.129897 | 316.794319 | 293.650864 |

Process conditioning clearly improves held-out-line generalization:

- Test RMSE improves from `175.127058` to `157.793227`.
- Validation RMSE improves from `340.584340` to `177.859802`.
- Test hot q90 RMSE improves from `351.525048` to `316.794319`.
- Test gradient q90 RMSE improves from `323.786011` to `293.650864`.

It also deliberately gives up some train-set fit, which is consistent with the branch goal: reduce line/process overfitting rather than memorize one thermal track.

Macro PINN artifacts:

```text
outputs/runs/ambench_multiline_process_temperature_a100_sxm4_40gb_v1_macro_pinn_minmax_no_process_v1/
outputs/runs/ambench_multiline_process_temperature_a100_sxm4_40gb_v1_macro_pinn_minmax_process_features_v1/
```

## Interpretation

This is the first branch since the real-microstructure diagnostics that produces a clean, physically meaningful positive signal: process metadata improves Macro PINN behavior on held-out line splits. That is stronger evidence than continuing to tune the four exact-line `mds2-2718` microscopy TIFF descriptors, because the signal comes from controlled AM-Bench process variation already present in `mds2-2716`.

The result is not yet a paper-facing win. The process-conditioned Macro PINN is still weaker than the simple train-mean baseline on global test RMSE, and the strong nonparametric baselines also struggle on held-out lines. The next claim should therefore be careful: process conditioning is a promising direction, not a completed model result.

## Decision

Phase 23 should close as a positive direction-selection node.

Recommended next step:

1. Keep the multi-line/process-conditioned branch as the mainline.
2. Add a small capacity/seed check for the process-conditioned Macro PINN before returning to closure or GNN conditioning.
3. Compare split construction by held-out process axis: spot-size holdout, speed holdout, and power holdout.
4. Only after the process-conditioned baseline is stable should sparse closure or graph-conditioned closure be reintroduced on the multi-line table.

Implementation follow-up: the converter now supports process-axis grouped splits (`laser_power`, `scan_speed`, `spot_size`, and full `process` condition), with the next server entry point at `scripts/server/run_multiline_process_holdout_splits_a100.sh`.
