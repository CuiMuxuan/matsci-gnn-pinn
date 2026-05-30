# AM-Bench Real Micro Graph-Conditioned Closure v1

## Context

This experiment evaluates the first real AM-Bench optical microscopy graph features inside the sparse closure branch. It replaces synthetic coordinate/RBF graph features with sample-level features derived from one real `mds2-2718` TIFF inspection.

- Code commit: `d4a4431`
- Script: `scripts/server/run_real_micro_graph_conditioned_closure_a100.sh`
- Log: `logs/ambench_real_micro_graph_conditioned_closure_a100_v1.log`
- Thermal data: `AMB2022-03 / mds2-2716`, active hot/gradient table
- Micro data: `AMB2022-03 / mds2-2718`, `AMB2022-718-SH1-BP1-P2-L2.1-3_m.tif`
- Graph features: `data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features.jsonl`

## Command

```bash
bash scripts/server/run_real_micro_graph_conditioned_closure_a100.sh \
  > logs/ambench_real_micro_graph_conditioned_closure_a100_v1.log 2>&1
```

The script runs two configurations with the same optimizer settings as the previous sparse closure and gated graph closure branch:

- `hidden_dim=256`, `layers=4`, `lr=1e-3`
- `closure_lr=1e-5`
- `closure_start_step=1500`
- `residual_sample_size=4096`
- `pde_weight=1e-6`
- `pde_field=normalized`
- `closure_graph_gate=0.25`
- `closure_graph_l1_weight=1e-4`

## Results

Primary numbers below are test split metrics.

| Method | Test RMSE | Test MAE | Test Relative L2 | Hot q90 RMSE | Gradient q90 RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sparse closure best, `closure_lr=1e-5` | 70.494433 | - | - | 31.542155 | 64.558069 |
| Gated coordinate-RBF graph, gate 0.25 | 72.840576 | - | - | 31.919197 | 61.623894 |
| Real micro graph, `g4` | 73.728702 | 58.618317 | 0.060323 | 116.225391 | 118.335960 |
| Real micro graph, `g8` | 79.072104 | 58.244222 | 0.064695 | 29.314642 | 65.371975 |

The `g8` run is the useful signal:

- It improves hot q90 RMSE relative to the current sparse closure best: `31.542155 -> 29.314642`.
- It keeps gradient q90 close to the sparse closure best: `64.558069 -> 65.371975`.
- It worsens global test RMSE: `70.494433 -> 79.072104`.

The `g4` run is not useful for local-region performance.

## Real Micro Features

`g4` source features:

```text
image_mask_fraction
node_mask_fraction_mean
node_mask_fraction_std
node_mean_intensity_norm_mean
```

Normalized `g4` values:

```text
[0.09810225665569305, 0.09810225665569305, 1.0, 0.0]
```

`g8` source features:

```text
image_mask_fraction
node_mask_fraction_mean
node_mask_fraction_std
node_mean_intensity_norm_mean
node_std_intensity_norm_mean
node_mean_intensity_norm_std
image_mean_intensity
image_std_intensity
```

Normalized `g8` values:

```text
[0.0008020295645110309, 0.0008020295645110309, 0.0035732961259782314, 0.0005005901912227273, 0.0, 2.993918678839691e-05, 1.0, 0.5398025512695312]
```

## Exported Expressions

`g4`:

```text
0.00608346611261368*T + 1.10150722321123e-6*g0 + 1.10150722321123e-6*g1 + 3.33523621520726e-6*g2 + 0.00608346611261368*t + 0.00638867216184735*x + 0.00577363511547446*y + 0.00656284857541323
```

`g8`:

```text
0.00646064104512334*T - 1.60488843903295e-6*g3 + 1.30578860080277e-6*g5 + 0.00646064104512334*t + 0.00693934690207243*x + 0.00624990370124578*y + 0.00710102962329984
```

The graph coefficients are small but not completely suppressed. Unlike the earlier synthetic coordinate/RBF sensitivity, retaining real micro graph terms does not catastrophically damage the hot-zone metric in the `g8` configuration.

## Interpretation

This is the first positive-ish Phase 17 signal, but it is not yet a paper-ready improvement. The result says:

- Real optical microscopy features can enter the closure branch and survive expression export.
- A single-image sample-level `g8` feature vector can improve hot-zone RMSE.
- The same feature vector hurts global test RMSE, so the current alignment is too coarse.

The most likely limitation is that one global microstructure vector is broadcast to every thermal residual point. It can behave like a process/sample modifier, but it cannot yet encode local spatial variation or multiple process-condition differences.

## Next Action

Do not expand hyperparameter sweeps yet. First improve the data side:

1. Add more `mds2-2718` TIFFs across `P/L/replicate` conditions.
2. Build a multi-record graph feature JSONL.
3. Add a scalar-statistics control using the same selected features.
4. Only then rerun real-micro closure with process/sample-aware selection.

If multi-image `mds2-2718` still cannot align cleanly to thermal runs, move to `mds2-2775` or ExaCA-generated microstructures.
