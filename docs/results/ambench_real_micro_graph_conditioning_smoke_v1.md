# AM-Bench Real Micro Graph Conditioning Smoke v1

## Context

This smoke test verifies that real AM-Bench optical microscopy features can enter the sparse closure training path. It is an interface and artifact test, not a performance claim.

- Code commit: `b943c84`
- Thermal table: `data/interim/ambench/2022_single_track/AMB2022-03/ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1.csv`
- Split: `outputs/data_splits/ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1_split.json`
- Micro graph features: `data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features.jsonl`
- Run: `outputs/runs/ambench_line_0_1_temperature_hot_gradient_real_micro_smoke_v1/`

## Feature Aggregation

Server command:

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
/home/vipuser/miniconda3/bin/conda run -n gnnpinn python -m gnnpinn.data.loaders.ambench_microstructure \
  --mode aggregate \
  --inspection outputs/data_audits/ambench_mds2_2718_micrograph_inspection.json \
  --jsonl-output data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features.jsonl \
  --csv-output data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features.csv \
  --output outputs/data_audits/ambench_mds2_2718_micrograph_feature_table_manifest.json
```

Result:

| Field | Value |
| --- | --- |
| records | `1` |
| sample id | `AMB2022-718-SH1-BP1-P2-L2.1-3_m` |
| `image_mask_fraction` | `0.10018239122755984` |
| `node_mask_fraction_mean` | `0.10018239122755984` |

## Training Smoke Command

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
/home/vipuser/miniconda3/bin/conda run -n gnnpinn python -m gnnpinn.train.macro_pinn \
  --table data/interim/ambench/2022_single_track/AMB2022-03/ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1.csv \
  --target temperature_C \
  --split-manifest outputs/data_splits/ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1_split.json \
  --output-dir outputs/runs/ambench_line_0_1_temperature_hot_gradient_real_micro_smoke_v1 \
  --steps 20 \
  --hidden-dim 32 \
  --layers 2 \
  --input-normalization minmax \
  --pde-weight 1e-8 \
  --closure-mode sparse_linear \
  --closure-feature T \
  --closure-feature x \
  --closure-feature y \
  --closure-feature t \
  --closure-graph-mode real_micro \
  --closure-graph-features data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features.jsonl \
  --closure-graph-sample-id AMB2022-718-SH1-BP1-P2-L2.1-3_m \
  --closure-graph-embedding-dim 4 \
  --closure-graph-gate 0.25 \
  --closure-graph-l1-weight 1e-4 \
  --closure-lr 1e-5 \
  --residual-sample-size 256 \
  --log-every 10 \
  --hot-quantile 0.9 \
  --gradient-quantile 0.9
```

## Smoke Metrics

The run used only 20 steps, so these values only confirm executable integration:

| Split | RMSE | MAE | Relative L2 |
| --- | ---: | ---: | ---: |
| train | 148.876019 | 132.965459 | 0.121753 |
| val | 147.787448 | 133.159515 | 0.120728 |
| test | 141.167300 | 124.322921 | 0.115499 |

Real micro graph metadata recorded in `metrics.json`:

| Field | Value |
| --- | --- |
| mode | `real_micro` |
| sample id | `AMB2022-718-SH1-BP1-P2-L2.1-3_m` |
| source features | `image_mask_fraction`, `node_mask_fraction_mean`, `node_mask_fraction_std`, `node_mean_intensity_norm_mean` |
| exported graph features | `g0`, `g1`, `g2`, `g3` |
| normalized values | `[0.09810225665569305, 0.09810225665569305, 1.0, 0.0]` |
| graph nodes / edges | `64 / 512` |

Exported sparse terms include real graph features:

```text
1, T, x, y, t, g0, g1, g2, g3
```

The expression includes small nonzero `g0/g1/g2` coefficients after the 20-step smoke. This only proves the plumbing and metadata path; it does not yet imply that real microscopy features improve prediction.

## Next Action

Run a controlled A100 comparison against the current sparse closure baseline:

- same active thermal table and split
- same best sparse closure optimizer settings
- `closure-graph-mode real_micro`
- compare `embedding_dim=4/8`, graph gate `0.1/0.25`, graph L1 `1e-4`
- include a scalar-statistics control using the same selected features without graph terminology

If the single-image sample-level feature remains weak, expand `mds2-2718` to multiple `P/L/replicate` TIFFs before moving to `mds2-2775` or ExaCA.

Implemented script for the next run:

```bash
bash scripts/server/run_real_micro_graph_conditioned_closure_a100.sh \
  > logs/ambench_real_micro_graph_conditioned_closure_a100_v1.log 2>&1
```
