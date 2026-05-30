# AM-Bench Real Micro Panel Frame-Cycle Closure v1

## Context

This experiment verifies the panel-aware `real_micro` path after the `mds2-2718` optional TIFF panel was downloaded locally and synced to the A100 server. It is a process/sample-aware plumbing test with a prototype alignment, not a physical process-to-microstructure ground truth.

- Code commit: `7ef0e1d`
- Server: A100-SXM4-40GB, `/root/matsci-gnn-pinn`
- Thermal source table: `data/interim/ambench/2022_single_track/AMB2022-03/ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1.csv`
- Aligned thermal table: `data/interim/ambench/2022_single_track/AMB2022-03/ambench_line_0_1_temperature_hot_gradient_mds2_2718_micro_panel_framecycle_v1.csv`
- Original split: `outputs/data_splits/ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1_split.json`
- Panel features: `data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features_panel.jsonl`
- Log: `logs/ambench_real_micro_graph_conditioned_closure_panel_framecycle_a100_v1.log`

## Data Preparation

Server-side panel build passed after installing `imagecodecs` for LZW-compressed TIFF decoding:

```bash
bash scripts/server/build_mds2_2718_micro_panel_a100.sh
```

The panel manifest reports:

| Field | Value |
| --- | ---: |
| micro records | 6 |
| graph/image features | 25 |
| graph nodes per image | 64 |
| graph edges per image | 512 |

Prototype alignment table:

```bash
bash scripts/server/create_mds2_2718_micro_panel_aligned_table_a100.sh
```

The alignment manifest reports:

| Field | Value |
| --- | ---: |
| rows | 14518 |
| unique frames | 71 |
| micro records used | 6 |

The default `frame_cycle` mapping assigns sorted `frame_index` values cyclically across the six panel sample IDs. This validates row-wise `micro_sample_id` selection but is not a physically validated alignment.

## Training Command

```bash
ACTIVE_ID=ambench_line_0_1_temperature_hot_gradient_mds2_2718_micro_panel_framecycle_v1 \
ACTIVE_TABLE=data/interim/ambench/2022_single_track/AMB2022-03/ambench_line_0_1_temperature_hot_gradient_mds2_2718_micro_panel_framecycle_v1.csv \
ACTIVE_SPLIT=outputs/data_splits/ambench_line_0_1_temperature_hot_gradient_a100_sxm4_40gb_v1_split.json \
MICRO_AGGREGATE=0 \
MICRO_FEATURES=data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features_panel.jsonl \
MICRO_SAMPLE_ID_COLUMN=micro_sample_id \
bash scripts/server/run_real_micro_graph_conditioned_closure_a100.sh \
  > logs/ambench_real_micro_graph_conditioned_closure_panel_framecycle_a100_v1.log 2>&1
```

The run uses the same sparse closure optimizer settings as the single-image real-micro comparison:

- `hidden_dim=256`, `layers=4`, `lr=1e-3`
- `closure_lr=1e-5`
- `closure_start_step=1500`
- `residual_sample_size=4096`
- `pde_weight=1e-6`
- `closure_graph_gate=0.25`
- `closure_graph_l1_weight=1e-4`

## Results

Primary numbers below are test split metrics.

| Method | Test RMSE | Test MAE | Test Relative L2 | Hot q90 RMSE | Gradient q90 RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sparse closure best, `closure_lr=1e-5` | 70.494433 | - | - | 31.542155 | 64.558069 |
| Single-image real micro, `g8` | 79.072104 | 58.244222 | 0.064695 | 29.314642 | 65.371975 |
| Panel frame-cycle real micro, `g4` | 103.994793 | 85.250663 | 0.085086 | 179.742978 | 175.162047 |
| Panel frame-cycle real micro, `g8` | 130.057854 | 106.024801 | 0.106410 | 213.906003 | 206.215815 |

The closure metadata confirms that the new row-wise selection path was used:

```text
sample_id_column = micro_sample_id
graph_features = data/processed/.../micro_graph_features_panel.jsonl
available_sample_ids = 6 panel records
```

## Interpretation

The experiment successfully validates the panel-aware software path:

- `load_field_table` preserved `micro_sample_id` row metadata.
- `RealMicroGraphFeatureProvider` selected micro records per residual point.
- The A100 script accepted a panel JSONL and skipped single-image aggregation.
- Both `g4` and `g8` runs completed with exported artifacts.

Scientifically, the frame-cycle prototype is a negative control. It strongly degrades both global and hot/gradient-region metrics, which is consistent with injecting a nonphysical sample assignment into the closure branch. The result should not be interpreted as evidence against real microstructure conditioning; it says the next useful step is a physically grounded process/sample mapping rather than arbitrary frame cycling.

## Next Action

Keep the panel-aware code path. Replace the prototype `frame_cycle` assignment with a defensible mapping from thermal process conditions to `mds2-2718` sample IDs, or move to a microstructure source where thermal and microscopy samples share an explicit experimental identifier. Only then rerun panel-aware closure as a performance claim.
