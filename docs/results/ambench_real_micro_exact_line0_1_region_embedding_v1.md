# Phase 22: Fixed Patch Embedding Real Micro Representation

## Run Context
- Code commit used for the server runs: `aefbe53`
- Server: Ubuntu 22.04.3, NVIDIA A100-SXM4-40GB
- Data: exact `Line_0_1` P3/P4 masked/unmasked TIFF panel
- Feature table: `data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features_line0_1_panel_region_embedding_v1.jsonl`

## Commands
```bash
bash scripts/server/build_mds2_2718_line0_1_micro_panel_region_embedding_a100.sh
bash scripts/server/run_real_micro_exact_line0_1_region_embedding_a100.sh
bash scripts/server/run_real_micro_exact_line0_1_region_embedding_seed_check_a100.sh
```

## Result
The fixed patch-embedding route is a weak positive, but not stable enough for a 3-seed expansion.

| Run | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Notes |
| --- | ---: | ---: | ---: | --- |
| `P4-L0-1_m/g4`, seed 0 | 66.116148 | 48.737337 | 68.469772 | Best Phase 22 seed-0 candidate |
| `P4-L0-1_m/g4`, seed 1 | 76.031581 | 42.437391 | 66.661596 | Focused seed check |
| `P4-L0-1_m/g4`, seed 2 | 78.599927 | 33.878222 | 66.525360 | Focused seed check |

For reference, Phase 21 `col_flip` seed-0 was `66.288087 / 49.074617 / 72.040545`, so Phase 22 improves slightly on all three metrics at seed 0. The gains do not hold across seeds, and `g8` is clearly unstable on the exact-line panel.

## Interpretation
- Frozen PCA patch embeddings preserve the local patch-route idea without introducing a trainable image encoder.
- The exact-line panel is still too small for stable model claims.
- The next branch should only expand if a stronger physical alignment or a larger microstructure source becomes available.

## Decision
Close this branch as a bounded, weak-positive diagnostic result.
