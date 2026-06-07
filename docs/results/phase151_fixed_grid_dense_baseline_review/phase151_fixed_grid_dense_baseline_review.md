# Phase 151 Fixed-Grid Dense Baseline Review

- Status: `phase151_fixed_grid_dense_baseline_closed_no_operator_gap`
- Leakage-safe source rows: `1`
- Phase 152 low-capacity dense design candidates: `0`
- Phase 151 model mechanism allowed: `false`
- Phase 151 model training allowed: `false`
- Operator training allowed now: `false`
- A100 training allowed now: `false`

## Tensor Manifests

| candidate_id | source_path | source_rows | summary_rows | target_column | target_columns | grid_index_columns | split_axis | split_contract_status | leakage_safe_split | train_rows | val_rows | test_rows | tensor_manifest_status | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ambench_line0_1_phase51_dense_csv | data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_phase51_dense.csv | 10205 | 35 | temperature_C | ["target_frame_mean", "target_frame_q90", "target_frame_std", "target_frame_range"] | frame_index,row_index,col_index | frame_block | diagnostic_frame_block_split_only | false | 21 | 7 | 7 | fixed_grid_summary_ready |  |
| ambench_line0_1_dense_a800_csv | data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_dense_a100_sxm4_40gb_v1.csv | 41054 | 70 | temperature_C | ["target_frame_mean", "target_frame_q90", "target_frame_std", "target_frame_range"] | frame_index,row_index,col_index | frame_block | diagnostic_frame_block_split_only | false | 42 | 14 | 14 | fixed_grid_summary_ready |  |
| ambench_multiline_process_dense_csv | data/interim/ambench/2022_single_track/AMB2022-03/ambench_multiline_process_temperature_a100_sxm4_40gb_v1.csv | 14842 | 231 | temperature_C | ["target_frame_mean", "target_frame_q90", "target_frame_std", "target_frame_range"] | frame_index,row_index,col_index | line_id | leakage_safe_line_group_split | true | 125 | 40 | 66 | fixed_grid_summary_ready |  |

## Split Contracts

| candidate_id | split_axis | split_contract_status | leakage_safe_split | group_count | train_rows | val_rows | test_rows | train_groups | val_groups | test_groups | rationale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ambench_line0_1_phase51_dense_csv | frame_block | diagnostic_frame_block_split_only | false | 35 | 21 | 7 | 7 | frame_block_train | frame_block_val | frame_block_test | single-group source can only be split by frame block; this is diagnostic, not operator-safe |
| ambench_line0_1_dense_a800_csv | frame_block | diagnostic_frame_block_split_only | false | 70 | 42 | 14 | 14 | frame_block_train | frame_block_val | frame_block_test | single-group source can only be split by frame block; this is diagnostic, not operator-safe |
| ambench_multiline_process_dense_csv | line_id | leakage_safe_line_group_split | true | 7 | 125 | 40 | 66 | ["Line_0_1", "Line_1_1_1", "Line_1_2_1", "Line_2_1_1"] | ["Line_2_2_1"] | ["Line_3_1_1", "Line_3_2_1"] | line_id groups are disjoint across train/val/test |

## Baseline Review

| candidate_id | target | split_contract_status | selected_feature_profile | selected_method | selected_validation_rmse | selected_validation_normalized_rmse | selected_test_rmse | selected_test_normalized_rmse | mean_validation_rmse | mean_test_rmse | validation_relative_improvement_over_mean | test_relative_improvement_over_mean | baseline_visible_gap | strong_baseline_solved | phase152_low_capacity_dense_design_candidate | status | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ambench_line0_1_phase51_dense_csv | target_frame_mean | diagnostic_frame_block_split_only | time_only | hist_gradient_boosting | 8.369028 | 0.071632 | 24.570275 | 0.210302 | 8.369028 | 24.570275 | 0.000000 | 0.000000 | false | false | false | blocked_no_leakage_safe_split | split is diagnostic only |
| ambench_line0_1_phase51_dense_csv | target_frame_q90 | diagnostic_frame_block_split_only | time_only | extra_trees | 27.681403 | 0.076445 | 91.087950 | 0.251550 | 27.840977 | 94.360027 | 0.005732 | 0.034677 | false | false | false | blocked_no_leakage_safe_split | split is diagnostic only |
| ambench_line0_1_phase51_dense_csv | target_frame_std | diagnostic_frame_block_split_only | time_only | extra_trees | 4.694750 | 0.037963 | 34.710021 | 0.280678 | 5.153724 | 34.294440 | 0.089057 | -0.012118 | true | false | false | blocked_no_leakage_safe_split | split is diagnostic only |
| ambench_line0_1_phase51_dense_csv | target_frame_range | diagnostic_frame_block_split_only | time_only | extra_trees | 0.000000 | 0.000000 | 129.083454 | 0.374124 | 16.466004 | 121.394864 | 1.000000 | -0.063335 | true | false | false | blocked_no_leakage_safe_split | split is diagnostic only |
| ambench_line0_1_dense_a800_csv | target_frame_mean | diagnostic_frame_block_split_only | time_only | knn | 6.990012 | 0.053825 | 34.108049 | 0.262643 | 6.986002 | 34.119059 | -0.000574 | 0.000323 | false | false | false | blocked_no_leakage_safe_split | split is diagnostic only |
| ambench_line0_1_dense_a800_csv | target_frame_q90 | diagnostic_frame_block_split_only | time_only | knn | 25.885329 | 0.071232 | 109.013791 | 0.299989 | 26.433281 | 110.177862 | 0.020730 | 0.010565 | false | false | false | blocked_no_leakage_safe_split | split is diagnostic only |
| ambench_line0_1_dense_a800_csv | target_frame_std | diagnostic_frame_block_split_only | time_only | knn | 4.158680 | 0.033707 | 39.404512 | 0.319381 | 4.153496 | 39.375768 | -0.001248 | -0.000730 | false | false | false | blocked_no_leakage_safe_split | split is diagnostic only |
| ambench_line0_1_dense_a800_csv | target_frame_range | diagnostic_frame_block_split_only | time_only | knn | 0.000000 | 0.000000 | 143.317806 | 0.402101 | 8.214954 | 139.003661 | 1.000000 | -0.031036 | true | false | false | blocked_no_leakage_safe_split | split is diagnostic only |
| ambench_multiline_process_dense_csv | target_frame_mean | leakage_safe_line_group_split | time_only | knn | 20.624803 | 0.089829 | 21.674013 | 0.094398 | 28.027423 | 29.398927 | 0.264121 | 0.262762 | true | true | false | blocked_strong_baseline_solved | non-neural strong baseline residual is below unsolved threshold |
| ambench_multiline_process_dense_csv | target_frame_q90 | leakage_safe_line_group_split | process_time | extra_trees | 10.486463 | 0.029231 | 5.883759 | 0.016401 | 57.039301 | 61.715298 | 0.816154 | 0.904663 | true | true | false | blocked_strong_baseline_solved | non-neural strong baseline residual is below unsolved threshold |
| ambench_multiline_process_dense_csv | target_frame_std | leakage_safe_line_group_split | process_time | extra_trees | 9.038344 | 0.061637 | 6.373920 | 0.043467 | 25.030652 | 23.583719 | 0.638909 | 0.729732 | true | true | false | blocked_strong_baseline_solved | non-neural strong baseline residual is below unsolved threshold |
| ambench_multiline_process_dense_csv | target_frame_range | leakage_safe_line_group_split | time_only | extra_trees | 13.218099 | 0.037421 | 2.650112 | 0.007503 | 56.055931 | 59.279419 | 0.764198 | 0.955295 | true | true | false | blocked_strong_baseline_solved | non-neural strong baseline residual is below unsolved threshold |

Next action: close neural-operator branch as diagnostic unless a new dense target/split is added
