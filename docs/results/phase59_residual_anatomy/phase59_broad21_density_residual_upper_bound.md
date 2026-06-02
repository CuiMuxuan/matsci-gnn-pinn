# Phase 59 Residual Upper-Bound Probe

Uses test for selection: `false`
Fit split: `train`
Selection split: `val`
Analysis split: `test`
Selected variant: `blend:broad_process_v1->mean:alpha=1`

## Selected Variant Metrics

| split | n | RMSE | hot q90 RMSE | gradient q90 RMSE | MAE | bias |
|---|---:|---:|---:|---:|---:|---:|
| test | 1129 | 139.725646 | 253.129723 | 198.003809 | 116.525079 | -35.568503 |
| train | 3476 | 118.046316 | 251.664869 | 167.660112 | 98.040839 | 0.000000 |
| val | 9190 | 138.500717 | 253.129723 | 194.626218 | 114.036633 | -18.783296 |

## Validation-Ranked Variants

| rank | variant | val RMSE | test RMSE | test hot q90 | test gradient q90 |
|---:|---|---:|---:|---:|---:|
| 1 | blend:broad_process_v1->mean:alpha=1 | 138.500717 | 139.725646 | 253.129723 | 198.003809 |
| 2 | blend:broad_process_v1->mean:alpha=0.75 | 140.133258 | 141.785617 | 257.031462 | 200.637306 |
| 3 | blend:broad_process_v1->mean:alpha=0.5 | 144.869162 | 144.766469 | 261.258946 | 203.812865 |
| 4 | blend:broad_process_v1->mean:alpha=0.25 | 152.419423 | 148.612799 | 265.796634 | 207.505602 |
| 5 | broad_process_v1:train_group_bias:time_bin | 162.132533 | 153.083447 | 269.653909 | 211.294892 |
| 6 | broad_process_v1:train_group_bias:frame_bin | 162.132533 | 153.083447 | 269.653909 | 211.294892 |
| 7 | broad_process_v1:train_global_bias | 162.142446 | 153.086051 | 269.693001 | 211.250815 |
| 8 | broad_process_v1:train_group_bias:line_id | 162.142446 | 153.086051 | 269.693001 | 211.250815 |
| 9 | broad_process_v1:train_group_bias:laser_power_W | 162.142446 | 153.086051 | 269.693001 | 211.250815 |
| 10 | broad_process_v1:train_group_bias:scan_speed_mm_s | 162.142446 | 153.086051 | 269.693001 | 211.250815 |

## Decision

validation-visible correction does not beat the reference on the analysis split
