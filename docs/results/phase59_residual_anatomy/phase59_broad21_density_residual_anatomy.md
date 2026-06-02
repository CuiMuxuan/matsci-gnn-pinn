# Phase 59 Residual Anatomy

Analysis split: `test`
Candidate: `broad_process_v1`
Reference: `mean`
Secondary reference: `no_process`

## Split Summary

| split | method | n | RMSE | MAE | bias |
|---|---|---:|---:|---:|---:|
| test | mean | 1129 | 139.725646 | 116.525079 | -35.568503 |
| test | no_process | 1129 | 226.518793 | 174.460834 | -166.617306 |
| test | broad_process_v1 | 1129 | 153.259459 | 125.542693 | -28.582547 |
| train | mean | 3476 | 118.046316 | 98.040839 | 0.000000 |
| train | no_process | 3476 | 85.085984 | 64.129127 | -0.075885 |
| train | broad_process_v1 | 3476 | 83.386937 | 62.183534 | -0.944905 |
| val | mean | 9190 | 138.500717 | 114.036633 | -18.783296 |
| val | no_process | 9190 | 181.956384 | 132.061031 | -96.602873 |
| val | broad_process_v1 | 9190 | 162.391962 | 124.611877 | -43.321610 |

## Worst Candidate-vs-Reference Groups

| field | value | n | candidate RMSE | reference RMSE | delta | secondary delta |
|---|---|---:|---:|---:|---:|---:|
| line_id | Line_1_1_1 | 368 | 165.425776 | 143.582200 | 21.843576 | -88.638657 |
| region | hot_q90 | 111 | 272.542708 | 253.129723 | 19.412985 | -152.489276 |
| time_bin | low | 462 | 150.448143 | 131.865568 | 18.582575 | -64.243223 |
| frame_bin | low | 462 | 150.448143 | 131.865568 | 18.582575 | -64.243223 |
| hot_region | hot_q90 | 150 | 270.628922 | 253.129723 | 17.499198 | -151.010383 |
| line_id | Line_1_1_2 | 387 | 154.110788 | 136.787462 | 17.323326 | -87.835503 |
| region | gradient_q90 | 75 | 177.673621 | 162.091182 | 15.582439 | -61.099888 |
| gradient_region | gradient_q90 | 114 | 211.688452 | 198.003809 | 13.684643 | -97.392271 |
| gradient_region | not_gradient_q90 | 1015 | 145.236024 | 131.577501 | 13.658523 | -70.040932 |
| laser_power_W | 285 | 1129 | 153.259459 | 139.725646 | 13.533813 | -73.259334 |
| scan_speed_mm_s | 960 | 1129 | 153.259459 | 139.725646 | 13.533813 | -73.259334 |
| spot_size_um | 49 | 1129 | 153.259459 | 139.725646 | 13.533813 | -73.259334 |
| process_tuple | laser_power_W=285.0__scan_speed_mm_s=960.0__spot_size_um=49.0 | 1129 | 153.259459 | 139.725646 | 13.533813 | -73.259334 |
| hot_region | not_hot_q90 | 979 | 125.958922 | 112.681848 | 13.277074 | -52.740547 |
| region | background | 904 | 120.676915 | 107.567849 | 13.109065 | -52.102734 |
| region | hot_q90+gradient_q90 | 39 | 265.106385 | 253.129723 | 11.976662 | -146.723885 |
| time_bin | mid | 352 | 156.294583 | 144.900916 | 11.393666 | -80.812003 |
| frame_bin | mid | 352 | 156.294583 | 144.900916 | 11.393666 | -80.812003 |
| time_bin | high | 315 | 153.918274 | 144.941885 | 8.976389 | -77.337069 |
| frame_bin | high | 315 | 153.918274 | 144.941885 | 8.976389 | -77.337069 |
| line_id | Line_1_1_3 | 374 | 139.293464 | 138.884409 | 0.409055 | -36.224930 |

## Decision

candidate loses to reference on the analysis split; inspect worst groups before model expansion
