# Phase 55 Spot-Size Route Seed Validation

Transfer gate: `seed_unstable_or_negative`

| dataset | route | aggregate gate | paired seed gate |
|---|---|---|---|
| broad12 | film/global_standard | yes | yes |
| broad21 | film/global_standard | no | no |

## Aggregate Metrics

| dataset | method | n | test RMSE mean +/- std | hot q90 mean +/- std | gradient q90 mean +/- std |
|---|---|---:|---:|---:|---:|
| broad12 | no_process | 1 | 260.807865 +/- 0.000000 | 478.337515 +/- 0.000000 | 434.594412 +/- 0.000000 |
| broad12 | broad_process_v1 | 1 | 139.085217 +/- 0.000000 | 235.696768 +/- 0.000000 | 221.604222 +/- 0.000000 |
| broad21 | no_process | 1 | 226.518789 +/- 0.000000 | 421.639304 +/- 0.000000 | 374.684937 +/- 0.000000 |
| broad21 | broad_process_v1 | 1 | 153.259455 +/- 0.000000 | 270.628922 +/- 0.000000 | 250.519935 +/- 0.000000 |

## Strong-Baseline Deltas

| dataset | metric | broad mean | best strong baseline | no-process mean | delta vs strong | delta vs no-process |
|---|---|---:|---:|---:|---:|---:|
| broad12 | rmse | 139.085217 | 140.201362 (mean) | 260.807865 | -1.116145 | -121.722647 |
| broad12 | hot_q90_rmse | 235.696768 | 253.374499 (mean) | 478.337515 | -17.677731 | -242.640747 |
| broad12 | gradient_q90_rmse | 221.604222 | 234.028899 (mean) | 434.594412 | -12.424677 | -212.990190 |
| broad21 | rmse | 153.259455 | 139.725646 (mean) | 226.518789 | 13.533809 | -73.259334 |
| broad21 | hot_q90_rmse | 270.628922 | 253.129723 (mean) | 421.639304 | 17.499198 | -151.010383 |
| broad21 | gradient_q90_rmse | 250.519935 | 231.780894 (mean) | 374.684937 | 18.739041 | -124.165002 |

## Per-Seed Metrics

| dataset | method | seed | test RMSE | hot q90 RMSE | gradient q90 RMSE |
|---|---|---:|---:|---:|---:|
| broad12 | no_process | 7 | 260.807865 | 478.337515 | 434.594412 |
| broad12 | broad_process_v1 | 7 | 139.085217 | 235.696768 | 221.604222 |
| broad21 | no_process | 7 | 226.518789 | 421.639304 | 374.684937 |
| broad21 | broad_process_v1 | 7 | 153.259455 | 270.628922 | 250.519935 |
