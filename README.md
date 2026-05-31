# matsci-gnn-pinn

`matsci-gnn-pinn` 是一个面向材料科学可复现实验的 GNN-PINN 研究原型。项目当前主线聚焦金属增材制造中的热场建模，以 NIST AM-Bench 2022 公开数据为真实数据入口，逐步推进到宏观 PINN、可学习闭合项、微观结构 GNN 表征和多尺度弱耦合。

这个仓库的目标不是先做一个泛化平台，而是先形成一个论文友好的代码项目：数据来源可追溯、实验命令可复现、指标产物可保存、模块边界清晰，方便后续扩展消融实验和投稿论文图表。

## 当前状态

- 已接入 AM-Bench 2022 / AMB2022-03 / `mds2-2716` 数据清单。
- 已提供 AM-Bench 文件下载、文件大小检查和 SHA256 校验入口。
- 已实现 thermography HDF5 到字段表 CSV 的转换。
- 已支持 raw `signal` 与校准 `temperature_C` 两种字段目标。
- 已支持 frame-based split manifest，避免随机点划分带来的时序泄漏。
- 已实现 mean baseline、Macro PINN 数据拟合入口和初步 PDE residual 训练开关。
- 已实现 hot/gradient active sampling、region-aware metrics 与服务器端复现实验脚本。
- 已实现 sparse closure 与 synthetic graph-conditioned closure 消融，当前转入真实/半真实 microstructure conditioning。
- 已新增 AM-Bench `mds2-2718` optical microscopy 下载/校验入口和 TIFF-to-coarse-micro-graph inspection 原型。
- 已固定 `mds2-2718` multi-image optional micro panel，并提供服务器脚本生成 panel-level graph feature table。
- 本地已完成 `mds2-2718` optional panel 下载、SHA256 校验、6 图 inspection 与 panel-level graph feature table 聚合。
- 已支持 `real_micro` panel feature JSONL 通过 field-table `micro_sample_id` 等列做逐点 sample-aware graph conditioning。
- 已预留 weak GNN-PINN coupling 模块。
- 本地 smoke/probe 与第一批 A100 dense calibrated temperature 实验已完成，当前正在推进 Macro PINN 稳健化和闭合项实验前置工作。

## Repository Layout

```text
configs/          数据源、字段映射和实验配置
data/             本地数据目录；raw/interim/processed 默认不提交大文件
docs/             数据说明、环境说明、路线文档、实验计划和结果记录
requirements/     CPU/GPU/PyG/科学计算依赖拆分
scripts/          环境检查等辅助脚本
src/gnnpinn/      Python package 源码
tests/            pytest 测试
```

核心模块：

```text
gnnpinn.data      数据审计、AM-Bench 下载、HDF5 转换、split manifest
gnnpinn.eval      baseline 与评估指标
gnnpinn.models    PINN、GNN、closure、coupled 原型模块
gnnpinn.physics   热方程 residual 等物理约束
gnnpinn.train     训练入口
```

## Environment

推荐 Python 3.11。通用 CPU/服务器基础环境：

```bash
conda env create -f environment.yml
conda activate gnnpinn
python -m pip install --no-deps -e .
python scripts/env/check_env.py
```

本地 Windows + CUDA 13.0 wheel 环境可使用：

```powershell
conda env create -f environment-cu130-local.yml
conda activate gnnpinn-cu130
python -m pip install --no-deps -e .
python -m pip install -r requirements/base.txt
python -m pip install -r requirements/science.txt
python -m pip install -r requirements/torch-local-cu130.txt
python scripts/env/check_env.py
```

Linux GPU 服务器建议按驱动选择 CUDA 依赖：

```bash
# Stable branch for V100 / A800 / A100 images
python -m pip install -r requirements/torch-cu118.txt
python -m pip install -r requirements/pyg-torch27-cu118.txt

# Use only when the image clearly supports CUDA 12.6
python -m pip install -r requirements/torch-cu126.txt
python -m pip install -r requirements/pyg-torch27-cu126.txt
```

## AM-Bench Data

当前第一批真实实验只需要 AMB2022-03 / `mds2-2716` 的 thermography 与 scan strategy 文件：

```text
data/raw/ambench/2022_single_track/AMB2022-03/mds2-2716/
  2716_README.txt
  Thermography/AMB2022-03-718-AMMT-StaringCamera_Signal.h5
  ScanStrategy/AMB2022-03-AMMT-718-Pad_XYPT.h5
```

下载并校验：

```bash
python -m gnnpinn.data.ambench_downloads \
  --root data/raw/ambench/2022_single_track/AMB2022-03/mds2-2716 \
  --download \
  --verify-sha256 \
  --output outputs/data_audits/ambench_mds2_2716_download_report.json
```

只校验已经手动下载或复制的数据：

```bash
python -m gnnpinn.data.ambench_downloads \
  --root data/raw/ambench/2022_single_track/AMB2022-03/mds2-2716 \
  --verify-sha256 \
  --output outputs/data_audits/ambench_mds2_2716_download_check.json
```

更完整的下载说明见 [docs/ambench_downloads.md](docs/ambench_downloads.md)。

Phase 17 真实微观组织入口使用 AMB2022-03 / `mds2-2718` optical microscopy：

```bash
python -m gnnpinn.data.ambench_downloads \
  --dataset-id mds2-2718 \
  --root data/raw/ambench/2022_single_track/AMB2022-03/mds2-2718 \
  --download \
  --verify-sha256 \
  --output outputs/data_audits/ambench_mds2_2718_download_report.json
```

扩展到第一版 multi-image optional panel：

```bash
python -m gnnpinn.data.ambench_downloads \
  --dataset-id mds2-2718 \
  --root data/raw/ambench/2022_single_track/AMB2022-03/mds2-2718 \
  --download \
  --include-optional \
  --verify-sha256 \
  --retries 3 \
  --timeout-seconds 300 \
  --resume-partial \
  --download-backend wget \
  --output outputs/data_audits/ambench_mds2_2718_micro_panel_download_report.json
```

下载后可先生成 coarse micro graph inspection：

```bash
python -m gnnpinn.data.loaders.ambench_microstructure \
  --image data/raw/ambench/2022_single_track/AMB2022-03/mds2-2718/Single_Track_Cross_Sections/AMB2022-718-SH1-BP1-P2-L2.1-3_m.tif \
  --sample-id AMB2022-718-SH1-BP1-P2-L2.1-3_m \
  --threshold-quantile 0.9 \
  --grid-rows 8 \
  --grid-cols 8 \
  --graph-k 4 \
  --output outputs/data_audits/ambench_mds2_2718_micrograph_inspection.json
```

将 inspection 聚合为 `real_micro` graph feature JSONL：

```bash
python -m gnnpinn.data.loaders.ambench_microstructure \
  --mode aggregate \
  --inspection outputs/data_audits/ambench_mds2_2718_micrograph_inspection.json \
  --jsonl-output data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features.jsonl \
  --csv-output data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features.csv \
  --output outputs/data_audits/ambench_mds2_2718_micrograph_feature_table_manifest.json
```

服务器上一键构建多图 panel inspection 与 feature table：

```bash
bash scripts/server/build_mds2_2718_micro_panel_a100.sh \
  > logs/ambench_mds2_2718_micro_panel_build_a100_v1.log 2>&1
```

Exact `Line_0_1` v2 rich-feature panel:

```bash
bash scripts/server/build_mds2_2718_line0_1_micro_panel_feature_v2_a100.sh \
  > logs/ambench_mds2_2718_line0_1_micro_panel_feature_v2_build_a100_v1.log 2>&1
```

The v2 feature schema keeps the coarse graph table but adds intensity distribution, threshold-mask geometry, and normalized texture/gradient descriptors. The `real_micro` provider prioritizes these descriptors for `g0..g7` before falling back to v1 coarse grid statistics.

Fixed patch-embedding exact `Line_0_1` panel:

```bash
bash scripts/server/build_mds2_2718_line0_1_micro_panel_region_embedding_a100.sh \
  > logs/ambench_mds2_2718_line0_1_region_embedding_build_a100_v1.log 2>&1
```

This route fits frozen PCA embeddings over lifted 8x8 patch descriptors and writes `region_embedding_*` fields for `--closure-graph-mode real_micro_region_embedding`.

Multi-line/process-conditioned thermal branch:

```bash
bash scripts/server/run_multiline_process_conditioned_thermal_a100.sh \
  > logs/ambench_multiline_process_conditioned_thermal_a100_v1.log 2>&1
```

This Phase 23 route builds a calibrated multi-line `mds2-2716` field table from representative `ThermalData/*/Signal` tracks, splits by `line_id`, and compares coordinate-only baselines/Macro PINN against process-conditioned variants using `laser_power_W`, `scan_speed_mm_s`, and `spot_size_um`.
The first A100 run shows the expected direction: process-conditioned Macro PINN improves held-out-line test RMSE from `175.127058` to `157.793227`, with hot q90 RMSE improving from `351.525048` to `316.794319`. It is a positive direction-selection result, but it has not yet beaten the train-mean baseline on global test RMSE.

Process-axis holdout checks:

```bash
bash scripts/server/run_multiline_process_holdout_splits_a100.sh \
  > logs/ambench_multiline_process_holdout_splits_a100_v1.log 2>&1
```

This runs the same seven-line table through `line`, `laser_power`, `scan_speed`, `spot_size`, and full process-condition grouped splits, so the next result can separate line memorization from true process-axis generalization.
The A100 holdout run is now documented in [docs/results/ambench_multiline_process_axis_holdout_v1.md](docs/results/ambench_multiline_process_axis_holdout_v1.md). Process features improve Macro PINN test RMSE on `line`, `laser_power`, `scan_speed`, and full `process` holdouts, with the largest gain on scan speed (`186.921887 -> 140.459979`), but they hurt the `spot_size` holdout (`208.741300 -> 227.573411`) and still do not beat the train-mean baseline. The next model-innovation branch is therefore structured process conditioning, starting with FiLM modulation rather than only concatenating process scalars.

FiLM process-conditioned Macro PINN comparison:

```bash
bash scripts/server/run_multiline_process_film_conditioned_a100.sh \
  > logs/ambench_multiline_process_film_conditioned_a100_v1.log 2>&1
```

This focused Phase 25 route compares the new `--input-conditioning-mode film` path on `line`, `scan_speed`, and `spot_size` holdouts.
If train-fitted feature normalization is too weak for single-axis holdouts, run the global process-feature normalization variant:

```bash
bash scripts/server/run_multiline_process_film_global_feature_norm_a100.sh \
  > logs/ambench_multiline_process_film_global_feature_norm_a100_v1.log 2>&1
```

Separate FiLM structure from process-feature normalization with concat/global-standard and hybrid controls:

```bash
bash scripts/server/run_multiline_process_concat_global_feature_norm_a100.sh \
  > logs/ambench_multiline_process_concat_global_feature_norm_a100_v1.log 2>&1

bash scripts/server/run_multiline_process_concat_film_global_feature_norm_a100.sh \
  > logs/ambench_multiline_process_concat_film_global_feature_norm_a100_v1.log 2>&1

bash scripts/server/run_multiline_process_concat_film_limited_global_feature_norm_a100.sh \
  > logs/ambench_multiline_process_concat_film_limited_global_feature_norm_a100_v1.log 2>&1
```

The Phase 25/26 result is documented in [docs/results/ambench_multiline_process_film_conditioned_v1.md](docs/results/ambench_multiline_process_film_conditioned_v1.md). Global process-feature standardization is essential. `scan_speed` favors concat + global standardization, while `spot_size` favors FiLM + global standardization and beats the train-mean baseline in a focused three-seed check. Simple concat+FiLM hybrid stacking, including a limited `--input-film-strength 0.25` variant, does not produce a universal architecture.

Split/process-aware routing diagnostics:

```bash
bash scripts/server/run_multiline_process_routed_global_feature_norm_a100.sh \
  > logs/ambench_multiline_process_routed_global_feature_norm_a100_v1.log 2>&1

bash scripts/server/run_multiline_process_axis_profile_a100.sh \
  > logs/ambench_multiline_process_axis_profile_a100_v1.log 2>&1
```

The Phase 27 result is documented in [docs/results/ambench_multiline_process_axis_routing_v1.md](docs/results/ambench_multiline_process_axis_routing_v1.md). Trainable dual-expert routed conditioning degraded on `line`, `scan_speed`, and `spot_size`. The explicit `--input-conditioning-profile process_axis_v1` route records the process-axis decision in artifacts and recovers the best known per-axis choices: `line -> concat/train-minmax`, `scan_speed -> concat/global_standard`, and `spot_size -> FiLM/global_standard`.

Process-axis profile validation:

```bash
PROFILE_SPLITS="laser_power process" \
  bash scripts/server/run_multiline_process_axis_profile_a100.sh \
  > logs/ambench_multiline_process_axis_profile_phase28_new_axes_a100_v1.log 2>&1

bash scripts/server/run_phase28_laser_power_profile_seed_check_a100.sh \
  > logs/ambench_phase28_laser_power_profile_seed_check_a100_v1.log 2>&1
```

The Phase 28 result is documented in [docs/results/ambench_multiline_process_axis_profile_validation_v1.md](docs/results/ambench_multiline_process_axis_profile_validation_v1.md). `process_axis_v1` now covers `laser_power_W` and full `process_condition`. `laser_power` shows a stable three-seed gain over no-process Macro PINN (`211.217281 +/- 0.443665 -> 147.980699 +/- 3.456300` test RMSE). Full `process` should use the line-like `concat/same` route; `concat/global_standard` was a negative diagnostic.

Broader process-dataset smoke:

```bash
DATASET_LIMIT=12 DATASET_ORDER=process_round_robin \
  bash scripts/server/run_phase29_broad_process_profile_smoke_a100.sh \
  > logs/ambench_phase29_broad12_rr_profile_smoke_a100_v1.log 2>&1
```

The Phase 29 result is documented in [docs/results/ambench_multiline_process_broader_profile_smoke_v1.md](docs/results/ambench_multiline_process_broader_profile_smoke_v1.md). The new `process_round_robin` dataset ordering verifies a balanced broad12 panel across `245/285/325 W`, `800/960/1200 mm/s`, and `49/67/82 um`. The old `process_axis_v1` profile transfers only partially: `spot_size` remains positive (`206.100512 -> 136.309183` test RMSE), but `scan_speed` and full `process` degrade. The next route should let the broad-data profile fall back to no-process modeling when process conditioning hurts.

Broad-data selector smoke:

```bash
DATASET_LIMIT=12 DATASET_ORDER=process_round_robin STEPS=500 N_ESTIMATORS=80 \
  bash scripts/server/run_phase30_broad_process_selector_smoke_a100.sh \
  > logs/ambench_phase30_broad12_rr_selector_smoke_a100_v1.log 2>&1

PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  /home/vipuser/miniconda3/bin/conda run -n gnnpinn \
  python scripts/server/summarize_phase30_broad_process_selector_smoke.py \
  --json-output outputs/reports/phase30_broad_process_selector_smoke_summary.json \
  --require-comparable
```

The Phase 30 result is documented in [docs/results/ambench_multiline_process_broad_selector_v1.md](docs/results/ambench_multiline_process_broad_selector_v1.md). `broad_process_v1` records a split-aware route and can fall back to no-process Macro PINN. On broad12 it removes the old profile regressions on `line` (`149.638162 -> 126.308616`), `scan_speed` (`226.454041 -> 186.173938`), and full `process` (`220.019735 -> 181.091525`) while preserving the positive `laser_power` concat/global-standard and `spot_size` FiLM/global-standard routes. The next server node is the same selector scaled to all 21 single-track `ThermalData/Line_*/Signal` datasets.

All-21 single-track selector scaling:

```bash
DATASET_LIMIT=21 DATASET_ORDER=process_round_robin STEPS=500 N_ESTIMATORS=80 \
  bash scripts/server/run_phase29_broad_process_profile_smoke_a100.sh \
  > logs/ambench_phase31_broad21_rr_process_axis_profile_a100_v1.log 2>&1

DATASET_LIMIT=21 DATASET_ORDER=process_round_robin STEPS=500 N_ESTIMATORS=80 \
  bash scripts/server/run_phase30_broad_process_selector_smoke_a100.sh \
  > logs/ambench_phase31_broad21_rr_selector_smoke_a100_v1.log 2>&1
```

The Phase 31 result is documented in [docs/results/ambench_multiline_process_broad_selector_broad21_v1.md](docs/results/ambench_multiline_process_broad_selector_broad21_v1.md). Broad21 confirms the selector story at all single-track lines: it preserves `laser_power` and `spot_size` positive routes, avoids the severe old-profile `scan_speed` regression (`469.347549 -> 227.128663`), and avoids full-process regression (`229.613547 -> 166.231596`). The selector is a conservative route guard; the next branch should either refine the line route or pivot to stronger broad-data model/data representation.

The Phase 32 refinement is documented in [docs/results/ambench_multiline_process_broad_selector_v2.md](docs/results/ambench_multiline_process_broad_selector_v2.md). `broad_process_v2` keeps the conservative `broad_process_v1` routes but changes `line_id` to concat/same. It recovers the small broad21 line gain (`126.194921 -> 125.449323`) but repeats the broad12 line regression (`126.308616 -> 149.638162`), so it is retained only as a diagnostic profile. The default broad-data route guard remains `broad_process_v1`; the next branch should move to stronger broad-data representation or closure/GNN reintegration.

生成带 `micro_sample_id` 的 prototype thermal 对齐表：

```bash
bash scripts/server/create_mds2_2718_micro_panel_aligned_table_a100.sh
```

## Quick Start

运行数据审计：

```bash
python -m gnnpinn.data.audit \
  --config configs/data/ambench_2022_single_track.yaml
```

将 AMB2022-03 thermography HDF5 转为小规模字段表：

```bash
python -m gnnpinn.data.loaders.ambench_hdf5 \
  --output data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_signal_subset.csv \
  --manifest outputs/data_audits/ambench_line_0_1_hdf5_conversion_manifest.json \
  --split-manifest outputs/data_splits/ambench_line_0_1_signal_subset_split.json
```

生成校准温度字段表：

```bash
python -m gnnpinn.data.loaders.ambench_hdf5 \
  --output data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_frame_subset.csv \
  --manifest outputs/data_audits/ambench_line_0_1_temperature_frame_conversion_manifest.json \
  --split-manifest outputs/data_splits/ambench_line_0_1_temperature_frame_split.json \
  --calibrate-temperature \
  --split-strategy frame \
  --min-signal 100
```

运行 split-aware mean baseline：

```bash
python -m gnnpinn.eval.field_baseline \
  --table data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_frame_subset.csv \
  --target temperature_C \
  --strategy mean \
  --split-manifest outputs/data_splits/ambench_line_0_1_temperature_frame_split.json \
  --output outputs/baselines/ambench_line_0_1_temperature_frame_mean_baseline.json
```

运行 Macro PINN smoke training：

```bash
python -m gnnpinn.train.macro_pinn \
  --table data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_frame_subset.csv \
  --target temperature_C \
  --split-manifest outputs/data_splits/ambench_line_0_1_temperature_frame_split.json \
  --output-dir outputs/runs/ambench_line_0_1_temperature_macro_pinn_frame_smoke \
  --steps 100 \
  --hidden-dim 32 \
  --layers 2 \
  --log-every 25
```

## Tests

```bash
python -m pytest -q --basetemp .pytest_tmp
```

本地 `gnnpinn-cu130` 环境可用：

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
$env:CONDA_NO_PLUGINS="true"
conda run -n gnnpinn-cu130 python -m pytest -q --basetemp .pytest_tmp
```

最近一次本地验证状态：`54 passed, 39 skipped`。

## Server Stage

本机阶段已经完成 AM-Bench 真实数据下载、HDF5 转换、calibrated temperature probe、baseline 和 Macro PINN smoke。A100 服务器阶段已完成 uniform dense 与 balanced hot/gradient active sampling 的第一轮真实实验。

- 推荐起步：`A800-SXM4-40GB`
- 更稳选择：`A100 80GB` 或 `A100-SXM4-80GB`

服务器阶段目标包括：

- dense calibrated temperature subset
- dense mean baseline
- Macro PINN data-only run
- Macro PINN + PDE residual run
- 多 seed / 多 split / 多采样策略消融
- 环境冻结与结果文档归档

当前关键结果：

- uniform dense minmax Macro PINN test RMSE: `51.655371`
- active hot/gradient Macro PINN test RMSE: `65.892559`
- active hot/gradient Macro PINN hot q90 RMSE: `30.868055`
- 3-seed active hot-zone candidate hot q90 RMSE: `17.033393 +/- 2.084327`
- sparse closure MVP 已跑通但未超过 data-only；下一步转向 residual/collocation sampling 与尺度化热源。
- random residual sampling 改善了 active closure，但仍弱于 data-only；下一步实现 hot/gradient residual sampling。
- hot/gradient residual sampling 比 random 更差；下一步转向 staged/warm-start closure fine-tuning。
- staged closure 起步有效但仍未超过 data-only；closure optimizer ablation 表明 `closure_lr=1e-5` 明显改善 hot/gradient 指标，但仍未超过 active data-only，下一步进入 GNN-conditioned closure 接口。
- 已完成 graph-L1 sensitivity；synthetic coordinate/RBF graph terms 一旦保留就会损害 hot/gradient 指标，下一步转向真实/半真实 microstructure conditioning。
- multi-line/process-conditioned thermal run 已完成；工艺特征让 Macro PINN 的 held-out-line test RMSE 从 `175.127058` 改善到 `157.793227`，hot q90 RMSE 从 `351.525048` 改善到 `316.794319`。该分支比继续扩展小规模 exact-line TIFF 手工特征更适合作为下一条主线，但仍需容量、split 和 seed 检查。
- process-axis holdout run 已完成；工艺特征在 `line`、`laser_power`、`scan_speed`、`process` 四类 holdout 上改善 Macro PINN，但在 `spot_size` holdout 上变差，且整体仍未超过 train-mean baseline。下一步进入 FiLM process-conditioned Macro PINN。
- 已新增 `--input-conditioning-mode concat|film`；FiLM 模式用工艺参数调制 hidden coordinate/time layers，默认仍为 concat 以兼容既有实验。
- 已新增 `--input-feature-normalization same|none|minmax|standard|global_minmax|global_standard`，用于把工艺标量的归一化从坐标/时间归一化中解耦。
- 已新增 `--input-conditioning-mode concat|film|concat_film` 与 `--input-film-strength`。Phase 25/26 表明工艺轴条件化是 split-sensitive：`spot_size` 的 FiLM/global-standard 分支最强，`scan_speed` 的 concat/global-standard 分支最强，简单 `concat_film` 叠加不是稳健通用解。
- 已新增 `--input-conditioning-mode routed` 与 `--input-conditioning-profile process_axis_v1`。Phase 27 表明可训练双专家 routed 路线退化，而显式 process-axis profile 能可复现地记录并恢复最佳 per-axis conditioning。
- Phase 28 已完成 `laser_power`/full `process` holdout 扩展与 `laser_power` focused seed check。`laser_power`、`scan_speed`、`spot_size` 均有三 seed process-conditioned gain；`line` 与 full `process` 仍弱于 train-mean baseline，下一步应扩大热/工艺数据或增强 baseline-facing 建模。
- Phase 29 已完成 broad12 process-balanced smoke。`process_round_robin` 选线覆盖多个功率、扫描速度和光斑尺寸；`spot_size` 的 FiLM/global-standard profile 仍强于 no-process 和 mean baseline，但 `scan_speed` 与 full `process` 在更宽数据面退化，下一步应做可回退 no-process 的 broad-data route selector/profile，再决定是否扩到 21 条 single-track lines。
- Phase 30 已完成 broad12 broad-data selector。`broad_process_v1` 把 `line`、`scan_speed` 和 full `process` 回退到 no-process，保留 `laser_power` concat/global-standard 与 `spot_size` FiLM/global-standard；summary 脚本会检查 manifest/split 签名，避免 tiny smoke 与 full run 混比。下一步扩到 21 条 single-track lines。
- Phase 31 已完成 broad21 all single-track selector scaling。all-21 结果确认 `broad_process_v1` 能保留 `laser_power`/`spot_size` 正向路由并避免 `scan_speed`/full `process` 负迁移；当前 A100-SXM4-40GB 仍足够。
- Phase 32 已完成 `broad_process_v2` 诊断。v2 只把 `line_id` 改为 concat/same；它改善 broad21 line，但明显伤害 broad12 line，因此不替代 `broad_process_v1` 默认路线。下一步转向更强 broad-data representation 或 closure/GNN reintegration。
- Phase 33 已完成 fixed Fourier spacetime representation 诊断。`--spacetime-encoding fourier` 与 `--spacetime-fourier-bands` 已实现并记录到 metrics/checkpoint；但 broad12 同口径结果在所有 split 均弱于 `broad_process_v1`，因此 Fourier 不替代 raw coordinate/time basis，下一步应转向 closure/GNN reintegration 或更结构化的数据表示。

详细命令见 [docs/server_runbook.md](docs/server_runbook.md)，完整推进方案见 [docs/server_execution_plan.md](docs/server_execution_plan.md)。

## Documentation

- [docs/data_direction_report.md](docs/data_direction_report.md): 公开数据方向评估。
- [docs/research_route.md](docs/research_route.md): 方向一与方向三关系及路线可行性。
- [docs/repo_architecture.md](docs/repo_architecture.md): 代码仓库结构和模块职责。
- [docs/experiment_plan.md](docs/experiment_plan.md): 实验矩阵、指标和配置命名。
- [docs/environment.md](docs/environment.md): 本地与远程环境迁移。
- [docs/server_environment_snapshot.md](docs/server_environment_snapshot.md): 当前 A100 服务器 Ubuntu/GPU/CUDA/Conda 环境快照。
- [docs/ambench_downloads.md](docs/ambench_downloads.md): AM-Bench 下载、断点处理和校验。
- [docs/data_cards/ambench_2022_optical_microscopy_mds2_2718.md](docs/data_cards/ambench_2022_optical_microscopy_mds2_2718.md): AM-Bench optical microscopy 数据卡。
- [docs/server_runbook.md](docs/server_runbook.md): 云 GPU 阶段运行手册。
- [docs/server_execution_plan.md](docs/server_execution_plan.md): 后续服务器研发推进总方案。
- [docs/closure_mvp_execution_plan.md](docs/closure_mvp_execution_plan.md): sparse closure MVP 与后续 residual sampling 方案。
- [docs/results/ambench_dense_temperature_server_v1.md](docs/results/ambench_dense_temperature_server_v1.md): 第一轮 A100 dense temperature 结果。
- [docs/results/ambench_dense_stage_a_normalization_baselines.md](docs/results/ambench_dense_stage_a_normalization_baselines.md): 归一化与强 baseline 结果。
- [docs/results/ambench_dense_region_metrics_q90.md](docs/results/ambench_dense_region_metrics_q90.md): q90 区域指标结果。
- [docs/results/ambench_dense_active_sampling_v1.md](docs/results/ambench_dense_active_sampling_v1.md): hot/gradient active sampling 结果。
- [docs/results/ambench_macro_pinn_screen_matrix_v1.md](docs/results/ambench_macro_pinn_screen_matrix_v1.md): Macro PINN 小矩阵与 3 seed 稳健性结果。
- [docs/results/ambench_sparse_closure_mvp_v1.md](docs/results/ambench_sparse_closure_mvp_v1.md): sparse source closure MVP 诊断结果。
- [docs/results/ambench_sparse_closure_residual_sampling_v1.md](docs/results/ambench_sparse_closure_residual_sampling_v1.md): random residual sampling closure 结果。
- [docs/results/ambench_sparse_closure_region_residual_sampling_v1.md](docs/results/ambench_sparse_closure_region_residual_sampling_v1.md): hot/gradient residual sampling 诊断结果。
- [docs/results/ambench_sparse_closure_staged_v1.md](docs/results/ambench_sparse_closure_staged_v1.md): staged sparse closure 诊断结果。
- [docs/results/ambench_sparse_closure_optimizer_ablation_v1.md](docs/results/ambench_sparse_closure_optimizer_ablation_v1.md): closure learning-rate 与 backbone freezing 消融结果。
- [docs/results/ambench_graph_conditioned_closure_toy_v1.md](docs/results/ambench_graph_conditioned_closure_toy_v1.md): toy/static graph-conditioned closure 接口验证结果。
- [docs/results/ambench_graph_conditioned_closure_coordinate_rbf_v1.md](docs/results/ambench_graph_conditioned_closure_coordinate_rbf_v1.md): coordinate RBF per-point graph-conditioned closure 诊断结果。
- [docs/results/ambench_graph_conditioned_closure_gated_v1.md](docs/results/ambench_graph_conditioned_closure_gated_v1.md): gated graph-conditioned closure 诊断结果。
- [docs/results/ambench_graph_closure_l1_sensitivity_v1.md](docs/results/ambench_graph_closure_l1_sensitivity_v1.md): graph-conditioned closure L1 sensitivity 与分支决策。
- [docs/results/ambench_microstructure_mds2_2718_entry_v1.md](docs/results/ambench_microstructure_mds2_2718_entry_v1.md): AM-Bench optical microscopy 下载、校验和 coarse micro graph 首次真实入口。
- [docs/results/ambench_real_micro_graph_conditioning_smoke_v1.md](docs/results/ambench_real_micro_graph_conditioning_smoke_v1.md): real micro graph feature 接入 sparse closure 的服务器 smoke。
- [docs/results/ambench_real_micro_graph_conditioned_closure_v1.md](docs/results/ambench_real_micro_graph_conditioned_closure_v1.md): real micro graph-conditioned sparse closure 首轮 A100 对比结果。
- [docs/results/ambench_real_micro_graph_conditioned_closure_panel_framecycle_v1.md](docs/results/ambench_real_micro_graph_conditioned_closure_panel_framecycle_v1.md): multi-image real micro panel frame-cycle prototype 对齐结果。
- [docs/results/ambench_real_micro_exact_line0_1_closure_v1.md](docs/results/ambench_real_micro_exact_line0_1_closure_v1.md): exact `Line_0_1` P3/P4 real micro closure 与 focused seed check 结果。
- [docs/results/ambench_real_micro_exact_line0_1_feature_v2_closure_v1.md](docs/results/ambench_real_micro_exact_line0_1_feature_v2_closure_v1.md): exact `Line_0_1` v2 rich micrograph features 对照结果。
- [docs/results/ambench_real_micro_exact_line0_1_feature_v2_no_norm_closure_v1.md](docs/results/ambench_real_micro_exact_line0_1_feature_v2_no_norm_closure_v1.md): exact `Line_0_1` v2 no-normalization real-micro 诊断结果。
- [docs/results/ambench_real_micro_exact_line0_1_region_closure_v1.md](docs/results/ambench_real_micro_exact_line0_1_region_closure_v1.md): exact `Line_0_1` deterministic region-level real-micro 诊断与 focused seed check 结果。
- [docs/results/ambench_real_micro_exact_line0_1_region_registration_v1.md](docs/results/ambench_real_micro_exact_line0_1_region_registration_v1.md): exact `Line_0_1` region coordinate-registration 消融与 `col_flip` focused seed check 结果。
- [docs/results/ambench_real_micro_exact_line0_1_region_embedding_v1.md](docs/results/ambench_real_micro_exact_line0_1_region_embedding_v1.md): exact `Line_0_1` fixed patch embedding 结果与 focused seed check 结果。
- [docs/results/ambench_multiline_process_conditioned_thermal_v1.md](docs/results/ambench_multiline_process_conditioned_thermal_v1.md): multi-line/process-conditioned thermal modeling 的首轮 A100 对比结果。
- [docs/results/ambench_multiline_process_axis_holdout_v1.md](docs/results/ambench_multiline_process_axis_holdout_v1.md): process-axis grouped holdout 对比结果与 FiLM 分支决策。
- [docs/results/ambench_multiline_process_film_conditioned_v1.md](docs/results/ambench_multiline_process_film_conditioned_v1.md): FiLM、global process-feature normalization、concat/global-standard、hybrid 条件化与 focused seed check 结果。
- [docs/results/ambench_multiline_process_axis_routing_v1.md](docs/results/ambench_multiline_process_axis_routing_v1.md): routed dual-expert 负结果与 `process_axis_v1` 显式轴感知 profile 结果。
- [docs/results/ambench_multiline_process_axis_profile_validation_v1.md](docs/results/ambench_multiline_process_axis_profile_validation_v1.md): `process_axis_v1` 的 `laser_power`/full `process` 扩展、强 baseline 汇总与 `laser_power` focused seed check。
- [docs/results/ambench_multiline_process_broader_profile_smoke_v1.md](docs/results/ambench_multiline_process_broader_profile_smoke_v1.md): Phase 29 broad12 process-balanced dataset smoke、baseline-facing profile transfer 诊断与下一步 route-selector 决策。
- [docs/results/ambench_multiline_process_broad_selector_v1.md](docs/results/ambench_multiline_process_broad_selector_v1.md): Phase 30 broad-data selector、no-process fallback 元数据验证与 broad12 可比性门禁结果。
- [docs/results/ambench_multiline_process_broad_selector_broad21_v1.md](docs/results/ambench_multiline_process_broad_selector_broad21_v1.md): Phase 31 all-21 single-track selector scaling 与 broad21 可比性门禁结果。
- [docs/results/ambench_multiline_process_broad_selector_v2.md](docs/results/ambench_multiline_process_broad_selector_v2.md): Phase 32 `broad_process_v2` line-route diagnostic 与 broad12/broad21 可比性门禁结果。
- [docs/results/ambench_multiline_process_fourier_spacetime_v1.md](docs/results/ambench_multiline_process_fourier_spacetime_v1.md): Phase 33 fixed Fourier spacetime representation 诊断、broad12 可比性门禁结果与不扩到 broad21 的决策。

Real micro graph closure 对比脚本：

```bash
bash scripts/server/run_real_micro_graph_conditioned_closure_a100.sh \
  > logs/ambench_real_micro_graph_conditioned_closure_a100_v1.log 2>&1
```

## Public Resources

- NIST AM-Bench: https://www.nist.gov/ambench
- AM-Bench benchmark test data: https://www.nist.gov/ambench/benchmark-test-data
- NIST PDR `mds2-2716`: https://data.nist.gov/od/id/mds2-2716
- ExaCA: https://github.com/LLNL/ExaCA
- PFHub benchmarks: https://pages.nist.gov/pfhub/benchmarks/

## License and Data Notes

代码许可证待正式补充。AM-Bench 数据不直接提交到本仓库；请通过 NIST PDR 或项目内 manifest 下载，并在使用时遵守原始数据源的引用与许可要求。

