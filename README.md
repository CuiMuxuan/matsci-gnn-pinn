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

最近一次本地验证状态：`58 passed, 2 skipped`。

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

详细命令见 [docs/server_runbook.md](docs/server_runbook.md)，完整推进方案见 [docs/server_execution_plan.md](docs/server_execution_plan.md)。

## Documentation

- [docs/data_direction_report.md](docs/data_direction_report.md): 公开数据方向评估。
- [docs/research_route.md](docs/research_route.md): 方向一与方向三关系及路线可行性。
- [docs/repo_architecture.md](docs/repo_architecture.md): 代码仓库结构和模块职责。
- [docs/experiment_plan.md](docs/experiment_plan.md): 实验矩阵、指标和配置命名。
- [docs/environment.md](docs/environment.md): 本地与远程环境迁移。
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

