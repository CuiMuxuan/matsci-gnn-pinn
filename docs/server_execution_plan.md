# 服务器推进完整执行方案

## 目标与当前基线

本方案用于后续在远程 A100/A800 服务器上持续推进 `matsci-gnn-pinn` 项目，目标是把当前已跑通的 AM-Bench dense 热场实验推进到可写论文的 GNN-PINN 多尺度耦合实验矩阵。

当前已完成节点：

- GitHub、本地、服务器三端代码已同步到 `85323b0`。
- 远程服务器：Ubuntu 22, NVIDIA A100-SXM4-40GB, CUDA 12.4。
- Conda 环境：`gnnpinn`, Python 3.11, PyTorch 2.5.0+cu124。
- AM-Bench `mds2-2716` 已在服务器下载并 SHA256 校验通过。
- 已生成 dense calibrated temperature subset，`41054` 个有效点。
- 已完成 dense mean baseline、Macro PINN data-only、Macro PINN + PDE residual。
- 结果文档：`docs/results/ambench_dense_temperature_server_v1.md`。

当前关键结论：

| Method | Test RMSE | Test MAE | Test Relative L2 |
| --- | ---: | ---: | ---: |
| Mean baseline | 108.387552 | 85.212557 | 0.091644 |
| Macro PINN data-only | 148.265776 | 102.813809 | 0.125361 |
| Macro PINN + PDE residual | 151.394768 | 106.453019 | 0.128007 |

这说明服务器端到端流程已经可用，但当前 Macro PINN 尚未超过简单 baseline。下一阶段必须先提升数据表示、采样、归一化、训练和 baseline 体系，再进入 closure/GNN/coupling。

## 执行原则

1. GitHub `main` 是代码与文档真源。
2. 服务器 `/root/matsci-gnn-pinn` 负责运行实验，保留 `data/`、`outputs/`、`logs/`。
3. 本地仓库负责审阅、必要代码修改、文档合并和最终推送。
4. `data/raw/`、`data/interim/`、`outputs/`、`logs/` 不提交 Git。
5. 每个实验必须有唯一 run id、命令、数据 manifest、split manifest、metrics、环境冻结文件和结果说明。
6. 远程服务器可直接读、写、改；删除服务器文件前必须单独确认。
7. 每轮服务器实验结束后必须同步三端状态：

```bash
# local
git status --short --branch
git rev-parse --short HEAD
git ls-remote origin refs/heads/main

# server
cd /root/matsci-gnn-pinn
git fetch origin
git status --short --branch
git rev-parse --short HEAD
git rev-parse --short origin/main
```

## 服务器固定入口

登录：

```bash
ssh -i ~/.ssh/matsci_gnnpinn_a100 -p 25820 root@223.109.239.30
```

项目目录：

```bash
cd /root/matsci-gnn-pinn
```

环境检查：

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
/home/vipuser/miniconda3/bin/conda run -n gnnpinn python scripts/env/check_env.py

PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
/home/vipuser/miniconda3/bin/conda run -n gnnpinn python -m pytest -q --basetemp .pytest_tmp
```

后台运行推荐：

```bash
tmux new-session -d -s <session_name> "<command> > logs/<run_id>.log 2>&1"
tmux ls
tail -f logs/<run_id>.log
```

## 阶段 A：修正当前 Macro PINN 短板

### A1. 增加坐标、时间、目标值归一化审计

目标：确认当前 Macro PINN 失败是否主要来自输入尺度不良、时间尺度过小、像素坐标与物理坐标混用。

执行内容：

- 在数据转换 manifest 中记录 `x/y/t/temperature_C` 的 min/max/mean/std。
- 在训练 metrics 中记录输入归一化参数。
- 增加 `--normalize-coordinates` 或配置化 normalization。
- 对比 raw pixel coordinate、min-max coordinate、standardized coordinate。

验收：

- 至少得到 3 个 data-only run。
- test RMSE 明显接近或低于 mean baseline，或能清楚证明模型容量/采样仍是主因。

建议 run ids：

```text
ambench_dense_a100_norm_none_data_only_v1
ambench_dense_a100_norm_minmax_data_only_v1
ambench_dense_a100_norm_standard_data_only_v1
```

### A2. 增加强 baseline

目标：mean baseline 太弱，不能支撑论文结论。需要加入更强的非物理对照。

优先实现：

- kNN / RBF interpolation baseline。
- RandomForest / ExtraTrees baseline。
- small MLP data-only baseline。
- frame-wise persistence 或 temporal interpolation baseline。

指标：

| Baseline | 必需指标 |
| --- | --- |
| Mean | MAE, RMSE, relative L2 |
| kNN/RBF | MAE, RMSE, relative L2, fit/predict time |
| Tree ensemble | MAE, RMSE, relative L2, feature importance |
| MLP | MAE, RMSE, relative L2, train time |

验收：

- 至少一个强 baseline 作为后续 Macro PINN 和 closure 的主要对照。
- 如果 Macro PINN 仍弱于强 baseline，先优化模型和采样，不进入论文主张。

### A3. 熔池活跃区域采样

目标：当前 frame split 中大量背景/弱变化点会稀释训练信号。应更关注高温区、温度梯度大区和时间变化强区。

执行内容：

- 基于 `temperature_C` 阈值采样：例如 top temperature quantile。
- 基于 `signal` 或 `temperature_C` 的空间梯度采样。
- 每个 frame 保留背景点、边界点、高温点的固定比例。
- 在 manifest 记录采样策略。
- 已实现 HDF5 转换器参数：`--sampling-mode uniform|hot|gradient|hot_gradient|balanced_hot_gradient`、`--hot-quantile`、`--gradient-quantile`、`--background-fraction`、`--max-points-per-frame`。
- 服务器一键脚本：`scripts/server/run_dense_active_sampling_a100.sh`，会生成 balanced hot/gradient 数据集，并运行 mean/kNN/RandomForest/ExtraTrees baseline 与 minmax Macro PINN。

建议样本集：

```text
dense_uniform_v1
dense_hot_quantile_v1
dense_gradient_band_v1
dense_balanced_hot_bg_v1
```

验收：

- 采样后 test split 仍有足够点数。
- melt-pool/hot-zone 指标单独报告，而不是只看全局 RMSE。

推荐首轮服务器命令：

```bash
cd /root/matsci-gnn-pinn
bash scripts/server/run_dense_active_sampling_a100.sh > logs/ambench_dense_active_sampling_a100_v1.log 2>&1
```

## 阶段 B：把 Macro PINN 做成可靠基线

### B1. 训练矩阵

在修正归一化和采样后，运行以下矩阵：

| Group | Variables |
| --- | --- |
| Model width | 64, 128, 256 |
| Layers | 3, 4, 6 |
| Steps | 2000, 5000, 10000 |
| LR | 1e-3, 3e-4, 1e-4 |
| Seeds | 0, 1, 2 |
| PDE weight | 0, 1e-8, 1e-7, 1e-6, 1e-5 |

第一轮不要全组合，采用逐步筛选：

1. 固定 `pde_weight=0`，找 data-only 最优容量和学习率。
2. 固定最佳 data-only 配置，扫描 PDE weight。
3. 固定最佳 PDE weight，跑 3 seeds。

当前筛选脚本：

```bash
cd /root/matsci-gnn-pinn
bash scripts/server/run_macro_pinn_screen_matrix_a100.sh > logs/ambench_macro_pinn_screen_matrix_a100_v1.log 2>&1
```

首轮脚本固定 `--input-normalization minmax`、`pde_weight=0`、q90 区域指标，在 uniform dense 与 balanced hot/gradient active 两张表上分别跑 4 个配置：`h64/l3/lr1e-3`、`h128/l4/lr1e-3`、`h128/l4/lr3e-4`、`h256/l4/lr1e-3`。如果该筛选显示一个配置稳定领先，再补 3 seed 与 PDE/closure 分支。

筛选后 seed-check 脚本：

```bash
cd /root/matsci-gnn-pinn
bash scripts/server/run_macro_pinn_seed_check_a100.sh > logs/ambench_macro_pinn_seed_check_a100_v1.log 2>&1
```

该脚本只补候选配置的 seed 1/2：uniform dense 的 `h128/l4/lr1e-3`，active 表的 `h128/l4/lr3e-4` 与 `h256/l4/lr1e-3`。

验收：

- data-only Macro PINN 至少接近强 baseline。
- PDE residual 不能显著恶化 test split。
- 至少有 3 seed 的均值和标准差。

### B2. PDE residual 合理化

目标：当前 PDE residual 使用常数 `conductivity=1.0`、`rho_cp=1.0`，只适合代码验证。下一步必须引入更合理的物理尺度。

执行内容：

- 梳理 AMB2022-03 材料与工艺参数。
- 引入温度/材料尺度归一化后的无量纲 PDE residual。
- 加入热源项或把热源项作为 closure。
- 只在局部 collocation points 上计算 PDE residual，控制显存。

验收：

- PDE residual loss 数值范围稳定。
- PDE weight 扫描呈现可解释趋势。
- PDE run 至少在 OOD frame 或 hot-zone 指标上不劣于 data-only。

## 阶段 C：方向一入口，闭合项发现

### C1. 热源/有效参数 closure MVP

目标：把“发现完整本构”降为“发现闭合项/修正项”，作为方向一可落地入口。

第一版 closure 目标：

```text
heat residual = known transient/diffusion terms + q_closure(features)
```

候选 features：

```text
T, x, y, t, grad_T_norm, laplacian_T, laser_power_W, scan_speed_mm_s, spot_size_um
```

执行内容：

- 实现 trainable sparse closure module。
- 加 L1 penalty 和 thresholding。
- 用 SymPy 导出表达式。
- 对比 fixed PDE、black-box closure、sparse closure。

验收：

- sparse closure test/hot-zone 指标优于 fixed PDE。
- active terms 稳定，3 seeds 下主项一致。
- 导出表达式可写入论文附录。

当前执行状态：

- 已实现 `--closure-mode sparse_linear`、`--pde-field normalized`、closure feature/library/threshold/L1 参数。
- 已完成首轮 sparse source closure MVP 与低权重修正扫描。
- 结果文档：`docs/results/ambench_sparse_closure_mvp_v1.md`。
- 结论：功能链路可复现，但当前 all-point residual formulation 未超过 data-only；`pde_weight=1e-3/1e-4` 明显过强，`1e-6` 可稳定但 hot/gradient 指标仍弱。
- 下一步应先实现 residual/collocation sampling，而不是扩大 sparse library 或跑 3 seed。

推荐下一步 C1b：

```text
Add residual sampling:
- --residual-sample-size
- --residual-sampling-mode random|hot|gradient|hot_gradient
- --residual-sampling-seed
```

验收：在 `pde_weight=1e-6` 附近，closure run 至少不显著恶化 data-only hot q90 与 gradient q90 指标。

当前 C1b 状态：

- 已实现 random residual sampling：`--residual-sample-size` 与 `--residual-sampling-seed`。
- 已完成 `2048/4096` random residual sampled closure 实验。
- 结果文档：`docs/results/ambench_sparse_closure_residual_sampling_v1.md`。
- 结论：random residual sampling 明显改善 active closure 的全局与梯度指标，但仍不超过 data-only；下一步应实现 hot/gradient residual sampling modes。

当前 C1c 状态：

- 已实现 `--residual-sampling-mode random|hot|gradient|hot_gradient`。
- 已完成 active 表上的 hot/gradient/hot_gradient residual sampling。
- 结果文档：`docs/results/ambench_sparse_closure_region_residual_sampling_v1.md`。
- 结论：region-aware residual sampling 比 random 更差，说明当前 residual/closure 形式本身仍过强；下一步应做 staged training 或 warm-started closure fine-tuning。

推荐下一步 C1d：

```text
Staged closure training:
- --closure-start-step
- optionally --freeze-backbone-after-closure-start
- optionally --closure-lr
```

验收：closure 在后期作为小修正项打开时，不显著损害 data-only hot q90 与 gradient q90，并保存可解释表达式。

当前 C1d 状态：

- 已实现 `--closure-start-step`。
- 已完成 active 表 staged closure 训练。
- 结果文档：`docs/results/ambench_sparse_closure_staged_v1.md`。
- 结论：`closure_start_step=1500` 明显优于过早打开 closure，但仍未超过 data-only；下一步需要把 closure coefficients 与 Macro PINN backbone 的优化分离。

推荐下一步 C1e：

```text
Separate closure optimizer behavior:
- --closure-lr
- --freeze-backbone-after-closure-start
```

验收：在冻结或低学习率 closure fine-tuning 下，closure 不显著损害 active data-only hot q90，并能输出稳定的稀疏表达式。

当前 C1e 本地实现状态：

- 已实现 `--closure-lr`，closure coefficients 可以使用独立学习率；未指定时默认等于 `--lr`。
- 已实现 `--freeze-backbone-after-closure-start`，在 PDE/closure residual 真正启用后冻结 Macro PINN backbone，只微调 closure coefficients。
- metrics/checkpoint 会保存 optimizer 配置；history 会保存 `backbone_frozen` 标记。
- 本地测试：`tests/test_macro_pinn_train.py` 已覆盖 closure lr 与冻结逻辑。
- 已完成 A100 optimizer ablation。
- 结果文档：`docs/results/ambench_sparse_closure_optimizer_ablation_v1.md`。
- 结论：`closure_lr=1e-5` 且 backbone 继续训练是当前最佳 C1 设置，test RMSE `70.494433`、hot q90 RMSE `31.542155`、gradient q90 RMSE `64.558069`；冻结 backbone 没有带来收益。该路线比 staged baseline 明显更稳，但仍未超过 active data-only 的 hot q90。

下一轮服务器命令：

```bash
bash scripts/server/run_sparse_closure_optimizer_ablation_a100.sh \
  > logs/ambench_sparse_closure_optimizer_ablation_a100_v1.log 2>&1
```

实验矩阵固定 active 表、`closure_start_step=1500`、`residual_sample_size=4096`、`pde_weight=1e-6`，只比较：

| closure lr | Backbone | 目的 |
| ---: | --- | --- |
| `1e-4` | trainable | 比主网络慢一档学习 closure |
| `1e-5` | trainable | 极低 closure lr，测试 source 小扰动 |
| `1e-4` | frozen after warmup | 数据拟合先收敛，后续只学习 closure |
| `1e-5` | frozen after warmup | 最保守 closure fine-tuning |

若冻结方案仍明显弱于 data-only，则 C1 路线应转入“closure 负结果 + GNN-conditioned/toy interface”并推进阶段 D，而不是继续堆叠 sparse polynomial order。

### C2. Closure 消融

必跑对照：

| Method | 目的 |
| --- | --- |
| Macro PINN data-only | 无物理下限 |
| fixed PDE PINN | 固定物理项 |
| black-box neural closure | 非解释闭合 |
| sparse closure | 方向一主方法 |

验收：

- sparse closure 不只是在 train split 好，在 val/test 或 OOD frame 上也稳定。

## 阶段 D：GNN 微观结构编码

### D1. 先做 synthetic/toy graph 到真实接口

目标：在真实 AM-Bench 微观组织数据未完全对齐前，先把图接口、GNN encoder、coarse grainer 与 closure 接通。

执行内容：

- 使用现有 `MicroGNNEncoder` 跑 toy graph。
- 增加 graph batch schema。
- 将 micro embedding 拼接到 closure features。
- 在 `MacroPINN` closure 训练入口中加入可选 graph embedding provider，先支持 toy/static embedding。
- 复用 C1 最佳设置：`closure_lr=1e-5`、`closure_start_step=1500`、`residual_sample_size=4096`。
- 与 sparse closure baseline 在 active AM-Bench 表上做同 split 对照。
- 先做 synthetic micro embedding 消融，确认训练接口可用。

验收：

- GNN-conditioned closure forward/backward 可跑。
- artifact 中保存 graph schema、embedding dim、coarse parameters。
- 至少有一个 toy/static graph-conditioned run 能生成完整 metrics/checkpoint/expression 或明确负结果。

当前 D1 本地实现状态：

- 已新增 `ToyStaticGraphEmbeddingProvider`，用 `MicroGNNEncoder` 编码 deterministic toy graph。
- `MacroPINN` training CLI 已支持 `--closure-graph-mode toy_static`。
- closure features 会自动加入 `g0/g1/...`，并在 metrics/checkpoint 中保存 graph conditioning metadata。
- 已新增服务器脚本：`scripts/server/run_graph_conditioned_closure_toy_a100.sh`。
- 已完成 toy/static graph-conditioned closure 服务器实验。
- 结果文档：`docs/results/ambench_graph_conditioned_closure_toy_v1.md`。
- 结论：方向三训练接口已跑通，但 static global graph embedding 退化 hot/gradient 指标；下一步应做 per-point 或 region-aware graph conditioning，而不是继续使用全局常量 embedding。

下一轮服务器命令：

```bash
bash scripts/server/run_graph_conditioned_closure_toy_a100.sh \
  > logs/ambench_graph_conditioned_closure_toy_a100_v1.log 2>&1
```

下一步 D1b：

```text
Region-aware graph conditioning:
- deterministic anchors/RBF graph features from normalized x,y,t
- per-point g0/g1/... instead of global constants
- same active AM-Bench split and C1 best optimizer settings
```

当前 D1b 本地实现状态：

- 已新增 `CoordinateRBFGraphFeatureProvider`。
- `MacroPINN` training CLI 支持 `--closure-graph-mode coordinate_rbf`。
- `g0/g1/...` 现在可以是每个 residual point 的 anchor/RBF graph features，而不是全局常量。
- 已新增服务器脚本：`scripts/server/run_graph_conditioned_closure_coordinate_rbf_a100.sh`。
- 已完成 coordinate RBF 服务器实验。
- 结果文档：`docs/results/ambench_graph_conditioned_closure_coordinate_rbf_v1.md`。
- 结论：`g6_ls0_25` 比 static toy graph 更好，test RMSE `68.237717`，但 hot q90 `50.900600` 与 gradient q90 `71.879264` 仍弱于 sparse closure best；下一步应做 graph contribution gating，而不是继续扩大 graph feature library。

下一轮服务器命令：

```bash
bash scripts/server/run_graph_conditioned_closure_coordinate_rbf_a100.sh \
  > logs/ambench_graph_conditioned_closure_coordinate_rbf_a100_v1.log 2>&1
```

下一步 D1c：

```text
Gated graph-conditioned closure:
- q = q_sparse + alpha * q_graph
- small/fixed or bounded learnable alpha
- separate graph coefficient penalty
- acceptance criterion: no hot q90 degradation relative to sparse closure best
```

当前 D1c 本地实现状态：

- 已新增 `--closure-graph-gate`，graph terms 对 source 的贡献按 gate 缩放。
- 已新增 `--closure-graph-l1-weight`，graph coefficients 与 base sparse coefficients 分开惩罚。
- 已新增服务器脚本：`scripts/server/run_graph_conditioned_closure_gated_a100.sh`。
- 已完成 gated graph closure 服务器实验。
- 结果文档：`docs/results/ambench_graph_conditioned_closure_gated_v1.md`。
- 结论：gate `0.25` 将 gradient q90 推到 `61.623894`，但 graph terms 被 `graph_l1_weight=1e-4` 几乎完全压没；下一步只做最小 graph-L1 sensitivity，不再扩大矩阵。

下一轮服务器命令：

```bash
bash scripts/server/run_graph_conditioned_closure_gated_a100.sh \
  > logs/ambench_graph_conditioned_closure_gated_a100_v1.log 2>&1
```

下一步 D1d：

```text
Graph L1 sensitivity:
- fixed gate=0.25
- graph_l1_weight=1e-5 and 1e-6
- require graph terms survive thresholding without hot q90 regression
```

当前 D1d 本地实现状态：

- 已新增服务器脚本：`scripts/server/run_graph_conditioned_closure_graph_l1_sensitivity_a100.sh`。
- 已完成 graph L1 sensitivity 服务器实验。
- 结果文档：`docs/results/ambench_graph_closure_l1_sensitivity_v1.md`。
- 结论：弱化 graph L1 后 graph terms 能保留，但 hot q90 和 gradient q90 急剧退化；当前 synthetic coordinate/RBF graph conditioning 不应继续扩大。

下一轮服务器命令：

```bash
bash scripts/server/run_graph_conditioned_closure_graph_l1_sensitivity_a100.sh \
  > logs/ambench_graph_conditioned_closure_graph_l1_sensitivity_a100_v1.log 2>&1
```

D1 阶段决策：

- C1 sparse closure 是当前最可信的可解释闭包基线。
- synthetic graph-conditioned closure 已跑通接口，但没有形成正结果。
- 下一步进入 D2：真实/半真实微观组织数据对齐，或用 ExaCA/PFHub 生成受控 micro graph 作为半真实条件。

### D2. 接入真实/半真实微观数据

数据优先级：

1. AM-Bench 同源 optical microscopy / cross-section microstructure。
2. ExaCA 生成补充组织数据。
3. PFHub/相场基准作为受控验证。

执行内容：

- 为每个数据源建立 data card。
- 建立样品级、区域级、晶粒级三层对齐策略。
- 先做区域级组织统计，不强行逐像素对齐。

验收：

- 至少一个真实或半真实 micro graph dataset 可复现生成。
- GNN-conditioned closure 相比手工统计特征有提升或给出明确失败原因。

当前 D2 实现状态：

- 已完成 AM-Bench microstructure 数据源评估。
- Phase 17 第一优先数据源确定为 `mds2-2718` optical microscopy，而不是直接进入更大的 `mds2-2775`。
- 原因：`mds2-2718` 与 AMB2022-03 同源，包含 single-track/pad optical microscopy、melt-pool cross-section measurement XLSX 和 TIFF checksum；全量约 10.6 GB，可先用一个代表性 TIFF 做低风险入口。
- 已新增 manifest：`configs/data/ambench_mds2_2718_sources.yaml`。
- 已将 `gnnpinn.data.ambench_downloads` 泛化为 `--dataset-id mds2-2716|mds2-2718`。
- 已新增显微图像 inspection / coarse graph 入口：`gnnpinn.data.loaders.ambench_microstructure`，并解析 `BP/P/L/replicate/view/masked` 等文件名元数据。
- 已新增 data card：`docs/data_cards/ambench_2022_optical_microscopy_mds2_2718.md`。

下一轮服务器命令：

```bash
cd /root/matsci-gnn-pinn
git pull --ff-only

PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
/home/vipuser/miniconda3/bin/conda run -n gnnpinn python -m gnnpinn.data.ambench_downloads \
  --dataset-id mds2-2718 \
  --root data/raw/ambench/2022_single_track/AMB2022-03/mds2-2718 \
  --download \
  --verify-sha256 \
  --output outputs/data_audits/ambench_mds2_2718_download_report.json

PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
/home/vipuser/miniconda3/bin/conda run -n gnnpinn python -m gnnpinn.data.loaders.ambench_microstructure \
  --image data/raw/ambench/2022_single_track/AMB2022-03/mds2-2718/Single_Track_Cross_Sections/AMB2022-718-SH1-BP1-P2-L2.1-3_m.tif \
  --sample-id AMB2022-718-SH1-BP1-P2-L2.1-3_m \
  --threshold-quantile 0.9 \
  --grid-rows 8 \
  --grid-cols 8 \
  --graph-k 4 \
  --output outputs/data_audits/ambench_mds2_2718_micrograph_inspection.json
```

D2 下一步研发：

```text
1. Convert one or more TIFF inspections into sample-level graph feature JSONL.
2. Add a graph-conditioning provider that reads real micro graph JSON rather than synthetic RBF anchors.
3. Compare real micro graph conditioning against scalar hand-crafted microscopy statistics.
4. If `mds2-2718` sample-level alignment is too weak, move to `mds2-2775` or ExaCA-generated microstructures as the second route.
```

当前 D2b 本地实现状态：

- 已新增 `--mode aggregate`，可将 inspection JSON 聚合成 `micro_graph_features.jsonl` 与 CSV。
- 已新增 `RealMicroGraphFeatureProvider`。
- `MacroPINN` training CLI 已支持 `--closure-graph-mode real_micro`、`--closure-graph-features`、`--closure-graph-sample-id`。
- 本地已新增 panel-aware 选择入口：当 thermal field table 带有 `micro_sample_id` 等逐行对齐列时，可用 `--closure-graph-sample-id-column micro_sample_id` 从 panel-level JSONL 按 residual point 选择 micro record。
- real micro graph features 暂时作为样品级固定特征注入 `g0/g1/...`，用于替代 synthetic RBF graph terms 进行接口验证。
- 本地测试通过：full suite `66 passed, 2 skipped`。

下一轮服务器命令：

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
/home/vipuser/miniconda3/bin/conda run -n gnnpinn python -m gnnpinn.data.loaders.ambench_microstructure \
  --mode aggregate \
  --inspection outputs/data_audits/ambench_mds2_2718_micrograph_inspection.json \
  --jsonl-output data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features.jsonl \
  --csv-output data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features.csv \
  --output outputs/data_audits/ambench_mds2_2718_micrograph_feature_table_manifest.json
```

完整 real-micro closure 对比脚本：

```bash
bash scripts/server/run_real_micro_graph_conditioned_closure_a100.sh \
  > logs/ambench_real_micro_graph_conditioned_closure_a100_v1.log 2>&1
```

默认运行 `embedding_dim=4/8`、gate `0.25`、graph L1 `1e-4`，使用与 gated coordinate-RBF 分支相同的 sparse closure 优化设置，便于和 `docs/results/ambench_graph_conditioned_closure_gated_v1.md` 对齐比较。
当使用已构建的 panel-level feature table 时，可设置 `MICRO_AGGREGATE=0`、`MICRO_FEATURES=data/processed/.../micro_graph_features_panel.jsonl`，并通过 `MICRO_SAMPLE_ID_COLUMN=micro_sample_id` 让训练入口逐行选择 micro record。

当前 D2c 决策：

- 单张 real-micro sample-level feature 的 `g8` run 对 hot q90 有改善，但 global RMSE 退化，说明一张全局 TIFF 广播到所有 thermal residual point 仍太粗。
- 下一步不扩大 closure 超参扫描，先把 `mds2-2718` 从单图扩展到多图 panel。
- `configs/data/ambench_mds2_2718_sources.yaml` 的 optional files 已固定一个小面板：P2/L2.1 masked + unmasked、P1/L3.1 masked、P3/L0 masked、P4/L0 masked + unmasked。
- 下载器已支持 `--include-optional`，也支持 `--file-id` 指定 required 或 optional 文件。
- 下载器已支持 `--retries`、`--timeout-seconds`、`--resume-partial` 与 `--download-backend curl|wget`，用于处理 NIST PDR 偶发 read timeout；服务器脚本默认 `DOWNLOAD_RETRIES=3`、`DOWNLOAD_TIMEOUT_SECONDS=300`、`DOWNLOAD_BACKEND=wget`，并启用续传。
- 本地 Windows 路径已完成 optional panel 下载、SHA256 校验、6 图 inspection 和 panel feature table 聚合；下一步应同步 `data/raw/.../mds2-2718`、`outputs/data_audits/mds2_2718_micro_panel/` 和 `data/processed/.../micro_graph_features_panel.*` 到服务器。
- 服务器环境需要 `imagecodecs` 才能读取 LZW-compressed TIFF；若 inspection 报 `<COMPRESSION.LZW: 5> requires the 'imagecodecs' package`，先执行 `/home/vipuser/miniconda3/bin/conda run -n gnnpinn python -m pip install imagecodecs`。

下一轮服务器命令：

```bash
bash scripts/server/build_mds2_2718_micro_panel_a100.sh \
  > logs/ambench_mds2_2718_micro_panel_build_a100_v1.log 2>&1
```

该脚本只构建数据资产，不启动长训练。预期产物：

```text
outputs/data_audits/ambench_mds2_2718_micro_panel_download_report.json
outputs/data_audits/mds2_2718_micro_panel/*_inspection.json
outputs/data_audits/ambench_mds2_2718_micro_panel_feature_table_manifest.json
data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features_panel.jsonl
data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features_panel.csv
```

验收：

- download report 中 required files ready，optional panel 文件存在且 SHA256 通过。
- feature table manifest 的 `n_records=6`。
- 每条 record 均带有可解析的 `process/line/replicate/masked` metadata。
- 通过该表生成或补充 thermal field table 的 `micro_sample_id` 对齐列后，使用 `--closure-graph-sample-id-column` 跑 process/sample-aware `real_micro` closure，而不是继续广播单一图像特征。

生成 prototype 对齐表：

```bash
bash scripts/server/create_mds2_2718_micro_panel_aligned_table_a100.sh
```

默认输出：

```text
data/interim/ambench/2022_single_track/AMB2022-03/ambench_line_0_1_temperature_hot_gradient_mds2_2718_micro_panel_framecycle_v1.csv
outputs/data_audits/ambench_line_0_1_temperature_hot_gradient_mds2_2718_micro_panel_framecycle_v1_manifest.json
```

该表使用 `frame_cycle` prototype 映射：按排序后的 `frame_index` 循环分配 6 个 panel `sample_id` 到 `micro_sample_id`。它只用于验证逐行 `real_micro` 选择链路，不作为真实 process-to-microstructure ground truth。

运行 panel-aware closure：

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

当前 D2d 结果：

- `frame_cycle` prototype 对齐链路已在 A100 跑通，结果文档为 `docs/results/ambench_real_micro_graph_conditioned_closure_panel_framecycle_v1.md`。
- `g4` test RMSE `103.994793`，hot q90 RMSE `179.742978`，gradient q90 RMSE `175.162047`。
- `g8` test RMSE `130.057854`，hot q90 RMSE `213.906003`，gradient q90 RMSE `206.215815`。
- 该结果明显差于 sparse closure best 与 single-image real micro `g8`，应视为非物理 sample assignment 的负面控制；下一步应寻找真实 process/sample alignment，而不是继续调该 frame-cycle mapping 的超参。

基于 `AMB2022-718-SH1-MeltPool_Cross-Section_Measurement_Results.xlsx` 的同工艺对齐：

- 当前 active thermal table 来自 `Line_0_1`，工艺参数为 `285 W / 960 mm/s / 67 um`。
- XLSX 中 BP1/P3 与 BP1/P4 的 `Line 0_*` 样本对应同一组工艺参数。
- 因此下一轮不再使用 frame-cycle，而是固定 same-process micro sample 运行 `P3/L0` 与 `P4/L0`：

```bash
bash scripts/server/run_real_micro_same_process_line0_a100.sh \
  > logs/ambench_real_micro_same_process_line0_a100_v1.log 2>&1
```

如果 single-seed 结果出现正向信号，优先做小 seed check，而不是立刻扩大模型：

```bash
bash scripts/server/run_real_micro_same_process_seed_check_a100.sh \
  > logs/ambench_real_micro_same_process_seed_check_a100_v1.log 2>&1
```

## 阶段 E：方向三弱双向耦合

### E1. Weak coupling MVP

目标：实现最小方向三闭环，控制训练难度。

流程：

```text
MacroPINN(x, t) -> local thermal state
MicroGNN(graph, local state/history) -> micro embedding
CoarseGrainer(micro embedding) -> effective params
MacroPINN(x, t, effective params) -> updated field
```

执行内容：

- 冻结 MicroGNN 与 CoarseGrainer，训练 MacroPINN。
- 每 N epoch 更新一次 effective params。
- 只在 hot-zone 或 gradient-band 点反馈。
- 对比 one-way 和 two-way。

验收：

- weak coupling 不发散。
- two-way 至少在一个 OOD/hot-zone 指标优于 one-way。
- 训练成本可记录、可复现。

### E2. 论文主实验矩阵

主表方法：

| Method | 必跑 |
| --- | --- |
| Mean / strong data baseline | yes |
| Pure Macro PINN | yes |
| Fixed PDE PINN | yes |
| PINN + black-box closure | yes |
| PINN + sparse closure | yes |
| GNN-conditioned closure | yes |
| One-way GNN-PINN | yes |
| Two-way GNN-PINN | yes |

主指标：

```text
temperature RMSE / MAE / relative L2
hot-zone RMSE
melt-pool contour error or IoU
PDE residual
closure sparsity
OOD frame/process error
training time and GPU memory
```

## 阶段 F：论文资产生成

### F1. 结果表与图

每个结果必须能由命令重建：

```bash
python -m gnnpinn.eval.export_tables --run outputs/runs/<run_id>
python -m gnnpinn.viz.paper_figures --run outputs/runs/<run_id>
```

需要生成：

- 数据流程图。
- GNN-PINN 框架图。
- 温度场预测对比图。
- hot-zone / melt-pool 局部误差图。
- closure active terms 图。
- 消融柱状图。
- 多 seed 均值方差表。

### F2. 论文叙事门槛

进入论文初稿前必须满足：

- 强 baseline 已建立。
- 至少一个 PINN/closure 模型在关键 split 上优于强 baseline 或给出明确可发表的负结果分析。
- sparse closure 有稳定项或可解释失败原因。
- GNN/coupling 至少完成 MVP forward/backward 与一个对照实验。
- 所有核心结果有命令、config、artifact、环境冻结。

## 每轮服务器实验标准流程

### 1. 同步代码

```bash
cd /root/matsci-gnn-pinn
git fetch origin
git status --short --branch
git pull --ff-only
```

### 2. 检查环境

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
/home/vipuser/miniconda3/bin/conda run -n gnnpinn python scripts/env/check_env.py

PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
/home/vipuser/miniconda3/bin/conda run -n gnnpinn python -m pytest -q --basetemp .pytest_tmp
```

### 3. 启动实验

```bash
mkdir -p logs
tmux new-session -d -s <session_name> "<command> > logs/<run_id>.log 2>&1"
```

### 4. 监控

```bash
tmux ls
tail -n 120 logs/<run_id>.log
nvidia-smi
```

### 5. 冻结环境

```bash
RUN_DIR=outputs/runs/<run_id>
/home/vipuser/miniconda3/bin/conda env export -n gnnpinn --from-history > "$RUN_DIR/conda_from_history.yml"
/home/vipuser/miniconda3/bin/conda run -n gnnpinn python -m pip freeze > "$RUN_DIR/pip_freeze.txt"
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 /home/vipuser/miniconda3/bin/conda run -n gnnpinn python scripts/env/check_env.py > "$RUN_DIR/env_report.txt"
nvidia-smi > "$RUN_DIR/nvidia_smi.txt"
```

### 6. 写结果文档

结果文档放入：

```text
docs/results/<experiment_name>.md
```

必须包含：

- run context。
- dataset and split。
- command/config。
- metrics table。
- artifact paths。
- interpretation。
- next action。

### 7. 同步三端

如果只新增文档/脚本/config：

```bash
git add <files>
git commit -m "<message>"
git push origin main
```

如果服务器没有 GitHub 凭据，则从本地拉取服务器提交后推送：

```bash
git -c core.sshCommand="ssh -i C:/Users/cjh02/.ssh/matsci_gnnpinn_a100 -p 25820 -o IdentitiesOnly=yes" \
  fetch ssh://root@223.109.239.30/root/matsci-gnn-pinn main:refs/remotes/server/main
git merge --ff-only server/main
git push origin main
```

然后服务器执行：

```bash
cd /root/matsci-gnn-pinn
git fetch origin
git status --short --branch
git rev-parse --short HEAD
git rev-parse --short origin/main
```

## 立即下一步建议

下一轮服务器研发按以下顺序推进：

1. 实现坐标/时间归一化与数据统计记录。
2. 增加 kNN/RBF 或 tree ensemble 强 baseline。
3. 重新跑 dense data-only normalization ablation。
4. 设计 hot-zone / gradient-band 采样。
5. 用最优采样和归一化重跑 Macro PINN data-only 与 PDE weight scan。
6. 若 Macro PINN 接近或超过强 baseline，进入 sparse closure MVP；否则先继续修正数据表示和训练策略。

这条路线优先解决当前真实结果暴露出的短板，比直接进入 GNN/coupling 更稳。GNN 与方向三仍是最终目标，但必须建立在可靠宏观热场基线和可解释 closure 上。
