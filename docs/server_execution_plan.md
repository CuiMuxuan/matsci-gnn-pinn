# 服务器推进完整执行方案

## 目标与当前基线

本方案用于后续在远程 A100/A800 服务器上持续推进 `matsci-gnn-pinn` 项目，目标是把 AM-Bench thermal/process 实验推进到可写论文的 route-guarded Macro PINN 结果，并在通过明确 gate 后继续尝试更多模型创新和模型架构。

当前权威推进状态：

- Phase 55 固定采样 broad12/broad21 `spot_size` 三 seed 结果是当前 frozen paper-facing floor。
- Phase 58 stronger-baseline stress 和 broad15 auxiliary panel 支持该 fixed-sampling floor，但 alternate-density broad21 暴露 density-sensitive boundary。
- Phase 59 no-test-leakage upper-bound probe 选择 `blend:broad_process_v1->mean:alpha=1`，因此 density failure 不能直接驱动新模型分支。
- Phase 60 已生成 manuscript evidence package，并记录 `block_density_failure_driven_model_expansion`。
- Phase 61 已生成 manuscript results/methods/caption draft package，并已完成 Phase 66 local/GitHub/server closeout/sync。
- Phase 68 已生成 validation-visible signal scorecard；当前 `opened_trainable_candidates=0`，下一步是 non-training `spot_size` signal probe 或 manuscript v0 claim audit。
- Phase 69 已完成 non-training `spot_size` signal probe；Candidate A 保持 `paused_no_training_signal`，当前不进入 A100 seed-7 训练，也不请求 A100-SXM4-80GB。
- Phase 70 已完成 route-policy non-training audit；Candidate B 为 `blocked_no_validation_visible_route_policy_signal`，当前不进入 route-policy 或 mixture-of-experts 训练。
- Phase 71 已完成 data-registration non-training audit；Candidate C 为 `blocked_by_registration_data`，当前不进入 heat-kernel/Green's-function/source-path A100 训练。
- Phase 74 已生成 manuscript v0 claim-audit package；当前主张锁定为 fixed-sampling broad12/broad21 `spot_size` floor，文献/venue claim 仍需后续核验，当前没有 trainable model branch 打开。
- 当前 A100-SXM4-40GB 仍足够处理 package regeneration、非训练审计和首轮小门槛实验；只有进入大规模 learned image encoder、更大多线表、多模型 ensemble 或 40GB 显存实测不足时，才向用户请求 A100-SXM4-80GB。

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
ssh -i ~/.ssh/matsci_gnnpinn_a100 -p 22036 root@223.109.239.30
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

同工艺 seed check 显示 seed 0 有正向信号，但 seed 1/2 不稳定。进一步检查 XLSX 后，`Line_0_1` 的精确显微对应应为 `P3/P4-L0-1`，而不是前一轮使用的 `P3/P4-L0-2`。下一步构建 exact-line panel：

```bash
bash scripts/server/build_mds2_2718_line0_1_micro_panel_a100.sh \
  > logs/ambench_mds2_2718_line0_1_micro_panel_build_a100_v1.log 2>&1

bash scripts/server/run_real_micro_exact_line0_1_a100.sh \
  > logs/ambench_real_micro_exact_line0_1_a100_v1.log 2>&1
```

Exact-line seed-0 sweep 的最佳候选是 `P4-L0-1_m/g4`：test RMSE `67.891122`，hot q90 RMSE `42.174310`，gradient q90 RMSE `68.729519`。该单 seed 的 global test RMSE 优于 sparse closure best `70.494433`，但 hot/gradient 指标没有超过 sparse closure。随后只对该候选做 focused seed check：

```bash
bash scripts/server/run_real_micro_exact_line0_1_seed_check_a100.sh \
  > logs/ambench_real_micro_exact_line0_1_p4_masked_g4_seedcheck_a100_v1.log 2>&1
```

Focused seed check 结果为 test RMSE `77.355537 +/- 10.649284`，hot q90 RMSE `59.851597 +/- 15.391250`，gradient q90 RMSE `80.681988 +/- 11.027500`。结论：exact-line alignment 比 same-process `L0-2` 更合理，但当前 sample-level hand-crafted real-micro feature 仍不够稳定，不能作为模型创新主结果。结果文档：`docs/results/ambench_real_micro_exact_line0_1_closure_v1.md`。

Phase 19 的下一步不是扩大 closure 超参，而是升级显微图像特征。v2 inspection 增加 intensity distribution、threshold-mask geometry、texture/gradient descriptors，并让 `RealMicroGraphFeatureProvider` 的 `g4/g8` 优先使用这些材料图像特征。服务器命令：

```bash
bash scripts/server/build_mds2_2718_line0_1_micro_panel_feature_v2_a100.sh \
  > logs/ambench_mds2_2718_line0_1_micro_panel_feature_v2_build_a100_v1.log 2>&1

bash scripts/server/run_real_micro_exact_line0_1_feature_v2_a100.sh \
  > logs/ambench_real_micro_exact_line0_1_feature_v2_a100_v1.log 2>&1
```

验收：v2 manifest `n_records=4`，closure metadata 的 `source_feature_names` 前 8 项应包含 `mask_centroid_*`、`mask_bbox_area_fraction`、`mask_span_*`、`mask_perimeter_fraction`、`gradient_magnitude_q90_norm` 等，而不是只使用 v1 的 coarse grid 均值/方差。

v2 rich-feature sweep 已完成，结果文档为 `docs/results/ambench_real_micro_exact_line0_1_feature_v2_closure_v1.md`。最佳 seed-0 为 `P4-L0-1/g8`：test RMSE `71.676666`，hot q90 RMSE `79.314583`，gradient q90 RMSE `90.549052`。结论：更多全局手工显微统计没有带来稳定提升。随后完成 no-normalization 诊断，结果文档为 `docs/results/ambench_real_micro_exact_line0_1_feature_v2_no_norm_closure_v1.md`。

```bash
bash scripts/server/run_real_micro_exact_line0_1_feature_v2_no_norm_a100.sh \
  > logs/ambench_real_micro_exact_line0_1_feature_v2_no_norm_a100_v1.log 2>&1

bash scripts/server/run_real_micro_exact_line0_1_feature_v2_no_norm_seed_check_a100.sh \
  > logs/ambench_real_micro_exact_line0_1_feature_v2_no_norm_p3_masked_g4_seedcheck_a100_v1.log 2>&1
```

No-normalization 的最佳 seed-0 global 结果为 `P3-L0-1_m/g8`：test RMSE `65.136753`。最佳 local-region seed-0 候选为 `P3-L0-1_m/g4`：test RMSE `67.843165`，hot q90 RMSE `40.724303`，gradient q90 RMSE `66.187221`。但 focused 3-seed check 不稳定：test RMSE `87.739183 +/- 33.551307`，hot q90 `94.058742 +/- 89.125747`，gradient q90 `108.484789 +/- 72.430034`。

Phase 19 决策：停止继续扩展全局手工 sample-level micrograph scalar descriptors。下一步进入 Phase 20，把微观组织信号移动到 residual point 附近。第一条低风险路线是 deterministic patch/region-level provider：聚合 inspection JSON 时保留 8x8 grid node features，训练时通过 `--closure-graph-mode real_micro_region` 按归一化 `x/y` 选择最近 patch 的局部特征。

```bash
bash scripts/server/run_real_micro_exact_line0_1_region_a100.sh \
  > logs/ambench_real_micro_exact_line0_1_region_a100_v1.log 2>&1

bash scripts/server/run_real_micro_exact_line0_1_region_seed_check_a100.sh \
  > logs/ambench_real_micro_exact_line0_1_region_seedcheck_a100_v1.log 2>&1
```

Region-level seed-0 sweep 和 focused seed check 已完成，结果文档为 `docs/results/ambench_real_micro_exact_line0_1_region_closure_v1.md`。最佳 seed-0 候选是 `P4-L0-1/g8`：test RMSE `74.072293`，hot q90 RMSE `21.377514`，gradient q90 RMSE `62.469599`。但 3-seed check 不稳定：test RMSE `84.664217 +/- 20.923602`，hot q90 `77.140936 +/- 66.143679`，gradient q90 `98.097679 +/- 48.567944`。

Phase 20 决策：deterministic region-level features 比继续扩展 global scalar descriptors 更有局部信号，但当前坐标映射和 nearest-patch 选择还不稳定，不能作为模型创新主结果。下一步优先做小规模 coordinate-registration ablation（row/col swap、flip、smooth interpolation），或在该问题被限定后转向 fixed learned patch embeddings。该路线仍不需要 A100-SXM4-80GB；只有进入 dense learned image encoder、更大图像 backbone 或多工况联合训练且当前 40GB 卡无法完成时，才向用户请求新服务器。

Phase 21 coordinate-registration ablation 已完成，结果文档为 `docs/results/ambench_real_micro_exact_line0_1_region_registration_v1.md`。seed-0 最佳全局变体是 `col_flip`：test RMSE `66.288087`，hot q90 `49.074617`，gradient q90 `72.040545`。但 `col_flip` focused 3-seed 仍不稳定：test RMSE `79.911958 +/- 20.066421`，hot q90 `81.615574 +/- 61.088806`，gradient q90 `97.241934 +/- 48.236581`。结论：坐标注册消融能确认局部微观特征路线比 global scalar 更有诊断价值，但 deterministic nearest-patch provider 仍不是稳定性能分支。下一步不应继续堆手工 patch scalar，应转向 fixed learned patch embeddings 或更强物理配准的 microstructure source；当前仍不需要 A100-SXM4-80GB。

Phase 22 固定 patch embedding 分支已进入实现。该分支先不训练重型图像编码器，而是在 exact `Line_0_1` inspection 的 8x8 patch descriptors 上拟合冻结 PCA embedding，并用 `--closure-graph-mode real_micro_region_embedding` 按 residual point 坐标选择局部 embedding。构建脚本不会覆盖 v1/v2/region scalar 产物：

```bash
bash scripts/server/build_mds2_2718_line0_1_micro_panel_region_embedding_a100.sh \
  > logs/ambench_mds2_2718_line0_1_region_embedding_build_a100_v1.log 2>&1
```

首轮 seed-0 小矩阵沿用 Phase 21 最好的 `col_flip` 坐标注册，跑 exact-line 四个 P3/P4 masked/unmasked 样本的 `g4/g8`：

```bash
bash scripts/server/run_real_micro_exact_line0_1_region_embedding_a100.sh \
  > logs/ambench_real_micro_exact_line0_1_region_embedding_a100_v1.log 2>&1
```

验收：feature manifest 中 `region_embedding_dim=8`，训练 artifact 的 graph conditioning metadata 记录 `region_embedding_metadata_by_sample_id`；若 seed-0 没有同时改善全局 test RMSE 与 hot/gradient q90，则不做 3-seed 扩大。该分支仍适合当前 A100-SXM4-40GB。

Phase 22 seed-0 小矩阵完成后，若只出现单个候选正向信号，使用 focused seed check，而不是扩大整张矩阵：

```bash
bash scripts/server/run_real_micro_exact_line0_1_region_embedding_seed_check_a100.sh \
  > logs/ambench_real_micro_exact_line0_1_region_embedding_p4_masked_g4_seedcheck_a100_v1.log 2>&1
```

Phase 23 已切换到 multi-line / process-conditioned thermal modeling。服务器 HDF5 盘点脚本：

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
/home/vipuser/miniconda3/bin/conda run -n gnnpinn python scripts/server/inspect_ambench_hdf5_lines.py
```

该盘点确认 `mds2-2716` 的单轨 `Line_*` 数据集有真实工艺多样性：`245/285/325 W`、`800/960/1200 mm/s`、`49/67/82 um`。首轮服务器比较使用代表性七条线，生成 line-split 多行温度表，并比较 coordinate-only 与 process-conditioned baseline/Macro PINN：

```bash
bash scripts/server/run_multiline_process_conditioned_thermal_a100.sh \
  > logs/ambench_multiline_process_conditioned_thermal_a100_v1.log 2>&1
```

验收：manifest 记录七条 `dataset_paths` 和每条线的 process parameters；split manifest 为 `line_id_order`；metrics/checkpoint 中 process-conditioned Macro PINN 的 `input_features.columns` 应为 `laser_power_W`、`scan_speed_mm_s`、`spot_size_um`。若该分支在 line-held-out split 上优于 no-process-feature 版本，应优先写结果文档并作为比 exact-line micro TIFF 更强的论文主线候选。

Phase 23 首轮完整 A100 run 已完成，结果文档为 `docs/results/ambench_multiline_process_conditioned_thermal_v1.md`。七条 single-track 数据生成 `14,842` 个 calibrated temperature 点，line-held-out split 为 train `8,087`、val `1,884`、test `4,871`。process-conditioned Macro PINN 相比 coordinate-only Macro PINN 有清晰泛化收益：test RMSE `175.127058 -> 157.793227`，val RMSE `340.584340 -> 177.859802`，test hot q90 RMSE `351.525048 -> 316.794319`，test gradient q90 RMSE `323.786011 -> 293.650864`。

同时，该分支还没有超过 train-mean baseline 的 global test RMSE `128.668856`。因此 Phase 23 结论应表述为正向方向选择，而不是最终模型主结果。下一步优先做 process-conditioned Macro PINN 的容量、seed 和按工艺轴 holdout split 检查；在该基线稳定前，不要急着把 sparse closure 或 real-micro graph conditioning 接回多线表。

Phase 24 的第一步是把 split 从 `line_id` 扩展到工艺轴 holdout。转换器现在支持：

```text
--split-strategy laser_power
--split-strategy scan_speed
--split-strategy spot_size
--split-strategy process
```

服务器一键脚本：

```bash
bash scripts/server/run_multiline_process_holdout_splits_a100.sh \
  > logs/ambench_multiline_process_holdout_splits_a100_v1.log 2>&1
```

该脚本会分别生成 `line / laser_power / scan_speed / spot_size / process` grouped split 的多线温度表，并沿用相同 baseline 与 Macro PINN no-process/process-feature 对照。验收重点不是单个 split 的偶然胜负，而是 process-conditioned Macro PINN 是否在多个工艺轴 holdout 上稳定改善 no-process 版本，并是否开始接近或超过 mean/strong baseline。

Phase 24 A100 run 已完成，结果文档为 `docs/results/ambench_multiline_process_axis_holdout_v1.md`。process features 在 `line`、`laser_power`、`scan_speed`、`process` 四个 grouped holdout 上改善 Macro PINN；其中 scan-speed test RMSE 从 `186.921887` 降到 `140.459979`。但 `spot_size` holdout 变差，test RMSE 从 `208.741300` 升到 `227.573411`，hot q90 和 gradient q90 也同步退化。所有 Macro PINN 结果仍未超过 train-mean baseline。

因此 Phase 24 关闭为“模型创新入口确认”，不是最终性能 claim。下一步进入 FiLM-style process-conditioned Macro PINN：保持 concat 作为默认基线，新增由 `laser_power_W / scan_speed_mm_s / spot_size_um` 调制 hidden coordinate/time layers 的结构化条件化模式。首轮 A100 对比应至少覆盖 `line`、`scan_speed`、`spot_size` 三个 split，分别代表原始正向 split、最强正向 process-axis split 和当前失败 split。

Phase 25 FiLM 首轮服务器命令：

```bash
bash scripts/server/run_multiline_process_film_conditioned_a100.sh \
  > logs/ambench_multiline_process_film_conditioned_a100_v1.log 2>&1
```

验收：metrics/checkpoint 中 `input_features.conditioning_mode` 应为 `film`；产物 run id 使用 `*_film_a100_sxm4_40gb_v1_macro_pinn_minmax_process_film_v1`，不得覆盖 Phase 23/24 的 concat 结果。若 FiLM 至少修复 `spot_size` 退化并保持 `line`/`scan_speed` 改善，再做 seed check；否则下一步考虑 process embedding/hypernetwork 或 mixture-of-experts，而不是直接把 sparse closure/GNN 接回多线表。

FiLM v1 首轮结果是负向诊断：它没有保留 concat 在 `line` 和 `scan_speed` 上的收益，只避免了 `spot_size` 的大幅退化。一个直接原因是单轴 holdout 中 train-fitted minmax 会让某些工艺特征在训练集上变成常数，FiLM 生成器难以学习有效调制。已新增独立 feature normalization：

```text
--input-feature-normalization same|none|minmax|standard|global_minmax|global_standard
```

下一轮先跑全表工艺特征标准化的 FiLM：

```bash
bash scripts/server/run_multiline_process_film_global_feature_norm_a100.sh \
  > logs/ambench_multiline_process_film_global_feature_norm_a100_v1.log 2>&1
```

Phase 25/26 后续消融已经完成，结果文档为 `docs/results/ambench_multiline_process_film_conditioned_v1.md`。关键结论：

- `scan_speed`：`concat + global_standard` 是最强单 seed 模型，focused 3-seed 均值相对 no-process 稳定改善，但 global RMSE 仍略高于 train-mean baseline。
- `spot_size`：`FiLM + global_standard` 是第一条在 spot-size holdout 上三 seed 均值超过 train-mean baseline 的神经 Macro PINN 分支，hot q90 和 gradient q90 也同步更优。
- `line`：原始 `concat + train-minmax` 仍是最不差的神经条件化路径，但所有神经模型仍弱于 mean baseline。
- `concat_film + global_standard` 和 `concat_film + global_standard + --input-film-strength 0.25` 都没有形成通用改进；满强度 hybrid 在 line split 崩溃，受限强度缓解但仍弱于最佳单路径方法。

当前代码已新增：

```text
--input-conditioning-mode concat|film|concat_film|routed
--input-film-strength <float>
--input-feature-normalization global_standard
--input-route-film-prior <float>
--freeze-input-route
--input-conditioning-profile process_axis_v1
```

Phase 27 split/process-aware routing 已完成，结果文档为 `docs/results/ambench_multiline_process_axis_routing_v1.md`。关键结论：

- 可训练 dual-expert routed 模式是负结果。即使 gate 保持接近先验，两个新专家混合训练仍弱于最佳单一路径：`line` test RMSE `219.861259`，`scan_speed` `150.293172`，`spot_size` `273.800445`。
- 显式 `process_axis_v1` profile 是当前可复现路线。它读取 grouped split manifest 的 `group_key`，并记录 selected route：`line_id -> concat/same`、`scan_speed_mm_s -> concat/global_standard`、`spot_size_um -> film/global_standard`。
- `process_axis_v1` 恢复了最佳单路径结果：`line` test RMSE `157.793227`，`scan_speed` `133.430469`，`spot_size` `142.351582`。
- `scan_speed` 与 `spot_size` 的 seed-7 profile 结果均低于 train-mean baseline；`line` 仍弱于 train-mean baseline。

Phase 28 process-axis profile validation 已完成，结果文档为 `docs/results/ambench_multiline_process_axis_profile_validation_v1.md`。关键结论：

- `process_axis_v1` 已扩展到 `laser_power_W` 与 full `process_condition`。当前路线为：`laser_power_W -> concat/global_standard`，`process_condition -> concat/same`。
- `laser_power` 是新增正向轴。focused 3-seed 结果显示 test RMSE 从 no-process `211.217281 +/- 0.443665` 改善到 `147.980699 +/- 3.456300`，hot q90 从 `391.157932 +/- 2.057337` 改善到 `260.268960 +/- 23.249762`。
- full `process` 不应使用 `global_standard`。`concat/global_standard` 诊断 run 退化到 test RMSE `183.009876`；修正为 line-like `concat/same` 后恢复 `157.793227`。
- `laser_power`、`scan_speed`、`spot_size` 现在都有三 seed process-conditioned gain；但 `line` 与 full `process` 仍弱于 train-mean baseline，因此不能把当前 profile 表述为 universal model。

Phase 29 broader process-dataset smoke 已完成，结果文档为 `docs/results/ambench_multiline_process_broader_profile_smoke_v1.md`。关键实现：

- `gnnpinn.data.loaders.ambench_hdf5` 支持 `--dataset-regex`、`--dataset-limit` 和 `--dataset-order sorted|process_round_robin`。
- `process_round_robin` 会按 `(laser_power, scan_speed, spot_size)` 分组轮询，再应用 `--dataset-limit`，避免简单 lexicographic first-N 只覆盖一个工艺组合。
- `scripts/server/run_phase29_broad_process_profile_smoke_a100.sh` 使用 broad12 process-balanced panel，并保留 Phase 23-28 产物。
- `scripts/server/summarize_phase29_broad_process_profile_smoke.py` 汇总 broad smoke 的 manifest、baseline 和 Macro PINN profile 指标。

关键结论：

- naive first-9 sorted smoke 只覆盖 `285 W / 960 mm/s`，不能作为 process-generalization 证据。
- broad12 process-balanced panel 覆盖 `245/285/325 W`、`800/960/1200 mm/s`、`49/67/82 um` 和 7 个 full-process tuples。
- 旧 `process_axis_v1` 只在 `spot_size` 上稳定转移：test RMSE `206.100512 -> 136.309183`，且强于 mean baseline `151.850578`。
- `line` 在 broad12 上更偏向 no-process Macro PINN；`scan_speed` 和 full `process` 在旧 profile 下退化。

因此下一步不应继续扩大 blind mixture-of-experts，也不应立即把 sparse closure/GNN 接回 broad process table。Phase 30 应实现 broad-data conditional route selector/profile，让每个 split 可以选择 no-process、concat 或 FiLM 路线；若 broad12 修正版 selector 成立，再扩到全部 21 条 single-track lines。该路线仍适合当前 A100-SXM4-40GB；只有引入 learned image encoder、更大多线表或大规模 mixture-of-experts 并实际超出 40GB 时，才需要向用户请求 A100-SXM4-80GB。

Phase 30 broad-data selector 已完成，结果文档为 `docs/results/ambench_multiline_process_broad_selector_v1.md`。关键实现：

- `MacroPINN` 新增 `--input-conditioning-profile broad_process_v1`。
- `broad_process_v1` 可根据 split `group_key` 选择 no-process、concat/global-standard 或 FiLM/global-standard。
- no-process route 会清空 `input_feature_columns`，并在 metrics/checkpoint 中记录 requested/selected/effective profile metadata；服务器 tiny smoke 已确认 `features_enabled=false` 且 checkpoint `param_dim=0`。
- `scripts/server/run_phase30_broad_process_selector_smoke_a100.sh` 使用 broad12 process-balanced panel，并保留 Phase 23-29 产物。
- `scripts/server/summarize_phase30_broad_process_selector_smoke.py` 增加 manifest/split signature check 与 `--require-comparable`，避免 tiny smoke artifact 和 full Phase 29 artifact 混比。

关键结论：

- full broad12 selector run 已通过 `--require-comparable` 门禁，所有五个 split 的 Phase 29/30 artifact 可比。
- `line` 回退 no-process，避免旧 profile 退化：test RMSE `149.638162 -> 126.308616`。
- `scan_speed` 回退 no-process，避免旧 profile 退化：`226.454041 -> 186.173938`。
- full `process` 回退 no-process，避免旧 profile 退化：`220.019735 -> 181.091525`。
- `laser_power` 保留 concat/global-standard，改善 no-process Macro PINN：`167.614004 -> 140.753534`。
- `spot_size` 保留 FiLM/global-standard，是 broad12 最强正向工艺条件化轴：`206.100512 -> 136.309183`，且优于 mean baseline `151.850578`。

Phase 30 关闭为 broad-data selector validation，不是最终 universal model。下一步 Phase 31 应用同一 selector 扩展到全部 21 条 single-track `ThermalData/Line_*/Signal` 数据集。为了保持 summary 的 manifest/split comparability gate 有同配置参照，需要先跑 broad21 的旧 profile counterpart，再跑新 selector：

```bash
DATASET_LIMIT=21 DATASET_ORDER=process_round_robin STEPS=500 N_ESTIMATORS=80 \
  bash scripts/server/run_phase29_broad_process_profile_smoke_a100.sh \
  > logs/ambench_phase31_broad21_rr_process_axis_profile_a100_v1.log 2>&1

DATASET_LIMIT=21 DATASET_ORDER=process_round_robin STEPS=500 N_ESTIMATORS=80 \
  bash scripts/server/run_phase30_broad_process_selector_smoke_a100.sh \
  > logs/ambench_phase31_broad21_rr_selector_smoke_a100_v1.log 2>&1
```

验收：生成 broad21 manifest/split、mean/kNN/ExtraTrees baselines、no-process Macro PINN 和 `broad_process_v1` artifacts；summary 必须通过 `--require-comparable`。如果 broad21 仍保持“避免负迁移并保留 spot-size/laser-power 正向路由”的形态，再进入更强 broad-data profile 或 closure/GNN reintegration。

Phase 31 broad21 all single-track selector scaling 已完成，结果文档为 `docs/results/ambench_multiline_process_broad_selector_broad21_v1.md`。关键结论：

- broad21 summary 通过 `--require-comparable`，旧 profile counterpart 与新 selector 在同一 21-line dataset/split/signature 下比较。
- `laser_power` 保留 concat/global-standard 路由，test RMSE `192.833317 -> 178.040331`，但仍弱于 mean baseline `131.741364`。
- `spot_size` 保留 FiLM/global-standard 路由，test RMSE `210.423419 -> 147.389475`，略优于 mean baseline `149.185412`。
- `scan_speed` 旧 profile 在 broad21 上严重退化到 `469.347549`，`broad_process_v1` 回退 no-process 到 `227.128663`。
- full `process` 旧 profile 退化到 `229.613547`，`broad_process_v1` 回退 no-process 到 `166.231596`。
- `line` 在 broad21 上旧 profile 略好于 conservative no-process route (`125.449323` vs `126.194921`)，提示下一步可以考虑 broad21-specific line route refinement。

Phase 31 关闭为 broad21 selector scaling。下一步不需要 A100-SXM4-80GB。建议 Phase 32 在两个方向中择一推进：其一，做 broad-data profile v2，把 line route 从 no-process 修正为 concat/same 并保留 scan_speed/process fallback；其二，保留当前 conservative selector，转入更强 broad-data representation 或 closure/GNN reintegration。

Phase 32 broad-data profile v2 diagnostic 已完成，结果文档为 `docs/results/ambench_multiline_process_broad_selector_v2.md`。关键实现：

- `MacroPINN` 新增 `--input-conditioning-profile broad_process_v2`。
- `broad_process_v2` 相比 `broad_process_v1` 只把 `line_id` route 改为 `concat/same`；`scan_speed_mm_s` 与 full `process_condition` 仍回退 no-process，`laser_power_W` 与 `spot_size_um` 保持正向路线。
- `scripts/server/run_phase30_broad_process_selector_smoke_a100.sh` 新增 `PROCESS_CONDITIONING_PROFILE`、`PROCESS_FEATURE_TAG`、`PROCESS_PROFILE_RUN_TAG` 环境变量，v2 使用 `broad_process_profile_v2`，不覆盖 Phase 30/31 artifact。
- `scripts/server/summarize_phase30_broad_process_selector_smoke.py --include-broad-process-v2` 可在同一可比性门禁中汇总 v1/v2。

验收结果：

- broad12 和 broad21 v2 summary 均通过 `--require-comparable`。
- broad21 `line` 从 `broad_process_v1`/no-process `126.194921` 改善到 v2 `125.449323`。
- broad12 `line` 从 `broad_process_v1`/no-process `126.308616` 退化到 v2 `149.638162`。
- 其它 split 与 v1 一致：`laser_power` 和 `spot_size` 保留正向路线，`scan_speed` 和 full `process` 保留 no-process fallback。

因此 Phase 32 关闭为 diagnostic refinement，不更换默认 selector。`broad_process_v1` 仍是更稳的 broad-data route guard；`broad_process_v2` 仅保留为 broad21 line-specific diagnostic。下一步 Phase 33 应转向更强 broad-data representation 或 closure/GNN reintegration，而不是继续手动调 selector。当前 A100-SXM4-40GB 仍足够，无需请求 A100-SXM4-80GB。

Phase 33 fixed Fourier spacetime representation diagnostic 已完成，结果文档为 `docs/results/ambench_multiline_process_fourier_spacetime_v1.md`。关键实现：

- `MacroPINN` 新增 `spacetime_encoding=raw|fourier` 和 `spacetime_fourier_bands`。
- CLI 新增 `--spacetime-encoding raw|fourier` 与 `--spacetime-fourier-bands`，metrics/checkpoint 记录 encoding、band 数和实际 spacetime input dimension。
- `scripts/server/run_multiline_process_conditioned_thermal_a100.sh` 新增 `SPACETIME_ENCODING` 与 `SPACETIME_FOURIER_BANDS`。
- 新增 `scripts/server/run_phase33_broad_fourier_selector_smoke_a100.sh`，使用 `broad_process_v1` 路由并以 non-overwriting run tag 写入 Fourier artifact。
- `scripts/server/summarize_phase30_broad_process_selector_smoke.py --include-broad-process-fourier` 可在 manifest/split comparability gate 中纳入 Phase 33 artifact。

验收结果：

- 服务器 targeted torch suite 通过：`29 passed, 9 warnings`；bash/py_compile 通过。
- full broad12 Fourier/4 summary 通过 `--require-comparable`。
- Fourier/4 在所有 broad12 split 上均弱于 `broad_process_v1`：`line` `126.308616 -> 168.277643`，`laser_power` `140.753534 -> 189.900869`，`scan_speed` `186.173938 -> 199.328381`，`spot_size` `136.309183 -> 213.024115`，full `process` `181.091525 -> 243.410025`。
- `SPACETIME_FOURIER_BANDS=1` 的 spot-size 低成本检查仍未追上既有 `broad_process_v1`：test RMSE `153.271041` vs `136.309183`。

因此 Phase 33 关闭为 negative representation diagnostic。Fourier basis 已实现并保留为可复现实验选项，但不替代默认 raw coordinate/time basis，也没有证据支持扩到 broad21。下一步应转向更结构化的分支：closure/GNN reintegration、broad selector 下的 learned residual correction，或改变采样/对齐的数据表示，而不是继续堆固定 coordinate basis。当前 A100-SXM4-40GB 仍足够，无需请求 A100-SXM4-80GB。

Phase 34 learned residual correction diagnostic 已完成并关闭为负结果，结果文档为 `docs/results/ambench_multiline_process_residual_correction_v1.md`。先导 sparse closure probes 在 strongest broad12 `spot_size` route 上退化：

- `broad_process_v1` spot-size baseline: test RMSE `136.309183`, hot q90 `165.228535`, gradient q90 `169.049295`。
- `pde_weight=1e-4`, `closure_start_step=100`: test RMSE `151.152357`, hot q90 `197.625638`, gradient q90 `191.064629`。
- lite closure: test RMSE `158.792203`, hot q90 `258.922260`, gradient q90 `237.740336`。

因此 Phase 34 没有继续加强 PDE/closure residual，而是在 base Macro PINN 输出上加一个弱 residual MLP correction：

```text
prediction = MacroPINN(coords, time, process?) + scale * ResidualMLP([coords, time, process?])
```

关键实现：

- CLI 新增 `--residual-correction-mode none|mlp`、hidden dim/layers/scale/lr/start-step 控制；默认 `none`，不改变旧实验。
- residual MLP 使用归一化后的 coordinate/time 与已选择的 process features，最后一层零初始化，初始预测等于 base Macro PINN。
- metrics/checkpoint 记录 `residual_correction` metadata；summary 脚本新增 `--include-broad-process-residual` 并继续执行 manifest/split comparability gate。
- `scripts/server/run_phase34_broad_residual_correction_a100.sh` 默认只跑 broad12 `spot_size`，不覆盖 Phase 30/33 artifact。

focused A100 验证命令：

```bash
PROFILE_SPLITS=spot_size DATASET_LIMIT=12 DATASET_ORDER=process_round_robin \
STEPS=500 N_ESTIMATORS=80 \
  bash scripts/server/run_phase34_broad_residual_correction_a100.sh \
  > logs/phase34_broad12_spot_size_residual_mlp_a100_v1.log 2>&1

python scripts/server/summarize_phase30_broad_process_selector_smoke.py \
  --dataset-limit 12 \
  --dataset-order process_round_robin \
  --split spot_size \
  --include-broad-process-residual \
  --json-output outputs/reports/phase34_broad12_spot_size_residual_mlp_summary.json \
  --require-comparable
```

验证结果：

- `broad_residual_mlp` 通过 manifest/split comparability gate，但只把 global test RMSE 从 `136.309183` 改为 `136.294049`，hot q90 从 `165.228535` 退化到 `192.514042`，gradient q90 从 `169.049295` 退化到 `187.270890`。
- 弱残差设置 `scale=0.03`, `lr=1e-4`, `start=300` 基本保持区域指标（hot q90 `165.248033`, gradient q90 `169.062741`），但 global test RMSE `136.327054` 没有收益。

结论：learned residual correction 保留为可复现诊断选项，不扩到全 broad12 或 broad21。下一步 Phase 35 直接针对 hot-zone/gradient-band supervised error，而不是再加 unconstrained residual head。

Phase 35 train-split region-weighted data loss 已进入实现与 focused A100 验证阶段，结果文档为 `docs/results/ambench_multiline_process_region_weighted_loss_v1.md`。关键实现：

- CLI 新增 `--data-loss-weighting none|hot|gradient|hot_gradient`、`--data-loss-hot-quantile`、`--data-loss-gradient-quantile`、`--data-loss-region-weight`；默认 `none`，不改变旧实验。
- 加权 selector 只在 optimization train split 上拟合阈值；gradient selector 也只在训练点子集内计算邻域差分，避免 test/val 目标值泄漏进 loss 权重。
- supervised data loss 使用 `sum(weight_i * squared_error_i) / sum(weight_i)`，因此权重改变区域贡献，不改变 loss 的整体尺度定义。
- metrics/checkpoint 记录 `data_loss_weighting` metadata；summary 脚本新增 `--include-broad-region-weighted` 并继续执行 manifest/split comparability gate。
- `scripts/server/run_phase35_broad_region_weighted_loss_a100.sh` 默认只跑 broad12 `spot_size`，不覆盖 Phase 30/34 artifact。

首轮 focused A100 命令：

```bash
PROFILE_SPLITS=spot_size DATASET_LIMIT=12 DATASET_ORDER=process_round_robin \
STEPS=500 N_ESTIMATORS=80 \
  bash scripts/server/run_phase35_broad_region_weighted_loss_a100.sh \
  > logs/phase35_broad12_spot_size_region_hotgrad_w2_a100_v1.log 2>&1

python scripts/server/summarize_phase30_broad_process_selector_smoke.py \
  --dataset-limit 12 \
  --dataset-order process_round_robin \
  --split spot_size \
  --include-broad-region-weighted \
  --json-output outputs/reports/phase35_broad12_spot_size_region_hotgrad_w2_summary.json \
  --require-comparable
```

首轮 focused A100 结果：

| Method | Region weight | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | 判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| `broad_process_v1` |  | 136.309183 | 165.228535 | 169.049295 | baseline |
| `rw125` | 1.25 | 139.773470 | 160.431178 | 168.187786 | global 接近，区域收益小 |
| `rw135` | 1.35 | 140.881887 | 199.804301 | 192.678431 | 负诊断 |
| `rw15` | 1.50 | 143.462665 | 128.090811 | 150.283022 | 区域收益最大，global 退化中等 |
| `rw2` | 2.00 | 153.645414 | 132.990923 | 154.642037 | 区域收益明显但 global 退化过大 |

下一步不继续加大权重；改为对 `rw15` 与 `rw125` 做 paired model-seed check：

```bash
SEEDS="1 2" REGION_WEIGHTED_TAGS="rw15 rw125" STEPS=500 \
  bash scripts/server/run_phase35_region_weighted_seed_check_a100.sh \
  > logs/phase35_broad12_spot_size_region_weighted_seed_check_a100_v1.log 2>&1

python scripts/server/summarize_phase35_region_weighted_seed_check.py \
  --json-output outputs/reports/phase35_broad12_spot_size_region_weighted_seed_check_summary.json \
  --require-complete
```

seed check 结果：

| Method | Test RMSE mean +/- std | Hot q90 mean +/- std | Gradient q90 mean +/- std | 判断 |
| --- | ---: | ---: | ---: | --- |
| `broad_process_v1` | 136.384782 +/- 0.467526 | 162.125337 +/- 4.909788 | 165.282182 +/- 5.270236 | baseline |
| `rw15` | 142.730123 +/- 1.890466 | 163.186816 +/- 52.968498 | 170.560643 +/- 34.779012 | 单 seed 区域收益不稳定 |
| `rw125` | 140.638507 +/- 4.227388 | 177.884187 +/- 37.767560 | 177.816410 +/- 26.335924 | 保守权重也未稳定改善 |

结论：Phase 35 关闭为负诊断。保留 data-loss weighting 作为可复现分析选项，但不扩到其它 split。下一步应转向更结构化的 process/microstructure 表示或 graph-conditioned branch，而不是继续调 scalar loss 权重。当前分支不需要 A100-SXM4-80GB。

Phase 36 structured process-neighborhood RBF graph feature branch 已进入实现与 focused A100 验证阶段，结果文档为 `docs/results/ambench_multiline_process_process_graph_rbf_v1.md`。关键实现：

- CLI 新增 `--process-graph-feature-mode none|rbf`、`--process-graph-feature-column`、`--process-graph-feature-count`、`--process-graph-length-scale`、`--process-graph-fit-scope train|global`；默认 `none`，不改变旧实验。
- RBF 分支从 `laser_power_W`、`scan_speed_mm_s`、`spot_size_um` 等 row metadata 构造 standardized process vectors，选择唯一 process anchors，并把相似度作为 `process_graph_rbf_*` 特征追加到 Macro PINN 条件输入。
- 该分支可与普通 process scalars 同用，也可在 `broad_process_v1` 对 `scan_speed`/full `process` 回退 no-process scalars 后，通过显式 `--process-graph-feature-column` 形成 graph-only 输入。
- metrics/checkpoint 记录 `process_graph_features` metadata，包括列、fit scope、anchor 数、length scale、最终 effective feature names 和 checkpoint `param_dim`。
- `scripts/server/run_multiline_process_conditioned_thermal_a100.sh` 新增 `PROCESS_GRAPH_*` 环境变量；`scripts/server/run_phase36_broad_process_graph_rbf_a100.sh` 默认 focused 跑 broad12 `spot_size` 与 `laser_power`，run tag 使用短名 `pg_rbf` 避免 Windows/server 长路径问题。
- `scripts/server/summarize_phase30_broad_process_selector_smoke.py --include-broad-process-graph-rbf` 可在 manifest/split comparability gate 中纳入 Phase 36 artifacts，并显示 graph mode/anchor/scope。

focused A100 验证命令：

```bash
PROFILE_SPLITS="spot_size laser_power" DATASET_LIMIT=12 DATASET_ORDER=process_round_robin \
STEPS=500 N_ESTIMATORS=80 \
  bash scripts/server/run_phase36_broad_process_graph_rbf_a100.sh \
  > logs/phase36_broad12_process_graph_rbf_a100_v1.log 2>&1

python scripts/server/summarize_phase30_broad_process_selector_smoke.py \
  --dataset-limit 12 \
  --dataset-order process_round_robin \
  --split spot_size \
  --split laser_power \
  --include-broad-process-graph-rbf \
  --json-output outputs/reports/phase36_broad12_process_graph_rbf_summary.json \
  --require-comparable
```

focused/seed-check 结果：

| Dataset/Split | Method | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | 判断 |
| --- | --- | ---: | ---: | ---: | --- |
| broad12 `spot_size` | `broad_process_v1` | 136.384782 | 162.125337 | 165.282182 | baseline |
| broad12 `spot_size` | `pg_rbf_global` | 148.632815 | 255.706330 | 236.036198 | 负诊断 |
| broad12 `laser_power` | `broad_process_v1` | 143.639451 | 266.975170 | 225.572273 | baseline |
| broad12 `laser_power` | `pg_rbf_global` | 140.689962 | 245.732430 | 211.178946 | broad12 有信号 |
| broad21 `laser_power` | `broad_process_v1` | 168.816296 | 263.145682 | 229.087591 | baseline |
| broad21 `laser_power` | `pg_rbf_global` | 271.516427 | 257.153407 | 291.054335 | 不稳定 |

结论：Phase 36 关闭为 process-neighborhood scalar-graph 负/不稳定诊断。保留 `process_graph_rbf_*` 作为可复现实验选项，但不继续调 anchor/length-scale。当前分支不需要 A100-SXM4-80GB。

Phase 37 strong-baseline residualized Macro PINN 已完成 focused A100 验证并关闭为负诊断，结果文档为 `docs/results/ambench_multiline_process_target_residual_v1.md`。关键实现：

- CLI 新增 `--target-residual-baseline none|mean|knn|extra_trees`、`--target-residual-baseline-feature-column`、`--target-residual-baseline-n-neighbors`、`--target-residual-baseline-n-estimators`、`--target-residual-baseline-random-state`；默认 `none`，不改变旧实验。
- 启用后先在 optimization train split 上拟合 baseline，再让 Macro PINN 学习 `target - baseline_prediction`；评估时把 baseline prediction 加回。
- target normalization 现在记录 `target_space`，启用残差 baseline 时为 `residual`；metrics/checkpoint 记录 baseline strategy、feature columns、fit split、fit points 和 train residual RMSE。
- 当前残差 baseline 分支仅支持 data-only training；若 `pde_weight > 0` 会报错，避免对不可微 tree/kNN baseline 计算 PDE residual 产生含混语义。
- `scripts/server/run_multiline_process_conditioned_thermal_a100.sh` 新增 `TARGET_RESIDUAL_BASELINE*` 环境变量；`scripts/server/run_phase37_broad_target_residual_a100.sh` 默认 focused 跑 broad12 `spot_size` 与 `laser_power`，run tag 使用 `target_resid_et`。
- `scripts/server/summarize_phase30_broad_process_selector_smoke.py --include-broad-target-residual` 可在 manifest/split comparability gate 中纳入 Phase 37 artifacts，并显示 target residual baseline metadata。

focused A100 验证命令：

```bash
PROFILE_SPLITS="spot_size laser_power" DATASET_LIMIT=12 DATASET_ORDER=process_round_robin \
STEPS=500 N_ESTIMATORS=80 \
  bash scripts/server/run_phase37_broad_target_residual_a100.sh \
  > logs/phase37_broad12_target_residual_et_a100_v1.log 2>&1

python scripts/server/summarize_phase30_broad_process_selector_smoke.py \
  --dataset-limit 12 \
  --dataset-order process_round_robin \
  --split spot_size \
  --split laser_power \
  --include-broad-target-residual \
  --json-output outputs/reports/phase37_broad12_target_residual_summary.json \
  --require-comparable
```

结果：broad12 `spot_size` 的 `target_resid_et` 基本跟随弱 ExtraTrees process baseline，测试指标为 `207.575682 / 357.674336 / 326.647214`，明显弱于 `broad_process_v1` 的 `136.309183 / 165.228535 / 169.049295`。broad12 `laser_power` 也退化到 `172.134324 / 340.026811 / 265.502092`，而 ExtraTrees train residual RMSE 为 `0.000000`，说明 baseline 已在 train split 上过拟合并没有留下可学的平滑 residual。结论：Phase 37 保留为可复现实验控制，不做 seed check，不扩 broad21。

Phase 38 residual Macro PINN backbone 已完成 focused A100 验证并关闭为负诊断，结果文档为 `docs/results/ambench_multiline_process_residual_backbone_v1.md`。关键实现：

- CLI 新增 `--backbone-mode mlp|residual` 与 `--backbone-residual-scale`；默认 `mlp`，不改变旧实验。
- `residual` 模式在 concat/routed-concat expert 中使用 same-width hidden residual MLP；FiLM/concat-FiLM route 在首层投影后对同维 hidden transitions 加 residual skip。
- metrics/checkpoint 新增 `backbone` metadata，记录 mode、residual scale、hidden width、layer count 和参数量。
- `scripts/server/run_multiline_process_conditioned_thermal_a100.sh` 新增 `BACKBONE_MODE` 与 `BACKBONE_RESIDUAL_SCALE` 环境变量。
- `scripts/server/run_phase38_broad_residual_backbone_a100.sh` 默认 focused 跑 broad12 `spot_size` 与 `laser_power`，run tag 使用 `res_backbone`。
- `scripts/server/summarize_phase30_broad_process_selector_smoke.py --include-broad-residual-backbone` 可在 manifest/split comparability gate 中纳入 Phase 38 artifacts，并显示 backbone metadata。

focused A100 验证命令：

```bash
PROFILE_SPLITS="spot_size laser_power" DATASET_LIMIT=12 DATASET_ORDER=process_round_robin \
STEPS=500 N_ESTIMATORS=80 BACKBONE_RESIDUAL_SCALE=0.5 \
  bash scripts/server/run_phase38_broad_residual_backbone_a100.sh \
  > logs/phase38_broad12_residual_backbone_a100_v1.log 2>&1

python scripts/server/summarize_phase30_broad_process_selector_smoke.py \
  --dataset-limit 12 \
  --dataset-order process_round_robin \
  --split spot_size \
  --split laser_power \
  --include-broad-residual-backbone \
  --json-output outputs/reports/phase38_broad12_residual_backbone_summary.json \
  --require-comparable
```

验收：先比较 `spot_size` 和 `laser_power` against mean/kNN/ExtraTrees、no-process、`process_axis_v1`、`broad_process_v1` 和 Phase 37 负诊断。若 residual backbone 在至少一个已正向 split 上同时改善 global/hot q90/gradient q90 且没有明显不稳定，再做 paired seed check；若退化，则关闭为 backbone diagnostic，下一步转向更物理对齐的数据表示或更强但仍 train-split-safe 的 process-conditioned architecture。当前分支不需要 A100-SXM4-80GB。

结果：broad12 `spot_size` residual backbone 给出轻微 global RMSE 改善，但 hot/gradient 明显退化：`136.025906 / 205.891992 / 197.838013` vs `broad_process_v1` 的 `136.309183 / 165.228535 / 169.049295`。broad12 `laser_power` 区域指标略有改善但 global RMSE 明显变差：`159.276166 / 239.054654 / 214.876025` vs `140.753534 / 254.473291 / 215.411533`。结论：Phase 38 不做 seed check，不扩 broad21，保留 residual backbone 为可复现实验选项。

Phase 39 process-conditioned output affine calibration 当前进入实现与 focused A100 验证阶段，结果文档为 `docs/results/ambench_multiline_process_output_affine_v1.md`。关键实现：

- CLI 新增 `--output-affine-mode none|linear`、`--output-affine-scale` 与 `--output-affine-lr`；默认 `none`，不改变旧实验。
- `linear` 模式从 active process/input features 预测 `(gamma, beta)`，在 normalized target space 中应用 `prediction <- (1 + scale * gamma) * prediction + scale * beta`。
- 输出 affine head 零初始化，因此初始行为是 identity，可与 `broad_process_v1` 做同口径比较。
- metrics/checkpoint 新增 `output_affine` metadata，记录 mode、input dim、scale、learning rate、参数量与 identity initialization，并保存 `output_affine_state_dict`。
- `scripts/server/run_multiline_process_conditioned_thermal_a100.sh` 新增 `OUTPUT_AFFINE_MODE`、`OUTPUT_AFFINE_SCALE` 与 `OUTPUT_AFFINE_LR` 环境变量。
- `scripts/server/run_phase39_broad_output_affine_a100.sh` 默认 focused 跑 feature-active broad12 `spot_size` 与 `laser_power`，run tag 使用 `out_affine`。
- `scripts/server/summarize_phase30_broad_process_selector_smoke.py --include-broad-output-affine` 可在 manifest/split comparability gate 中纳入 Phase 39 artifacts，并显示 output affine metadata。

focused A100 验证命令：

```bash
PROFILE_SPLITS="spot_size laser_power" DATASET_LIMIT=12 DATASET_ORDER=process_round_robin \
STEPS=500 N_ESTIMATORS=80 OUTPUT_AFFINE_SCALE=0.5 \
  bash scripts/server/run_phase39_broad_output_affine_a100.sh \
  > logs/phase39_broad12_output_affine_a100_v1.log 2>&1

python scripts/server/summarize_phase30_broad_process_selector_smoke.py \
  --dataset-limit 12 \
  --dataset-order process_round_robin \
  --split spot_size \
  --split laser_power \
  --include-broad-output-affine \
  --json-output outputs/reports/phase39_broad12_output_affine_summary.json \
  --require-comparable
```

验收：先比较 `spot_size` 和 `laser_power` against mean/kNN/ExtraTrees、no-process、`process_axis_v1`、`broad_process_v1` 和 Phase 38 负诊断。若 output affine 在至少一个已正向 split 上同时改善 global/hot q90/gradient q90 且没有明显不稳定，再做 paired seed check；若退化，则关闭为 output-calibration diagnostic，下一步转向更强数据表示或更结构化 process-conditioned architecture。当前分支不需要 A100-SXM4-80GB。

结果：

| Dataset/Split | Method | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | 判断 |
| --- | --- | ---: | ---: | ---: | --- |
| broad12 `spot_size` | `broad_process_v1` | 136.309183 | 165.228535 | 169.049295 | baseline |
| broad12 `spot_size` | `broad_output_affine` | 137.814723 | 170.105606 | 173.564283 | 负 |
| broad12 `laser_power` | `broad_process_v1` | 140.753534 | 254.473291 | 215.411533 | baseline |
| broad12 `laser_power` | `broad_output_affine` | 139.161435 | 238.174812 | 207.673483 | 正 |
| broad21 `laser_power` | `broad_process_v1` | 178.040331 | 296.909567 | 254.954359 | baseline |
| broad21 `laser_power` | `broad_output_affine` | 210.939830 | 407.352779 | 326.056523 | 不迁移 |

broad12 `laser_power` paired seed check 结果：`broad_process_v1` 三 seed 均值为 `148.172067 / 296.570673 / 242.768775`，`broad_output_affine` 三 seed 均值为 `136.412542 / 267.616397 / 224.432279`，delta 为 `-11.759525 / -28.954276 / -18.336496`。结论：Phase 39 关闭为 broad12 局部正但 broad21 不迁移的诊断分支，不做 broad21 seed check，不作为论文主 claim。

Phase 40 进一步测试更小 output-affine scale 是否能修复 broad21 transfer，结果仍为负：

| Dataset/Split | Method | Scale | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | 判断 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| broad21 `laser_power` | `broad_process_v1` | n/a | 178.040331 | 296.909567 | 254.954359 | baseline |
| broad21 `laser_power` | `broad_output_affine` | 0.25 | 195.281223 | 383.819303 | 305.364963 | 负 |
| broad21 `laser_power` | `broad_output_affine` | 0.10 | 224.145034 | 412.785926 | 334.457714 | 负 |

结论：不要继续调 output-affine scale。Phase 41 转向 physics-derived process representation，结果文档为 `docs/results/ambench_multiline_process_derived_process_features_v1.md`。关键实现：

- CLI 新增 `--input-derived-process-features none|am_energy_v1`；默认 `none`，不改变旧实验。
- `am_energy_v1` 从 `laser_power_W`、`scan_speed_mm_s`、`spot_size_um` 派生 `P/v`、`P/(v*d)`、`P/(v*d^2)` 与 `d/v` 形式的 AM 工艺特征，并加入 `input_features.effective_columns`。
- `scripts/server/run_multiline_process_conditioned_thermal_a100.sh` 新增 `PROCESS_DERIVED_FEATURE_MODE` 环境变量。
- `scripts/server/run_multiline_process_conditioned_thermal_a100.sh` 新增 `PROCESS_FEATURE_COLUMNS`，可用 `PROCESS_FEATURE_COLUMNS=""` 跑 derived-only 诊断。
- `scripts/server/run_phase41_broad_derived_process_features_a100.sh` 默认 focused 跑 broad21 `laser_power`，run tag 使用 `phys_proc`。
- `scripts/server/summarize_phase30_broad_process_selector_smoke.py --include-broad-derived-process` 可在 manifest/split comparability gate 中纳入 Phase 41 artifacts，并显示 derived-process metadata。

focused A100 验证命令：

```bash
PROFILE_SPLITS=laser_power DATASET_LIMIT=21 DATASET_ORDER=process_round_robin \
STEPS=500 N_ESTIMATORS=80 \
  bash scripts/server/run_phase41_broad_derived_process_features_a100.sh \
  > logs/phase41_broad21_laser_power_derived_process_a100_v1.log 2>&1

python scripts/server/summarize_phase30_broad_process_selector_smoke.py \
  --dataset-limit 21 \
  --dataset-order process_round_robin \
  --split laser_power \
  --include-broad-derived-process \
  --json-output outputs/reports/phase41_broad21_laser_power_derived_process_summary.json \
  --require-comparable
```

验收：先比较 `broad_derived_process` against mean/kNN/ExtraTrees、no-process、`process_axis_v1` 与 `broad_process_v1`。若 broad21 `laser_power` 同时改善 global/hot q90/gradient q90 且不落后于最强 baseline，再做 broad12/broad21 paired validation；若退化，则关闭为物理派生特征诊断并继续 pivot。当前分支不需要 A100-SXM4-80GB。

结果：raw process scalars + `am_energy_v1` 在 broad21 `laser_power` 上改善 hot/gradient 但严重牺牲 global：`212.704856 / 221.878476 / 238.794848` vs `broad_process_v1` `178.040331 / 296.909567 / 254.954359`。derived-only `am_energy_v1` 去掉 raw scalars 后在 broad21 上转为正向：`171.892969 / 211.624381 / 207.270255`，但 broad12 同口径检查退化为 `162.766699 / 303.019663 / 254.346542` vs `broad_process_v1` `140.753534 / 254.473291 / 215.411533`。结论：Phase 41 不做 seed expansion；下一步进入 Phase 42，检查 validation metrics 能否在 raw process scalars 与 derived-only representation 之间做不看 test 的选择，或者转向更强 baseline-facing architecture。

Phase 42 validation-selection 检查结果文档为 `docs/results/ambench_multiline_process_validation_selection_v1.md`。脚本：

```bash
python scripts/server/summarize_phase42_validation_selection.py \
  --json-output outputs/reports/phase42_laser_power_validation_selection_summary.json
```

结论：simple validation selector 不可信。broad12 val RMSE/gradient 能选 raw process，broad21 val gradient 能选 derived-only，但 broad21 val RMSE 错选 raw，broad21 val hot q90 错选 raw+derived。下一步不应添加 hand-coded validation-selected raw/derived profile，而应转向更强 baseline-facing architecture 或训练目标。当前已接入 `--prediction-anchor-weight` 作为新的 baseline-facing 训练目标；下一轮 broad12/broad21 `laser_power` focused validation 入口是 `scripts/server/run_phase42_broad_prediction_anchor_a100.sh`。

Prediction-anchor focused validation 结果记录在 `docs/results/ambench_multiline_process_prediction_anchor_v1.md`。`weight=0.05` 与 `0.01` 都改善 broad12 `laser_power`，但 broad21 global RMSE 退化：`0.05` 为 `192.755869 / 320.927695 / 275.299993`，`0.01` 为 `200.097570 / 292.510967 / 267.746425`，均弱于 `broad_process_v1` 的 `178.040331 / 296.909567 / 254.954359`。结论：prediction anchor 关闭为 split-local 诊断，不做 seed expansion；下一步应转向更强 process representation 或 architecture，而不是继续调 scalar output shrinkage。

Phase 43 当前分支为 `process_encoder_v1`，结果文档为 `docs/results/ambench_multiline_process_process_encoder_v1.md`。关键实现：

- `MacroPINN` 新增 `process_encoder_mode=none|linear` 与 `process_encoder_dim`。
- `linear` encoder 对 leading input features identity 初始化，用于从 raw process route guard 起步。
- 首轮 focused validation 使用 raw process scalars + `am_energy_v1`，再压缩到 3 维 latent 后进入 `broad_process_v1` route。
- metrics/checkpoint 记录 `process_encoder` metadata；summary 脚本支持 `--include-broad-process-encoder`。

focused A100 命令：

```bash
PROFILE_SPLITS=laser_power DATASET_LIMITS="12 21" DATASET_ORDER=process_round_robin \
PROCESS_DERIVED_FEATURE_MODE=am_energy_v1 PROCESS_ENCODER_MODE=linear PROCESS_ENCODER_DIM=3 \
PROCESS_FEATURE_TAG=proc_enc STEPS=500 N_ESTIMATORS=80 \
  bash scripts/server/run_phase43_broad_process_encoder_a100.sh \
  > logs/phase43_laser_power_process_encoder_a100_v1.log 2>&1
```

验收：先比较 `broad_process_encoder` against mean/kNN/ExtraTrees、no-process、`process_axis_v1`、`broad_process_v1`、derived-only `am_energy_v1` 和 Phase 42 prediction-anchor diagnostics。若 broad12 与 broad21 `laser_power` 同时改善或保持 global RMSE 且改善 hot/gradient，再做 seed expansion；若仍是 split-local，则关闭为 representation diagnostic。

结果：`process_encoder_v1` 是 broad21-positive 但 broad12-negative。`linear`/`7->3` identity-initialized encoder 在 broad12 `laser_power` 上退化到 `189.137331 / 369.311362 / 293.900869`，而 broad21 `laser_power` 相比 `broad_process_v1` 改善到 `172.459317 / 264.292100 / 237.096411`。这比 Phase 41 derived-only 的 broad21 全局表现还弱，因此仍属于 split-local diagnostic，不做 seed expansion。

结论：Phase 43 关闭为 representation diagnostic。编码器能在 broad21 上朝正确方向移动，但 broad12 退化过大，说明当前分支仍不足以处理广12/broad21 transfer split。下一步若继续，应转向更明确的过程-group balance/objective，而不是继续堆 process encoder 宽度。

#### Phase 44：process-group balanced objective

目标：保持 `broad_process_v1` 路由守门不变，先测试 train split 内过程组平衡的 supervised objective，而不是继续增加 process encoder 宽度或 scalar anchor。

实现：

- `gnnpinn.train.macro_pinn` 新增 `--data-loss-group-balance-column` 与 `--data-loss-group-balance-strength`。
- `process_condition` 是内置组合键，来自 `(laser_power_W, scan_speed_mm_s, spot_size_um)`。
- group balance 使用 train split 内 inverse-frequency 权重，并与已有 hot/gradient region weighting 正交相乘；metrics/checkpoint 记录 `data_loss_group_balance` 与组合后的 `data_loss_objective`。
- summary 脚本支持 `--include-broad-process-group-balance`。

focused A100 命令：

```bash
PROFILE_SPLITS=laser_power DATASET_LIMITS="12 21" DATASET_ORDER=process_round_robin \
PROCESS_FEATURE_TAG=group_bal PROCESS_DERIVED_FEATURE_MODE=am_energy_v1 \
DATA_LOSS_GROUP_BALANCE_COLUMN=process_condition DATA_LOSS_GROUP_BALANCE_STRENGTH=1.0 \
STEPS=500 N_ESTIMATORS=80 \
  bash scripts/server/run_phase44_broad_group_balance_a100.sh \
  > logs/phase44_laser_power_group_balance_a100_v1.log 2>&1
```

验收：比较 `broad_process_group_balance` against mean/kNN/ExtraTrees、no-process、`process_axis_v1`、`broad_process_v1`、Phase 41 derived-only、Phase 42 prediction-anchor 与 Phase 43 process encoder。只有 broad12 与 broad21 `laser_power` 同时改善或保持 global RMSE 且改善 hot/gradient，才做 seed expansion；若仍是 split-local，关闭为 objective diagnostic。

验收结果：Phase 44 已关闭为负向 objective diagnostic。两个 summary 均通过 `--require-comparable`：

- broad12 `laser_power`: `broad_process_v1` 为 `140.753534 / 254.473291 / 215.411533`，`broad_process_group_balance` 退化到 `189.364413 / 356.845339 / 289.133792`。
- broad21 `laser_power`: `broad_process_v1` 为 `178.040331 / 296.909567 / 254.954359`，`broad_process_group_balance` 为 `212.704856 / 221.878476 / 238.794848`。

结论：该分支只在 broad21 region metrics 上有改善，但 global RMSE 明显退化，且 broad12 全部指标退化。不要 seed expansion；继续保留 `broad_process_v1` 作为 broad-data route guard。

#### Phase 45：baseline-guarded process expert Gate 1

目标：在继续实现任何 trainable MoE 或更复杂 process expert 前，先做 prediction-level 上界验证。Phase 33-44 已经显示小模块/标量目标反复产生 split-local tradeoff；Phase 45 先检查已有专家预测是否包含一个可由 train/validation 选择的、能同时压过 `broad_process_v1` 和强 classical baseline 的组合。

实现：

- `gnnpinn.eval.field_baseline` 新增 `--prediction-output`，可导出 mean/kNN/ExtraTrees 的 row-aligned prediction CSV。
- `gnnpinn.train.macro_pinn` 新增 `--prediction-output` 与 `--prediction-method-name`，可导出 Macro PINN 变体的 row-aligned prediction CSV。
- `scripts/server/phase45_prediction_stack_probe.py` 读取多个 prediction CSV，在 train/validation split 上拟合 simplex-constrained nonnegative stack weights，并报告 split/region metrics；输出中显式记录 `uses_test_for_selection=false`。
- `scripts/server/run_multiline_process_conditioned_thermal_a100.sh` 新增 `PREDICTION_OUTPUT_DIR`，默认空，不改变旧 artifacts。
- `scripts/server/run_phase45_prediction_stack_probe_a100.sh` 生成 broad12/broad21 `laser_power` 的 mean、kNN process、ExtraTrees process、no-process Macro PINN、`broad_process_v1`、derived-only `am_energy_v1` predictions，并运行 stack probe。

focused A100 命令：

```bash
PROFILE_SPLITS=laser_power DATASET_LIMITS="12 21" DATASET_ORDER=process_round_robin \
STEPS=500 N_ESTIMATORS=80 WEIGHT_STEP=0.1 \
  bash scripts/server/run_phase45_prediction_stack_probe_a100.sh \
  > logs/phase45_prediction_stack_probe_a100_v1.log 2>&1
```

验收：只有 prediction stack 在 broad12 与 broad21 `laser_power` 上同时改善或保持 global RMSE，并改善 hot q90 与 gradient q90，且能面对最强 classical baseline，才进入 `baseline_guarded_expert_v1` 训练模型实现。若 stack probe 失败，则不要实现 MoE；应转向数据/任务定义或其他有真实可学习剩余误差的 paper-facing 方向。

验收结果：Phase 45 Gate 1 已关闭为负向诊断。两个 summary 均生成：

- broad12 `laser_power`: stack 使用 `0.4` ExtraTrees process + `0.6` `broad_process_v1`，test 为 `143.594749 / 286.768169 / 229.955382`，弱于 mean `132.965887 / 242.427068 / 208.105836` 和 `broad_process_v1` `140.753540 / 254.473291 / 215.411533`。
- broad21 `laser_power`: stack 使用 `0.1` mean + `0.4` ExtraTrees process + `0.3` no-process Macro PINN + `0.2` derived-only，test 为 `160.353180 / 321.588710 / 255.299857`，弱于 mean `131.741364 / 237.730958 / 205.133029`，且 hot/gradient 弱于 `broad_process_v1` `178.040335 / 296.909567 / 254.954359`。

结论：不要实现 `baseline_guarded_expert_v1`，不要 seed expansion，也不需要 A100-SXM4-80GB。当前 expert pool 没有可由 train/validation 稳定选择并迁移到 broad12+broad21 `laser_power` 的组合优势。

#### Phase 46：Bayesian inverse-closure PINN feasibility gate

目标：Phase 45 关闭后，不再继续包装同一批 expert predictions。下一步 paper-facing 方向应改变问题表述：从 full-field global RMSE 竞争，转到 sparse-data inverse closure/source discovery with calibrated uncertainty。

Phase 46 已完成本地 feasibility validation，结果文档为 `docs/results/ambench_bayesian_inverse_closure_pinn_plan_v1.md`。当前结论是不启动 A100 broad12/broad21 扩展。

实现：

- `scripts/server/phase46_bayesian_inverse_closure_probe.py` 实现轻量 Bayesian linear posterior over low-dimensional source/closure proxy features。
- `tests/test_phase46_bayesian_inverse_closure_probe.py` 覆盖 synthetic parameter recovery 与 table-mode sparse sampling summary。
- probe 比较 `random` 与 `uncertainty_source` sparse sampling。
- 不做 full Bayesian neural weights。

本地验证命令：

```bash
PYTHONPATH=src python -X utf8 scripts/server/phase46_bayesian_inverse_closure_probe.py \
  --mode synthetic \
  --synthetic-grid 16 \
  --synthetic-frames 8 \
  --synthetic-noise-std 8.0 \
  --initial-size 48 \
  --acquisition-size 144 \
  --repeats 5 \
  --json-output outputs/reports/phase46_synthetic_bayesian_inverse_closure_probe_summary.json

PYTHONPATH=src python -X utf8 scripts/server/phase46_bayesian_inverse_closure_probe.py \
  --mode table \
  --table data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_medium_probe.csv \
  --target temperature_C \
  --split-manifest outputs/data_splits/ambench_line_0_1_temperature_medium_probe_split.json \
  --initial-size 64 \
  --acquisition-size 192 \
  --repeats 5 \
  --json-output outputs/reports/phase46_line0_1_temperature_medium_probe_bayesian_inverse_closure_summary.json
```

结果：

| Probe | Strategy | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Coverage / recovery |
| --- | --- | ---: | ---: | ---: | --- |
| synthetic | random | 8.002149 | 7.835315 | 8.655339 | coverage `0.913936`, source recovery `1.0` |
| synthetic | uncertainty_source | 8.195683 | 7.749243 | 8.478088 | coverage `0.907090`, source recovery `1.0` |
| local AM-Bench `Line_0_1` | random | 171.880449 | 99.651322 | 128.046258 | coverage `0.688421` |
| local AM-Bench `Line_0_1` | uncertainty_source | 132.956146 | 146.360946 | 131.690693 | coverage `0.629474` |

结论：

```text
Phase 46 关闭为负向诊断。
Synthetic source parameters 可辨识，但 uncertainty-guided sampling 没有守住 global/hot/gradient/calibration 联合门槛。
Local AM-Bench sparse proxy 改善 global RMSE，但 hot q90 明显退化，coverage 不足。
```

不要将当前 Phase 46 扩展到 A100 broad12/broad21。若以后重启，应先在本地改进 AM-Bench 对齐的 heat-source feature family、posterior calibration 和 multi-objective acquisition，再重新跑本地 gate。

#### Phase 47：lightweight physics-guided closure attention probe

目标：先验证最容易落地的 PINN + attention 思路，即在 Phase 46 的低维 inverse-closure proxy 上加入 deterministic physics-guided gating，而不是直接实现 CNN/GCN/Transformer/meta-learning。

实现：

- 复用 `scripts/server/phase46_bayesian_inverse_closure_probe.py`。
- 新增 `--feature-mode base|physics_attention`。
- `physics_attention` 用 source prior 与 source-prior spatial gradient 生成 attention score，并追加 `attn_*` source-like features。
- Macro PINN 训练路径未改动。

本地验证命令：

```bash
PYTHONPATH=src python -X utf8 scripts/server/phase46_bayesian_inverse_closure_probe.py \
  --mode synthetic \
  --feature-mode base \
  --synthetic-grid 16 \
  --synthetic-frames 8 \
  --synthetic-noise-std 8.0 \
  --initial-size 48 \
  --acquisition-size 144 \
  --repeats 5 \
  --json-output outputs/reports/phase47_synthetic_base_closure_probe_summary.json

PYTHONPATH=src python -X utf8 scripts/server/phase46_bayesian_inverse_closure_probe.py \
  --mode synthetic \
  --feature-mode physics_attention \
  --synthetic-grid 16 \
  --synthetic-frames 8 \
  --synthetic-noise-std 8.0 \
  --initial-size 48 \
  --acquisition-size 144 \
  --repeats 5 \
  --json-output outputs/reports/phase47_synthetic_physics_attention_closure_probe_summary.json

PYTHONPATH=src python -X utf8 scripts/server/phase46_bayesian_inverse_closure_probe.py \
  --mode table \
  --feature-mode base \
  --table data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_medium_probe.csv \
  --target temperature_C \
  --split-manifest outputs/data_splits/ambench_line_0_1_temperature_medium_probe_split.json \
  --initial-size 64 \
  --acquisition-size 192 \
  --repeats 5 \
  --json-output outputs/reports/phase47_line0_1_base_closure_probe_summary.json

PYTHONPATH=src python -X utf8 scripts/server/phase46_bayesian_inverse_closure_probe.py \
  --mode table \
  --feature-mode physics_attention \
  --table data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_medium_probe.csv \
  --target temperature_C \
  --split-manifest outputs/data_splits/ambench_line_0_1_temperature_medium_probe_split.json \
  --initial-size 64 \
  --acquisition-size 192 \
  --repeats 5 \
  --json-output outputs/reports/phase47_line0_1_physics_attention_closure_probe_summary.json
```

结果：

| Probe | Feature mode | Strategy | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Coverage |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| synthetic | base | random | 8.002149 | 7.835315 | 8.655339 | 0.913936 |
| synthetic | base | uncertainty_source | 8.195683 | 7.749243 | 8.478088 | 0.907090 |
| synthetic | physics_attention | random | 8.029186 | 8.149420 | 8.846463 | 0.914425 |
| synthetic | physics_attention | uncertainty_source | 8.178254 | 7.893498 | 8.450195 | 0.911980 |
| local AM-Bench `Line_0_1` | base | random | 171.880449 | 99.651322 | 128.046258 | 0.688421 |
| local AM-Bench `Line_0_1` | base | uncertainty_source | 132.956146 | 146.360946 | 131.690693 | 0.629474 |
| local AM-Bench `Line_0_1` | physics_attention | random | 134.384901 | 155.581884 | 141.281181 | 0.852632 |
| local AM-Bench `Line_0_1` | physics_attention | uncertainty_source | 108.580442 | 221.871098 | 173.180299 | 0.783158 |

结论：

```text
Phase 47 关闭为负向诊断。
physics_attention 在本地 AM-Bench sparse proxy 上改善 global RMSE 和 coverage，
但显著伤害 hot q90 与 gradient q90；synthetic 上也没有稳定优势。
```

不要将当前 lightweight attention/gating 分支扩展到 CNN、GCN、Transformer attention、meta-learning 或 A100 broad12/broad21。若以后重启，先设计不牺牲 hot/gradient 的 region-aware source/closure gate。

#### Phase 48：region-preserving Bayesian inverse-closure acquisition gate

目标：修正 Phase 46/47 暴露的核心失败模式：global RMSE 可以改善，但 hot q90 与 gradient q90 被牺牲。Phase 48 只测试 acquisition/calibration 层，不改变 Macro PINN 主训练路径。

实现：

- `region_quota_uncertainty`：为 source-prior hot 和 source-prior-gradient 区域保留采样配额。
- `pareto_source_gradient`：按 posterior uncertainty、source prior、source-prior gradient 组合打分。
- `validation_selected_region_policy`：只用 validation objective 在候选策略中选择。
- `--calibration-mode conformal90`：用 validation residual nonconformity 缩放 90% predictive interval。
- `--require-region-preservation`：要求 active strategy 同时不恶化 hot q90 与 gradient q90。

本地验证命令：

```bash
PYTHONPATH=src python -X utf8 scripts/server/phase46_bayesian_inverse_closure_probe.py \
  --mode synthetic \
  --feature-mode base \
  --strategy region_quota_uncertainty \
  --strategy pareto_source_gradient \
  --strategy validation_selected_region_policy \
  --active-strategy validation_selected_region_policy \
  --calibration-mode conformal90 \
  --require-region-preservation \
  --synthetic-grid 16 \
  --synthetic-frames 8 \
  --synthetic-noise-std 8.0 \
  --initial-size 48 \
  --acquisition-size 144 \
  --repeats 5 \
  --json-output outputs/reports/phase48_synthetic_region_preserving_inverse_closure_summary.json

PYTHONPATH=src python -X utf8 scripts/server/phase46_bayesian_inverse_closure_probe.py \
  --mode table \
  --feature-mode base \
  --strategy region_quota_uncertainty \
  --strategy pareto_source_gradient \
  --strategy validation_selected_region_policy \
  --active-strategy validation_selected_region_policy \
  --calibration-mode conformal90 \
  --require-region-preservation \
  --table data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_medium_probe.csv \
  --target temperature_C \
  --split-manifest outputs/data_splits/ambench_line_0_1_temperature_medium_probe_split.json \
  --initial-size 64 \
  --acquisition-size 192 \
  --repeats 5 \
  --json-output outputs/reports/phase48_line0_1_region_preserving_inverse_closure_summary.json
```

结果：

| Probe | Strategy | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Coverage |
| --- | --- | ---: | ---: | ---: | ---: |
| synthetic | random | 8.002149 | 7.835315 | 8.655339 | 0.914425 |
| synthetic | uncertainty_source | 8.065546 | 7.722530 | 8.527884 | 0.912469 |
| synthetic | region_quota_uncertainty | 8.208480 | 7.763360 | 8.467253 | 0.905134 |
| synthetic | pareto_source_gradient | 8.165143 | 7.749178 | 8.477412 | 0.908068 |
| synthetic | validation_selected_region_policy | 8.113055 | 7.728250 | 8.458095 | 0.910513 |
| local AM-Bench `Line_0_1` | random | 171.880449 | 99.651322 | 128.046258 | 0.772632 |
| local AM-Bench `Line_0_1` | uncertainty_source | 116.038931 | 181.367100 | 148.307919 | 0.922105 |
| local AM-Bench `Line_0_1` | region_quota_uncertainty | 205.832757 | 60.956245 | 129.545936 | 0.682105 |
| local AM-Bench `Line_0_1` | pareto_source_gradient | 208.200707 | 69.184290 | 139.286502 | 0.736842 |
| local AM-Bench `Line_0_1` | validation_selected_region_policy | 205.832757 | 60.956245 | 129.545936 | 0.682105 |

结论：

```text
Phase 48 关闭为负向诊断。
Acquisition/calibration 层可以在 global 与 hot 之间做 tradeoff，
但尚不能同时守住 global/hot/gradient/coverage。
```

不要将 Phase 48 扩展到 A100 broad12/broad21。下一步应先改物理 source/closure feature family，例如 moving heat-source 参数、heat-kernel/Green's-function basis，或带明确物理参数的 synthetic-to-AM-Bench bridge。

#### Phase 49：heat-kernel / Green's-function physical feature family gate

目标：验证是否能通过更物理的 source/closure feature family 修复 Phase 48 的 tradeoff，而不是继续改 acquisition 或加神经模块。

实现：

- `--feature-mode heat_kernel`
- 追加多尺度 moving heat-source diffusion-kernel proxy features：
  - `heat_kernel_d{diffusion}_tau{decay}`
  - `source_hot_x_gradient`
- 保持 Bayesian linear inverse-closure proxy，不改变 Macro PINN 主训练路径。

本地验证命令：

```bash
PYTHONPATH=src python -X utf8 scripts/server/phase46_bayesian_inverse_closure_probe.py \
  --mode synthetic \
  --feature-mode heat_kernel \
  --synthetic-grid 16 \
  --synthetic-frames 8 \
  --synthetic-noise-std 8.0 \
  --initial-size 48 \
  --acquisition-size 144 \
  --repeats 5 \
  --json-output outputs/reports/phase49_synthetic_heat_kernel_inverse_closure_summary.json

PYTHONPATH=src python -X utf8 scripts/server/phase46_bayesian_inverse_closure_probe.py \
  --mode table \
  --feature-mode heat_kernel \
  --table data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_medium_probe.csv \
  --target temperature_C \
  --split-manifest outputs/data_splits/ambench_line_0_1_temperature_medium_probe_split.json \
  --initial-size 64 \
  --acquisition-size 192 \
  --repeats 5 \
  --json-output outputs/reports/phase49_line0_1_heat_kernel_inverse_closure_summary.json
```

结果摘要：

| Probe | Mode / strategy | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Coverage |
| --- | --- | ---: | ---: | ---: | ---: |
| synthetic | base / uncertainty_source | 8.065546 | 7.722530 | 8.527884 | 0.912469 |
| synthetic | heat_kernel / uncertainty_source | 8.208595 | 7.927758 | 8.521494 | 0.924205 |
| synthetic | heat_kernel + validation-selected region | 8.231446 | 8.066628 | 8.625961 | 0.921760 |
| local AM-Bench `Line_0_1` | base / uncertainty_source | 116.038931 | 181.367100 | 148.307919 | 0.703158 |
| local AM-Bench `Line_0_1` | heat_kernel / uncertainty_source | 124.831560 | 162.742524 | 140.899931 | 0.682105 |
| local AM-Bench `Line_0_1` | heat_kernel + validation-selected region | 198.389759 | 74.617728 | 132.642762 | 0.825263 |

结论：

```text
Phase 49 关闭为 synthetic-positive / AM-Bench-local-negative 诊断。
Heat-kernel feature family 在 synthetic + region acquisition 下有信号，
但本地 AM-Bench 仍然在 global RMSE 与 hot/gradient 之间 tradeoff。
```

不要将 Phase 49 扩展到 A100 broad12/broad21。下一步应转向显式 nonlinear moving-source parameter inversion，例如反演 source center offset、width、temporal decay、amplitude，而不是继续追加线性 proxy features。

#### Phase 50：explicit nonlinear moving-source parameter inversion gate

目标：从 Phase 49 的线性 heat-kernel proxy 转为显式低维 moving-source 参数反演，判断该问题表述是否比 full-field sparse proxy 更适合作为 paper-facing 方向。

实现：

- 新脚本：`scripts/server/phase50_moving_source_inversion_probe.py`
- 反演参数：
  - `start_x`
  - `span_x`
  - `center_y`
  - `sine_y_amp`
  - `core_width`
  - `tail_width`
  - `tail_decay`
- 对每个 nonlinear parameter candidate，在 train selected observations 上拟合 linear amplitude/background coefficients。
- 只用 validation objective 选 candidate，test labels 不参与拟合或选择。

本地验证命令：

```bash
PYTHONPATH=src python -X utf8 scripts/server/phase50_moving_source_inversion_probe.py \
  --mode synthetic \
  --grid-mode fast \
  --synthetic-grid 16 \
  --synthetic-frames 8 \
  --synthetic-noise-std 8.0 \
  --initial-size 48 \
  --acquisition-size 144 \
  --repeats 5 \
  --json-output outputs/reports/phase50_synthetic_moving_source_inversion_summary.json

PYTHONPATH=src python -X utf8 scripts/server/phase50_moving_source_inversion_probe.py \
  --mode table \
  --grid-mode fast \
  --table data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_medium_probe.csv \
  --target temperature_C \
  --split-manifest outputs/data_splits/ambench_line_0_1_temperature_medium_probe_split.json \
  --initial-size 64 \
  --acquisition-size 192 \
  --repeats 5 \
  --json-output outputs/reports/phase50_line0_1_moving_source_inversion_summary.json
```

结果：

| Probe | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Coverage | Parameter recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| synthetic moving-source inversion | 8.175431 | 8.012787 | 8.291478 | 0.887042 | 1.000000 |
| local AM-Bench `Line_0_1` moving-source inversion | 149.222337 | 133.745276 | 126.979221 | 0.745263 | n/a |

结论：

```text
Phase 50 关闭为 synthetic-positive / AM-Bench-local-negative 诊断。
Synthetic 参数可辨识，但 local AM-Bench sparse proxy 仍不支持稳定 paper-facing claim。
```

不要将 Phase 50 扩展到 A100 broad12/broad21。下一步应换 gate：用更密的 calibrated AM-Bench subset 先拟合 source parameters，再做 sparse downsampling；或把 paper-facing target 改成 parameter-identifiability + calibrated uncertainty，而不是继续追当前 `Line_0_1` sparse full-field RMSE。

#### Phase 51：dense-to-sparse source-parameter transfer gate

目标：验证 Phase 50 的失败是否只是 sparse observations 不足导致参数不可辨识。如果更密的 AM-Bench `Line_0_1` calibrated table 能先识别 source parameters，并且这些参数能迁移到 sparse coefficient refit，就可以继续设计 broad12/broad21 sparse validation；否则关闭当前 normalized moving-source grid。

实现：

- 新脚本：`scripts/server/phase51_dense_source_parameter_transfer_probe.py`
- 三条可比路径：
  - `sparse_search`：sparse observations 上搜索 source parameters 并拟合 coefficients。
  - `dense_params_sparse_theta`：dense observations 上识别 source parameters，再只用 sparse observations 重拟合 coefficients。
  - `dense_upper_bound`：dense observations 上识别 source parameters 并拟合 coefficients。
- gate 同时要求 global RMSE、hot q90 RMSE、gradient q90 RMSE、coverage 通过；coverage-only positive 不算通过。

A100 dense table 生成命令：

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 /home/vipuser/miniconda3/bin/conda run -n gnnpinn \
  python -m gnnpinn.data.loaders.ambench_hdf5 \
  --sample-id amb2022_03_line_0_1_temperature_phase51_dense_local \
  --output data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_phase51_dense.csv \
  --manifest outputs/data_audits/phase51_line0_1_temperature_dense_manifest.json \
  --split-manifest outputs/data_splits/phase51_line0_1_temperature_dense_split.json \
  --calibrate-temperature \
  --split-strategy frame \
  --min-signal 100 \
  --frame-step 10 \
  --max-frames 60 \
  --row-step 4 \
  --max-rows 160 \
  --col-step 2 \
  --max-cols 152
```

A100 validation commands:

```bash
PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 /home/vipuser/miniconda3/bin/conda run -n gnnpinn \
  python -X utf8 scripts/server/phase51_dense_source_parameter_transfer_probe.py \
  --mode synthetic \
  --grid-mode fast \
  --synthetic-grid 16 \
  --synthetic-frames 8 \
  --synthetic-noise-std 8.0 \
  --sparse-fit-size 192 \
  --dense-fit-size 512 \
  --repeats 5 \
  --json-output outputs/reports/phase51_synthetic_dense_source_parameter_transfer_summary.json

PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 /home/vipuser/miniconda3/bin/conda run -n gnnpinn \
  python -X utf8 scripts/server/phase51_dense_source_parameter_transfer_probe.py \
  --mode table \
  --table data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_phase51_dense.csv \
  --target temperature_C \
  --split-manifest outputs/data_splits/phase51_line0_1_temperature_dense_split.json \
  --grid-mode fast \
  --sparse-fit-size 512 \
  --dense-fit-size 4096 \
  --repeats 5 \
  --json-output outputs/reports/phase51_line0_1_dense_source_parameter_transfer_summary.json
```

结果：

| Probe/path | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE | Coverage |
| --- | ---: | ---: | ---: | ---: |
| synthetic `sparse_search` | 7.719156 | 8.411110 | 8.336583 | 0.937897 |
| synthetic `dense_upper_bound` | 7.612343 | 7.882416 | 7.818963 | 0.942298 |
| AM-Bench `sparse_search` | 132.593587 | 181.224995 | 129.999543 | 0.832464 |
| AM-Bench `dense_params_sparse_theta` | 132.221241 | 182.115709 | 130.153119 | 0.843128 |
| AM-Bench `dense_upper_bound` | 128.049763 | 185.851461 | 128.169027 | 0.876540 |

结论：

```text
Phase 51 关闭为 synthetic-positive / AM-Bench-dense-negative 诊断。
Dense fitting 不是充分解；当前 normalized moving-source grid 仍将误差推向 hot region。
```

不要将当前 moving-source grid 扩展到 broad12/broad21。下一步应改变 source representation：读取 `ScanStrategy/AMB2022-03-AMMT-718-Pad_XYPT.h5`，构建 physically registered source-path/time-lag features，并先在同一个 A100 dense `Line_0_1` gate 上验证 global/hot/gradient 是否能同时守住。

#### Phase 52：physically registered source-path feature gate

目标：验证 AM-Bench scan-strategy XYPT 文件是否能为 Phase 51 dense `Line_0_1` table 提供物理注册 source-path features。如果数据对象或坐标系统不兼容，必须先关闭该 gate，不能把 pad scan strategy 硬套到 single-track thermography。

实现：

- 新脚本：`scripts/server/phase52_registered_source_path_probe.py`
- 先检查兼容性：
  - XYPT group 是否是 `Xpad/Ypad`；
  - field table 的 `line_id` 与 `dataset_path` 是 pad 还是 `Line_*`；
  - XYPT 坐标单位与 table 坐标范围是否可安全重合；
  - 只有兼容时才构造 registered source-path / time-lag features。

A100 validation command:

```bash
PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 /home/vipuser/miniconda3/bin/conda run -n gnnpinn \
  python -X utf8 scripts/server/phase52_registered_source_path_probe.py \
  --scan-strategy data/raw/ambench/2022_single_track/AMB2022-03/mds2-2716/ScanStrategy/AMB2022-03-AMMT-718-Pad_XYPT.h5 \
  --table data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_phase51_dense.csv \
  --target temperature_C \
  --split-manifest outputs/data_splits/phase51_line0_1_temperature_dense_split.json \
  --json-output outputs/reports/phase52_line0_1_registered_source_path_summary.json
```

数据检查结果：

| Source | Points | Power-on points | Power-on segments | X range | Y range |
| --- | ---: | ---: | ---: | --- | --- |
| `XYPT/Xpad` | 117,999 | 12,267 | 47 | `[-0.6520, 13.6520]` mm | `[28.2470, 33.2470]` mm |
| `XYPT/Ypad` | 25,499 | 12,528 | 24 | `[-1.6104, 2.9834]` mm | `[27.0940, 34.3990]` mm |
| Phase 51 table | 10,205 | n/a | n/a | camera pixels `0..302` | camera pixels `0..636` |

Formal decision:

```text
negative: scan strategy file contains pad XYPT groups but table is a single-track Line_* dataset
```

结论：不要将 `AMB2022-03-AMMT-718-Pad_XYPT.h5` 用作 `ThermalData/Line_0_1/Signal` 的 registered source path。source-inversion broad12/broad21 扩展继续阻塞，除非找到 aligned single-track scan-path source，或改为 pad thermography table gate。

#### Phase 53：source-path data pivot gate

目标：完成 Phase 52 后的最后数据兼容性检查，判断当前 `mds2-2716` bundle 是否还有可支撑 paper-facing source-path inversion 的 aligned 数据对象。如果没有，则停止 source-inversion broad12/broad21 扩展，把论文主线转回 broad-data process-conditioned route guard。

新增脚本：

```bash
python -X utf8 scripts/server/phase53_source_path_data_pivot_gate.py \
  --json-output outputs/reports/phase53_source_path_data_pivot_summary.json

bash scripts/server/run_phase53_source_path_pivot_gate_a100.sh
```

关键 inventory：

| Item | Value |
| --- | --- |
| Thermography groups | 27 |
| Single-track thermography groups | 21 `Line_*` groups |
| Pad thermography groups | `X_pad1`, `X_pad2`, `Y_pad1`, `Y_pad1_SS`, `Y_pad2`, `Y_pad2_SS` |
| Scan-strategy file | `AMB2022-03-AMMT-718-Pad_XYPT.h5` |
| XYPT groups | `XYPT/Xpad`, `XYPT/Ypad` |
| Single-track scan-path groups | none |
| HDF5 camera-pixel to galvo-mm registration metadata | none found |

Formal inventory decision:

```text
negative: scan strategy exposes pad XYPT only; thermography has pad tables, but no HDF5 camera-pixel to galvo-mm registration metadata was found
```

Pad diagnostic-only rescale gate:

| Pad | Rows | Base RMSE / hot / gradient / cov90 | Registered-path diagnostic RMSE / hot / gradient / cov90 | Decision |
| --- | ---: | --- | --- | --- |
| `X_pad1` | 101 | `127.723559 / 190.914071 / 178.656473 / 1.000000` | `157.298079 / 170.275221 / 170.508595 / 1.000000` | negative: global RMSE worsens |
| `Y_pad1` | 372 | `153.859838 / 218.406789 / 183.031892 / 0.966102` | `221.030923 / 241.662999 / 203.527790 / 0.796610` | negative |

Phase 52 guard was also tightened: coordinate compatibility now checks span ratio in addition to range overlap. This prevents camera-pixel spans such as `0..300` / `0..620` from being treated as safely comparable to XYPT galvo-mm spans.

结论：

- 当前 AMB2022-03 bundle 没有 aligned single-track scan-path data。
- Pad thermography + pad XYPT 只能做 independent-rescale diagnostic，不能作为 paper-facing physical registration。
- 即使放宽到 rescale diagnostic，`X_pad1/Y_pad1` 也没有通过 combined global/hot/gradient gate。
- 不运行 source-inversion broad12/broad21 validation。
- Phase 54 应转向 broad-data process-conditioned route guard / process-axis selector 的论文贡献整合。

#### Phase 54：paper-facing process route claim boundary

目标：把 Phase 46-53 的 source-inversion / Bayesian PINN / registered source-path work 冻结为诊断分支，并把当前可投稿主线收敛到 `broad_process_v1` 的 broad-data route guard 贡献边界。

新增脚本：

```bash
python -X utf8 scripts/server/summarize_phase54_process_route_claim_boundary.py \
  --input outputs/reports/phase54_broad12_claim_boundary_input_summary.json \
  --input outputs/reports/phase54_broad21_claim_boundary_input_summary.json \
  --json-output outputs/reports/phase54_process_route_claim_boundary_summary.json \
  --markdown-output outputs/reports/phase54_process_route_claim_boundary_summary.md \
  --require-comparable
```

结果分类：

| Classification | Splits |
| --- | --- |
| paper-claim positive | `broad12:line`, `broad12:spot_size`, `broad21:line`, `broad21:spot_size` |
| route-guard positive | `broad12:laser_power`, `broad12:process`, `broad12:scan_speed`, `broad21:laser_power`, `broad21:process`, `broad21:scan_speed` |
| incomplete metric | none |
| diagnostic negative | none |
| incomparable | none |

论文边界：

- Clean process-conditioned strong-baseline-positive splits 是 broad12 和 broad21 `spot_size`。broad12 FiLM/global-standard 为 `136.309183 / 165.228535 / 169.049295`，优于 mean `151.850578 / 252.554440 / 233.119660`；broad21 FiLM/global-standard 为 `147.389475 / 163.081706 / 177.908136`，优于 mean `149.185412 / 251.976794 / 231.072566`。
- `line` 在 broad12/broad21 也压过 strong baselines，但 route 是 no-process fallback，因此只能支持 route guard，不支持“process conditioning 改善 line holdout”的表述。
- `laser_power`、`scan_speed` 和 full `process` 是 route-guard-only，不应 seed-expand 为强 baseline claim。
- broad21 `spot_size` 的 hot q90 已补齐。缺失原因是 target q90 浮点插值阈值略高于观测最大值，导致 region selector 选到 `0` 个 hot 点；`region_metric_tables` 现已将 quantile threshold clamp 到观测 min/max。

结论：Phase 54 关闭为 claim-boundary consolidation。下一步只对 broad12/broad21 `spot_size` 做 seed validation，不扩展 route-guard-only 轴。

#### Phase 55：spot-size transferable route seed validation

目标：把 Phase 54 的 broad12/broad21 `spot_size` 单 seed 正例升级为 seed-robust、可迁移的 paper-facing 模型贡献。

新增脚本：

```bash
DATASET_LIMITS="12 21" \
SEEDS="1 2" \
SUMMARY_SEEDS="7 1 2" \
STEPS=500 \
bash scripts/server/run_phase55_spot_size_route_seed_check_a100.sh \
  > logs/phase55_spot_size_route_seed_check_a100.log 2>&1
```

验证命令：

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
/home/vipuser/miniconda3/bin/conda run -n gnnpinn \
python -X utf8 scripts/server/summarize_phase55_spot_size_seed_check.py \
  --dataset-limit 12 \
  --dataset-limit 21 \
  --seed 7 \
  --seed 1 \
  --seed 2 \
  --require-complete \
  --require-pass
```

结果：

| Dataset | Method | N | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| broad12 | no-process | 3 | `238.093690 +/- 29.836097` | `424.409003 +/- 53.733799` | `382.799174 +/- 48.263668` |
| broad12 | `broad_process_v1` | 3 | `136.384782 +/- 0.467526` | `162.125337 +/- 4.909788` | `165.282182 +/- 5.270236` |
| broad21 | no-process | 3 | `217.922642 +/- 5.308273` | `401.488520 +/- 15.153059` | `360.868300 +/- 17.032414` |
| broad21 | `broad_process_v1` | 3 | `146.002303 +/- 1.118699` | `164.313888 +/- 3.548500` | `174.735839 +/- 2.301005` |

结论：Phase 55 通过 `seed_robust_transfer_positive`。当前可写成 paper-facing model contribution 的内容是 explicit broad-data process route guard，其中 `spot_size -> FiLM/global-standard` 是跨 broad12/broad21、跨 seeds 7/1/2 的稳定 process-conditioned 正例。`line` 仍是 no-process fallback route-guard evidence；`laser_power`、`scan_speed` 和 full `process` 仍是 route-guard-only。

#### Phase 56：manuscript-facing table/figure package

目标：把 Phase 54/55 的机器可读结果转换成论文主表、route-guard 边界表、负诊断 appendix 和可编辑 figure assets。

生成命令：

```bash
python -X utf8 scripts/server/build_phase56_manuscript_package.py \
  --manifest-output outputs/reports/phase56_manuscript_package_manifest.json
```

输出：

```text
docs/results/phase56_manuscript_package/phase56_manuscript_table_figure_package.md
docs/results/phase56_manuscript_package/phase56_main_spot_size_seed_positive_table.csv
docs/results/phase56_manuscript_package/phase56_route_guard_boundary_table.csv
docs/results/phase56_manuscript_package/phase56_negative_diagnostic_appendix_table.csv
docs/results/phase56_manuscript_package/phase56_spot_size_seed_validation_figure.svg
docs/results/phase56_manuscript_package/phase56_spot_size_seed_validation_figure.png
outputs/reports/phase56_manuscript_package_manifest.json
```

用途：

- 主文主表：`phase56_main_spot_size_seed_positive_table.csv`。
- 主文或结果图：`phase56_spot_size_seed_validation_figure.svg/png`。
- supplement/appendix：route-guard boundary 和 negative diagnostic tables。
- caption/source trace：`phase56_manuscript_table_figure_package.md`。

结论：Phase 56 完成结果包装，不新增训练证据。下一步进入 results text / figure caption / discussion boundary drafting。

#### Phase 57：claim governance and future-branch contract

目标：在继续任何新模型分支前，把当前 paper-facing floor、claim ledger 和 no-test-leakage gate 固化成机器可读合同。

生成命令：

```bash
python -X utf8 scripts/server/build_phase57_claim_governance.py
```

输出：

```text
docs/results/phase57_claim_governance/phase57_claim_contract.json
docs/results/phase57_claim_governance/phase57_claim_ledger.csv
docs/results/phase57_claim_governance/phase57_claim_governance.md
docs/results/phase57_claim_governance/phase57_claim_governance_manifest.json
```

结论：Phase 57 将 broad12/broad21 fixed-sampling `spot_size` seeds 7/1/2 固化为 frozen floor。未来任何模型候选都必须在 broad12 和 broad21 上同时守住 test RMSE、hot q90 RMSE 和 gradient q90 RMSE，且 route selection、hyperparameter selection、seed expansion 不能使用 test labels。

#### Phase 58：clean repro and stress evidence

目标：先证明当前 Phase 55/56/57 package 可从 GitHub clean checkout 重建，再用 stronger baselines、sampling density 和 auxiliary process panel 压测固定采样 `spot_size` claim。

关键输出：

```text
docs/results/phase58_clean_repro/phase58_clean_repro_manifest.json
docs/results/phase58_stronger_baseline_stress/phase58_stronger_baseline_stress_summary.md
docs/results/phase58_sampling_panel_stress/phase58_sampling_density_stress_summary.md
docs/results/phase58_sampling_panel_stress/phase58_process_panel_stress_summary.md
docs/results/ambench_phase58_clean_repro_stress_plan_v1.md
```

结论：

- Stronger-baseline stress 通过；random forest 和 histogram gradient boosting 没有压过 frozen Phase 55 `spot_size` floor。
- Auxiliary broad15 panel seed-7 为正：`138.855456 / 158.622677 / 165.869192`，优于 mean baseline。
- Alternate-density broad21 `spot_size` 为边界失败：`153.259455 / 270.628922 / 250.519935`，弱于 mean `139.725646 / 253.129723 / 231.780894`。因此 fixed-sampling claim 可保留，但不能写成 density-invariant robustness。

#### Phase 59：residual anatomy and upper-bound gate

目标：解释 Phase 58 alternate-density broad21 失败是否提供可学习的新模型信号，或只是 route boundary。

关键输出：

```text
docs/results/phase59_residual_anatomy/phase59_broad21_density_residual_anatomy.md
docs/results/phase59_residual_anatomy/phase59_broad21_density_residual_upper_bound.md
docs/results/ambench_phase59_residual_anatomy_v1.md
```

结论：失败是结构化的，但不是当前 no-test-leakage gate 下可支持模型扩展的信号。Worst slices 集中在 `Line_1_1_1`、hot q90、low time/frame bins 和 `Line_1_1_2`；但 train/val-only upper-bound probe 选择 `blend:broad_process_v1->mean:alpha=1`，且 `uses_test_for_selection=false`。因此不要从 density failure 直接进入 Candidate A/B 训练分支。

#### Phase 60：post-Phase-59 manuscript evidence package

目标：把 Phase 55-59 证据合成当前主文/补充材料的可审计入口，并给后续模型创新分支写入 gate。

生成命令：

```bash
python -X utf8 scripts/server/build_phase60_manuscript_evidence_package.py
```

输出：

```text
docs/results/phase60_manuscript_evidence_package/phase60_manuscript_evidence_package.md
docs/results/phase60_manuscript_evidence_package/phase60_main_spot_size_seed_positive_table.csv
docs/results/phase60_manuscript_evidence_package/phase60_route_guard_boundary_table.csv
docs/results/phase60_manuscript_evidence_package/phase60_stress_boundary_table.csv
docs/results/phase60_manuscript_evidence_package/phase60_appendix_negative_diagnostic_table.csv
docs/results/phase60_manuscript_evidence_package/phase60_next_branch_gate_table.csv
docs/results/phase60_manuscript_evidence_package/phase60_manuscript_evidence_package_manifest.json
```

结论：Phase 60 不新增训练证据。当前 manifest 记录 main rows `6`、route rows `8`、stress/boundary rows `19`、appendix rows `14`，并给出 `block_density_failure_driven_model_expansion`。下一步优先进入 Phase 61 manuscript results/methods/caption draft package；若用户要求继续模型创新，则只能从 Phase 60 next-branch gate 中未阻塞的新验证信号进入。

#### Phase 61：manuscript results/methods/caption draft package

目标：把 Phase 60 evidence package 转成主文 Results、Methods、caption 草稿和 claim-to-evidence crosswalk，作为 manuscript v0 的可审计输入。该阶段不新增训练证据。

生成命令：

```bash
python -X utf8 scripts/server/build_phase61_manuscript_draft_package.py
```

输出：

```text
docs/results/phase61_manuscript_draft_package/phase61_results_draft.md
docs/results/phase61_manuscript_draft_package/phase61_methods_draft.md
docs/results/phase61_manuscript_draft_package/phase61_table_figure_captions.md
docs/results/phase61_manuscript_draft_package/phase61_claim_evidence_crosswalk.csv
docs/results/phase61_manuscript_draft_package/phase61_literature_gap_register.csv
docs/results/phase61_manuscript_draft_package/phase61_manuscript_draft_package.md
docs/results/phase61_manuscript_draft_package/phase61_manuscript_draft_package_manifest.json
```

当前 Phase 61 manifest 记录 claim anchors `11`、literature gaps `3`、result draft files `2`、caption files `1`。Writing gate 是 `draft_ready_for_internal_results_methods; needs_verification_for_literature_context`。模型扩展 gate 继续继承 Phase 60 的 `block_density_failure_driven_model_expansion`。

结论：Phase 61 是 manuscript-facing package，不是模型扩展入口。下一步 Phase 66 应先完成 local/GitHub/server 三端同步和服务器可复现检查，再进入 manuscript v0 claim audit 与 validation-visible signal mining。

#### Phase 68：validation-visible signal mining scorecard

目标：把 Phase 59-61 的 evidence boundary 转成候选模型创新分支的 machine-readable scorecard 和 next action queue。该阶段不新增训练证据，也不直接打开 A100 训练分支。

生成命令：

```bash
python -X utf8 scripts/server/build_phase68_validation_signal_scorecard.py
```

输出：

```text
docs/results/phase68_validation_signal_scorecard/phase68_candidate_signal_scorecard.csv
docs/results/phase68_validation_signal_scorecard/phase68_next_action_queue.csv
docs/results/phase68_validation_signal_scorecard/phase68_validation_signal_scorecard.md
docs/results/phase68_validation_signal_scorecard/phase68_validation_signal_scorecard_manifest.json
```

当前 manifest 记录 candidate rows `6`、action rows `5`、opened trainable candidates `0`。Candidate A 仍是 `paused_no_training_signal`，Candidate B 是 `blocked_by_phase59_validation_gate`，Candidate C 是 `blocked_by_registration_data`，larger architecture branch 需要先过 local/synthetic identifiability gate，external dataset branch 只开放 data planning。

结论：Phase 68 允许后续继续尝试更多模型创新和模型架构，但必须先从 non-training signal probe 或 local/synthetic gate 进入，不能从 Phase 58/59 density failure 直接开训。A100-SXM4-80GB 的触发条件是已通过门槛的训练分支实测或明确预计超过当前 40GB 显存。

#### Phase 69：spot-size non-training signal probe

目标：执行 Phase 68 的 `P68-SPOT-SIGNAL` action，用已有 fixed-sampling、density stress、auxiliary panel 和 Phase 59 upper-bound 证据判断 Candidate A 是否能进入 bounded physical `spot_size` parameterization 的 A100 seed-7 gate。该阶段不新增训练证据。

生成命令：

```bash
python -X utf8 scripts/server/build_phase69_spot_size_signal_probe.py
```

输出：

```text
docs/results/phase69_spot_size_signal_probe/phase69_spot_size_signal_probe_table.csv
docs/results/phase69_spot_size_signal_probe/phase69_candidate_a_gate.json
docs/results/phase69_spot_size_signal_probe/phase69_spot_size_signal_probe.md
docs/results/phase69_spot_size_signal_probe/phase69_spot_size_signal_probe_manifest.json
```

当前 manifest 记录 signal rows `15`，其中 `pass=12`、`boundary=3`。Fixed-sampling broad12/broad21 和 broad15 auxiliary panel 支持当前 floor，但 alternate-density broad21 的 RMSE、hot q90 RMSE、gradient q90 RMSE 都是 strong-baseline boundary；Phase 59 upper-bound 继续从 validation 选择 `blend:broad_process_v1->mean:alpha=1`。

结论：Candidate A 当前 gate 为 `paused_no_training_signal`，不进入 A100 seed-7 训练，也不请求 A100-SXM4-80GB。下一步应进入 manuscript v0 claim audit，或继续做 non-training route-policy/data-registration 探测。

#### Phase 70：route-policy non-training audit

目标：执行 Phase 68 的 `P68-ROUTE-POLICY` action，用现有 main table、route-guard table、stress/boundary table 和 Phase 59 upper-bound 证据判断 Candidate B 是否能进入 validation-auditable route policy gate。该阶段不新增训练证据，也不启动 mixture-of-experts 或 trainable policy。

生成命令：

```bash
python -X utf8 scripts/server/build_phase70_route_policy_audit.py
```

输出：

```text
docs/results/phase70_route_policy_audit/phase70_route_policy_audit_table.csv
docs/results/phase70_route_policy_audit/phase70_candidate_b_gate.json
docs/results/phase70_route_policy_audit/phase70_route_policy_audit.md
docs/results/phase70_route_policy_audit/phase70_route_policy_audit_manifest.json
```

当前 manifest 记录 audit rows `29`。Route guard 能保留 fixed-sampling `spot_size` floor，并保留 no-process `line` fallback；但 `laser_power`、`scan_speed`、full `process` 等 boundary axes 仍落后 strong baselines，Phase 59 density upper-bound 也选择 mean fallback。Candidate B gate 为 `blocked_no_validation_visible_route_policy_signal`。

结论：不要实现 trainable route policy、mixture-of-experts 或更高容量路由器。下一步应进入 manuscript v0 claim audit，或执行 `P68-DATA-REGISTRATION`，检查 Candidate C 是否有新的物理注册数据入口。

#### Phase 71：data-registration non-training audit

目标：执行 Phase 68 的 `P68-DATA-REGISTRATION` action，用 Phase 52/53 注册阻塞证据、Phase 60 next-branch gate 和 Phase 68 scorecard 判断 Candidate C 是否能进入 heat-kernel、Green's-function 或 source-path fixed-feature gate。该阶段不新增训练证据，也不启动 A100 模型训练。

生成命令：

```bash
python -X utf8 scripts/server/build_phase71_data_registration_audit.py
```

输出：

```text
docs/results/phase71_data_registration_audit/phase71_data_registration_audit_table.csv
docs/results/phase71_data_registration_audit/phase71_candidate_c_gate.json
docs/results/phase71_data_registration_audit/phase71_data_registration_audit.md
docs/results/phase71_data_registration_audit/phase71_data_registration_audit_manifest.json
```

当前 manifest 记录 audit rows `7`，其中 blocking rows `5`、diagnostic rows `2`、aligned target count `0`。当前 AM-Bench bundle 有 `XYPT/Xpad` 和 `XYPT/Ypad`，也有 pad thermography tables，但没有 paper-facing single-track scan-path registration，也没有 pad camera-pixel 到 galvo-mm 的 HDF5 注册元数据。`X_pad1/Y_pad1` independent-rescale diagnostics 仍是 appendix negative/boundary evidence。

结论：Candidate C 当前 gate 为 `blocked_by_registration_data`。不要运行 heat-kernel、Green's-function、source-path broad12/broad21 A100 训练；下一步应进入 manuscript v0 claim audit，或单独规划外部/新增 registered target 数据卡。

#### Phase 74：manuscript v0 evidence-locked claim audit

目标：把 Phase 60/61/68-71 的 evidence boundary 汇总成内部可审阅 manuscript v0 包。该阶段不新增训练证据，不新增未核验文献 claim，不打开 A100 训练分支。

生成命令：

```bash
python -X utf8 scripts/server/build_phase74_manuscript_v0_claim_audit.py
```

输出：

```text
docs/results/phase74_manuscript_v0_claim_audit/phase74_manuscript_v0_evidence_locked.md
docs/results/phase74_manuscript_v0_claim_audit/phase74_claim_audit_table.csv
docs/results/phase74_manuscript_v0_claim_audit/phase74_table_figure_inventory.csv
docs/results/phase74_manuscript_v0_claim_audit/phase74_model_boundary_register.csv
docs/results/phase74_manuscript_v0_claim_audit/phase74_manuscript_v0_claim_audit_package.md
docs/results/phase74_manuscript_v0_claim_audit/phase74_manuscript_v0_claim_audit_manifest.json
```

当前 manifest 记录 claim-audit rows `13`、supported claim rows `12`、unsupported v0 claim rows `1`、inventory rows `13`、boundary rows `9`、literature gap rows `3`。Writing gate 为 `ready_for_internal_manuscript_review`，但 Introduction/Related Work 与 target-venue style claim 仍需后续文献和目标期刊核验。

结论：当前 paper-facing 主线应写成 route-guarded Macro PINN + fixed-sampling `spot_size` transfer floor。Candidate A/B/C、大架构和 source-path 分支只能作为 gated future work 或 appendix/boundary evidence；不要从当前 Phase 74 直接进入 A100 训练。

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
git -c core.sshCommand="ssh -i C:/Users/cjh02/.ssh/matsci_gnnpinn_a100 -p 22036 -o IdentitiesOnly=yes" \
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

Phase 74 manuscript v0 claim-audit package 已生成。Candidate A/B/C 都没有通过开训门槛；下一步不要直接启动 broad12/broad21 A100 训练，应进入 Phase 75 local/synthetic identifiability gate，或先解决 Phase 74 的 literature/venue gaps。

优先级：

1. 本地和 A100 服务器复现 Phase 74 manuscript v0 claim-audit package，确认 claim-audit rows `13`、boundary rows `9`、trainable model opened `false`。
2. 若继续模型创新，进入 Phase 75 local/synthetic identifiability gate：选择一个候选方向，先做小型可识别性/机制有效性验证。
3. 若继续论文写作，解决 Phase 74/61 的 literature gaps 和 target-venue style gaps，再进入 Introduction/Related Work 写作。
4. Candidate A/B/C 只有在新的 train/validation-visible signal 或 registered target 出现并通过各自非训练 gate 时才重新打开。
5. 只有当已通过门槛的训练分支实测或明确预计超过当前 40GB 显存时，才向用户请求 A100-SXM4-80GB。
