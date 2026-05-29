# 实验计划、指标与复现矩阵

## 实验总原则

本项目的实验必须同时满足科研和工程两类要求：

- 科研上能回答：方向一为什么有价值，方向三为什么优于串联/弱耦合。
- 工程上能复现：每个实验都有配置、命令、数据版本、随机种子、指标和产物。

## 实验阶段矩阵

| 编号 | 实验 | 目标 | 数据 | 模型 | 关键指标 |
|---|---|---|---|---|---|
| E0 | 数据审计 | 验证真实公开数据可用 | AM-Bench 最小子集 | 无 | completeness, field coverage |
| E1 | 数据 baseline | 建立无物理 ML 基线 | AM-Bench 单道/局部数据 | MLP/CNN surrogate | RMSE, MAE |
| E2 | 宏观 PINN | 验证自写 PINN 内核 | 热场/熔池数据 | MacroPINN | temperature error, PDE residual |
| E3 | 稀疏闭合发现 | 方向一最小实现 | AM-Bench 热/力数据 | MacroPINN + SparseClosure | error, sparsity, stability |
| E4 | GNN 条件化闭合 | 微观结构感知 | 微观组织图/统计 | MicroGNN + SparseClosure | generalization gain |
| E5 | 弱双向耦合 | 方向三最小闭环 | AM-Bench + 组织图 | CoupledGNNPINN | coupled improvement |
| E6 | 完整闭环 | 核心论文实验 | 多工况 AM-Bench | Full coupled model | all metrics + ablation |
| E7 | 备选数据迁移 | 验证方法泛化 | 电池/多孔/PFHub | selected modules | transfer metrics |

## E0: 数据审计

目标：确认不是“模型写完后发现数据不够”。

未来命令草案：

```bash
python -m gnnpinn.data.audit --config configs/data/ambench_2022_in625_single_track.yaml
```

输出：

- `data_card.md`
- `field_mapping.csv`
- `missingness_report.json`
- `sample_manifest.json`
- `split_preview.json`

指标：

| 指标 | 含义 |
|---|---|
| file_read_success_rate | 可读取文件比例。 |
| metadata_coverage | 关键元数据字段覆盖率。 |
| observation_coverage | 温度/组织/性能等目标字段覆盖率。 |
| split_validity | train/val/test 是否满足工况隔离或样品隔离。 |

## E1: 数据 baseline

目标：建立不含 PDE 和 GNN 的基础误差下限。

模型：

- MLP surrogate。
- CNN/UNet surrogate，如果输入是场量图像。
- 简单物理经验模型。

未来命令草案：

```bash
python -m gnnpinn.train.baseline --config configs/experiment/exp01_data_baseline.yaml
```

指标：

- RMSE。
- MAE。
- normalized RMSE。
- melt pool width/depth error。
- OOD process split error。

当前已落地的真实数据 smoke 结果见：

- [results/ambench_line_0_1_signal_split_smoke.md](results/ambench_line_0_1_signal_split_smoke.md)：raw digital level `signal` 的 320 点随机 split smoke。
- [results/ambench_line_0_1_temperature_frame_probe.md](results/ambench_line_0_1_temperature_frame_probe.md)：calibrated `temperature_C` 的 1044 点 frame-split probe。

## E2: 宏观 PINN

目标：验证自写 PyTorch PINN 内核、autograd 高阶导数和 PDE residual。

推荐第一版 PDE：

```text
rho * c_p * dT/dt = div(k * grad T) + Q
```

第一版可以把复杂项简化为：

- 常数或温度依赖热导率。
- 参数化热源。
- 边界热损失。

未来命令草案：

```bash
python -m gnnpinn.train.macro_pinn --config configs/experiment/exp02_macro_pinn.yaml
```

指标：

| 类别 | 指标 |
|---|---|
| 数据误差 | temperature RMSE, MAE, relative L2 |
| 物理误差 | heat equation residual, boundary residual, initial residual |
| 几何误差 | melt pool contour IoU, width/depth error |
| 泛化 | unseen power/speed split error |

## E3: 稀疏闭合发现

目标：方向一的最小实现。

形式：

```text
R = R_known + q_sparse(features)
q_sparse = Theta(features) @ xi
```

候选特征：

- `T`
- `grad_T`
- `laplacian_T`
- `t`
- `laser_power`
- `scan_speed`
- `thermal_history`

未来命令草案：

```bash
python -m gnnpinn.train.sparse_closure --config configs/experiment/exp03_sparse_closure.yaml
python -m gnnpinn.closure.export_symbolic --run outputs/runs/exp03_sparse_closure
```

对照：

- fixed-physics PINN。
- neural black-box closure。
- sparse closure。

指标：

| 指标 | 目标 |
|---|---|
| validation error | 低于 fixed-physics PINN。 |
| active_terms | 越少越好，但不能牺牲验证误差。 |
| coefficient_stability | 不同 seed 下主要项稳定。 |
| extrapolation_error | 未见工艺条件下优于 black-box closure。 |
| physical_violation_rate | 物理边界违反率低。 |

## E4: GNN 条件化闭合

目标：把 `think2.md` 的第二阶段落成代码实验。

形式：

```text
micro_embedding = MicroGNN(graph)
q_sparse = SparseClosure(features, micro_embedding)
```

未来命令草案：

```bash
python -m gnnpinn.data.build_graphs --config configs/data/ambench_2022_in718_microstructure.yaml
python -m gnnpinn.train.micro_gnn --config configs/experiment/exp04_gnn_conditioned_closure.yaml
```

图输入：

- 节点：晶粒/相区/局部 cell。
- 边：邻接、距离、晶界关系、取向差。
- 全局：工艺参数、材料牌号、热历史摘要。

指标：

| 指标 | 含义 |
|---|---|
| micro_stat_error | 晶粒尺寸、取向、相分数等统计误差。 |
| distribution_distance | Wasserstein/KL/EMD 等分布距离。 |
| closure_error_gain | 相比无 GNN 的闭合项误差降低。 |
| OOD_gain | 新工艺、新区域、新样品上的提升。 |

## E5: 弱双向耦合

目标：方向三最小闭环，不追求一步到位的全耦合。

未来命令草案：

```bash
python -m gnnpinn.train.coupled --config configs/experiment/exp05_weak_coupling.yaml
```

训练策略：

```text
1. load pretrained MacroPINN
2. load pretrained MicroGNN
3. load pretrained SparseClosure
4. freeze MicroGNN and CoarseGrainer
5. train MacroPINN with low-frequency feedback
6. unfreeze selected closure parameters
7. evaluate against one-way coupling and pure PINN
```

关键指标：

| 指标 | 含义 |
|---|---|
| macro_improvement | 相比 pure PINN 的宏观场误差下降。 |
| micro_improvement | 相比 pure GNN 的组织预测误差下降。 |
| coupling_residual | 宏观局部状态与介观平均状态一致性。 |
| feedback_stability | 多次反馈后误差是否发散。 |
| training_cost | 显存、训练时长、吞吐。 |

## E6: 完整闭环与消融

目标：核心论文实验。

未来命令草案：

```bash
python -m gnnpinn.train.coupled --config configs/experiment/exp06_full_coupling.yaml
python -m gnnpinn.eval.run --run outputs/runs/exp06_full_coupling
python -m gnnpinn.eval.export_tables --run outputs/runs/exp06_full_coupling
python -m gnnpinn.viz.paper_figures --run outputs/runs/exp06_full_coupling
```

消融矩阵：

| 消融 | 操作 | 预期回答 |
|---|---|---|
| no_physics | 移除 PDE residual | 物理约束是否提升外推。 |
| no_sparse | 用 black-box closure 替代 sparse closure | 可解释稀疏项是否必要。 |
| no_gnn | 移除微观结构 GNN | 微观组织信息是否贡献增益。 |
| one_way | 只保留 PINN -> GNN | 反馈是否必要。 |
| no_coarse_grain | 移除粗粒化反馈 | 微观到宏观桥接是否必要。 |
| frozen_closure | 闭合项固定不更新 | 动态本构/闭合是否必要。 |
| different_gnn | GCN/GAT/MPNN 对比 | 图结构选择是否影响结果。 |

论文主表建议：

| Method | Temp RMSE | Melt pool IoU | Microstructure error | OOD error | PDE residual | Active terms |
|---|---:|---:|---:|---:|---:|---:|
| MLP/CNN surrogate | | | | | | |
| Pure PINN | | | | | | |
| PINN + black-box closure | | | | | | |
| PINN + sparse closure | | | | | | |
| GNN-conditioned closure | | | | | | |
| One-way GNN-PINN | | | | | | |
| Two-way coupled GNN-PINN | | | | | | |

## E7: 备选方向迁移

目标：如果 AM-Bench 数据对齐过难，或者论文需要展示方法泛化，迁移到更容易控制的数据方向。

### 电池电极

```bash
python -m gnnpinn.data.prepare --config configs/data/battery_nrel_microstructure.yaml
python -m gnnpinn.train.coupled --config configs/experiment/exp07_battery_transfer.yaml
```

适合指标：

- effective diffusivity error。
- tortuosity error。
- porosity/connectivity error。
- electrochemical residual。

### 数字多孔介质

```bash
python -m gnnpinn.data.prepare --config configs/data/digital_porous_media.yaml
python -m gnnpinn.train.coupled --config configs/experiment/exp07_porous_transfer.yaml
```

适合指标：

- permeability error。
- pressure/flow PDE residual。
- pore-network graph error。
- macro transport prediction error。

### PFHub

```bash
python -m gnnpinn.data.prepare --config configs/data/pfhub_benchmark.yaml
python -m gnnpinn.train.sparse_closure --config configs/experiment/exp07_pfhub_equation_recovery.yaml
```

适合指标：

- PDE term recovery。
- field prediction error。
- interface evolution error。
- known-equation coefficient error。

## 复现要求

每个实验必须保存：

- resolved config。
- data card。
- split manifest。
- random seed。
- dependency versions。
- GPU type。
- training log。
- checkpoint。
- predictions。
- metrics。
- paper figures/tables。

每个论文图必须能由一个命令重建：

```bash
python -m gnnpinn.viz.paper_figures --figure fig3 --run outputs/runs/exp06_full_coupling
```

## 本地到服务器边界

本地继续推进的范围：

- 真实 HDF5 小样本转换。
- train/val/test-aware 指标管线。
- 小型 Macro PINN smoke。
- 温度校准公式验证。
- 时间/帧分组切分策略验证。

需要云 GPU 的边界：

- 单次训练使用数万至数十万 HDF5 采样点。
- 启用 PDE residual 且需要对大量坐标求导。
- 对多个 line/pad 工况进行多 seed 消融。
- 引入 GNN 条件化闭合或双向耦合训练。

当前本机已经完成 1044 点 calibrated temperature frame-split probe。下一步若继续向论文主结果推进，应转入云 GPU：优先 A800-SXM4-40GB；若要跑 dense sampling + PDE residual + 多 seed，建议 A100 80GB。

## 成功标准

第一版 MVP 成功标准：

- AM-Bench 最小子集可读取、可切分、可评估。
- Pure PINN 与至少一个 data-driven baseline 可跑。
- Sparse closure 相比 fixed-physics PINN 有稳定增益。
- GNN-conditioned closure 在至少一个 split 上显示微结构信息带来的增益。
- 弱双向耦合能跑完整 forward/backward，不发散，并完成与 one-way coupling 的对照。

核心论文成功标准：

- 在真实公开数据上，two-way coupled GNN-PINN 相比 pure PINN、pure GNN、one-way coupling 有明确优势。
- 稀疏闭合项能导出可解释表达式，并在未见工况上比 black-box closure 更稳。
- 所有核心图表可由配置和命令复现。
