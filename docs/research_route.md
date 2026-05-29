# 研究路线与 `think2.md` 可行性评估

## 问题：方向三是否必须以方向一为切入点？

结论：**不是必须，但强烈建议以方向一作为第一阶段入口。**

方向三的目标是宏观 PINN 与介观 GNN 的双向动态耦合。理论上可以直接实现：

```text
宏观 PINN -> 介观 GNN -> 粗粒化反馈 -> 宏观 PINN
```

但直接做会把所有困难同时暴露出来：

- 真实数据清洗和多模态对齐。
- 宏观 PDE residual 和边界/初始条件建模。
- 微观组织图构建、图动态更新和 GNN 训练。
- 微观到宏观的粗粒化映射。
- 联合训练中的梯度稳定性。
- 方向三核心主张所需的消融与对照。

如果没有一个可靠的中间“材料闭合/本构模块”，方向三的失败很难定位。因此方向一更适合作为入口：它先把最关键的“材料如何把场量和微观结构转化为等效响应”问题做成可验证模块。

## 对 `think2.md` 路线的判断

`think2.md` 提出的三阶段路线是合理的：

1. 单尺度 PINN + 稀疏/符号回归。
2. 引入 GNN 微观结构编码器。
3. 闭合宏观 PINN 与介观 GNN 的双向循环。

但需要做三处工程化修正，才能成为可执行代码项目。

### 修正一：把“完整本构方程发现”降为“闭合项/修正项发现”

第一版不要直接声称从真实 AM 数据中发现完整本构方程。更稳妥的表达是：

```text
已知物理方程 + 可学习闭合项 + 稀疏约束 -> 可解释修正关系
```

例如：

- 热源项修正。
- 温度相关有效热导率。
- 熔池边界附近的等效损失项。
- 应力/应变历史相关的局部材料参数修正。
- 由微结构 embedding 条件化的有效模量或屈服项。

这样既保留方向一的创新，又能用真实公开数据给出明确指标。

### 修正二：第一版方向三采用弱耦合，再进入强耦合

不要一开始让 GNN 在每个 PINN collocation point、每个训练 step 都反馈参数。推荐顺序：

1. **离线耦合**：训练好 PINN、GNN、粗粒化模块，再串联评估。
2. **弱在线耦合**：每隔若干 epoch 或若干时间窗口更新一次宏观参数。
3. **局部强耦合**：只在少量高价值采样点反馈，例如熔池边界、温度梯度大处、组织变化显著处。
4. **全耦合实验**：在 A100 80GB 级别资源上做最终论文实验。

### 修正三：微观组织不要求一开始做到逐点同源

AM-Bench 真实数据虽然强，但微观组织和热历史的空间/时间精确对齐可能复杂。第一版可以采用分层标签：

| 层级 | 数据形式 | 用途 |
|---|---|---|
| 样品级 | 工艺参数 -> 组织统计/力学性能 | 最早的可复现 baseline。 |
| 区域级 | 局部热历史 -> 区域组织统计 | 方向一和 GNN 条件化。 |
| 像素/晶粒级 | EBSD/显微图 -> 晶粒图 | 介观 GNN 和粗粒化。 |
| 时序级 | 仿真或原位数据 -> 组织演化 | 方向三完整闭环。 |

## 可执行路线

### Phase 0: 数据资格审查

目标：确认真实数据能支撑模型，而不是先写模型再找数据。

执行内容：

- 建立数据清单和 `data card`。
- 记录数据来源、下载入口、许可证、文件格式、样本数量、字段含义。
- 选择 AM-Bench 中一个最小问题，例如单道扫描或小区域热历史。
- 定义第一个 train/val/test split。
- 确认至少两个指标可以复现。

输出：

- 数据方向决策表。
- 字段映射表。
- 最小数据样本说明。
- 第一组 baseline 指标定义。

### Phase 1: 宏观 PINN 与真实数据基线

目标：搭建自写 PyTorch PINN 内核并在 AM-Bench 子问题上跑通。

推荐先做：

```text
输入: x, y, t, process parameters
输出: T(x, y, t)
约束: transient heat equation + boundary/initial/data loss
观测: AM-Bench thermography / melt pool geometry
```

执行内容：

- 写未来的 `MacroPINN` 接口。
- 实现 PDE residual、边界损失、数据损失的设计文档。
- 先对比纯数据 MLP/CNN surrogate，再对比 PINN。
- 输出温度误差、熔池几何误差和 PDE residual。

论文价值：

- 这是后续所有模块的可复现基线。
- 即使暂不接 GNN，也能证明数据和 PINN 管线有效。

### Phase 2: 方向一入口，闭合项发现

目标：把 PINN 中难以准确建模的材料项或源项表示为可学习模块，并用稀疏约束得到可解释形式。

建议形式：

```text
R_PDE = known_terms + closure_theta(features)
closure_theta(features) = sparse_library(features) @ coefficients
```

特征可以包括：

- `T`
- `grad_T`
- `time`
- `laser_power`
- `scan_speed`
- `thermal_history`
- 后续接入的 `micro_embedding`

执行内容：

- 构造候选项库，例如多项式、梯度项、历史项、温度依赖项。
- 使用 L1/Sequential thresholding 得到稀疏系数。
- 用 SymPy 导出可读公式。
- 比较 fixed-physics PINN、black-box closure、sparse closure。

验收指标：

- 验证集误差下降。
- PDE residual 不恶化。
- 闭合项稀疏。
- 在未见工艺条件上外推更稳。

### Phase 3: GNN 微观结构编码

目标：把微观组织纳入闭合项，形成“微观结构感知”的方向一模块。

图构建建议：

```text
节点: 晶粒、相区、孔隙/颗粒，或规则网格 cell
边: 邻接、晶界、距离阈值、取向差、界面类型
节点特征: 尺寸、取向、形貌、局部热历史、相类别
边特征: misorientation、界面长度、距离、温度梯度
全局特征: 工艺参数、材料牌号、扫描策略
```

执行内容：

- 从 EBSD/显微图/仿真组织建立图。
- 训练 `MicroGNNEncoder` 输出 `micro_embedding`。
- 将 `micro_embedding` 输入 `closure_theta`。
- 比较无微观结构、手工组织统计、GNN embedding 三种方案。

验收指标：

- 微结构统计预测误差。
- 闭合项验证误差下降。
- 跨工艺或跨区域泛化提升。

### Phase 4: 弱双向耦合

目标：实现方向三的最小闭环，但控制难度。

推荐流程：

```text
MacroPINN(x, t) -> local_state
MicroGNN(graph, local_state, history) -> micro_state_next
CoarseGrainer(micro_state_next) -> effective_params
MacroPINN(x, t, effective_params) -> updated_fields
```

执行内容：

- 先冻结 `MicroGNN` 与 `CoarseGrainer`，训练宏观 PINN。
- 再低频更新 effective parameters。
- 最后做局部采样点的端到端梯度回传。

验收指标：

- 相比 pure PINN，宏观场预测改善。
- 相比 one-way PINN -> GNN，组织/性能预测改善。
- 耦合迭代不发散。

### Phase 5: 完整闭环与论文实验

目标：把方向三作为核心论文贡献完整展示。

必须包含的对照：

- Pure PINN。
- Pure GNN / GNN surrogate。
- Direction 1 closure without GNN。
- GNN-conditioned closure。
- One-way coupling。
- Two-way coupling。
- Two-way coupling without sparse discovery。
- Two-way coupling without physical residual。

必须包含的论文图：

- 框架图。
- 数据流与多尺度映射图。
- 预测场量对比图。
- 微观组织预测图。
- 闭合项公式/稀疏项图。
- 消融柱状图或雷达图。
- 泛化工况误差图。

## 论文叙事建议

第一篇核心论文不宜只讲“我们把 GNN 和 PINN 组合了”。更强的叙事是：

```text
现有方法多为串联或弱物理约束。
本工作提出可微闭合项作为桥梁，
先从真实数据中发现材料响应修正关系，
再将其作为宏观 PINN 与介观 GNN 双向耦合的反馈通道。
```

贡献可以写成：

1. 提出一个面向真实公开 AM 数据的 GNN-PINN 多尺度耦合原型。
2. 建立可解释闭合项发现模块，使材料参数不再固定给定。
3. 设计微观组织图到宏观等效参数的可微反馈机制。
4. 在 AM-Bench 上进行可复现实验与消融，验证双向耦合优于纯 PINN、纯 GNN 与单向串联。

