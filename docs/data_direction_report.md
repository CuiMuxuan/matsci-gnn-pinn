# 公开数据方向评估报告

## 评估目标

本项目需要的数据方向必须服务两个目标：

1. 支撑方向一：GNN 引导或条件化的本构/闭合方程发现。
2. 支撑方向三：宏观 PINN 与介观 GNN 的多尺度双向耦合。

因此数据不能只适合普通性质预测。理想数据应同时包含：

- 工艺或外部载荷条件。
- 时空场量或 PDE 可约束观测，例如温度、位移、应变、浓度、电势、压力。
- 微观结构表征，例如 EBSD、晶粒图、相图、孔隙图、XCT、相场结果。
- 宏观性能或可验证输出，例如熔池几何、残余应力、力学性能、有效传输系数。
- 清晰许可证、公开入口、可复现下载和可拆分训练/验证/测试集。

## 评分维度

| 维度 | 含义 |
|---|---|
| 数据可获得性 | 是否公开、入口是否稳定、是否需要复杂授权。 |
| 同源完整性 | 工艺/场量/微结构/性能是否来自同一材料体系或同一实验链条。 |
| 方向一适配度 | 是否能做本构、闭合项、有效性质或源项发现。 |
| 方向三适配度 | 是否能做宏观场到介观结构再反馈宏观参数的闭环。 |
| 可复现实验性 | 是否容易定义 baseline、split、metric 和 artifact。 |
| 论文说服力 | 是否足以支撑材料计算/科学机器学习论文中的核心实验。 |

## 候选方向总览

| 候选方向 | 代表公开资源 | 数据便利性 | 方向一 | 方向三 | 综合判断 |
|---|---|---:|---:|---:|---|
| 金属增材制造热-组织-性能耦合 | NIST AM-Bench, ExaCA | 高 | 高 | 高 | **主线** |
| 锂离子电池电极微结构与退化 | NREL Battery Microstructure Library, Battery Archive 类数据 | 中 | 中高 | 中高 | 备选一 |
| 数字多孔介质微结构与传输 | Digital Porous Media Portal | 高 | 高 | 中 | 备选二 |
| 相场基准/PRISMS/PFHub | PFHub benchmarks | 高 | 中高 | 中 | 受控验证/降风险 |
| 复合材料 XCT/微结构有效性质 | Materials Data Facility | 中 | 中高 | 中 | 辅助/备选 |
| 晶体结构-弹性/热力学数据库 | Materials Project, JARVIS, NOMAD | 高 | 中 | 低 | 参数先验/辅助 |

## 主线：金属增材制造热-组织-性能耦合

### 数据来源

- NIST AM-Bench: https://www.nist.gov/ambench
- AM-Bench data management systems: https://www.nist.gov/ambench/am-bench-data-management-systems
- AM-Bench benchmark test data: https://www.nist.gov/ambench/benchmark-test-data
- ExaCA: https://github.com/LLNL/ExaCA
- ExascaleAM organization: https://github.com/ExascaleAM

### 为什么适合本项目

AM-Bench 是当前最适合本项目的主线，因为它不是单一微观图像数据集，也不是单一性质表，而是面向增材制造预测模型的公开基准生态。它覆盖金属 AM 中的工艺参数、原位测量、微观组织、残余变形/应变、机械行为和挑战问题，天然对应“工艺 -> 场量 -> 组织 -> 性能”的链条。

这与方向三高度一致：

```text
工艺参数
  -> 宏观 PINN: 温度场 / 位移场 / 应力场
  -> 介观 GNN: 晶粒组织 / 取向 / 局部组织统计
  -> 粗粒化: 等效热/力学参数
  -> 宏观 PINN: 更新 PDE 参数与闭合项
```

也与方向一兼容：

```text
已知热传导/力学平衡 PDE
  + 未知材料闭合项 q_theta(T, grad T, history, micro_embedding)
  + 稀疏/符号回归
  -> 可解释闭合关系或本构修正项
```

### 推荐起始子任务

第一版不要直接做完整热-力-组织三场耦合。推荐从 AM-Bench 中选择最小闭环：

1. 选择单道或少量扫描轨迹。
2. 先做温度场与熔池几何预测。
3. 再加入微观组织统计，例如晶粒尺寸、取向分布或晶粒形貌。
4. 最后加入等效参数反馈或力学响应。

### 可复现实验指标

| 层级 | 指标 |
|---|---|
| 宏观热场 | temperature RMSE/MAE, normalized RMSE, PDE residual, boundary residual |
| 熔池几何 | melt pool width/depth error, IoU, contour Hausdorff distance |
| 微观组织 | grain size distribution distance, orientation distribution similarity, texture metrics |
| 方向一闭合发现 | sparse term count, validation residual, equation stability, extrapolation error |
| 方向三耦合 | coupled vs pure-PINN improvement, coupled vs one-way coupling improvement, feedback stability |

### 风险

| 风险 | 应对 |
|---|---|
| 数据字段复杂，下载和元数据组织成本高 | 先建立 `data card` 与最小样本清单，完成数据审计后再建模。 |
| 微观组织与热历史不一定逐点同源 | 第一版使用局部/样品级组织统计，后续再做空间对齐。 |
| 完整热-力-组织耦合过重 | MVP 限定为热场 + 组织统计 + 等效参数反馈。 |
| PINN 在 3D 时空域训练慢 | 先做 2D/2.5D 切片，重型实验再上 A100 80GB。 |

## 备选一：锂离子电池电极微结构与退化

### 数据来源

- NREL battery microstructure library: https://www.nrel.gov/transportation/battery-microstructure-library-data
- Battery Archive 类循环数据可作为电化学退化补充，但需要另行验证同源性。

### 适合点

电池方向与材料化学关联强，天然存在电-化学-力学耦合。3D 电极微结构可以构图，宏观 PINN 可以求解浓度、电势或应力场，GNN 可以表示颗粒/孔隙网络或活性材料连接关系。

### 可做的方向一

- 发现有效扩散系数、反应源项、孔隙率-曲折度关系、局部退化源项。
- 将微结构图 embedding 条件化到电化学 PDE 的闭合项中。

### 可做的方向三

```text
宏观电化学 PINN
  -> 局部浓度/电势/应力历史
  -> 电极微结构 GNN
  -> 孔隙率、连通性、局部失活或裂纹状态
  -> 有效输运参数反馈
```

### 主要风险

公开数据常常不是同一实验链路：微结构来自一个来源，循环寿命来自另一个来源，力学失效来自第三个来源。要把它作为主线，需要先确认同源数据足够支撑闭环指标。否则它更适合做第二篇迁移验证，而不是第一版主线。

## 备选二：数字多孔介质微结构与传输

### 数据来源

- Digital Porous Media Portal: https://digitalporousmedia.org/

### 适合点

数据容易获取，3D 微 CT、孔隙结构和渗透率/毛细压力等宏观性质很适合从图或体素结构预测有效性质。PDE 也清晰，例如 Darcy flow、Stokes flow、扩散方程。

### 可做的方向一

- 从孔隙图结构发现或修正 Darcy/Forchheimer 闭合项。
- 学习孔隙网络 embedding 到渗透率、曲折度、有效扩散系数的关系。

### 可做的方向三

```text
宏观 Darcy/transport PINN
  -> 局部压力梯度/浓度梯度
  -> 孔隙网络 GNN
  -> 局部等效渗透率/扩散率
  -> 宏观 PINN 参数反馈
```

### 主要风险

该方向的“多尺度耦合”比较容易做成工程闭环，但材料化学叙事弱于 AM 和电池。它更适合作为方法降风险和快速跑通 pipeline 的备选，而不是最终主论文的首选应用。

## 备选三：PFHub/相场基准

### 数据来源

- PFHub benchmarks: https://pages.nist.gov/pfhub/benchmarks/

### 适合点

PFHub 提供受控 benchmark problems，适合验证 PINN、神经算子、GNN surrogate 和方程发现模块。它的优势是物理清楚、边界条件清楚、可生成大量一致数据。

### 局限

PFHub 更像受控数值基准，而不是真实实验主线。它可以用于：

- 调试 PINN PDE residual。
- 验证方程发现模块是否能从已知 PDE 中恢复关键项。
- 在 AM-Bench 数据对齐困难时提供备用 benchmark。

但如果最终论文主张真实材料系统中的多尺度耦合，PFHub 不宜作为唯一数据证据。

## 辅助方向：MDF 复合材料 XCT 与晶体数据库

### Materials Data Facility

MDF 提供多类材料数据集，包括 XCT、复合材料、模拟数据和高通量计算数据。它适合做辅助验证，例如微结构图像到有效属性的粗粒化模块。

入口：https://www.materialsdatafacility.org/

### Materials Project / JARVIS / NOMAD

晶体数据库适合提供弹性常数、热力学参数、晶体结构先验和材料参数标定。它们对方向一有帮助，但很难单独支撑方向三的动态多尺度闭环。因此建议作为参数先验或材料属性补全，不作为第一版主线。

## 数据方向决策

当前项目主线选择：

```text
主线: AM-Bench 金属增材制造热-组织-性能耦合
备选一: 电池电极微结构与退化
备选二: 数字多孔介质微结构与传输
备选三: PFHub/相场基准
辅助: MDF XCT, Materials Project/JARVIS/NOMAD
```

选择 AM-Bench 的原因不是“其他方向完全不可行”，而是 AM-Bench 最完整地覆盖本项目需要的四个环节：真实工艺输入、PDE 场量约束、微观组织表征、宏观性能验证。

## 数据准入门槛

进入任何模型训练前，每个候选数据方向都必须满足以下 gate：

| Gate | 要求 |
|---|---|
| G0: 入口稳定 | 有公开 URL、元数据、许可证或使用说明。 |
| G1: 最小样本可下载 | 能下载并读取一个最小样本，不依赖手工网页操作。 |
| G2: 字段映射明确 | 能定义输入、观测、标签、物理常数、边界/初始条件。 |
| G3: 至少两个指标可复现 | 例如温度 RMSE + 熔池 IoU，或渗透率误差 + PDE residual。 |
| G4: baseline 可跑 | 至少有 pure PINN、pure GNN 或传统回归 baseline。 |
| G5: 可写论文图 | 至少能产生一张数据流程图、一张预测对比图、一张误差/消融图。 |

