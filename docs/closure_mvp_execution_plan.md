# Closure MVP 执行方案

## 目标

本阶段把项目从“可靠的 data-only Macro PINN 基线”推进到“可学习热源/闭合项”的方向一入口。核心思路不是继续强行使用当前常数物性 PDE residual，而是实现一个最小可复现的 sparse source closure：

```text
scaled heat residual = transient/diffusion terms - q_closure(features)
```

其中 `q_closure` 由稀疏线性候选库学习，先服务于可复现实验和论文消融，再逐步扩展到更物理的热源尺度、材料参数和 GNN-conditioned closure。

## 已完成内容

### 数据与环境

- GitHub、本地、服务器三端已经同步到 `d6c43c5`。
- 远程服务器：Ubuntu 22, NVIDIA A100-SXM4-40GB, CUDA 12.4。
- Conda 环境：`gnnpinn`, Python 3.11, PyTorch 2.5.0+cu124。
- AM-Bench 2022 / AMB2022-03 / `mds2-2716` 已下载并 SHA256 校验通过。
- 已建立 frame-based split，避免随机点划分引入时序泄漏。

### A 阶段：数据表示与采样

- 已实现 calibrated `temperature_C` HDF5 转换。
- 已实现 `--input-normalization none|minmax|standard`。
- 已实现 mean、kNN、RandomForest、ExtraTrees baselines。
- 已实现 hot q90 与 gradient q90 区域指标。
- 已实现 HDF5 active sampling：
  - `uniform`
  - `hot`
  - `gradient`
  - `hot_gradient`
  - `balanced_hot_gradient`

关键结果：

| Result | Value |
| --- | ---: |
| uniform dense valid points | 41054 |
| active hot/gradient points | 14518 |
| uniform dense minmax Macro PINN test RMSE | 51.655371 |
| active hot/gradient Macro PINN hot q90 RMSE | 30.868055 |

### B 阶段：Macro PINN 稳健基线

已完成小型 screen matrix 与 3 seed check。当前固定 data-only 基线如下：

| Use | Dataset | Configuration | Key result |
| --- | --- | --- | ---: |
| Global reconstruction | uniform dense | `hidden_dim=128`, `layers=4`, `lr=1e-3`, `input_normalization=minmax` | test RMSE `61.222230 +/- 3.409577` |
| Hot-zone focused model | active hot/gradient | `hidden_dim=256`, `layers=4`, `lr=1e-3`, `input_normalization=minmax` | hot q90 RMSE `17.033393 +/- 2.084327` |

结论：data-only Macro PINN 已经足够稳定，可以作为 closure/PDE 分支的固定对照。

## 为什么下一步不继续旧 PDE scan

旧 PDE scan 使用常数 `conductivity=1.0`、`rho_cp=1.0`，并在物理温度尺度上直接计算 residual。已有结果显示任意正 `pde_weight` 都会显著恶化 minmax data-only 模型。继续扩大旧 scan 只会消耗 GPU，不能形成更强论文证据。

因此下一步改为：

1. 在 normalized/scaled field 上计算 residual，先控制数值尺度。
2. 学习 source/closure 项，而不是固定常数 source。
3. 用 L1 penalty 约束 closure 系数，形成可解释表达式。
4. 只与已经固定的 data-only 基线比较。

## C 阶段：Sparse Source Closure MVP

### C1. 代码功能

需要新增能力：

- `--pde-field physical|normalized`
- `--closure-mode none|sparse_linear`
- `--closure-feature <name>`，首轮使用 `T,x,y,t`
- `--closure-polynomial-order`
- `--closure-l1-weight`
- `--closure-threshold`
- metrics 中保存：
  - closure term names
  - coefficients
  - symbolic expression
  - source mean/std
  - pde loss
  - closure L1 loss

首轮 closure 特征：

```text
T, x, y, t
```

暂不默认加入 `grad_norm` 和 `laplacian`，避免一开始引入更高阶导数带来的成本和不稳定。后续如第一版 source closure 稳定，再扩展导数特征。

### C2. 首轮服务器实验

固定 data-only 最优配置，扫描较小 closure 权重：

| Dataset | Hidden | Layers | LR | PDE field | Closure | PDE weights |
| --- | ---: | ---: | ---: | --- | --- | --- |
| uniform dense | 128 | 4 | 1e-3 | normalized | sparse linear | `1e-4`, `1e-3` |
| active hot/gradient | 256 | 4 | 1e-3 | normalized | sparse linear | `1e-4`, `1e-3` |

固定：

```text
steps=2000
input_normalization=minmax
closure_features=T,x,y,t
closure_polynomial_order=1
closure_l1_weight=1e-5
hot_quantile=0.9
gradient_quantile=0.9
```

### C3. 验收条件

进入更大 closure 实验前必须满足：

- 训练不发散，metrics 与 checkpoint 正常输出。
- closure coefficients 与 symbolic expression 被保存。
- 至少一个 closure run 在 hot q90 或 gradient q90 上不劣于对应 data-only 基线。
- 如果 global RMSE 变差，需要能解释是 residual 权重、closure 表达能力还是采样目标导致。

### C4. 若首轮失败

按以下顺序修正，不直接进入 GNN/coupling：

1. 降低 `pde_weight` 到 `1e-5` 或 `1e-6`。
2. 保持 `pde_field=normalized`，调高/调低 `closure_l1_weight`。
3. 增加 `grad_norm` 特征，但只在 active 表上试跑。
4. 引入 mini-batch/collocation sampling，避免全量 residual 训练成本过高。

## 后续路线

如果 sparse source closure MVP 有正结果：

1. 跑 3 seed。
2. 增加 polynomial order 2。
3. 对比 black-box neural closure。
4. 加入导数特征。
5. 将 closure features 接入 micro/GNN embedding，进入 GNN-conditioned closure。

如果 sparse source closure MVP 没有正结果：

1. 写负结果分析。
2. 优先实现 residual/collocation sampling 与尺度化热源。
3. 暂缓方向三弱双向耦合。

## 当前推进：Optimizer Separation

staged closure 的最新结果显示，`closure_start_step=1500` 能显著缓解过早 residual 约束带来的退化，但 closure 仍然没有超过 data-only。当前判断是：问题不主要来自 sparse library 表达能力不足，而来自 closure coefficients 与 Macro PINN backbone 同步优化时相互牵扯。

因此下一步已经落地为 C1e：

- `--closure-lr`：closure coefficients 使用独立学习率。
- `--freeze-backbone-after-closure-start`：warmup 后冻结 Macro PINN backbone，只让 closure coefficients 解释 residual。
- `scripts/server/run_sparse_closure_optimizer_ablation_a100.sh`：A100 上固定 active 表、`closure_start_step=1500`、`residual_sample_size=4096`、`pde_weight=1e-6` 的四组消融。

推荐服务器执行：

```bash
bash scripts/server/run_sparse_closure_optimizer_ablation_a100.sh \
  > logs/ambench_sparse_closure_optimizer_ablation_a100_v1.log 2>&1
```

判据：

- 若低 `closure_lr` 或冻结 backbone 能把 hot q90 拉回接近 data-only，则继续做 3 seed 和 closure 稳定性分析。
- 若仍然大幅弱于 data-only，则保留 C1 作为可复现负结果，转入 GNN-conditioned closure 接口与 synthetic/toy graph 耦合，以服务方向三主线。

## Optimizer Ablation 结论

已完成 A100 四组 optimizer ablation。最佳组合为：

```text
closure_lr=1e-5
freeze_backbone_after_closure_start=false
closure_start_step=1500
residual_sample_size=4096
```

关键指标：

| Method | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE |
| --- | ---: | ---: | ---: |
| staged closure, start 1500, 4096 | 71.341922 | 56.310533 | 73.296876 |
| optimizer ablation best | 70.494433 | 31.542155 | 64.558069 |
| active data-only 3 seed mean | 60.143582 | 17.033393 | 63.622568 |

解释：

- 低 closure lr 明显缓解了 closure coefficients 过快增长的问题。
- 冻结 backbone 不如继续共同微调，说明当前 closure 仍需要场网络小幅适应。
- C1 已经从“不可用”推进到“可解释但未胜出”的技术基线。

下一步不扩大 polynomial order，先进入 GNN-conditioned closure 接口，检验方向三是否能在同一真实数据基线上提供额外收益。

## D1 接口落地

已将方向三的最小接口接入 closure 训练链路：

- `ToyStaticGraphEmbeddingProvider`：使用现有 `MicroGNNEncoder` 编码 deterministic toy graph。
- `--closure-graph-mode toy_static`：在 sparse closure 中自动加入 `g0/g1/...` 特征。
- metrics/checkpoint 保存 graph conditioning metadata，便于论文实验复现和后续替换真实微观组织 graph。

首轮服务器对照仍固定 C1 最佳设置：

```bash
bash scripts/server/run_graph_conditioned_closure_toy_a100.sh \
  > logs/ambench_graph_conditioned_closure_toy_a100_v1.log 2>&1
```

这一轮的目标不是立即声称真实组织耦合有效，而是验证方向三的训练接口、artifact、对照指标和论文实验骨架已经可运行。

服务器结果：

| Method | Test RMSE | Hot q90 RMSE | Gradient q90 RMSE |
| --- | ---: | ---: | ---: |
| sparse closure best | 70.494433 | 31.542155 | 64.558069 |
| toy/static graph-conditioned closure | 70.057147 | 76.367170 | 87.591751 |

结论：方向三接口已跑通，但 static global embedding 基本等价于额外常量项，无法改善局部热区和梯度带。下一步应做 per-point / region-aware graph conditioning，例如基于归一化 `x,y,t` 的 anchor/RBF graph features。
