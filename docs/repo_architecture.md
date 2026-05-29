# 未来代码仓库架构草案

本文档描述后续代码仓库应如何组织。当前阶段不创建这些代码文件，只把它们作为后续实现蓝图。

## 设计原则

- 科研原型优先：清晰、可改、可复现，比工程抽象完美更重要。
- 每个实验都由配置驱动，避免把论文设置写死在代码里。
- 所有模型模块都能单独训练、单独评估，再进入耦合训练。
- 方向一模块要能独立产出结果，也要能插入方向三闭环。
- 真实数据处理、仿真补充数据、模型训练、指标评估分层隔离。

## 建议目录结构

```text
GNN-PINN/
  README.md
  docs/
    data_direction_report.md
    research_route.md
    repo_architecture.md
    experiment_plan.md
    en/
      project_blueprint.md

  configs/
    data/
      ambench_2022_in625_single_track.yaml
      ambench_2022_in718_microstructure.yaml
      battery_nrel_microstructure.yaml
      digital_porous_media.yaml
      pfhub_benchmark.yaml
    model/
      macro_pinn_heat2d.yaml
      macro_pinn_thermoelastic2d.yaml
      micro_gnn_grain_graph.yaml
      closure_sparse_library.yaml
      coarse_grainer_vrh.yaml
      coupled_weak_feedback.yaml
    experiment/
      exp01_data_baseline.yaml
      exp02_macro_pinn.yaml
      exp03_sparse_closure.yaml
      exp04_gnn_conditioned_closure.yaml
      exp05_weak_coupling.yaml
      exp06_full_coupling.yaml
      exp07_ablation.yaml

  src/
    gnnpinn/
      data/
        registries.py
        schemas.py
        splits.py
        loaders/
          ambench.py
          battery.py
          porous_media.py
          pfhub.py
        transforms/
          fields.py
          microstructure.py
          graph_builder.py
          normalization.py

      physics/
        heat.py
        thermoelasticity.py
        diffusion.py
        balance_laws.py
        boundary_conditions.py
        material_priors.py

      models/
        pinn/
          macro_pinn.py
          coordinate_networks.py
          autograd_derivatives.py
        gnn/
          micro_gnn.py
          grain_graph_encoder.py
          pore_graph_encoder.py
        closure/
          sparse_library.py
          symbolic_export.py
          neural_closure.py
        coarse_grain/
          voigt_reuss_hill.py
          differentiable_homogenizer.py
          cnn_homogenizer.py
        coupled/
          weak_coupler.py
          feedback_scheduler.py
          coupled_system.py

      losses/
        pde_losses.py
        data_losses.py
        closure_losses.py
        graph_losses.py
        coupling_losses.py
        regularizers.py

      train/
        trainers.py
        curriculum.py
        optimizers.py
        checkpointing.py
      eval/
        metrics.py
        baselines.py
        uncertainty.py
        reports.py
      viz/
        fields.py
        microstructure.py
        equations.py
        paper_figures.py

  scripts/
    data/
      download_ambench.py
      prepare_ambench.py
      build_grain_graphs.py
    train/
      train_macro_pinn.py
      train_sparse_closure.py
      train_micro_gnn.py
      train_coupled.py
    eval/
      evaluate_experiment.py
      export_paper_tables.py
      export_paper_figures.py

  data/
    raw/
    interim/
    processed/
    external/

  outputs/
    runs/
    figures/
    tables/
    equations/

  tests/
    unit/
    integration/
    regression/
```

## 模块职责

### `data`

负责真实公开数据接入、字段标准化、切分和图构建。

核心对象草案：

```python
class DataCard:
    dataset_id: str
    source_url: str
    license: str
    raw_files: list[str]
    fields: dict
    metrics_supported: list[str]
```

```python
class FieldSample:
    coordinates: Tensor
    time: Tensor
    observations: dict[str, Tensor]
    boundary_conditions: dict
    process_parameters: dict
```

```python
class MicrostructureGraph:
    node_features: Tensor
    edge_index: Tensor
    edge_features: Tensor
    global_features: Tensor
    target_statistics: dict[str, Tensor]
```

### `physics`

负责已知 PDE、边界条件、物理常数和材料先验。

首批方程：

- transient heat equation
- thermoelastic equilibrium
- diffusion/transport equation
- balance laws

设计要求：

- PDE residual 只依赖张量和 autograd，不依赖具体模型类。
- 每个 residual 都能单独单元测试。
- 物理单位和归一化参数必须显式记录。

### `models.pinn`

负责宏观连续场求解。

建议接口：

```python
class MacroPINN(torch.nn.Module):
    def forward(self, coords, time, params=None):
        """Return field predictions such as temperature, displacement, stress."""
```

```python
class AutogradDerivatives:
    def grad(self, y, x): ...
    def div(self, vector_field, coords): ...
    def laplacian(self, scalar_field, coords): ...
```

### `models.gnn`

负责微观结构图编码和演化预测。

建议接口：

```python
class MicroGNNEncoder(torch.nn.Module):
    def forward(self, graph, local_state=None, history=None):
        """Return microstructure embedding or next microstructure state."""
```

输入可以是：

- 晶粒图。
- 孔隙图。
- 颗粒/相区图。
- 规则 cell 图。

### `models.closure`

方向一核心模块。负责把未知闭合项写成可训练、可稀疏化、可导出的模块。

建议接口：

```python
class SparseClosure(torch.nn.Module):
    def forward(self, features, micro_embedding=None):
        """Return closure correction term and active library coefficients."""
```

候选项库：

- 常数项。
- 多项式项。
- 梯度项。
- 历史项。
- 温度依赖项。
- GNN embedding 条件化项。

### `models.coarse_grain`

负责微观到宏观的等效参数映射。

第一版建议：

- `VoigtReussHillCoarseGrainer`：低风险解析/半解析 baseline。
- `DifferentiableHomogenizer`：可微映射，用于耦合。
- `CNNHomogenizer`：后期可用于体素组织图像。

### `models.coupled`

方向三核心模块。负责宏观 PINN 与介观 GNN 的信息传递、反馈调度和联合训练前向图。

建议接口：

```python
class CoupledGNNPINN(torch.nn.Module):
    def forward(self, macro_batch, micro_graphs, coupling_schedule):
        macro_fields = self.macro_pinn(...)
        local_state = self.sampler(macro_fields, micro_graphs)
        micro_state = self.micro_gnn(micro_graphs, local_state)
        effective_params = self.coarse_grainer(micro_state)
        updated_fields = self.macro_pinn(..., params=effective_params)
        return {
            "macro_fields": macro_fields,
            "micro_state": micro_state,
            "effective_params": effective_params,
            "updated_fields": updated_fields,
        }
```

### `losses`

损失函数必须可组合，并且每个 loss 都记录权重。

首批 loss：

- `MacroPDEResidualLoss`
- `BoundaryConditionLoss`
- `DataFitLoss`
- `SparseClosureLoss`
- `MicrostructurePredictionLoss`
- `CouplingConsistencyLoss`
- `PhysicalBoundsLoss`

### `train`

负责课程学习和模块冻结/解冻。

推荐训练策略：

```text
Stage A: train macro PINN only
Stage B: train sparse closure with macro PINN
Stage C: train micro GNN separately
Stage D: freeze GNN/coarse-grainer, train weak feedback PINN
Stage E: unfreeze selected modules for local end-to-end coupling
Stage F: full ablation and final training
```

## 未来命令约定

这些命令当前只是规划，后续代码实现时保持命名一致。

### 数据准备

```bash
python -m gnnpinn.data.prepare --config configs/data/ambench_2022_in625_single_track.yaml
python -m gnnpinn.data.audit --config configs/data/ambench_2022_in625_single_track.yaml
python -m gnnpinn.data.build_graphs --config configs/data/ambench_2022_in718_microstructure.yaml
```

### 单模块训练

```bash
python -m gnnpinn.train.macro_pinn --config configs/experiment/exp02_macro_pinn.yaml
python -m gnnpinn.train.sparse_closure --config configs/experiment/exp03_sparse_closure.yaml
python -m gnnpinn.train.micro_gnn --config configs/experiment/exp04_gnn_conditioned_closure.yaml
```

### 耦合训练

```bash
python -m gnnpinn.train.coupled --config configs/experiment/exp05_weak_coupling.yaml
python -m gnnpinn.train.coupled --config configs/experiment/exp06_full_coupling.yaml
```

### 评估与论文产物

```bash
python -m gnnpinn.eval.run --run outputs/runs/exp06_full_coupling
python -m gnnpinn.eval.export_tables --run outputs/runs/exp06_full_coupling
python -m gnnpinn.viz.paper_figures --run outputs/runs/exp06_full_coupling
python -m gnnpinn.closure.export_symbolic --run outputs/runs/exp03_sparse_closure
```

## 配置命名规范

```text
configs/data/{dataset}_{material}_{case}.yaml
configs/model/{module}_{variant}.yaml
configs/experiment/exp{number}_{short_name}.yaml
```

示例：

- `configs/data/ambench_2022_in625_single_track.yaml`
- `configs/model/macro_pinn_heat2d.yaml`
- `configs/model/closure_sparse_library.yaml`
- `configs/experiment/exp03_sparse_closure.yaml`

## 运行产物规范

每个 run 必须保存：

```text
outputs/runs/{run_id}/
  config_resolved.yaml
  metrics.json
  checkpoints/
  predictions/
  figures/
  tables/
  equations/
  logs/
  artifact_manifest.json
```

`artifact_manifest.json` 至少记录：

- 代码版本。
- 配置哈希。
- 数据版本/数据哈希。
- 随机种子。
- GPU 型号。
- 训练时长。
- 依赖版本。

## 测试策略

第一版至少需要三类测试：

| 类型 | 内容 |
|---|---|
| unit | autograd derivative、PDE residual、sparse library、graph builder。 |
| integration | 数据样本 -> 模型 forward -> loss -> backward。 |
| regression | 固定小样本和 seed 下，关键 metric 不显著漂移。 |

## 不建议第一版做的事

- 不要一开始引入过重的分布式训练框架。
- 不要先写通用科研平台。
- 不要把 AM-Bench、PFHub、电池和多孔介质全都同时实现。
- 不要把完整热-力-组织耦合作为第一个训练目标。
- 不要先追求漂亮 UI 或自动报告系统。

