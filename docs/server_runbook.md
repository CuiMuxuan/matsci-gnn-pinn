# 服务器阶段运行手册

## 何时需要租服务器

本机已经完成：

- AM-Bench `mds2-2716` 真实数据下载与 SHA256 校验。
- HDF5 thermography 读取。
- raw `signal` 字段表转换。
- `temperature_C` 校准转换。
- frame-based split。
- 1044 点 calibrated temperature probe。
- baseline 与 Macro PINN smoke。

下一步如果继续向论文主结果推进，建议租云 GPU，而不是继续在本机试探。触发条件：

- 采样点扩大到数万至数十万。
- 启用 PDE residual，需要对大量坐标做 autograd 求导。
- 跑多 seed、多 frame split 或多 line/pad 工况。
- 加 sparse closure、GNN-conditioned closure 或 coupled GNN-PINN。

## 推荐机器

优先：

```text
A800-SXM4-40GB
```

适合第一轮 dense sampling、Macro PINN + PDE residual、小规模多 seed。

更稳：

```text
A100 80GB / A100-SXM4-80GB
```

适合 dense sampling + PDE residual + 多 seed + 后续 GNN/coupled 训练。

V100-32GB 可以作为便宜试跑，但不建议作为最终主实验机器。

## 环境准备

Linux 服务器推荐：

```bash
git clone <your-repo-url> GNN-PINN
cd GNN-PINN

conda env create -f environment.yml
conda activate gnnpinn

python -m pip install --no-deps -e .
python -m pip install -r requirements/base.txt
python -m pip install -r requirements/science.txt
python -m pip install -r requirements/torch-cu118.txt
python -m pip install -r requirements/pyg-torch27-cu118.txt

python scripts/env/check_env.py
python -m pytest -q --basetemp .pytest_tmp
```

如果云镜像明确支持 CUDA 12.6，可把 `cu118` 替换为 `cu126`：

```bash
python -m pip install -r requirements/torch-cu126.txt
python -m pip install -r requirements/pyg-torch27-cu126.txt
```


## 后续推进总方案

第一批 A100 dense 实验完成后，后续服务器研发不再只按本手册的初始 smoke 流程推进，而应按 [服务器推进完整执行方案](server_execution_plan.md) 执行。该方案包含当前结果基线、三端同步规则、归一化与强 baseline、采样改进、Macro PINN 调参、sparse closure、GNN 条件化闭合、弱双向耦合和论文资产生成的阶段门槛。

## 下载 AM-Bench 数据

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 conda run -n gnnpinn python -m gnnpinn.data.ambench_downloads \
  --root data/raw/ambench/2022_single_track/AMB2022-03/mds2-2716 \
  --download \
  --verify-sha256 \
  --output outputs/data_audits/ambench_mds2_2716_download_report.json
```

只校验：

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 conda run -n gnnpinn python -m gnnpinn.data.ambench_downloads \
  --root data/raw/ambench/2022_single_track/AMB2022-03/mds2-2716 \
  --verify-sha256 \
  --output outputs/data_audits/ambench_mds2_2716_download_check.json
```

## 第一批服务器实验

目标：把本机 1044 点 probe 扩展为 dense calibrated temperature experiment。

### 1. 生成 dense calibrated subset

先用保守 dense 配置：

```bash
python -m gnnpinn.data.loaders.ambench_hdf5 \
  --sample-id amb2022_03_line_0_1_temperature_dense_a800_v1 \
  --output data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_dense_a800_v1.csv \
  --manifest outputs/data_audits/ambench_line_0_1_temperature_dense_a800_v1_manifest.json \
  --split-manifest outputs/data_splits/ambench_line_0_1_temperature_dense_a800_v1_split.json \
  --calibrate-temperature \
  --split-strategy frame \
  --min-signal 100 \
  --frame-step 5 \
  --max-frames 120 \
  --row-step 2 \
  --max-rows 320 \
  --col-step 2 \
  --max-cols 152
```

如果 CSV 超过可接受大小或训练太慢，把 `row-step` 和 `col-step` 调回 `4`。如果显存和时间充足，再把 `frame-step` 降为 `2` 或 `1`。

### 2. baseline

```bash
python -m gnnpinn.eval.field_baseline \
  --table data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_dense_a800_v1.csv \
  --target temperature_C \
  --strategy mean \
  --split-manifest outputs/data_splits/ambench_line_0_1_temperature_dense_a800_v1_split.json \
  --output outputs/baselines/ambench_line_0_1_temperature_dense_a800_v1_mean_baseline.json
```

### 3. Macro PINN data-only dense run

```bash
python -m gnnpinn.train.macro_pinn \
  --table data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_dense_a800_v1.csv \
  --target temperature_C \
  --split-manifest outputs/data_splits/ambench_line_0_1_temperature_dense_a800_v1_split.json \
  --output-dir outputs/runs/ambench_line_0_1_temperature_dense_macro_pinn_data_only_v1 \
  --steps 2000 \
  --hidden-dim 128 \
  --layers 4 \
  --device cuda \
  --log-every 100
```

### 4. Macro PINN PDE residual run

第一版 PDE residual 只是代码/性能验证，不应直接作为最终物理结论：

```bash
python -m gnnpinn.train.macro_pinn \
  --table data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_dense_a800_v1.csv \
  --target temperature_C \
  --split-manifest outputs/data_splits/ambench_line_0_1_temperature_dense_a800_v1_split.json \
  --output-dir outputs/runs/ambench_line_0_1_temperature_dense_macro_pinn_pde_v1 \
  --steps 2000 \
  --hidden-dim 128 \
  --layers 4 \
  --device cuda \
  --pde-weight 1e-6 \
  --conductivity 1.0 \
  --rho-cp 1.0 \
  --log-every 100
```

## 服务器实验记录要求

每个 run 完成后保存：

```bash
conda env export --from-history > outputs/runs/<run_id>/conda_from_history.yml
python -m pip freeze > outputs/runs/<run_id>/pip_freeze.txt
python scripts/env/check_env.py > outputs/runs/<run_id>/env_report.txt
nvidia-smi > outputs/runs/<run_id>/nvidia_smi.txt
```

## 停止条件

如果出现以下情况，应先回到代码侧优化，而不是继续烧 GPU：

- dense CSV 生成后训练仍然只是在 train split 下降，val/test 明显恶化。
- PDE residual 权重稍增就数值发散。
- 训练吞吐低到无法完成多 seed。
- 温度场采样点分布仍然高度稀疏或偏置，无法支撑论文结论。

## 服务器阶段完成标志

第一批服务器实验完成后，应至少得到：

- dense calibrated dataset manifest。
- dense frame split manifest。
- dense mean baseline 指标。
- dense Macro PINN data-only 指标。
- dense Macro PINN PDE 指标。
- 每个 run 的环境冻结文件。
- 一份新的 `docs/results/ambench_dense_temperature_server_v1.md`。

