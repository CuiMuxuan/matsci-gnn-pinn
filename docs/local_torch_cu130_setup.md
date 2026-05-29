# 本机 `torch 2.11.0+cu130` 环境配置

## 背景

当前 base 环境中导入 `torch` 时出现：

```text
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

这通常不是项目代码的问题，而是 base 环境中多个二进制包同时加载了 OpenMP 运行时。常见来源包括 Intel MKL、NumPy/SciPy、PyTorch、scikit-learn 或其他科学计算包。Intel OpenMP 检测到同一个进程里已有 `libiomp5md.dll` 实例后，会拒绝第二次初始化，导致进程退出。

不要用 `KMP_DUPLICATE_LIB_OK=TRUE` 作为正式方案。它只是绕过检查，可能导致性能异常或静默错误。正确方案是新建干净 conda 环境。

## 约束

本机使用 conda 时：

- 允许 `conda activate`。
- 允许 `conda run`。
- 禁止直接调用 conda 环境目录下的 Python 解释器。

本项目因此统一使用：

```powershell
conda run -n gnnpinn-cu130 python ...
```

或先：

```powershell
conda activate gnnpinn-cu130
python ...
```

## 本地 torch wheel

使用用户提供的本地 wheel：

```text
C:\Users\cjh02\Downloads\torch-2.11.0+cu130-cp311-cp311-win_amd64.whl
```

该 wheel 文件名要求 Python ABI 为 `cp311`，所以环境必须固定为 Python 3.11。

## 创建环境

本机 `cu130` 环境文件故意保持很小，只用 conda 安装 Python 3.11 与 pip，避免在第一步拉取 Jupyter/Qt/scikit-image 等大包导致网络中断。

```powershell
conda env create -f environment-cu130-local.yml
```

如果环境已存在：

```powershell
conda env update -n gnnpinn-cu130 -f environment-cu130-local.yml --prune
```

## 安装项目与 torch

```powershell
conda run -n gnnpinn-cu130 python -m pip install -r requirements/base.txt
conda run -n gnnpinn-cu130 python -m pip install --no-deps -e .
conda run -n gnnpinn-cu130 python -m pip install -r requirements/torch-local-cu130.txt
```

如果 pip 提示 torch 缺少依赖，再运行：

```powershell
conda run -n gnnpinn-cu130 python -m pip install filelock fsspec jinja2 networkx setuptools sympy typing_extensions
```

`requirements/base.txt` 包含 `numpy`，这是为了避免 PyTorch 在张量/数组转换路径上提示 NumPy 不可用。

需要完整科学计算栈时，再分层安装：

```powershell
conda run -n gnnpinn-cu130 python -m pip install -r requirements/science.txt
conda run -n gnnpinn-cu130 python -m pip install -r requirements/notebooks.txt
```

## 验证

Windows PowerShell 下建议先设置 UTF-8 输出，避免 `conda run` 捕获测试输出时触发 GBK 编码错误：

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
```

```powershell
conda run -n gnnpinn-cu130 python scripts/env/check_env.py
conda run -n gnnpinn-cu130 python -m pytest -q --basetemp .pytest_tmp
conda run -n gnnpinn-cu130 python -m gnnpinn.data.audit --config configs/data/ambench_2022_single_track.yaml
```

当前本机验证结果：

```text
Python: 3.11.15
torch: 2.11.0+cu130
torch.version.cuda: 13.0
torch.cuda.is_available: True
GPU: NVIDIA GeForce RTX 5070 Laptop GPU
pytest: 6 passed
```

## 关于 PyG

`torch-geometric` 与 `pyg_lib`、`torch_scatter`、`torch_sparse`、`torch_cluster`、`torch_spline_conv` 等扩展必须与 torch 和 CUDA wheel 精确匹配。

当前本机 torch 是 `2.11.0+cu130`，若 PyG 官方 wheel 尚未提供对应的：

```text
torch-2.11.0+cu130
```

则不要强装不匹配版本。可以先做：

- Phase 0 数据审计。
- Phase 1 PINN。
- Phase 2 数据 baseline。
- Phase 3 稀疏闭合。

GNN 阶段再根据可用 wheel 选择：

1. 等待/查找 `torch-2.11.0+cu130` 对应 PyG wheel。
2. 在远程服务器使用更常见的 `torch 2.7.0 + cu118/cu126` 分支。
3. 暂时用纯 PyTorch 写一个轻量 message passing baseline，后续替换为 PyG。
