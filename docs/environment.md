# Python 依赖环境与迁移方案

## 目标

本项目需要同时满足两件事：

1. 本机可用 `conda` 管理环境，快速运行数据审计、测试、Notebook 和后续轻量实验。
2. 后续迁移到 A800/V100/A100 云 GPU 时，CUDA、PyTorch、PyG 的版本组合可控，不因为服务器镜像差异而反复踩坑。

因此本项目采用两层环境策略：

```text
conda 管通用科学计算栈
pip 管 PyTorch / PyG 等 CUDA wheel
```

这样做的原因是：PyTorch 官方当前稳定安装页提供 pip CUDA wheel；PyG 官方文档也说明 PyTorch 2.5 之后 PyG 不再提供 conda 包，CUDA 相关扩展需要按 PyTorch 与 CUDA wheel 版本匹配安装。

## 推荐版本

| 组件 | 推荐 |
|---|---|
| Python | 3.11 |
| PyTorch | 2.7.0 |
| torchvision | 0.22.0 |
| torchaudio | 2.7.0 |
| PyG | 2.6+ |
| 默认 GPU wheel | CUDA 11.8 |
| 高版本 GPU wheel | CUDA 12.6 |

### 为什么默认推荐 CUDA 11.8

你计划使用的云 GPU 包括 V100-32GB、A800-SXM4-40GB、A100-80GB。CUDA 11.8 wheel 对 V100/A800/A100 的兼容面更宽，适合作为第一套稳定科研环境。

如果云服务器驱动较新，且镜像明确支持 CUDA 12.6，可以使用 `cu126` 分支获得更贴近当前 PyTorch 官方稳定页的组合。

### 关于本机默认 Python 3.13

当前机器默认 Python 检查结果为 3.13.9，但本项目不建议直接使用 base Python 做科研实验环境。原因是深度学习、图神经网络和科学计算包在 Python 3.13 上的组合更容易遇到二进制 wheel、OpenMP、CUDA 扩展兼容问题。

推荐始终使用：

```text
conda env create -f environment.yml
conda activate gnnpinn
```

也就是用独立的 Python 3.11 环境运行项目。

## 本地/CPU 环境

适合：

- 数据审计。
- 单元测试。
- 文档与 Notebook。
- 小规模无 GPU 实验。

创建环境：

```powershell
conda env create -f environment.yml
conda activate gnnpinn
python -m pip install --no-deps -e .
```

如需 CPU 版 PyTorch 和 PyG：

```powershell
python -m pip install -r requirements/torch-cpu.txt
python -m pip install -r requirements/pyg-torch27-cpu.txt
```

验证：

```powershell
python scripts/env/check_env.py
python -m pytest --basetemp .pytest_tmp
python -m gnnpinn.data.audit --config configs/data/ambench_2022_single_track.yaml
```

## 云 GPU 环境：CUDA 11.8 稳定分支

推荐用于：

- V100-32GB。
- A800-SXM4-40GB。
- A100-80GB，但服务器驱动不确定时。

Linux 服务器命令：

```bash
conda env create -f environment.yml
conda activate gnnpinn
python -m pip install --no-deps -e .
python -m pip install -r requirements/torch-cu118.txt
python -m pip install -r requirements/pyg-torch27-cu118.txt
python -m pip install -r requirements/research-extra.txt
python scripts/env/check_env.py
python -m pytest --basetemp .pytest_tmp
```

## 云 GPU 环境：CUDA 12.6 分支

推荐用于：

- 驱动较新的 A800/A100 云镜像。
- 需要贴近当前 PyTorch 官方稳定安装页时。

Linux 服务器命令：

```bash
conda env create -f environment.yml
conda activate gnnpinn
python -m pip install --no-deps -e .
python -m pip install -r requirements/torch-cu126.txt
python -m pip install -r requirements/pyg-torch27-cu126.txt
python -m pip install -r requirements/research-extra.txt
python scripts/env/check_env.py
python -m pytest --basetemp .pytest_tmp
```

## Windows 本机开发

PowerShell:

```powershell
conda env create -f environment.yml
conda activate gnnpinn
python -m pip install --no-deps -e .
python scripts/env/check_env.py
python -m pytest --basetemp .pytest_tmp
```

如果使用 `conda run`，建议设置 UTF-8 输出，避免 Windows GBK 编码导致 conda wrapper 在打印测试输出时崩溃：

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
conda run -n gnnpinn-cu130 python -m pytest -q --basetemp .pytest_tmp
```

如果只做 Phase 0 数据审计，不需要安装 PyTorch/PyG。

## 本机 CUDA 13.0 torch wheel 环境

当前机器可用的本地 wheel：

```text
C:\Users\cjh02\Downloads\torch-2.11.0+cu130-cp311-cp311-win_amd64.whl
```

该 wheel 需要 Python 3.11，因此本项目提供单独环境文件：

```text
environment-cu130-local.yml
requirements/torch-local-cu130.txt
```

详细步骤见：

```text
docs/local_torch_cu130_setup.md
```

注意：本机使用 conda 时禁止直接调用 conda 环境目录下的 Python 解释器。请使用 `conda activate gnnpinn-cu130` 后运行 `python ...`，或使用 `conda run -n gnnpinn-cu130 python ...`。

本机 `cu130` 环境采用最小 conda 环境：

```text
environment-cu130-local.yml: python=3.11 + pip
requirements/base.txt: 项目最小运行与测试依赖
requirements/science.txt: 后续科学计算依赖
requirements/notebooks.txt: Notebook/绘图依赖
```

这样可以先验证 torch wheel 与项目测试，再按需补装大包。

## 迁移到远程服务器的流程

1. 在本机提交或打包代码。
2. 远程服务器安装 Miniconda/Mambaforge。
3. 拉取仓库。
4. 根据 GPU/驱动选择 `cu118` 或 `cu126`。
5. 创建 conda 环境。
6. 安装本项目 editable 包。
7. 安装 PyTorch/PyG 对应 wheel。
8. 运行 `scripts/env/check_env.py`。
9. 运行 `python -m pytest --basetemp .pytest_tmp`。
10. 运行 AM-Bench 数据审计命令。

进入 dense AM-Bench 实验时，继续阅读：

```text
docs/server_runbook.md
```

## 版本冻结策略

推荐三层冻结：

| 层级 | 文件 | 用途 |
|---|---|---|
| 人类维护 | `environment.yml`, `requirements/*.txt` | 跨机器重建环境。 |
| 实验记录 | `outputs/runs/*/artifact_manifest.json` | 记录每次实验依赖版本。 |
| 服务器冻结 | `conda env export --from-history` 和 `pip freeze` | 论文最终实验归档。 |

最终论文实验完成后，在对应 run 目录保存：

```bash
conda env export --from-history > outputs/runs/{run_id}/conda_from_history.yml
python -m pip freeze > outputs/runs/{run_id}/pip_freeze.txt
python scripts/env/check_env.py > outputs/runs/{run_id}/env_report.txt
```

## 常见问题

### `check_env.py` 报 OpenMP/libiomp5 冲突怎么办？

这通常说明当前解释器环境里混入了多个 OpenMP 运行时，常见于 base conda 环境安装过多科学计算包。不要在论文实验里用这个 base 环境。创建独立的 `gnnpinn` 环境后再运行：

```bash
conda activate gnnpinn
python scripts/env/check_env.py
```

脚本会用子进程探测 `torch`，因此即使当前 base 环境导入 `torch` 失败，也能保留完整环境诊断输出。

### `conda run` 报 UnicodeEncodeError/GBK 怎么办？

这是 Windows 下 conda wrapper 打印子进程输出时的编码问题。项目测试本身可能已经完成，但 conda 在回显输出时崩溃。使用：

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
conda run -n gnnpinn-cu130 python -m pytest -q --basetemp .pytest_tmp
```

### 是否把 PyTorch 写进 `environment.yml`？

不建议。PyTorch/PyG 与 CUDA wheel 强绑定，不同云服务器驱动环境差异较大。把它们拆到 `requirements/torch-*.txt` 和 `requirements/pyg-*.txt` 更容易迁移。

### V100 应该用哪个分支？

优先 `cu118`。如果云镜像驱动很新，也可尝试 `cu126`，但第一版科研原型不需要冒这个风险。

### A800/A100 应该用哪个分支？

默认 `cu118` 稳定优先。最终重型实验时，如果服务器镜像确认支持 CUDA 12.6，再切到 `cu126`。

### 是否需要 Docker？

第一阶段不需要。等项目进入大规模训练和论文最终复现时，可以再补 `Dockerfile` 或基于 NVIDIA PyTorch/PyG 容器的方案。

## 官方参考

- PyTorch Get Started: https://pytorch.org/get-started/locally/
- PyG Installation: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
- Conda environment.yml specification: https://conda.org/learn/specifications/exchange/environment-yml
- Conda environment management: https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html
