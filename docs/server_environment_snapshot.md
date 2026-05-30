# Server Environment Snapshot

Snapshot date: 2026-05-31 00:52 Asia/Shanghai

This file records the active Ubuntu server and `gnnpinn` conda environment used for the current A100 training workflow. Treat it as the working server baseline for follow-up experiments unless a newer snapshot supersedes it.

## Access and Paths

```bash
ssh -i ~/.ssh/matsci_gnnpinn_a100 -p 25820 root@223.109.239.30
cd /root/matsci-gnn-pinn
```

Key paths:

| Item | Path |
| --- | --- |
| Project | `/root/matsci-gnn-pinn` |
| Conda base | `/home/vipuser/miniconda3` |
| Active env | `/home/vipuser/miniconda3/envs/gnnpinn` |
| Python | `/home/vipuser/miniconda3/envs/gnnpinn/bin/python` |
| Data root | `/root/matsci-gnn-pinn/data` |
| Outputs root | `/root/matsci-gnn-pinn/outputs` |
| Logs root | `/root/matsci-gnn-pinn/logs` |

## Operating System

| Field | Value |
| --- | --- |
| Hostname | `ubuntu22` |
| User | `root` |
| OS | Ubuntu 22.04.3 LTS (Jammy Jellyfish) |
| Kernel | `Linux ubuntu22 6.2.0-37-generic #38~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Thu Nov 2 18:01:13 UTC 2 x86_64` |
| CPU | Intel Xeon Gold 6430 |
| CPU count | 8 logical CPUs, 4 cores, 2 threads/core |
| RAM | 15 GiB |
| Swap | 5.9 GiB |

## GPU and CUDA

| Field | Value |
| --- | --- |
| GPU | NVIDIA A100-SXM4-40GB |
| GPU memory | 40960 MiB |
| Driver | 550.127.05 |
| CUDA runtime shown by `nvidia-smi` | 12.4 |
| MIG | Disabled |
| Idle snapshot | 14 MiB used, 0% GPU util |
| `nvcc` | Not found in `PATH` |

`nvidia-smi` header at snapshot time:

```text
NVIDIA-SMI 550.127.05
Driver Version: 550.127.05
CUDA Version: 12.4
GPU 0: NVIDIA A100-SXM4-40GB
```

## Conda

Conda version:

```text
conda 24.9.2
```

Environment list:

```text
base     /home/vipuser/miniconda3
gnnpinn  /home/vipuser/miniconda3/envs/gnnpinn
```

`gnnpinn` was created from history with:

```yaml
name: gnnpinn
channels:
  - https://repo.anaconda.com/pkgs/main
  - https://repo.anaconda.com/pkgs/r
dependencies:
  - pip
  - python=3.11
prefix: /home/vipuser/miniconda3/envs/gnnpinn
```

Always run server Python commands through the explicit conda path:

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
/home/vipuser/miniconda3/bin/conda run -n gnnpinn python <command>
```

## Project State

At snapshot time:

```text
Repository: /root/matsci-gnn-pinn
Branch: main...origin/main
Commit: 4c01d3a Document rich micro feature results
```

## Environment Check

`scripts/env/check_env.py` reported:

```text
Python: 3.11.15
Executable: /home/vipuser/miniconda3/envs/gnnpinn/bin/python
Platform: Linux-6.2.0-37-generic-x86_64-with-glibc2.35

torch: 2.5.0+cu124
torch.cuda.is_available: True
torch.version.cuda: 12.4
cuda.device_count: 1
cuda.device[0]: NVIDIA A100-SXM4-40GB
torch_geometric: not installed
```

Important note: this server has a working Torch/CUDA path but does not currently have `torch-geometric` installed. Existing project GNN and graph-conditioned closure experiments use the local PyTorch implementation, so this has not blocked the current Phase 17-19 workflow.

## Key Python Packages

Installed in `gnnpinn` at snapshot time:

| Package | Version |
| --- | --- |
| Python | 3.11.15 |
| torch | 2.5.0+cu124 |
| torchvision | 0.20.0+cu124 |
| torchaudio | 2.5.0+cu124 |
| torch-geometric | not installed |
| numpy | 2.4.4 |
| pandas | 3.0.3 |
| scipy | 1.17.1 |
| scikit-learn | 1.8.0 |
| scikit-image | 0.26.0 |
| imagecodecs | 2026.3.6 |
| h5py | 3.16.0 |
| PyYAML | 6.0.3 |
| pytest | 9.0.3 |
| sympy | 1.13.1 |
| matplotlib | not installed |
| pillow | 12.2.0 |
| tifffile | 2026.3.3 |
| xarray | 2026.4.0 |
| zarr | 3.1.6 |

## Common Commands

Check environment:

```bash
cd /root/matsci-gnn-pinn
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
/home/vipuser/miniconda3/bin/conda run -n gnnpinn python scripts/env/check_env.py
```

Run targeted tests:

```bash
cd /root/matsci-gnn-pinn
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
/home/vipuser/miniconda3/bin/conda run -n gnnpinn python -m pytest -q --basetemp .pytest_tmp
```

Run a background experiment:

```bash
cd /root/matsci-gnn-pinn
mkdir -p logs
tmux new-session -d -s <session_name> "<command> > logs/<run_id>.log 2>&1"
tmux ls
tail -f logs/<run_id>.log
```

Freeze a run environment:

```bash
RUN_DIR=outputs/runs/<run_id>
/home/vipuser/miniconda3/bin/conda env export -n gnnpinn --from-history > "$RUN_DIR/conda_from_history.yml"
/home/vipuser/miniconda3/bin/conda run -n gnnpinn python -m pip freeze > "$RUN_DIR/pip_freeze.txt"
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
/home/vipuser/miniconda3/bin/conda run -n gnnpinn python scripts/env/check_env.py > "$RUN_DIR/env_report.txt"
nvidia-smi > "$RUN_DIR/nvidia_smi.txt"
```

## Regenerate This Snapshot

Use this compact command pattern when the server environment changes:

```bash
cd /root/matsci-gnn-pinn
date -Is
cat /etc/os-release
uname -a
lscpu | sed -n '1,24p'
free -h
nvidia-smi
command -v nvcc >/dev/null 2>&1 && nvcc --version || echo "nvcc not found in PATH"
/home/vipuser/miniconda3/bin/conda --version
/home/vipuser/miniconda3/bin/conda env list
/home/vipuser/miniconda3/bin/conda env export -n gnnpinn --from-history
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
/home/vipuser/miniconda3/bin/conda run -n gnnpinn python scripts/env/check_env.py
/home/vipuser/miniconda3/bin/conda run -n gnnpinn python -m pip list --format=freeze
git status --short --branch
git log -1 --oneline
```
