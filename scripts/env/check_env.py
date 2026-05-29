"""Print a compact environment report for local and remote reproducibility."""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import subprocess
import sys


PACKAGES = [
    "yaml",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "sympy",
    "h5py",
    "zarr",
    "xarray",
    "torch",
    "torch_geometric",
]


DIST_NAMES = {
    "yaml": "PyYAML",
    "sklearn": "scikit-learn",
}


def package_version(import_name: str) -> str:
    dist_name = DIST_NAMES.get(import_name, import_name)
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def module_available(import_name: str) -> bool:
    try:
        importlib.import_module(import_name)
    except Exception:
        return False
    return True


def torch_report() -> str:
    code = """
import torch
print(f"torch.__version__: {torch.__version__}")
print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"cuda.device_count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        print(f"cuda.device[{idx}]: {torch.cuda.get_device_name(idx)}")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    stderr = result.stderr.strip() or result.stdout.strip()
    return f"torch import failed with exit code {result.returncode}\n{stderr}"


def main() -> int:
    print("GNN-PINN environment report")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print("")
    print("Packages:")
    for name in PACKAGES:
        print(f"  {name}: {package_version(name)}")

    print("")
    print("Torch/CUDA:")
    for line in torch_report().splitlines():
        print(f"  {line}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
