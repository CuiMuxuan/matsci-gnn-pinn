from __future__ import annotations

import subprocess
import sys

import pytest


def _torch_available() -> bool:
    result = subprocess.run(
        [sys.executable, "-c", "import torch"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


pytestmark = pytest.mark.skipif(
    not _torch_available(),
    reason="torch is not importable in the current environment; create the gnnpinn conda env to run PINN tests",
)


def test_laplacian_for_quadratic_field():
    import torch

    from gnnpinn.models.pinn.autograd_derivatives import laplacian

    coords = torch.tensor([[1.0, 2.0], [3.0, -1.0]], requires_grad=True)
    field = coords[:, 0] ** 2 + coords[:, 1] ** 2

    assert torch.allclose(laplacian(field, coords), torch.full((2,), 4.0))


def test_transient_heat_residual_for_known_solution():
    import torch

    from gnnpinn.physics.heat import HeatEquationParams, transient_heat_residual

    coords = torch.tensor([[1.0, 2.0], [3.0, -1.0]], requires_grad=True)
    time = torch.tensor([[0.5], [1.5]], requires_grad=True)
    temperature = coords[:, 0] ** 2 + coords[:, 1] ** 2 + 3.0 * time[:, 0]

    residual = transient_heat_residual(
        temperature,
        coords,
        time,
        params=HeatEquationParams(rho_cp=2.0, conductivity=0.5),
        source=4.0,
    )

    assert torch.allclose(residual, torch.zeros_like(residual))

