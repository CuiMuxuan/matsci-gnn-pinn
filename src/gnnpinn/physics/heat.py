"""Heat-equation residuals for AM-Bench starter experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from gnnpinn.models.pinn.autograd_derivatives import grad, laplacian


@dataclass(frozen=True)
class HeatEquationParams:
    """Constant-property heat equation parameters.

    Residual convention:

    `rho_cp * dT_dt - conductivity * laplacian(T) - source = 0`
    """

    rho_cp: float = 1.0
    conductivity: float = 1.0


def transient_heat_residual(
    temperature: Any,
    coords: Any,
    time: Any,
    params: HeatEquationParams | None = None,
    source: float | Any | Callable[[Any, Any], Any] = 0.0,
) -> Any:
    """Compute transient heat-equation residual for scalar temperature."""

    params = params or HeatEquationParams()
    spatial_laplacian = laplacian(temperature, coords)
    temporal_grad = grad(temperature, time)
    d_t = temporal_grad[:, 0] if temporal_grad.ndim > 1 else temporal_grad

    if callable(source):
        source_value = source(coords, time)
    else:
        source_value = source
    return params.rho_cp * d_t - params.conductivity * spatial_laplacian - source_value

