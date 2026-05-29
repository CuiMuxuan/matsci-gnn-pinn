"""Autograd derivative helpers used by PINN residuals."""

from __future__ import annotations

from typing import Any


def _torch() -> Any:
    import torch

    return torch


def grad(y: Any, x: Any) -> Any:
    """Return dy/dx for batched scalar outputs.

    `y` may have shape `(n,)` or `(n, 1)`, and `x` is expected to require
    gradients. The returned tensor has the same shape as `x`.
    """

    torch = _torch()
    if y.ndim > 1 and y.shape[-1] == 1:
        y = y.squeeze(-1)
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]


def divergence(vector_field: Any, coords: Any) -> Any:
    """Compute divergence of a batched vector field with respect to coords."""

    torch = _torch()
    if vector_field.shape[-1] > coords.shape[-1]:
        raise ValueError("vector_field cannot have more components than coords")

    div = torch.zeros(coords.shape[0], device=coords.device, dtype=coords.dtype)
    for component in range(vector_field.shape[-1]):
        component_grad = grad(vector_field[:, component], coords)
        div = div + component_grad[:, component]
    return div


def laplacian(scalar_field: Any, coords: Any) -> Any:
    """Compute the scalar Laplacian of a batched field."""

    torch = _torch()
    first_grad = grad(scalar_field, coords)
    lap = torch.zeros(coords.shape[0], device=coords.device, dtype=coords.dtype)
    for dim in range(coords.shape[-1]):
        second_grad = grad(first_grad[:, dim], coords)
        lap = lap + second_grad[:, dim]
    return lap

