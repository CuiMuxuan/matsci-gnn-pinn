"""Macro-scale PINN interface."""

from __future__ import annotations

from typing import Any

from .coordinate_networks import MLP, MLPConfig


def _torch() -> Any:
    import torch

    return torch


class MacroPINN(_torch().nn.Module):
    """Coordinate-based macro field model.

    The first implementation is intentionally minimal: concatenate spatial
    coordinates, time, and optional process/material parameters, then predict
    requested macro fields such as temperature or displacement.
    """

    def __init__(
        self,
        coord_dim: int,
        field_dim: int,
        param_dim: int = 0,
        hidden_dim: int = 128,
        num_hidden_layers: int = 4,
        activation: str = "tanh",
    ):
        super().__init__()
        self.coord_dim = coord_dim
        self.field_dim = field_dim
        self.param_dim = param_dim
        self.backbone = MLP(
            MLPConfig(
                input_dim=coord_dim + 1 + param_dim,
                output_dim=field_dim,
                hidden_dim=hidden_dim,
                num_hidden_layers=num_hidden_layers,
                activation=activation,
            )
        )

    def forward(self, coords: Any, time: Any, params: Any | None = None) -> Any:
        torch = _torch()
        if time.ndim == 1:
            time = time[:, None]
        inputs = [coords, time]
        if self.param_dim:
            if params is None:
                raise ValueError("params are required when param_dim > 0")
            inputs.append(params)
        return self.backbone(torch.cat(inputs, dim=-1))

