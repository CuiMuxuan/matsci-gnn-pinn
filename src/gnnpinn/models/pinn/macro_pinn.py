"""Macro-scale PINN interface."""

from __future__ import annotations

from typing import Any

from .coordinate_networks import MLP, MLPConfig, _activation


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
        conditioning_mode: str = "concat",
        film_strength: float = 1.0,
    ):
        super().__init__()
        torch = _torch()
        normalized_mode = conditioning_mode.lower()
        if normalized_mode not in {"concat", "film", "concat_film"}:
            raise ValueError(f"Unsupported conditioning mode: {conditioning_mode}")
        if normalized_mode in {"film", "concat_film"} and param_dim <= 0:
            raise ValueError(f"{normalized_mode} conditioning requires param_dim > 0")
        self.coord_dim = coord_dim
        self.field_dim = field_dim
        self.param_dim = param_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.conditioning_mode = normalized_mode
        self.film_strength = float(film_strength)
        if self.conditioning_mode == "concat":
            self.backbone = MLP(
                MLPConfig(
                    input_dim=coord_dim + 1 + param_dim,
                    output_dim=field_dim,
                    hidden_dim=hidden_dim,
                    num_hidden_layers=num_hidden_layers,
                    activation=activation,
                )
            )
            return

        self.activation = _activation(activation)
        self.hidden_layers = torch.nn.ModuleList()
        self.film_generators = torch.nn.ModuleList()
        in_dim = coord_dim + 1
        if self.conditioning_mode == "concat_film":
            in_dim += param_dim
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(torch.nn.Linear(in_dim, hidden_dim))
            generator = torch.nn.Linear(param_dim, 2 * hidden_dim)
            torch.nn.init.zeros_(generator.weight)
            torch.nn.init.zeros_(generator.bias)
            self.film_generators.append(generator)
            in_dim = hidden_dim
        self.output_layer = torch.nn.Linear(in_dim, field_dim)

    def forward(self, coords: Any, time: Any, params: Any | None = None) -> Any:
        torch = _torch()
        if time.ndim == 1:
            time = time[:, None]
        if self.conditioning_mode in {"film", "concat_film"}:
            if params is None:
                raise ValueError(f"params are required when conditioning_mode={self.conditioning_mode!r}")
            hidden_inputs = [coords, time]
            if self.conditioning_mode == "concat_film":
                hidden_inputs.append(params)
            hidden = torch.cat(hidden_inputs, dim=-1)
            for layer, generator in zip(self.hidden_layers, self.film_generators):
                hidden = self.activation(layer(hidden))
                gamma, beta = generator(params).chunk(2, dim=-1)
                hidden = hidden * (1.0 + self.film_strength * gamma) + self.film_strength * beta
            return self.output_layer(hidden)
        inputs = [coords, time]
        if self.param_dim:
            if params is None:
                raise ValueError("params are required when param_dim > 0")
            inputs.append(params)
        return self.backbone(torch.cat(inputs, dim=-1))
