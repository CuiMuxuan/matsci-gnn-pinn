"""Macro-scale PINN interface."""

from __future__ import annotations

import math
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
        route_film_prior: float = 0.5,
        route_trainable: bool = True,
    ):
        super().__init__()
        torch = _torch()
        normalized_mode = conditioning_mode.lower()
        if normalized_mode not in {"concat", "film", "concat_film", "routed"}:
            raise ValueError(f"Unsupported conditioning mode: {conditioning_mode}")
        if normalized_mode in {"film", "concat_film", "routed"} and param_dim <= 0:
            raise ValueError(f"{normalized_mode} conditioning requires param_dim > 0")
        if normalized_mode == "routed" and not 0.0 < route_film_prior < 1.0:
            raise ValueError("route_film_prior must be between 0 and 1")
        self.coord_dim = coord_dim
        self.field_dim = field_dim
        self.param_dim = param_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.conditioning_mode = normalized_mode
        self.film_strength = float(film_strength)
        self.route_film_prior = float(route_film_prior)
        self.route_trainable = bool(route_trainable)
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

        if self.conditioning_mode == "routed":
            self.concat_expert = MLP(
                MLPConfig(
                    input_dim=coord_dim + 1 + param_dim,
                    output_dim=field_dim,
                    hidden_dim=hidden_dim,
                    num_hidden_layers=num_hidden_layers,
                    activation=activation,
                )
            )
            self.route_gate = torch.nn.Linear(param_dim, 1)
            torch.nn.init.zeros_(self.route_gate.weight)
            torch.nn.init.constant_(self.route_gate.bias, math.log(route_film_prior / (1.0 - route_film_prior)))
            if not self.route_trainable:
                for parameter in self.route_gate.parameters():
                    parameter.requires_grad_(False)

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
        if self.conditioning_mode in {"film", "concat_film", "routed"}:
            if params is None:
                raise ValueError(f"params are required when conditioning_mode={self.conditioning_mode!r}")
            concat_output = None
            if self.conditioning_mode == "routed":
                concat_output = self.concat_expert(torch.cat([coords, time, params], dim=-1))
            hidden_inputs = [coords, time]
            if self.conditioning_mode == "concat_film":
                hidden_inputs.append(params)
            hidden = torch.cat(hidden_inputs, dim=-1)
            for layer, generator in zip(self.hidden_layers, self.film_generators):
                hidden = self.activation(layer(hidden))
                gamma, beta = generator(params).chunk(2, dim=-1)
                hidden = hidden * (1.0 + self.film_strength * gamma) + self.film_strength * beta
            film_output = self.output_layer(hidden)
            if self.conditioning_mode == "routed":
                gate = self.routing_gate(params)
                return (1.0 - gate) * concat_output + gate * film_output
            return film_output
        inputs = [coords, time]
        if self.param_dim:
            if params is None:
                raise ValueError("params are required when param_dim > 0")
            inputs.append(params)
        return self.backbone(torch.cat(inputs, dim=-1))

    def routing_gate(self, params: Any) -> Any:
        """Return the FiLM-expert weight for routed conditioning."""

        if self.conditioning_mode != "routed":
            raise ValueError("routing_gate is only available for conditioning_mode='routed'")
        torch = _torch()
        return torch.sigmoid(self.route_gate(params))

    def routing_summary(self, params: Any) -> dict[str, Any] | None:
        """Summarize routed conditioning gate values for run metadata."""

        if self.conditioning_mode != "routed":
            return None
        gate = self.routing_gate(params).detach().cpu().reshape(-1)
        return {
            "enabled": True,
            "film_prior": self.route_film_prior,
            "trainable": self.route_trainable,
            "film_gate_mean": float(gate.mean()),
            "film_gate_std": float(gate.std(unbiased=False)),
            "film_gate_min": float(gate.min()),
            "film_gate_max": float(gate.max()),
        }
