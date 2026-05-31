"""Coordinate-based neural networks for macro-scale PINNs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _torch() -> Any:
    import torch

    return torch


@dataclass(frozen=True)
class MLPConfig:
    input_dim: int
    output_dim: int
    hidden_dim: int = 128
    num_hidden_layers: int = 4
    activation: str = "tanh"


def _activation(name: str) -> Any:
    torch = _torch()
    activations = {
        "tanh": torch.nn.Tanh,
        "relu": torch.nn.ReLU,
        "gelu": torch.nn.GELU,
        "silu": torch.nn.SiLU,
    }
    try:
        return activations[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc


class MLP(_torch().nn.Module):
    """Small fully connected network for coordinate-to-field maps."""

    def __init__(self, config: MLPConfig):
        super().__init__()
        torch = _torch()
        layers: list[Any] = []
        in_dim = config.input_dim
        for _ in range(config.num_hidden_layers):
            layers.append(torch.nn.Linear(in_dim, config.hidden_dim))
            layers.append(_activation(config.activation))
            in_dim = config.hidden_dim
        layers.append(torch.nn.Linear(in_dim, config.output_dim))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: Any) -> Any:
        return self.network(x)


class ResidualMLP(_torch().nn.Module):
    """MLP with residual hidden transitions after the input projection."""

    def __init__(self, config: MLPConfig, residual_scale: float = 1.0):
        super().__init__()
        if residual_scale < 0.0:
            raise ValueError("residual_scale must be non-negative")
        torch = _torch()
        self.residual_scale = float(residual_scale)
        self.activation = _activation(config.activation)
        self.input_layer = torch.nn.Linear(config.input_dim, config.hidden_dim)
        self.hidden_layers = torch.nn.ModuleList(
            torch.nn.Linear(config.hidden_dim, config.hidden_dim)
            for _ in range(max(0, config.num_hidden_layers - 1))
        )
        self.output_layer = torch.nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, x: Any) -> Any:
        hidden = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            residual = hidden
            hidden = residual + self.residual_scale * self.activation(layer(hidden))
        return self.output_layer(hidden)
