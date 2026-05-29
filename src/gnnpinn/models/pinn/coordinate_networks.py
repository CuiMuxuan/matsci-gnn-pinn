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

