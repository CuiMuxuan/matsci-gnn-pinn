"""Minimal PyTorch microstructure graph encoder.

This module intentionally avoids a hard PyG dependency. It keeps the same
`node_features`/`edge_index` conventions so a PyG implementation can replace
it later without changing the coupling interface.
"""

from __future__ import annotations

from typing import Any


def _torch() -> Any:
    import torch

    return torch


class MicroGNNEncoder(_torch().nn.Module):
    """Small message-passing encoder for microstructure graphs."""

    def __init__(self, node_dim: int, hidden_dim: int = 64, output_dim: int = 32, steps: int = 2):
        super().__init__()
        torch = _torch()
        self.steps = steps
        self.node_in = torch.nn.Linear(node_dim, hidden_dim)
        self.message = torch.nn.Linear(hidden_dim, hidden_dim)
        self.update = torch.nn.GRUCell(hidden_dim, hidden_dim)
        self.readout = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, node_features: Any, edge_index: Any) -> Any:
        torch = _torch()
        hidden = torch.tanh(self.node_in(node_features))
        for _ in range(self.steps):
            src, dst = edge_index[0], edge_index[1]
            messages = self.message(hidden[src])
            aggregated = torch.zeros_like(hidden)
            aggregated.index_add_(0, dst, messages)
            hidden = self.update(aggregated, hidden)
        graph_embedding = hidden.mean(dim=0, keepdim=True)
        return self.readout(graph_embedding)

