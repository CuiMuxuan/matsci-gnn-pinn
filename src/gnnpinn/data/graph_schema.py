"""Backend-neutral graph schema for microstructure data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MicrostructureGraph:
    node_features: list[list[float]]
    edge_index: list[list[int]]
    edge_features: list[list[float]] = field(default_factory=list)
    global_features: dict[str, Any] = field(default_factory=dict)
    target_statistics: dict[str, float] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return len(self.node_features)

    @property
    def num_edges(self) -> int:
        return len(self.edge_index[0]) if self.edge_index else 0

