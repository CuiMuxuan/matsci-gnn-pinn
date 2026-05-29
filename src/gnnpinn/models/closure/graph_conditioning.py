"""Graph-conditioned closure helpers for early GNN-PINN coupling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gnnpinn.models.gnn import MicroGNNEncoder


def _torch() -> Any:
    import torch

    return torch


@dataclass(frozen=True)
class ToyStaticGraphConfig:
    node_feature_dim: int = 2
    hidden_dim: int = 16
    embedding_dim: int = 2
    steps: int = 1
    seed: int = 2026


@dataclass(frozen=True)
class CoordinateRBFGraphConfig:
    state_dim: int = 3
    embedding_dim: int = 4
    length_scale: float = 0.35
    normalize: bool = True


def graph_feature_names(embedding_dim: int, prefix: str = "g") -> list[str]:
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")
    return [f"{prefix}{index}" for index in range(embedding_dim)]


class ToyStaticGraphEmbeddingProvider(_torch().nn.Module):
    """Encode a tiny deterministic graph for interface-level coupling tests."""

    def __init__(self, config: ToyStaticGraphConfig):
        super().__init__()
        if config.node_feature_dim <= 0:
            raise ValueError("node_feature_dim must be positive")
        if config.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if config.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.config = config
        torch = _torch()
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(config.seed)
        try:
            self.encoder = MicroGNNEncoder(
                node_dim=config.node_feature_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.embedding_dim,
                steps=config.steps,
            )
        finally:
            torch.random.set_rng_state(rng_state)
        self.register_buffer("node_features", _toy_node_features(config.node_feature_dim))
        self.register_buffer("edge_index", _toy_edge_index())

    def forward(self) -> Any:
        return self.encoder(self.node_features, self.edge_index).reshape(-1)

    @property
    def feature_names(self) -> list[str]:
        return graph_feature_names(self.config.embedding_dim)

    def metadata(self) -> dict[str, Any]:
        return {
            "node_feature_dim": self.config.node_feature_dim,
            "hidden_dim": self.config.hidden_dim,
            "embedding_dim": self.config.embedding_dim,
            "steps": self.config.steps,
            "seed": self.config.seed,
            "feature_names": self.feature_names,
            "node_features": self.node_features.detach().cpu().tolist(),
            "edge_index": self.edge_index.detach().cpu().tolist(),
        }


def _toy_node_features(node_feature_dim: int) -> Any:
    torch = _torch()
    base = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    if node_feature_dim <= 2:
        return base[:, :node_feature_dim]
    extra_columns = []
    radius = torch.sqrt(((base - 0.5) ** 2).sum(dim=1, keepdim=True))
    coordinate_sum = base.sum(dim=1, keepdim=True)
    templates = [radius, coordinate_sum, torch.ones((base.shape[0], 1), dtype=torch.float32)]
    for index in range(node_feature_dim - 2):
        extra_columns.append(templates[index % len(templates)])
    return torch.cat([base, *extra_columns], dim=1)


def _toy_edge_index() -> Any:
    torch = _torch()
    return torch.tensor(
        [
            [0, 1, 0, 2, 1, 3, 2, 3, 1, 0, 2, 0, 3, 1, 3, 2],
            [1, 0, 2, 0, 3, 1, 3, 2, 0, 1, 0, 2, 1, 3, 2, 3],
        ],
        dtype=torch.long,
    )


class CoordinateRBFGraphFeatureProvider(_torch().nn.Module):
    """Deterministic per-point graph features from coordinate/time anchors."""

    def __init__(self, config: CoordinateRBFGraphConfig):
        super().__init__()
        if config.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if config.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if config.length_scale <= 0:
            raise ValueError("length_scale must be positive")
        self.config = config
        self.register_buffer("anchors", _coordinate_rbf_anchors(config.state_dim, config.embedding_dim))

    def forward(self, coords: Any, time: Any) -> Any:
        torch = _torch()
        state = torch.cat([coords, time.reshape(time.shape[0], -1)], dim=-1)
        if state.shape[-1] != self.config.state_dim:
            raise ValueError(f"Expected state_dim={self.config.state_dim}, got {state.shape[-1]}")
        anchors = self.anchors.to(device=state.device, dtype=state.dtype)
        distances = torch.sum((state[:, None, :] - anchors[None, :, :]) ** 2, dim=-1)
        features = torch.exp(-distances / (2.0 * self.config.length_scale**2))
        if self.config.normalize:
            denominator = features.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(features.dtype).eps)
            features = features / denominator
        return features

    @property
    def feature_names(self) -> list[str]:
        return graph_feature_names(self.config.embedding_dim)

    def metadata(self) -> dict[str, Any]:
        return {
            "state_dim": self.config.state_dim,
            "embedding_dim": self.config.embedding_dim,
            "length_scale": self.config.length_scale,
            "normalize": self.config.normalize,
            "feature_names": self.feature_names,
            "anchors": self.anchors.detach().cpu().tolist(),
        }


def _coordinate_rbf_anchors(state_dim: int, embedding_dim: int) -> Any:
    torch = _torch()
    if embedding_dim == 1:
        return torch.full((1, state_dim), 0.5, dtype=torch.float32)
    anchors = []
    for index in range(embedding_dim):
        fraction = index / (embedding_dim - 1)
        anchors.append(
            [
                ((fraction * (dimension + 1) * 0.6180339887498949) + 0.5 / (dimension + 2)) % 1.0
                for dimension in range(state_dim)
            ]
        )
    return torch.tensor(anchors, dtype=torch.float32)
