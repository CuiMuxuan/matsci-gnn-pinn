"""Graph-conditioned closure helpers for early GNN-PINN coupling."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
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


@dataclass(frozen=True)
class RealMicroGraphFeatureConfig:
    graph_features: str
    sample_id: str | None = None
    embedding_dim: int = 4
    normalize: bool = True


@dataclass(frozen=True)
class RealMicroRegionFeatureConfig:
    graph_features: str
    sample_id: str | None = None
    embedding_dim: int = 4
    normalize: bool = True
    row_source: str = "y"
    col_source: str = "x"
    flip_row: bool = False
    flip_col: bool = False
    selection: str = "nearest"
    inverse_distance_epsilon: float = 1e-6


@dataclass(frozen=True)
class RealMicroRegionEmbeddingFeatureConfig:
    graph_features: str
    sample_id: str | None = None
    embedding_dim: int = 4
    normalize: bool = True
    row_source: str = "y"
    col_source: str = "x"
    flip_row: bool = False
    flip_col: bool = False
    selection: str = "nearest"
    inverse_distance_epsilon: float = 1e-6


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


class RealMicroGraphFeatureProvider(_torch().nn.Module):
    """Provide sample-level features from real microstructure graph records."""

    def __init__(self, config: RealMicroGraphFeatureConfig):
        super().__init__()
        if config.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.config = config
        records = _load_real_micro_graph_records(Path(config.graph_features))
        if config.sample_id is not None:
            default_record = _find_real_micro_graph_record(records, config.sample_id, Path(config.graph_features))
        else:
            default_record = records[0]

        sample_ids: list[str] = []
        selected_feature_names: list[str] | None = None
        selected_values_by_sample: list[list[float]] = []
        selected_names_by_sample: dict[str, list[str]] = {}
        for record in records:
            sample_id = str(record.get("sample_id") or "")
            if not sample_id:
                raise ValueError("Every real micro graph feature record must contain a non-empty sample_id")
            feature_names = list(record["feature_names"])
            feature_values = [float(record["features"][name]) for name in feature_names]
            selected_names, selected_values = _select_real_micro_features(
                feature_names,
                feature_values,
                config.embedding_dim,
                normalize=config.normalize,
            )
            if selected_feature_names is None:
                selected_feature_names = selected_names
            elif selected_names != selected_feature_names:
                raise ValueError("All real micro graph feature records must expose the same selected feature names")
            sample_ids.append(sample_id)
            selected_names_by_sample[sample_id] = selected_names
            selected_values_by_sample.append(selected_values)

        torch = _torch()
        self.records = records
        self.record = default_record
        self.sample_ids = sample_ids
        self.sample_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
        self.default_sample_id = str(default_record.get("sample_id"))
        self.selected_feature_names = selected_feature_names or []
        self.selected_feature_names_by_sample = selected_names_by_sample
        self.register_buffer("feature_table", torch.tensor(selected_values_by_sample, dtype=torch.float32))
        default_index = self.sample_index[self.default_sample_id]
        self.register_buffer("features", self.feature_table[default_index : default_index + 1].clone())

    def forward(
        self,
        coords: Any | None = None,
        time: Any | None = None,
        sample_ids: list[str] | tuple[str, ...] | None = None,
    ) -> Any:
        if coords is None:
            return self.features.reshape(-1)
        if sample_ids is not None:
            if len(sample_ids) != coords.shape[0]:
                raise ValueError("sample_ids length must match coords batch size")
            torch = _torch()
            indices = torch.tensor(
                [self._sample_index_for_id(sample_id) for sample_id in sample_ids],
                dtype=torch.long,
                device=coords.device,
            )
            return self.feature_table.to(device=coords.device, dtype=coords.dtype).index_select(0, indices)
        return self.features.to(device=coords.device, dtype=coords.dtype).expand(coords.shape[0], -1)

    @property
    def feature_names(self) -> list[str]:
        return graph_feature_names(self.config.embedding_dim)

    def _sample_index_for_id(self, sample_id: str | None) -> int:
        resolved_sample_id = self.default_sample_id if sample_id in {None, ""} else str(sample_id)
        try:
            return self.sample_index[resolved_sample_id]
        except KeyError as exc:
            available = ", ".join(self.sample_ids)
            raise ValueError(
                f"Sample id {resolved_sample_id!r} not found in real micro graph feature records. "
                f"Available sample ids: {available}"
            ) from exc

    def metadata(self) -> dict[str, Any]:
        return {
            "graph_features": self.config.graph_features,
            "sample_id": self.record.get("sample_id"),
            "requested_sample_id": self.config.sample_id,
            "default_sample_id": self.default_sample_id,
            "available_sample_ids": self.sample_ids,
            "embedding_dim": self.config.embedding_dim,
            "normalize": self.config.normalize,
            "feature_names": self.feature_names,
            "source_feature_names": self.selected_feature_names,
            "source_feature_names_by_sample_id": self.selected_feature_names_by_sample,
            "features": self.features.detach().cpu().reshape(-1).tolist(),
            "sample_metadata": self.record.get("sample_metadata", {}),
            "graph_summary": self.record.get("graph_summary", {}),
        }


class RealMicroRegionFeatureProvider(_torch().nn.Module):
    """Select local micrograph patch features from real inspection grid nodes."""

    def __init__(self, config: RealMicroRegionFeatureConfig):
        super().__init__()
        if config.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if config.row_source not in {"x", "y"}:
            raise ValueError("row_source must be 'x' or 'y'")
        if config.col_source not in {"x", "y"}:
            raise ValueError("col_source must be 'x' or 'y'")
        if config.selection not in {"nearest", "inverse_distance"}:
            raise ValueError("selection must be 'nearest' or 'inverse_distance'")
        if config.inverse_distance_epsilon <= 0:
            raise ValueError("inverse_distance_epsilon must be positive")
        self.config = config
        records = _load_real_micro_graph_records(Path(config.graph_features))
        if config.sample_id is not None:
            default_record = _find_real_micro_graph_record(records, config.sample_id, Path(config.graph_features))
        else:
            default_record = records[0]

        sample_ids: list[str] = []
        selected_feature_names: list[str] | None = None
        selected_names_by_sample: dict[str, list[str]] = {}
        region_feature_names_by_sample: dict[str, list[str]] = {}
        region_counts_by_sample: dict[str, int] = {}
        region_tables: list[list[list[float]]] = []
        region_centers: list[list[list[float]]] = []
        expected_region_count: int | None = None
        for record in records:
            sample_id = str(record.get("sample_id") or "")
            if not sample_id:
                raise ValueError("Every real micro region feature record must contain a non-empty sample_id")
            region_feature_names = list(record.get("region_feature_names") or [])
            region_features = record.get("region_features") or []
            if not region_feature_names or not region_features:
                raise ValueError(
                    "real_micro_region requires records with region_feature_names and region_features. "
                    "Regenerate the micro feature JSONL with the current ambench_microstructure aggregate path."
                )
            centers = _real_micro_region_centers(region_feature_names, region_features)
            selected_names, selected_table = _select_real_micro_region_features(
                region_feature_names,
                region_features,
                config.embedding_dim,
                normalize=config.normalize,
            )
            if selected_feature_names is None:
                selected_feature_names = selected_names
            elif selected_names != selected_feature_names:
                raise ValueError("All real micro region records must expose the same selected feature names")
            if expected_region_count is None:
                expected_region_count = len(selected_table)
            elif len(selected_table) != expected_region_count:
                raise ValueError("All real micro region records must expose the same number of regions")
            sample_ids.append(sample_id)
            selected_names_by_sample[sample_id] = selected_names
            region_feature_names_by_sample[sample_id] = region_feature_names
            region_counts_by_sample[sample_id] = len(selected_table)
            region_tables.append(selected_table)
            region_centers.append(centers)

        torch = _torch()
        self.records = records
        self.record = default_record
        self.sample_ids = sample_ids
        self.sample_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
        self.default_sample_id = str(default_record.get("sample_id"))
        self.selected_feature_names = selected_feature_names or []
        self.selected_feature_names_by_sample = selected_names_by_sample
        self.region_feature_names_by_sample = region_feature_names_by_sample
        self.region_counts_by_sample = region_counts_by_sample
        self.register_buffer("region_feature_table", torch.tensor(region_tables, dtype=torch.float32))
        self.register_buffer("region_centers", torch.tensor(region_centers, dtype=torch.float32))
        default_index = self.sample_index[self.default_sample_id]
        self.register_buffer("features", self.region_feature_table[default_index, 0:1, :].clone())

    def forward(
        self,
        coords: Any | None = None,
        time: Any | None = None,
        sample_ids: list[str] | tuple[str, ...] | None = None,
    ) -> Any:
        if coords is None:
            return self.features.reshape(-1)
        if coords.shape[-1] < 2:
            raise ValueError("real_micro_region requires at least two coordinate dimensions")
        torch = _torch()
        if sample_ids is not None:
            if len(sample_ids) != coords.shape[0]:
                raise ValueError("sample_ids length must match coords batch size")
            sample_indices = torch.tensor(
                [self._sample_index_for_id(sample_id) for sample_id in sample_ids],
                dtype=torch.long,
                device=coords.device,
            )
        else:
            sample_indices = torch.full(
                (coords.shape[0],),
                self.sample_index[self.default_sample_id],
                dtype=torch.long,
                device=coords.device,
            )
        query = _real_micro_region_query(
            coords,
            row_source=self.config.row_source,
            col_source=self.config.col_source,
            flip_row=self.config.flip_row,
            flip_col=self.config.flip_col,
        )
        centers = self.region_centers.to(device=coords.device, dtype=coords.dtype).index_select(0, sample_indices)
        distances = torch.sum((query[:, None, :] - centers) ** 2, dim=-1)
        tables = self.region_feature_table.to(device=coords.device, dtype=coords.dtype).index_select(0, sample_indices)
        if self.config.selection == "inverse_distance":
            weights = 1.0 / (distances + self.config.inverse_distance_epsilon)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
            return torch.sum(tables * weights[:, :, None], dim=1)
        region_indices = torch.argmin(distances, dim=-1)
        batch_indices = torch.arange(coords.shape[0], dtype=torch.long, device=coords.device)
        return tables[batch_indices, region_indices, :]

    @property
    def feature_names(self) -> list[str]:
        return graph_feature_names(self.config.embedding_dim)

    def _sample_index_for_id(self, sample_id: str | None) -> int:
        resolved_sample_id = self.default_sample_id if sample_id in {None, ""} else str(sample_id)
        try:
            return self.sample_index[resolved_sample_id]
        except KeyError as exc:
            available = ", ".join(self.sample_ids)
            raise ValueError(
                f"Sample id {resolved_sample_id!r} not found in real micro region feature records. "
                f"Available sample ids: {available}"
            ) from exc

    def metadata(self) -> dict[str, Any]:
        return {
            "graph_features": self.config.graph_features,
            "sample_id": self.record.get("sample_id"),
            "requested_sample_id": self.config.sample_id,
            "default_sample_id": self.default_sample_id,
            "available_sample_ids": self.sample_ids,
            "embedding_dim": self.config.embedding_dim,
            "normalize": self.config.normalize,
            "feature_names": self.feature_names,
            "source_feature_names": self.selected_feature_names,
            "source_feature_names_by_sample_id": self.selected_feature_names_by_sample,
            "region_feature_names_by_sample_id": self.region_feature_names_by_sample,
            "region_counts_by_sample_id": self.region_counts_by_sample,
            "coordinate_mapping": {
                "row_source": self.config.row_source,
                "col_source": self.config.col_source,
                "flip_row": self.config.flip_row,
                "flip_col": self.config.flip_col,
                "query_row": _real_micro_region_query_description(
                    self.config.row_source,
                    self.config.flip_row,
                ),
                "query_col": _real_micro_region_query_description(
                    self.config.col_source,
                    self.config.flip_col,
                ),
                "selection": self.config.selection,
                "inverse_distance_epsilon": self.config.inverse_distance_epsilon,
            },
            "sample_metadata": self.record.get("sample_metadata", {}),
            "graph_summary": self.record.get("graph_summary", {}),
        }


class RealMicroRegionEmbeddingFeatureProvider(_torch().nn.Module):
    """Select fixed low-dimensional local patch embeddings from real inspection grids."""

    def __init__(self, config: RealMicroRegionEmbeddingFeatureConfig):
        super().__init__()
        if config.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if config.row_source not in {"x", "y"}:
            raise ValueError("row_source must be 'x' or 'y'")
        if config.col_source not in {"x", "y"}:
            raise ValueError("col_source must be 'x' or 'y'")
        if config.selection not in {"nearest", "inverse_distance"}:
            raise ValueError("selection must be 'nearest' or 'inverse_distance'")
        if config.inverse_distance_epsilon <= 0:
            raise ValueError("inverse_distance_epsilon must be positive")
        self.config = config
        records = _load_real_micro_graph_records(Path(config.graph_features))
        if config.sample_id is not None:
            default_record = _find_real_micro_graph_record(records, config.sample_id, Path(config.graph_features))
        else:
            default_record = records[0]

        sample_ids: list[str] = []
        selected_feature_names: list[str] | None = None
        selected_names_by_sample: dict[str, list[str]] = {}
        region_embedding_names_by_sample: dict[str, list[str]] = {}
        region_embedding_metadata_by_sample: dict[str, Any] = {}
        region_counts_by_sample: dict[str, int] = {}
        region_tables: list[list[list[float]]] = []
        region_centers: list[list[list[float]]] = []
        expected_region_count: int | None = None
        for record in records:
            sample_id = str(record.get("sample_id") or "")
            if not sample_id:
                raise ValueError("Every real micro region embedding record must contain a non-empty sample_id")
            region_feature_names = list(record.get("region_feature_names") or [])
            region_features = record.get("region_features") or []
            embedding_feature_names = list(record.get("region_embedding_feature_names") or [])
            embedding_features = record.get("region_embedding_features") or []
            if not region_feature_names or not region_features:
                raise ValueError(
                    "real_micro_region_embedding requires records with region_feature_names and region_features"
                )
            if not embedding_feature_names or not embedding_features:
                raise ValueError(
                    "real_micro_region_embedding requires records with region_embedding_feature_names "
                    "and region_embedding_features. Regenerate the JSONL aggregate with --region-embedding-dim."
                )
            centers = _real_micro_region_centers(region_feature_names, region_features)
            selected_names, selected_table = _select_real_micro_region_embedding_features(
                embedding_feature_names,
                embedding_features,
                config.embedding_dim,
                normalize=config.normalize,
            )
            if selected_feature_names is None:
                selected_feature_names = selected_names
            elif selected_names != selected_feature_names:
                raise ValueError("All real micro region embedding records must expose the same selected names")
            if expected_region_count is None:
                expected_region_count = len(selected_table)
            elif len(selected_table) != expected_region_count:
                raise ValueError("All real micro region embedding records must expose the same number of regions")
            if len(centers) != len(selected_table):
                raise ValueError("Region center count must match region embedding row count")
            sample_ids.append(sample_id)
            selected_names_by_sample[sample_id] = selected_names
            region_embedding_names_by_sample[sample_id] = embedding_feature_names
            region_embedding_metadata_by_sample[sample_id] = record.get("region_embedding_metadata", {})
            region_counts_by_sample[sample_id] = len(selected_table)
            region_tables.append(selected_table)
            region_centers.append(centers)

        torch = _torch()
        self.records = records
        self.record = default_record
        self.sample_ids = sample_ids
        self.sample_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
        self.default_sample_id = str(default_record.get("sample_id"))
        self.selected_feature_names = selected_feature_names or []
        self.selected_feature_names_by_sample = selected_names_by_sample
        self.region_embedding_names_by_sample = region_embedding_names_by_sample
        self.region_embedding_metadata_by_sample = region_embedding_metadata_by_sample
        self.region_counts_by_sample = region_counts_by_sample
        self.register_buffer("region_feature_table", torch.tensor(region_tables, dtype=torch.float32))
        self.register_buffer("region_centers", torch.tensor(region_centers, dtype=torch.float32))
        default_index = self.sample_index[self.default_sample_id]
        self.register_buffer("features", self.region_feature_table[default_index, 0:1, :].clone())

    def forward(
        self,
        coords: Any | None = None,
        time: Any | None = None,
        sample_ids: list[str] | tuple[str, ...] | None = None,
    ) -> Any:
        if coords is None:
            return self.features.reshape(-1)
        if coords.shape[-1] < 2:
            raise ValueError("real_micro_region_embedding requires at least two coordinate dimensions")
        torch = _torch()
        if sample_ids is not None:
            if len(sample_ids) != coords.shape[0]:
                raise ValueError("sample_ids length must match coords batch size")
            sample_indices = torch.tensor(
                [self._sample_index_for_id(sample_id) for sample_id in sample_ids],
                dtype=torch.long,
                device=coords.device,
            )
        else:
            sample_indices = torch.full(
                (coords.shape[0],),
                self.sample_index[self.default_sample_id],
                dtype=torch.long,
                device=coords.device,
            )
        query = _real_micro_region_query(
            coords,
            row_source=self.config.row_source,
            col_source=self.config.col_source,
            flip_row=self.config.flip_row,
            flip_col=self.config.flip_col,
        )
        centers = self.region_centers.to(device=coords.device, dtype=coords.dtype).index_select(0, sample_indices)
        distances = torch.sum((query[:, None, :] - centers) ** 2, dim=-1)
        tables = self.region_feature_table.to(device=coords.device, dtype=coords.dtype).index_select(0, sample_indices)
        if self.config.selection == "inverse_distance":
            weights = 1.0 / (distances + self.config.inverse_distance_epsilon)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
            return torch.sum(tables * weights[:, :, None], dim=1)
        region_indices = torch.argmin(distances, dim=-1)
        batch_indices = torch.arange(coords.shape[0], dtype=torch.long, device=coords.device)
        return tables[batch_indices, region_indices, :]

    @property
    def feature_names(self) -> list[str]:
        return graph_feature_names(self.config.embedding_dim)

    def _sample_index_for_id(self, sample_id: str | None) -> int:
        resolved_sample_id = self.default_sample_id if sample_id in {None, ""} else str(sample_id)
        try:
            return self.sample_index[resolved_sample_id]
        except KeyError as exc:
            available = ", ".join(self.sample_ids)
            raise ValueError(
                f"Sample id {resolved_sample_id!r} not found in real micro region embedding records. "
                f"Available sample ids: {available}"
            ) from exc

    def metadata(self) -> dict[str, Any]:
        return {
            "graph_features": self.config.graph_features,
            "sample_id": self.record.get("sample_id"),
            "requested_sample_id": self.config.sample_id,
            "default_sample_id": self.default_sample_id,
            "available_sample_ids": self.sample_ids,
            "embedding_dim": self.config.embedding_dim,
            "normalize": self.config.normalize,
            "feature_names": self.feature_names,
            "source_feature_names": self.selected_feature_names,
            "source_feature_names_by_sample_id": self.selected_feature_names_by_sample,
            "region_embedding_feature_names_by_sample_id": self.region_embedding_names_by_sample,
            "region_embedding_metadata_by_sample_id": self.region_embedding_metadata_by_sample,
            "region_counts_by_sample_id": self.region_counts_by_sample,
            "coordinate_mapping": {
                "row_source": self.config.row_source,
                "col_source": self.config.col_source,
                "flip_row": self.config.flip_row,
                "flip_col": self.config.flip_col,
                "query_row": _real_micro_region_query_description(
                    self.config.row_source,
                    self.config.flip_row,
                ),
                "query_col": _real_micro_region_query_description(
                    self.config.col_source,
                    self.config.flip_col,
                ),
                "selection": self.config.selection,
                "inverse_distance_epsilon": self.config.inverse_distance_epsilon,
            },
            "sample_metadata": self.record.get("sample_metadata", {}),
            "graph_summary": self.record.get("graph_summary", {}),
        }


def _load_real_micro_graph_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Real micro graph feature file not found: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"Real micro graph feature file is empty: {path}")
    return records


def _find_real_micro_graph_record(records: list[dict[str, Any]], sample_id: str, path: Path) -> dict[str, Any]:
    for record in records:
        if record.get("sample_id") == sample_id:
            return record
    raise ValueError(f"Sample id {sample_id!r} not found in real micro graph feature file: {path}")


def _load_real_micro_graph_record(path: Path, sample_id: str | None) -> dict[str, Any]:
    records = _load_real_micro_graph_records(path)
    if sample_id is None:
        return records[0]
    return _find_real_micro_graph_record(records, sample_id, path)


def _select_real_micro_features(
    feature_names: list[str],
    feature_values: list[float],
    embedding_dim: int,
    *,
    normalize: bool,
) -> tuple[list[str], list[float]]:
    if len(feature_names) != len(feature_values):
        raise ValueError("feature_names and feature_values must have the same length")
    preferred_names = [
        "image_mask_fraction",
        "mask_centroid_row_norm",
        "mask_centroid_col_norm",
        "mask_bbox_area_fraction",
        "mask_span_row_norm",
        "mask_span_col_norm",
        "mask_perimeter_fraction",
        "gradient_magnitude_q90_norm",
        "mask_fill_fraction",
        "mask_anisotropy",
        "gradient_magnitude_mean_norm",
        "image_intensity_iqr_norm",
        "image_entropy_32bin",
        "node_mask_fraction_mean",
        "node_mask_fraction_std",
        "node_mean_intensity_norm_mean",
        "node_std_intensity_norm_mean",
        "node_mean_intensity_norm_std",
        "image_mean_intensity",
        "image_std_intensity",
    ]
    by_name = dict(zip(feature_names, feature_values))
    selected_names = [name for name in preferred_names if name in by_name]
    for name in feature_names:
        if name not in selected_names:
            selected_names.append(name)
        if len(selected_names) >= embedding_dim:
            break
    selected_names = selected_names[:embedding_dim]
    selected_values = [float(by_name[name]) for name in selected_names]
    if len(selected_values) < embedding_dim:
        selected_names.extend(f"padding_{index}" for index in range(embedding_dim - len(selected_values)))
        selected_values.extend(0.0 for _ in range(embedding_dim - len(selected_values)))
    if normalize:
        selected_values = _minmax_normalize_vector(selected_values)
    return selected_names, selected_values


def _minmax_normalize_vector(values: list[float]) -> list[float]:
    if not values:
        return values
    minimum = min(values)
    maximum = max(values)
    scale = maximum - minimum
    if scale == 0:
        return [0.0 for _ in values]
    return [(value - minimum) / scale for value in values]


def _real_micro_region_centers(
    region_feature_names: list[str],
    region_features: list[list[float]],
) -> list[list[float]]:
    try:
        row_index = region_feature_names.index("center_row_norm")
        col_index = region_feature_names.index("center_col_norm")
    except ValueError as exc:
        raise ValueError(
            "real_micro_region records must include center_row_norm and center_col_norm region features"
        ) from exc
    return [[float(region[row_index]), float(region[col_index])] for region in region_features]


def _real_micro_region_query(
    coords: Any,
    *,
    row_source: str,
    col_source: str,
    flip_row: bool,
    flip_col: bool,
) -> Any:
    row = _real_micro_region_coordinate(coords, row_source)
    col = _real_micro_region_coordinate(coords, col_source)
    if flip_row:
        row = 1.0 - row
    if flip_col:
        col = 1.0 - col
    torch = _torch()
    return torch.stack([row, col], dim=-1)


def _real_micro_region_coordinate(coords: Any, source: str) -> Any:
    if source == "x":
        return coords[:, 0].clamp(0.0, 1.0)
    if source == "y":
        return coords[:, 1].clamp(0.0, 1.0)
    raise ValueError("region coordinate source must be 'x' or 'y'")


def _real_micro_region_query_description(source: str, flipped: bool) -> str:
    axis = "0" if source == "x" else "1"
    expression = f"coords[:, {axis}] clamped to [0, 1]"
    return f"1 - ({expression})" if flipped else expression


def _select_real_micro_region_features(
    region_feature_names: list[str],
    region_features: list[list[float]],
    embedding_dim: int,
    *,
    normalize: bool,
) -> tuple[list[str], list[list[float]]]:
    if len(region_feature_names) != len(region_features[0]):
        raise ValueError("region_feature_names length must match region feature width")
    preferred_names = [
        "mask_fraction",
        "mean_intensity_norm",
        "std_intensity_norm",
        "center_row_norm",
        "center_col_norm",
    ]
    selected_names = [name for name in preferred_names if name in region_feature_names]
    for name in region_feature_names:
        if name not in selected_names:
            selected_names.append(name)
        if len(selected_names) >= embedding_dim:
            break
    selected_names = selected_names[:embedding_dim]
    selected_indices = [region_feature_names.index(name) for name in selected_names]
    selected_table = [[float(region[index]) for index in selected_indices] for region in region_features]
    if len(selected_names) < embedding_dim:
        padding_count = embedding_dim - len(selected_names)
        selected_names.extend(f"padding_{index}" for index in range(padding_count))
        for row in selected_table:
            row.extend(0.0 for _ in range(padding_count))
    if normalize:
        selected_table = _minmax_normalize_columns(selected_table)
    return selected_names, selected_table


def _minmax_normalize_columns(rows: list[list[float]]) -> list[list[float]]:
    if not rows:
        return rows
    width = len(rows[0])
    columns = [[row[index] for row in rows] for index in range(width)]
    minimums = [min(column) for column in columns]
    maximums = [max(column) for column in columns]
    normalized_rows: list[list[float]] = []
    for row in rows:
        normalized_row = []
        for index, value in enumerate(row):
            scale = maximums[index] - minimums[index]
            normalized_row.append(0.0 if scale == 0 else (value - minimums[index]) / scale)
        normalized_rows.append(normalized_row)
    return normalized_rows


def _select_real_micro_region_embedding_features(
    embedding_feature_names: list[str],
    embedding_features: list[list[float]],
    embedding_dim: int,
    *,
    normalize: bool,
) -> tuple[list[str], list[list[float]]]:
    if not embedding_features:
        raise ValueError("region embedding features must contain at least one row")
    if len(embedding_feature_names) != len(embedding_features[0]):
        raise ValueError("region_embedding_feature_names length must match embedding feature width")
    selected_names = embedding_feature_names[:embedding_dim]
    selected_indices = [embedding_feature_names.index(name) for name in selected_names]
    selected_table = [[float(region[index]) for index in selected_indices] for region in embedding_features]
    if len(selected_names) < embedding_dim:
        padding_count = embedding_dim - len(selected_names)
        selected_names.extend(f"padding_{index}" for index in range(padding_count))
        for row in selected_table:
            row.extend(0.0 for _ in range(padding_count))
    if normalize:
        selected_table = _minmax_normalize_columns(selected_table)
    return selected_names, selected_table
