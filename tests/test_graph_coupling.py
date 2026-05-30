from __future__ import annotations

import subprocess
import sys

import pytest

from gnnpinn.data.graph_schema import MicrostructureGraph
from gnnpinn.data.transforms import knn_edges_2d


def _torch_available() -> bool:
    result = subprocess.run(
        [sys.executable, "-c", "import torch"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


torchmark = pytest.mark.skipif(
    not _torch_available(),
    reason="torch is not importable in the current environment",
)


def test_knn_edges_and_graph_schema():
    edge_index = knn_edges_2d([(0.0, 0.0), (1.0, 0.0), (0.0, 2.0)], k=1)
    graph = MicrostructureGraph(
        node_features=[[1.0], [2.0], [3.0]],
        edge_index=edge_index,
    )

    assert graph.num_nodes == 3
    assert graph.num_edges == len(edge_index[0])
    assert graph.num_edges > 0


@torchmark
def test_micro_gnn_encoder_forward_shape():
    import torch

    from gnnpinn.models.gnn import MicroGNNEncoder

    encoder = MicroGNNEncoder(node_dim=2, hidden_dim=8, output_dim=4, steps=1)
    node_features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

    output = encoder(node_features, edge_index)

    assert tuple(output.shape) == (1, 4)


@torchmark
def test_toy_static_graph_embedding_provider_metadata():
    from gnnpinn.models.closure import ToyStaticGraphConfig, ToyStaticGraphEmbeddingProvider

    provider = ToyStaticGraphEmbeddingProvider(
        ToyStaticGraphConfig(node_feature_dim=3, hidden_dim=8, embedding_dim=3, steps=1, seed=13)
    )

    embedding = provider()
    metadata = provider.metadata()

    assert tuple(embedding.shape) == (3,)
    assert provider.feature_names == ["g0", "g1", "g2"]
    assert metadata["embedding_dim"] == 3
    assert metadata["node_feature_dim"] == 3
    assert len(metadata["node_features"]) == 4


@torchmark
def test_coordinate_rbf_graph_feature_provider_is_per_point():
    import torch

    from gnnpinn.models.closure import CoordinateRBFGraphConfig, CoordinateRBFGraphFeatureProvider

    provider = CoordinateRBFGraphFeatureProvider(
        CoordinateRBFGraphConfig(state_dim=3, embedding_dim=4, length_scale=0.4)
    )
    coords = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    time = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

    features = provider(coords, time)
    metadata = provider.metadata()

    assert tuple(features.shape) == (2, 4)
    assert torch.allclose(features.sum(dim=1), torch.ones(2))
    assert not torch.allclose(features[0], features[1])
    assert metadata["feature_names"] == ["g0", "g1", "g2", "g3"]


@torchmark
def test_real_micro_graph_feature_provider_reads_jsonl(tmp_path):
    import json
    import torch

    from gnnpinn.models.closure import RealMicroGraphFeatureConfig, RealMicroGraphFeatureProvider

    feature_path = tmp_path / "features.jsonl"
    feature_path.write_text(
        json.dumps(
            {
                "sample_id": "sample_a",
                "sample_metadata": {"process": "P2"},
                "feature_names": [
                    "image_mask_fraction",
                    "node_mask_fraction_mean",
                    "node_mask_fraction_std",
                    "node_mean_intensity_norm_mean",
                ],
                "features": {
                    "image_mask_fraction": 0.1,
                    "node_mask_fraction_mean": 0.2,
                    "node_mask_fraction_std": 0.05,
                    "node_mean_intensity_norm_mean": 0.7,
                },
                "graph_summary": {"num_nodes": 64},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    provider = RealMicroGraphFeatureProvider(
        RealMicroGraphFeatureConfig(
            graph_features=str(feature_path),
            sample_id="sample_a",
            embedding_dim=3,
            normalize=False,
        )
    )
    coords = torch.zeros((2, 2), dtype=torch.float32)
    time = torch.zeros((2, 1), dtype=torch.float32)
    features = provider(coords, time)
    metadata = provider.metadata()

    assert tuple(features.shape) == (2, 3)
    assert torch.allclose(features[0], features[1])
    assert metadata["sample_id"] == "sample_a"
    assert metadata["feature_names"] == ["g0", "g1", "g2"]
    assert metadata["source_feature_names"] == [
        "image_mask_fraction",
        "node_mask_fraction_mean",
        "node_mask_fraction_std",
    ]


@torchmark
def test_real_micro_graph_feature_provider_selects_per_sample_id(tmp_path):
    import json
    import torch

    from gnnpinn.models.closure import RealMicroGraphFeatureConfig, RealMicroGraphFeatureProvider

    feature_path = tmp_path / "features.jsonl"
    records = [
        {
            "sample_id": "sample_a",
            "sample_metadata": {"process": "P1"},
            "feature_names": ["image_mask_fraction", "node_mask_fraction_mean"],
            "features": {"image_mask_fraction": 0.1, "node_mask_fraction_mean": 0.2},
            "graph_summary": {"num_nodes": 64},
        },
        {
            "sample_id": "sample_b",
            "sample_metadata": {"process": "P2"},
            "feature_names": ["image_mask_fraction", "node_mask_fraction_mean"],
            "features": {"image_mask_fraction": 0.8, "node_mask_fraction_mean": 0.4},
            "graph_summary": {"num_nodes": 64},
        },
    ]
    feature_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    provider = RealMicroGraphFeatureProvider(
        RealMicroGraphFeatureConfig(
            graph_features=str(feature_path),
            sample_id="sample_a",
            embedding_dim=2,
            normalize=False,
        )
    )
    coords = torch.zeros((3, 2), dtype=torch.float32)
    time = torch.zeros((3, 1), dtype=torch.float32)

    features = provider(coords, time, sample_ids=["sample_a", "sample_b", "sample_a"])
    metadata = provider.metadata()

    assert tuple(features.shape) == (3, 2)
    assert torch.allclose(features[0], torch.tensor([0.1, 0.2]))
    assert torch.allclose(features[1], torch.tensor([0.8, 0.4]))
    assert torch.allclose(features[2], features[0])
    assert metadata["available_sample_ids"] == ["sample_a", "sample_b"]


@torchmark
def test_real_micro_graph_feature_provider_prefers_material_image_features(tmp_path):
    import json

    from gnnpinn.models.closure import RealMicroGraphFeatureConfig, RealMicroGraphFeatureProvider

    feature_path = tmp_path / "features_v2.jsonl"
    feature_path.write_text(
        json.dumps(
            {
                "sample_id": "sample_a",
                "sample_metadata": {"process": "P4"},
                "feature_names": [
                    "image_mean_intensity",
                    "image_std_intensity",
                    "image_mask_fraction",
                    "mask_centroid_row_norm",
                    "mask_centroid_col_norm",
                    "mask_bbox_area_fraction",
                    "mask_span_row_norm",
                    "mask_span_col_norm",
                    "mask_perimeter_fraction",
                    "gradient_magnitude_q90_norm",
                ],
                "features": {
                    "image_mean_intensity": 10.0,
                    "image_std_intensity": 2.0,
                    "image_mask_fraction": 0.1,
                    "mask_centroid_row_norm": 0.6,
                    "mask_centroid_col_norm": 0.4,
                    "mask_bbox_area_fraction": 0.2,
                    "mask_span_row_norm": 0.3,
                    "mask_span_col_norm": 0.5,
                    "mask_perimeter_fraction": 0.05,
                    "gradient_magnitude_q90_norm": 0.8,
                },
                "graph_summary": {"num_nodes": 64},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    provider = RealMicroGraphFeatureProvider(
        RealMicroGraphFeatureConfig(
            graph_features=str(feature_path),
            sample_id="sample_a",
            embedding_dim=8,
            normalize=False,
        )
    )
    metadata = provider.metadata()

    assert metadata["source_feature_names"] == [
        "image_mask_fraction",
        "mask_centroid_row_norm",
        "mask_centroid_col_norm",
        "mask_bbox_area_fraction",
        "mask_span_row_norm",
        "mask_span_col_norm",
        "mask_perimeter_fraction",
        "gradient_magnitude_q90_norm",
    ]


@torchmark
def test_weak_coupler_runs_two_macro_passes():
    import torch

    from gnnpinn.models.coupled import WeakGNNPINNCoupler

    calls = {"macro": 0}

    def macro(coords, time, params=None):
        calls["macro"] += 1
        offset = 0.0 if params is None else float(params.mean())
        return coords.sum(dim=-1, keepdim=True) + time + offset

    def micro(node_features, edge_index):
        return node_features.mean(dim=0, keepdim=True)

    def coarse(micro_embedding):
        return micro_embedding.mean(dim=-1, keepdim=True)

    coupler = WeakGNNPINNCoupler(macro, micro, coarse)
    state = coupler.forward(
        coords=torch.tensor([[1.0, 2.0]]),
        time=torch.tensor([[0.5]]),
        node_features=torch.tensor([[1.0], [3.0]]),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
    )

    assert calls["macro"] == 2
    assert float(state.updated_fields.item()) > float(state.macro_fields.item())
