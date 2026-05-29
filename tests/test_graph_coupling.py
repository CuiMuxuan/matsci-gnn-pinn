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

