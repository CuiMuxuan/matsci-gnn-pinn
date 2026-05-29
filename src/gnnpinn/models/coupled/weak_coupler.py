"""Minimal weak-coupling orchestration for direction 3 prototypes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class WeakCouplingState:
    macro_fields: Any
    micro_embedding: Any
    effective_params: Any
    updated_fields: Any


class WeakGNNPINNCoupler:
    """Orchestrate macro -> micro -> coarse-grain -> macro feedback."""

    def __init__(
        self,
        macro_model: Callable[..., Any],
        micro_encoder: Callable[..., Any],
        coarse_grainer: Callable[[Any], Any],
    ):
        self.macro_model = macro_model
        self.micro_encoder = micro_encoder
        self.coarse_grainer = coarse_grainer

    def forward(
        self,
        coords: Any,
        time: Any,
        node_features: Any,
        edge_index: Any,
        params: Any | None = None,
    ) -> WeakCouplingState:
        macro_fields = self.macro_model(coords, time, params=params)
        micro_embedding = self.micro_encoder(node_features, edge_index)
        effective_params = self.coarse_grainer(micro_embedding)
        updated_fields = self.macro_model(coords, time, params=effective_params)
        return WeakCouplingState(
            macro_fields=macro_fields,
            micro_embedding=micro_embedding,
            effective_params=effective_params,
            updated_fields=updated_fields,
        )

