"""Shared data schemas for field observations and baseline evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FieldSample:
    """A tabular field-observation sample.

    The schema is intentionally backend-neutral: values are stored as Python
    lists so the loader works before NumPy/Pandas/Torch are required by tests.
    Training code can convert these lists into arrays or tensors later.
    """

    sample_id: str
    source_path: Path
    coordinates: list[list[float]]
    time: list[float]
    observations: dict[str, list[float]]
    process_parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return len(self.coordinates)

    def require_observation(self, name: str) -> list[float]:
        try:
            return self.observations[name]
        except KeyError as exc:
            raise KeyError(f"Observation not found in sample {self.sample_id}: {name}") from exc

