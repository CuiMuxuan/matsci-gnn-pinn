"""Utilities for train/validation/test split manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    split_path = Path(path)
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "splits" not in payload:
        raise ValueError(f"Split manifest must contain a 'splits' object: {split_path}")
    return payload


def split_indices(split_manifest: dict[str, Any], split_name: str) -> list[int]:
    try:
        indices = split_manifest["splits"][split_name]
    except KeyError as exc:
        raise KeyError(f"Split not found: {split_name}") from exc
    return [int(index) for index in indices]


def subset_sequence(values: list[Any], indices: list[int]) -> list[Any]:
    return [values[index] for index in indices]


def subset_field_sample(sample: Any, indices: list[int]) -> dict[str, Any]:
    """Return list-backed sample fields indexed by a split."""

    return {
        "coordinates": subset_sequence(sample.coordinates, indices),
        "time": subset_sequence(sample.time, indices),
        "observations": {
            name: subset_sequence(values, indices)
            for name, values in sample.observations.items()
        },
    }
