"""Sparse candidate libraries for closure/equation discovery."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Any


def _torch() -> Any:
    import torch

    return torch


@dataclass(frozen=True)
class LibraryTerm:
    name: str
    powers: tuple[int, ...]


@dataclass(frozen=True)
class SparseLibraryConfig:
    feature_names: tuple[str, ...]
    polynomial_order: int = 2
    include_bias: bool = True
    include_linear: bool = True


class SparseLibrary:
    """Build a polynomial candidate matrix for sparse closure discovery."""

    def __init__(self, config: SparseLibraryConfig):
        if config.polynomial_order < 1:
            raise ValueError("polynomial_order must be >= 1")
        self.config = config
        self.terms = build_terms(config)

    def transform(self, features: Any) -> Any:
        """Return candidate matrix with shape `(n_samples, n_terms)`."""

        torch = _torch()
        if features.shape[-1] != len(self.config.feature_names):
            raise ValueError(
                f"Expected {len(self.config.feature_names)} features, got {features.shape[-1]}"
            )
        columns = []
        for term in self.terms:
            if sum(term.powers) == 0:
                columns.append(torch.ones(features.shape[0], device=features.device, dtype=features.dtype))
                continue
            value = torch.ones(features.shape[0], device=features.device, dtype=features.dtype)
            for idx, power in enumerate(term.powers):
                if power:
                    value = value * features[:, idx] ** power
            columns.append(value)
        return torch.stack(columns, dim=-1)

    @property
    def term_names(self) -> list[str]:
        return [term.name for term in self.terms]


def build_terms(config: SparseLibraryConfig) -> list[LibraryTerm]:
    feature_count = len(config.feature_names)
    terms: list[LibraryTerm] = []
    if config.include_bias:
        terms.append(LibraryTerm(name="1", powers=tuple(0 for _ in range(feature_count))))

    min_order = 1 if config.include_linear else 2
    for order in range(min_order, config.polynomial_order + 1):
        for combo in combinations_with_replacement(range(feature_count), order):
            powers = [0 for _ in range(feature_count)]
            for idx in combo:
                powers[idx] += 1
            terms.append(
                LibraryTerm(
                    name=_format_term(config.feature_names, tuple(powers)),
                    powers=tuple(powers),
                )
            )
    return terms


def _format_term(feature_names: tuple[str, ...], powers: tuple[int, ...]) -> str:
    pieces: list[str] = []
    for name, power in zip(feature_names, powers):
        if power == 1:
            pieces.append(name)
        elif power > 1:
            pieces.append(f"{name}^{power}")
    return "*".join(pieces)

