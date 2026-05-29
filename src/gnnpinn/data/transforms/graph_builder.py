"""Simple graph-building helpers for early microstructure prototypes."""

from __future__ import annotations

import math


def knn_edges_2d(points: list[tuple[float, float]], k: int = 2, bidirectional: bool = True) -> list[list[int]]:
    """Build a small k-nearest-neighbor edge index for 2D points."""

    if k <= 0:
        raise ValueError("k must be positive")
    sources: list[int] = []
    targets: list[int] = []
    for i, point in enumerate(points):
        distances = [
            (j, _distance(point, other))
            for j, other in enumerate(points)
            if j != i
        ]
        for j, _ in sorted(distances, key=lambda item: item[1])[:k]:
            sources.append(i)
            targets.append(j)
            if bidirectional:
                sources.append(j)
                targets.append(i)
    return [sources, targets]


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

