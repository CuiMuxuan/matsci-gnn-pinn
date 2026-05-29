"""Dependency-light scalar regression metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence


def mse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    pairs = _paired(y_true, y_pred)
    return sum((true - pred) ** 2 for true, pred in pairs) / len(pairs)


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return math.sqrt(mse(y_true, y_pred))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    pairs = _paired(y_true, y_pred)
    return sum(abs(true - pred) for true, pred in pairs) / len(pairs)


def relative_l2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    pairs = _paired(y_true, y_pred)
    numerator = math.sqrt(sum((true - pred) ** 2 for true, pred in pairs))
    denominator = math.sqrt(sum(true**2 for true, _ in pairs))
    if denominator == 0:
        raise ValueError("relative_l2 denominator is zero")
    return numerator / denominator


def normalized_rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    pairs = _paired(y_true, y_pred)
    values = [true for true, _ in pairs]
    span = max(values) - min(values)
    if span == 0:
        raise ValueError("normalized_rmse range is zero")
    return rmse(y_true, y_pred) / span


def _paired(y_true: Sequence[float], y_pred: Sequence[float]) -> list[tuple[float, float]]:
    if len(y_true) != len(y_pred):
        raise ValueError("Metric inputs must have the same length")
    if not y_true:
        raise ValueError("Metric inputs cannot be empty")
    return [(float(true), float(pred)) for true, pred in zip(y_true, y_pred)]

