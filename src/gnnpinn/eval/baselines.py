"""Simple no-training baselines for early field-data checks."""

from __future__ import annotations

from collections.abc import Sequence

from .metrics import mae, normalized_rmse, relative_l2, rmse


def constant_predictions(values: Sequence[float], strategy: str = "mean") -> list[float]:
    """Return constant predictions using a simple statistic of target values."""

    if not values:
        raise ValueError("Cannot build a baseline for an empty target")
    if strategy == "mean":
        value = sum(float(item) for item in values) / len(values)
    elif strategy == "first":
        value = float(values[0])
    elif strategy == "zero":
        value = 0.0
    else:
        raise ValueError(f"Unsupported baseline strategy: {strategy}")
    return [value for _ in values]


def regression_metric_table(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float | str]:
    """Compute the standard field-regression metric table."""

    metrics: dict[str, float | str] = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }
    try:
        metrics["relative_l2"] = relative_l2(y_true, y_pred)
    except ValueError as exc:
        metrics["relative_l2"] = f"undefined: {exc}"
    try:
        metrics["normalized_rmse"] = normalized_rmse(y_true, y_pred)
    except ValueError as exc:
        metrics["normalized_rmse"] = f"undefined: {exc}"
    return metrics
