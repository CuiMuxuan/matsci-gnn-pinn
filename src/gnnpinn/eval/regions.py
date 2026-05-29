"""Region-aware metric helpers for field predictions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from gnnpinn.eval.baselines import regression_metric_table


def region_metric_tables(
    sample: Any,
    target: str,
    y_pred: Sequence[float],
    indices: list[int] | None = None,
    hot_quantiles: list[float] | None = None,
    gradient_quantiles: list[float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute metrics for target-defined hot zones and gradient bands."""

    active_indices = indices or list(range(sample.n_points))
    y_true_all = sample.require_observation(target)
    output: dict[str, dict[str, Any]] = {}
    for quantile in hot_quantiles or []:
        selected = _quantile_region_indices(y_true_all, active_indices, quantile, high=True)
        output[f"hot_q{_quantile_label(quantile)}"] = _region_payload(
            y_true_all,
            y_pred,
            selected,
            selector={"kind": "target_quantile", "target": target, "quantile": quantile},
        )
    if gradient_quantiles:
        scores = _spatial_gradient_scores(sample, y_true_all)
        for quantile in gradient_quantiles:
            selected = _quantile_region_indices(scores, active_indices, quantile, high=True)
            output[f"gradient_q{_quantile_label(quantile)}"] = _region_payload(
                y_true_all,
                y_pred,
                selected,
                selector={"kind": "spatial_gradient_quantile", "target": target, "quantile": quantile},
            )
    return output


def _region_payload(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    indices: list[int],
    selector: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "n_points": len(indices),
        "selector": selector,
    }
    if indices:
        payload["metrics"] = regression_metric_table(
            [y_true[index] for index in indices],
            [y_pred[index] for index in indices],
        )
    else:
        payload["metrics"] = {}
    return payload


def _quantile_region_indices(values: Sequence[float], indices: list[int], quantile: float, high: bool) -> list[int]:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")
    if not indices:
        return []
    selected_values = [float(values[index]) for index in indices]
    threshold = _quantile(selected_values, quantile)
    if high:
        return [index for index in indices if float(values[index]) >= threshold]
    return [index for index in indices if float(values[index]) <= threshold]


def _quantile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _quantile_label(quantile: float) -> str:
    return str(int(round(quantile * 100))).zfill(2)


def _spatial_gradient_scores(sample: Any, values: Sequence[float]) -> list[float]:
    frame_values = sample.observations.get("frame_index")
    row_values = sample.observations.get("row_index")
    col_values = sample.observations.get("col_index")
    if row_values is None or col_values is None:
        coord_columns = list(sample.metadata.get("coordinate_columns") or [])
        if "y" in coord_columns and "x" in coord_columns:
            row_pos = coord_columns.index("y")
            col_pos = coord_columns.index("x")
            row_values = [row[row_pos] for row in sample.coordinates]
            col_values = [row[col_pos] for row in sample.coordinates]
        else:
            return [0.0 for _ in range(sample.n_points)]

    frames = frame_values if frame_values is not None else [0.0 for _ in range(sample.n_points)]
    groups: dict[float, list[int]] = {}
    for index, frame in enumerate(frames):
        groups.setdefault(float(frame), []).append(index)

    scores = [0.0 for _ in range(sample.n_points)]
    for group_indices in groups.values():
        rows = sorted({float(row_values[index]) for index in group_indices})
        cols = sorted({float(col_values[index]) for index in group_indices})
        row_neighbors = _axis_neighbors(rows)
        col_neighbors = _axis_neighbors(cols)
        by_position = {
            (float(row_values[index]), float(col_values[index])): index
            for index in group_indices
        }
        for index in group_indices:
            row = float(row_values[index])
            col = float(col_values[index])
            local_scores: list[float] = []
            for neighbor_row in row_neighbors.get(row, []):
                neighbor = by_position.get((neighbor_row, col))
                if neighbor is not None:
                    distance = abs(neighbor_row - row) or 1.0
                    local_scores.append(abs(float(values[index]) - float(values[neighbor])) / distance)
            for neighbor_col in col_neighbors.get(col, []):
                neighbor = by_position.get((row, neighbor_col))
                if neighbor is not None:
                    distance = abs(neighbor_col - col) or 1.0
                    local_scores.append(abs(float(values[index]) - float(values[neighbor])) / distance)
            if local_scores:
                scores[index] = max(local_scores)
    return scores


def _axis_neighbors(values: list[float]) -> dict[float, list[float]]:
    neighbors: dict[float, list[float]] = {}
    for position, value in enumerate(values):
        current: list[float] = []
        if position > 0:
            current.append(values[position - 1])
        if position + 1 < len(values):
            current.append(values[position + 1])
        neighbors[value] = current
    return neighbors
