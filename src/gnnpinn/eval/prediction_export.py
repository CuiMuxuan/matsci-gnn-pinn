"""Row-aligned prediction export helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence


def split_name_by_index(split_manifest: dict[str, Any] | None, n_points: int) -> list[str]:
    """Return one split label per row index."""

    labels = ["all" for _ in range(n_points)]
    if split_manifest is None:
        return labels
    for split_name, indices in split_manifest.get("splits", {}).items():
        for index in indices:
            labels[int(index)] = str(split_name)
    return labels


def write_prediction_csv(
    path: str | Path,
    *,
    sample: Any,
    target: str,
    y_true: Sequence[float],
    y_pred: Sequence[float],
    split_manifest: dict[str, Any] | None = None,
    method: str = "",
) -> None:
    """Write row-aligned predictions for stack/probe analysis."""

    if len(y_true) != len(y_pred):
        raise ValueError("Prediction export requires y_true and y_pred to have the same length")
    if len(y_true) != sample.n_points:
        raise ValueError("Prediction export length must match sample.n_points")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coordinate_columns = list(sample.metadata.get("coordinate_columns") or [])
    time_column = sample.metadata.get("time_column") or "time"
    split_labels = split_name_by_index(split_manifest, sample.n_points)
    row_metadata = sample.metadata.get("row_metadata", {})
    metadata_columns = sorted(str(column) for column in row_metadata)
    fieldnames = [
        "row_index",
        "split",
        "sample_id",
        "method",
        *coordinate_columns,
        time_column,
        target,
        "prediction",
        "error",
        "abs_error",
        *metadata_columns,
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, (truth, prediction) in enumerate(zip(y_true, y_pred)):
            error = float(prediction) - float(truth)
            row = {
                "row_index": row_index,
                "split": split_labels[row_index],
                "sample_id": sample.sample_id,
                "method": method,
                time_column: float(sample.time[row_index]),
                target: float(truth),
                "prediction": float(prediction),
                "error": error,
                "abs_error": abs(error),
            }
            for coord_index, column in enumerate(coordinate_columns):
                row[column] = float(sample.coordinates[row_index][coord_index])
            for column in metadata_columns:
                row[column] = row_metadata[column][row_index]
            writer.writerow(row)
