"""Load simple local field-observation tables for early baselines."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from gnnpinn.data.schemas import FieldSample


COORD_COLUMNS = ("x", "y", "z")
TIME_COLUMNS = ("t", "time")


def load_field_table(
    path: str | Path,
    observation_columns: list[str] | None = None,
    sample_id: str | None = None,
) -> FieldSample:
    """Load a CSV or JSON field table.

    CSV columns should include coordinate columns such as `x`, `y`, optional
    `z`, and optional time column `t` or `time`. Remaining numeric columns are
    treated as observations when `observation_columns` is not provided.
    """

    path = Path(path)
    if path.suffix.lower() == ".csv":
        rows = _read_csv_rows(path)
    elif path.suffix.lower() == ".json":
        rows = _read_json_rows(path)
    else:
        raise ValueError(f"Unsupported field table extension: {path.suffix}")

    if not rows:
        raise ValueError(f"Field table is empty: {path}")

    coordinate_columns = [name for name in COORD_COLUMNS if name in rows[0]]
    if not coordinate_columns:
        raise ValueError("Field table must contain at least one coordinate column among x, y, z")

    time_column = next((name for name in TIME_COLUMNS if name in rows[0]), None)
    if observation_columns is None:
        excluded = set(coordinate_columns)
        if time_column:
            excluded.add(time_column)
        observation_columns = [
            column
            for column in rows[0]
            if column not in excluded and _is_float_like(rows[0][column])
        ]

    coordinates = [
        [_to_float(row[column], column) for column in coordinate_columns]
        for row in rows
    ]
    time = [
        _to_float(row[time_column], time_column) if time_column else 0.0
        for row in rows
    ]
    observations = {
        column: [_to_float(row[column], column) for row in rows]
        for column in observation_columns
    }

    return FieldSample(
        sample_id=sample_id or path.stem,
        source_path=path,
        coordinates=coordinates,
        time=time,
        observations=observations,
        metadata={
            "coordinate_columns": coordinate_columns,
            "time_column": time_column,
            "observation_columns": observation_columns,
        },
    )


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "rows" in data:
        data = data["rows"]
    if not isinstance(data, list):
        raise ValueError("JSON field table must be a list of rows or contain a 'rows' list")
    return data


def _is_float_like(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _to_float(value: Any, column: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Column {column!r} contains non-numeric value: {value!r}") from exc

