"""Inspect local raw tables and draft an explicit field-table mapping."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


def inspect_table(path: str | Path, sample_rows: int = 5) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        rows = _read_csv(path)
    elif path.suffix.lower() == ".json":
        rows = _read_json(path)
    else:
        raise ValueError(f"Unsupported table extension for inspection: {path.suffix}")
    if not rows:
        raise ValueError(f"Table is empty: {path}")
    columns = list(rows[0].keys())
    numeric_columns = [column for column in columns if _is_numeric_column(rows, column)]
    return {
        "path": str(path),
        "n_rows": len(rows),
        "columns": columns,
        "numeric_columns": numeric_columns,
        "sample_rows": rows[:sample_rows],
    }


def draft_mapping(
    inspection: dict[str, Any],
    output: str | Path,
    sample_id: str,
    dataset_id: str = "ambench_2022_single_track",
) -> dict[str, Any]:
    columns = inspection["columns"]
    numeric = set(inspection["numeric_columns"])
    guessed = {
        "x": _guess(columns, numeric, ("x", "x_mm", "X_mm", "position_x", "X")),
        "y": _guess(columns, numeric, ("y", "y_mm", "Y_mm", "position_y", "Y")),
        "z": _guess(columns, numeric, ("z", "z_mm", "Z_mm", "position_z", "Z")),
        "t": _guess(columns, numeric, ("t", "time", "time_s", "Time", "Time_s")),
    }
    guessed = {key: value for key, value in guessed.items() if value}
    target = _guess(columns, numeric, ("T", "temperature", "temperature_K", "Temperature", "temp"))
    observations = {"T": target} if target else {}
    output = Path(output)
    return {
        "dataset_id": dataset_id,
        "sample_id": sample_id,
        "source": inspection["path"],
        "output": str(output),
        "split_manifest": str(output.parent.parent / "splits" / f"{sample_id}_split.json"),
        "columns": guessed,
        "observations": observations,
        "process_parameters": {},
        "splits": {
            "train_fraction": 0.7,
            "val_fraction": 0.15,
            "test_fraction": 0.15,
            "seed": 7,
        },
        "metadata": {
            "mapping_status": "draft",
            "action_required": "Verify guessed column names against the AM-Bench source file before conversion.",
        },
    }


def _guess(columns: list[str], numeric: set[str], candidates: tuple[str, ...]) -> str | None:
    lower_map = {column.lower(): column for column in columns}
    for candidate in candidates:
        exact = lower_map.get(candidate.lower())
        if exact and exact in numeric:
            return exact
    for column in columns:
        if column in numeric:
            lowered = column.lower()
            if any(candidate.lower() in lowered for candidate in candidates):
                return column
    return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "rows" in data:
        data = data["rows"]
    if not isinstance(data, list):
        raise ValueError("JSON table must be a list of rows or contain a 'rows' list")
    return data


def _is_numeric_column(rows: list[dict[str, Any]], column: str) -> bool:
    for row in rows:
        value = row.get(column)
        if value in ("", None):
            continue
        try:
            float(value)
        except (TypeError, ValueError):
            return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True, type=Path, help="Raw CSV/JSON table to inspect.")
    parser.add_argument("--sample-id", default="ambench_sample", help="Sample id for mapping draft.")
    parser.add_argument(
        "--draft-output",
        type=Path,
        help="Optional path for a mapping YAML draft. Inspection JSON is always printed.",
    )
    parser.add_argument(
        "--field-output",
        type=Path,
        default=Path("data/processed/ambench/2022_single_track/field_tables/ambench_sample.csv"),
        help="Field-table output path to place in the mapping draft.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inspection = inspect_table(args.table)
    print(json.dumps(inspection, indent=2, ensure_ascii=False))
    if args.draft_output:
        mapping = draft_mapping(inspection, output=args.field_output, sample_id=args.sample_id)
        args.draft_output.parent.mkdir(parents=True, exist_ok=True)
        args.draft_output.write_text(yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"Wrote: {args.draft_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

