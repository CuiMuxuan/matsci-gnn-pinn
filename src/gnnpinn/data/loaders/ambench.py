"""AM-Bench adapter from explicitly mapped raw tables to field-table schema."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import yaml


STANDARD_COLUMNS = ("x", "y", "z", "t")


def load_mapping(path: str | Path) -> dict[str, Any]:
    mapping_path = Path(path)
    data = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Mapping must be a YAML object: {mapping_path}")
    for key in ("source", "output", "columns", "observations"):
        if key not in data:
            raise ValueError(f"Mapping missing required key {key!r}: {mapping_path}")
    return data


def convert_mapped_table(mapping: dict[str, Any]) -> dict[str, Any]:
    source = Path(mapping["source"])
    output = Path(mapping["output"])
    split_manifest = Path(mapping["split_manifest"]) if mapping.get("split_manifest") else None
    rows = _read_csv(source)
    if not rows:
        raise ValueError(f"Source table is empty: {source}")

    converted = [_convert_row(row, mapping) for row in rows]
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output, converted)

    manifest = {
        "dataset_id": mapping.get("dataset_id"),
        "sample_id": mapping.get("sample_id", output.stem),
        "source": str(source),
        "output": str(output),
        "n_rows": len(converted),
        "columns": list(converted[0].keys()),
        "process_parameters": mapping.get("process_parameters", {}),
        "metadata": mapping.get("metadata", {}),
    }
    if split_manifest:
        split_payload = build_split_manifest(
            n_rows=len(converted),
            sample_id=manifest["sample_id"],
            split_config=mapping.get("splits", {}),
        )
        split_manifest.parent.mkdir(parents=True, exist_ok=True)
        split_manifest.write_text(json.dumps(split_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest["split_manifest"] = str(split_manifest)
    return manifest


def build_split_manifest(n_rows: int, sample_id: str, split_config: dict[str, Any]) -> dict[str, Any]:
    train_fraction = float(split_config.get("train_fraction", 0.7))
    val_fraction = float(split_config.get("val_fraction", 0.15))
    test_fraction = float(split_config.get("test_fraction", 0.15))
    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Split fractions must sum to 1.0")

    indices = list(range(n_rows))
    rng = random.Random(int(split_config.get("seed", 7)))
    rng.shuffle(indices)
    train_end = int(round(train_fraction * n_rows))
    val_end = train_end + int(round(val_fraction * n_rows))
    return {
        "sample_id": sample_id,
        "n_rows": n_rows,
        "seed": int(split_config.get("seed", 7)),
        "splits": {
            "train": sorted(indices[:train_end]),
            "val": sorted(indices[train_end:val_end]),
            "test": sorted(indices[val_end:]),
        },
    }


def _convert_row(row: dict[str, str], mapping: dict[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for standard, source_col in mapping["columns"].items():
        if standard not in STANDARD_COLUMNS:
            raise ValueError(f"Unsupported standard coordinate/time column: {standard}")
        converted[standard] = _float(row, source_col)
    for standard, source_col in mapping["observations"].items():
        converted[standard] = _float(row, source_col)
    for name, value in (mapping.get("process_parameters") or {}).items():
        converted[name] = value
    return converted


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _float(row: dict[str, str], column: str) -> float:
    if column not in row:
        raise KeyError(f"Source column not found: {column}")
    return float(row[column])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", required=True, type=Path, help="YAML mapping file.")
    parser.add_argument("--manifest", type=Path, help="Optional conversion manifest JSON output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mapping = load_mapping(args.mapping)
    manifest = convert_mapped_table(mapping)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote: {args.manifest}")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

