#!/usr/bin/env python3
"""Create a streamed, no-training preflight package for AMB2022-01.

The package records what is available locally, fixes the B6/B7 development and
B8 held-out split, and explicitly leaves coordinate/time registration as a
separate gate.  It never materializes the full XYPT HDF5 dataset.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "AMB2022_01_DATA_ROOT",
        "/root/matsci-gnn-pinn-data/raw/ambench/2022_3d_build/AMB2022-01/mds2-2607",
    )
)
THERMOCOUPLE_CHANNELS = {"B6": ("P2", "P3", "Chamber"), "B7": ("P2", "P3", "Chamber"), "B8": ("P2", "Chamber")}
DOWNSTREAM_RECORDS = ("mds2-2692", "mds2-2711", "mds2-2767")


def _json_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "tolist"):
        value = value.tolist()
    return str(value)


def summarize_h5(path: Path, *, max_datasets: int) -> dict[str, Any]:
    """Read HDF5 metadata only; dataset values are never materialized."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        return {"available": False, "reason": f"h5py unavailable: {exc}"}

    datasets: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        def visit(name: str, node: Any) -> None:
            if not isinstance(node, h5py.Dataset) or len(datasets) >= max_datasets:
                return
            datasets.append(
                {
                    "path": name,
                    "shape": list(node.shape),
                    "dtype": str(node.dtype),
                    "ndim": int(node.ndim),
                    "attrs": {str(key): _json_value(value) for key, value in node.attrs.items()},
                }
            )

        handle.visititems(visit)
        return {
            "available": True,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "root_keys": sorted(handle.keys()),
            "root_attrs": {str(key): _json_value(value) for key, value in handle.attrs.items()},
            "dataset_count_reported": len(datasets),
            "dataset_scan_capped": len(datasets) >= max_datasets,
            "datasets": datasets,
        }


def summarize_thermocouple_csv(path: Path) -> dict[str, Any]:
    """Stream a thermocouple CSV and calculate per-column numeric ranges."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        ranges: dict[str, list[float | None]] = {field: [None, None] for field in fields}
        row_count = 0
        for row in reader:
            row_count += 1
            for field, raw in row.items():
                try:
                    value = float(raw) if raw is not None else math.nan
                except ValueError:
                    continue
                if not math.isfinite(value):
                    continue
                low, high = ranges[field]
                ranges[field] = [value if low is None or value < low else low, value if high is None or value > high else high]
    numeric_ranges = {field: bounds for field, bounds in ranges.items() if bounds[0] is not None}
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "fields": fields,
        "row_count": row_count,
        "numeric_ranges": numeric_ranges,
    }


def build_split_rows(data_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for build in ("B6", "B7", "B8"):
        filename = f"AMB2022-01-AMMT-{build}-Thermocouple.csv"
        is_holdout = build == "B8"
        rows.append(
            {
                "build_id": build,
                "role": "held_out_real_build" if is_holdout else "development_build",
                "split": "test_frozen" if is_holdout else "development_only",
                "thermocouple_file": str(data_root / "Thermocouples" / filename),
                "temperature_channels": ";".join(THERMOCOUPLE_CHANNELS[build]),
                "xypt_file": str(data_root / "Scan_Strategy" / "AMB2022-01-AMMT-XYPT_v1.h5"),
                "coordinate_time_mapping_state": "not_yet_registered",
                "future_information_allowed": "false",
                "model_selection_allowed": "false" if is_holdout else "true",
            }
        )
    return rows


def fetch_downstream_metadata(record_id: str) -> dict[str, Any]:
    url = f"https://data.nist.gov/od/id/{record_id}"
    with urllib.request.urlopen(url, timeout=20) as response:
        payload = json.load(response)
    return {
        "record_id": record_id,
        "record_url": url,
        "doi": payload.get("doi"),
        "title": payload.get("title"),
        "license": payload.get("license"),
        "component_count": len(payload.get("components", [])),
        "download_deferred": True,
        "reason": "sample_and_coordinate_join_required_before_large_downstream_download",
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_protocol(path: Path, *, split_rows: list[dict[str, str]], preflight: dict[str, Any]) -> None:
    lines = [
        "# AMB2022-01 CPU Preflight Protocol",
        "",
        "## Frozen Split",
        "",
        "- B6 and B7 are development-only builds.",
        "- B8 is a frozen real-build holdout: no model selection, normalization fit, or threshold tuning may use it.",
        "- XYPT is an input history source, not a temperature label.",
        "",
        "## Mapping Gate",
        "",
        "1. Identify the thermocouple clock origin and units from the README and HDF5 metadata.",
        "2. Register build, layer, scan segment, coordinate frame, and time with an auditable mapping table.",
        "3. Reject any feature needing future scan commands relative to a thermocouple observation.",
        "4. Run shuffled-history and no-history negative controls before GPU training.",
        "",
        "## Preflight State",
        "",
        f"- HDF5 metadata available: `{preflight['h5']['available']}`",
        f"- Thermocouple files found: `{len(preflight['thermocouples'])}/3`",
        f"- Training allowed now: `{preflight['gate']['model_training_allowed']}`",
        "",
        "## Split Rows",
        "",
    ]
    for row in split_rows:
        lines.append(f"- {row['build_id']}: {row['split']} ({row['temperature_channels']})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_preflight(data_root: Path, *, include_downstream_metadata: bool, max_h5_datasets: int) -> tuple[dict[str, Any], list[dict[str, str]]]:
    h5_path = data_root / "Scan_Strategy" / "AMB2022-01-AMMT-XYPT_v1.h5"
    h5 = summarize_h5(h5_path, max_datasets=max_h5_datasets) if h5_path.exists() else {"available": False, "reason": "XYPT file missing"}
    thermocouples: dict[str, Any] = {}
    for build in ("B6", "B7", "B8"):
        path = data_root / "Thermocouples" / f"AMB2022-01-AMMT-{build}-Thermocouple.csv"
        thermocouples[build] = summarize_thermocouple_csv(path) if path.exists() else {"available": False, "reason": "file missing"}
    downstream = [fetch_downstream_metadata(record) for record in DOWNSTREAM_RECORDS] if include_downstream_metadata else []
    split_rows = build_split_rows(data_root)
    all_thermocouples = all("row_count" in item for item in thermocouples.values())
    gate = {
        "status": "cpu_preflight_ready_mapping_required" if h5.get("available") and all_thermocouples else "cpu_preflight_incomplete",
        "model_training_allowed": False,
        "a100_training_allowed_now": False,
        "held_out_build": "B8",
        "next_action": "create audited thermocouple-time-to-XYPT coordinate mapping before GPU training",
    }
    return {
        "phase": 179,
        "generated_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "data_root": str(data_root),
        "h5": h5,
        "thermocouples": thermocouples,
        "downstream_metadata": downstream,
        "gate": gate,
    }, split_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-h5-datasets", type=int, default=128)
    parser.add_argument("--skip-downstream-metadata", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preflight, split_rows = build_preflight(
        args.data_root,
        include_downstream_metadata=not args.skip_downstream_metadata,
        max_h5_datasets=args.max_h5_datasets,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "phase179_amb2022_01_cpu_preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
    _write_csv(args.output_dir / "phase179_amb2022_01_split_manifest.csv", split_rows)
    _write_protocol(args.output_dir / "phase179_amb2022_01_cpu_preflight_protocol.md", split_rows=split_rows, preflight=preflight)
    print(json.dumps(preflight["gate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
