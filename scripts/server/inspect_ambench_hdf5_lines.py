"""Inspect AM-Bench thermography HDF5 line datasets on the server.

This is intentionally read-only. It prints a compact JSON summary of
ThermalData/* groups so follow-up runs can choose physically meaningful
multi-line/process-conditioned subsets without fragile shell quoting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


THERMAL_HDF5 = Path(
    "data/raw/ambench/2022_single_track/AMB2022-03/mds2-2716/"
    "Thermography/AMB2022-03-718-AMMT-StaringCamera_Signal.h5"
)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def inspect_lines(path: Path = THERMAL_HDF5) -> list[dict[str, Any]]:
    import h5py

    with h5py.File(path, "r") as handle:
        thermal_data = handle["ThermalData"]
        summary: list[dict[str, Any]] = []
        for line_name in sorted(thermal_data.keys()):
            line_group = thermal_data[line_name]
            line_payload = {
                "line": line_name,
                "attrs": {key: _jsonable(value) for key, value in line_group.attrs.items()},
                "datasets": {},
            }
            for dataset_name in sorted(line_group.keys()):
                dataset = line_group[dataset_name]
                line_payload["datasets"][dataset_name] = {
                    "path": f"ThermalData/{line_name}/{dataset_name}",
                    "shape": list(dataset.shape),
                    "dtype": str(dataset.dtype),
                    "attrs": {key: _jsonable(value) for key, value in dataset.attrs.items()},
                }
            summary.append(line_payload)
    return summary


def main() -> int:
    print(json.dumps(inspect_lines(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
