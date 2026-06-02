"""Phase 53 source-path data pivot gate.

This read-only gate decides whether the current AMB2022-03 mds2-2716 bundle
contains a source-path object that is safely aligned to the thermography table
used by the source-inversion branch. It deliberately separates paper-facing
registration evidence from diagnostic-only pad experiments that require
independent coordinate rescaling.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_DATASET_ROOT = Path("data/raw/ambench/2022_single_track/AMB2022-03/mds2-2716")
DEFAULT_THERMAL_HDF5 = DEFAULT_DATASET_ROOT / "Thermography/AMB2022-03-718-AMMT-StaringCamera_Signal.h5"
DEFAULT_SCAN_ROOT = DEFAULT_DATASET_ROOT / "ScanStrategy"


def _h5py() -> Any:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("h5py is required to inspect AM-Bench HDF5 files") from exc
    return h5py


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


def _attrs(obj: Any) -> dict[str, Any]:
    return {str(key): _jsonable(value) for key, value in obj.attrs.items()}


def inspect_thermal_hdf5(path: Path) -> dict[str, Any]:
    h5py = _h5py()
    groups: list[dict[str, Any]] = []
    registration_attr_keys: set[str] = set()
    with h5py.File(path, "r") as handle:
        thermal = handle["ThermalData"]
        _collect_registration_attr_keys(handle, registration_attr_keys)
        for name in sorted(thermal.keys()):
            group = thermal[name]
            _collect_registration_attr_keys(group, registration_attr_keys)
            children: dict[str, Any] = {}
            for child_name in sorted(group.keys()):
                child = group[child_name]
                if hasattr(child, "shape"):
                    _collect_registration_attr_keys(child, registration_attr_keys)
                    children[child_name] = {
                        "path": f"ThermalData/{name}/{child_name}",
                        "shape": [int(value) for value in child.shape],
                        "dtype": str(child.dtype),
                        "attrs": _attrs(child),
                    }
            groups.append(
                {
                    "name": name,
                    "category": _thermal_category(name),
                    "attrs": _attrs(group),
                    "children": children,
                }
            )
    categories = {
        category: sum(1 for item in groups if item["category"] == category)
        for category in ["line", "pad", "other"]
    }
    return {
        "path": str(path),
        "groups": groups,
        "categories": categories,
        "line_groups": [item["name"] for item in groups if item["category"] == "line"],
        "pad_groups": [item["name"] for item in groups if item["category"] == "pad"],
        "registration_attr_keys": sorted(registration_attr_keys),
    }


def _thermal_category(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("line_"):
        return "line"
    if re.match(r"^[xy]_pad", lowered):
        return "pad"
    return "other"


def _collect_registration_attr_keys(obj: Any, output: set[str]) -> None:
    pattern = re.compile(r"(galvo|pixel|camera|coord|register|homography|affine|scale|offset)", re.IGNORECASE)
    for key in obj.attrs.keys():
        if pattern.search(str(key)):
            output.add(str(key))


def inspect_scan_strategy_root(path: Path) -> dict[str, Any]:
    files = sorted(path.rglob("*.h5")) if path.exists() else []
    return {
        "path": str(path),
        "hdf5_files": [inspect_scan_strategy_file(file_path) for file_path in files],
    }


def inspect_scan_strategy_file(path: Path) -> dict[str, Any]:
    h5py = _h5py()
    groups: list[str] = []
    datasets: dict[str, dict[str, Any]] = {}
    with h5py.File(path, "r") as handle:
        root_keys = sorted(handle.keys())

        def visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Group):
                groups.append(name)
            elif isinstance(obj, h5py.Dataset):
                datasets[name] = {
                    "shape": [int(value) for value in obj.shape],
                    "dtype": str(obj.dtype),
                }

        handle.visititems(visit)
    xypt_groups = [name for name in groups if name.startswith("XYPT/")]
    line_like_groups = [
        name
        for name in groups
        if _looks_like_single_track_scan_object(name)
    ]
    return {
        "path": str(path),
        "name": path.name,
        "root_keys": root_keys,
        "groups": groups,
        "datasets": datasets,
        "xypt_groups": xypt_groups,
        "line_like_groups": line_like_groups,
        "pad_xypt_only": bool(xypt_groups) and not line_like_groups and all("pad" in name.lower() for name in xypt_groups),
    }


def _looks_like_single_track_scan_object(name: str) -> bool:
    lowered = name.lower()
    if "pad" in lowered:
        return False
    return "line" in lowered or "track" in lowered


def pad_xypt_matches(thermal: dict[str, Any], scan_root: dict[str, Any]) -> list[dict[str, Any]]:
    xypt_groups = {
        group
        for file_payload in scan_root["hdf5_files"]
        for group in file_payload.get("xypt_groups", [])
    }
    matches: list[dict[str, Any]] = []
    for pad_name in thermal["pad_groups"]:
        normalized = pad_name.lower().replace("_", "")
        if normalized.startswith("xpad"):
            xypt = "XYPT/Xpad"
        elif normalized.startswith("ypad"):
            xypt = "XYPT/Ypad"
        else:
            xypt = None
        matches.append(
            {
                "thermal_group": pad_name,
                "dataset_path": f"ThermalData/{pad_name}/Signal",
                "xypt_group": xypt if xypt in xypt_groups else None,
                "matched": bool(xypt and xypt in xypt_groups),
                "diagnostic_only_reason": (
                    "requires independent camera-pixel and galvo-mm rescaling unless an external registration is supplied"
                ),
            }
        )
    return matches


def decision_payload(thermal: dict[str, Any], scan_root: dict[str, Any], matches: list[dict[str, Any]]) -> dict[str, Any]:
    scan_files = scan_root["hdf5_files"]
    has_single_track_scan = any(file_payload.get("line_like_groups") for file_payload in scan_files)
    has_pad_match = any(match["matched"] for match in matches)
    has_registration_attrs = bool(thermal["registration_attr_keys"])
    paper_ready = has_single_track_scan or (has_pad_match and has_registration_attrs)
    if paper_ready:
        reason = "aligned source-path evidence is available"
    elif has_pad_match:
        reason = (
            "scan strategy exposes pad XYPT only; thermography has pad tables, "
            "but no HDF5 camera-pixel to galvo-mm registration metadata was found"
        )
    else:
        reason = "no aligned single-track source path or pad XYPT/thermography match was found"
    return {
        "status": "positive" if paper_ready else "negative",
        "paper_facing_source_path_ready": bool(paper_ready),
        "single_track_scan_path_available": bool(has_single_track_scan),
        "pad_registered_diagnostic_available": bool(has_pad_match),
        "hdf5_registration_metadata_available": bool(has_registration_attrs),
        "source_inversion_broad12_broad21_blocked": not bool(paper_ready),
        "reason": reason,
        "recommended_next": (
            "run the registered source-path gate on aligned data"
            if paper_ready
            else "use pad runs only as diagnostic rescale probes, then pivot to the stronger process-conditioned broad-data route guard unless external registration is obtained"
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    thermal = inspect_thermal_hdf5(args.thermal_hdf5)
    scan_root = inspect_scan_strategy_root(args.scan_root)
    matches = pad_xypt_matches(thermal, scan_root)
    return {
        "mode": "phase53_source_path_data_pivot",
        "dataset_root": str(args.dataset_root),
        "thermal": thermal,
        "scan_strategy": scan_root,
        "pad_xypt_matches": matches,
        "decision": decision_payload(thermal, scan_root, matches),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--thermal-hdf5", type=Path, default=DEFAULT_THERMAL_HDF5)
    parser.add_argument("--scan-root", type=Path, default=DEFAULT_SCAN_ROOT)
    parser.add_argument("--json-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text, encoding="utf-8")
        print(f"Wrote: {args.json_output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
