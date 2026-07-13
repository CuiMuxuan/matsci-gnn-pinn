#!/usr/bin/env python3
"""Audit the registered AMB2022-01 trigger-layer-space supervision route.

Phase 180 correctly blocks a claim that the XYPT trigger vector alone is an
absolute wall-clock schedule for the full build.  This gate tests a narrower,
but auditable, route: per-layer XYPT commands and StaringCamera triggers are
joined to NIST's spatially registered TAM/SCR fields.  Passing this gate
permits dataset construction only; it does not permit a raw-frame causal or
absolute-time trajectory claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SCAN_ROOT = Path(
    os.environ.get(
        "AMB2022_01_DATA_ROOT",
        "/root/matsci-gnn-pinn-data/raw/ambench/2022_3d_build/AMB2022-01/mds2-2607",
    )
)
DEFAULT_THERMOGRAPHY_ROOT = Path(
    os.environ.get(
        "AMB2022_01_THERMOGRAPHY_ROOT",
        "/root/matsci-gnn-pinn-data/raw/ambench/2022_3d_build/AMB2022-01/"
        "mds2-2715/official",
    )
)
DEFAULT_FILE_LIST = DEFAULT_THERMOGRAPHY_ROOT / "_filelisting.csv"
SCAN_FILENAME = "AMB2022-01-AMMT-XYPT_v1.h5"
SCAN_REFERENCE_MARKER = "mds2-2607/Scan_Strategy/AMB2022-01-AMMT-XYPT_v1.h5"
BUILD_IDS = ("B6", "B7", "B8")
TARGET_NAMES = ("TAM", "SCR")


def _attr_text(value: Any) -> str:
    """Return a stable text representation for scalar HDF5 attributes."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _attr_text(value[0])
        return ",".join(_attr_text(item) for item in value)
    return str(value)


def _attr_float(value: Any) -> float:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected scalar attribute, received {value!r}")
        value = value[0]
    return float(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pdr_file_list(path: Path) -> dict[str, dict[str, Any]]:
    """Read the NIST PDR CSV while ignoring its comment-prefixed header."""
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        row = next(csv.reader([line]))
        if len(row) != 6:
            raise ValueError(f"Unexpected PDR file-list row: {line!r}")
        relative_path, size, file_type, mime_type, sha256, url = row
        records[relative_path] = {
            "byte_size": int(size),
            "file_type": file_type,
            "mime_type": mime_type,
            "sha256": sha256,
            "source_url": url,
        }
    if not records:
        raise ValueError(f"No files found in PDR file list: {path}")
    return records


def _dataset_lengths(group: Any, names: Iterable[str]) -> dict[str, int]:
    lengths: dict[str, int] = {}
    for name in names:
        if name not in group:
            raise KeyError(f"Missing XYPT dataset {group.name}/{name}")
        shape = tuple(int(value) for value in group[name].shape)
        if len(shape) == 1:
            lengths[name] = shape[0]
        elif len(shape) == 2 and 1 in shape:
            # MATLAB HDF5 exports vectors as either 1-by-n or n-by-1 arrays.
            lengths[name] = max(shape)
        else:
            raise ValueError(f"Expected vector-shaped XYPT dataset {group.name}/{name}, got {shape}")
    return lengths


def inspect_scan_strategy(path: Path) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("h5py is required for the trigger-layer-space gate") from exc

    with h5py.File(path, "r") as handle:
        xypt = handle["XYPT"]
        layer_ids = sorted((int(value) for value in xypt.keys()))
        contiguous_layers = layer_ids == list(range(1, len(layer_ids) + 1))
        digital_rate = _attr_float(xypt.attrs["digital_rate"])
        trigger_bit2 = _attr_text(xypt.attrs.get("Trigger_bit2", ""))
        x_units = _attr_text(handle["Calibration"].attrs.get("X_units", ""))
        y_units = _attr_text(handle["Calibration"].attrs.get("Y_units", ""))
        layer_height_um = _attr_float(handle["AMB2022-01-AMMT"].attrs["layerthickness"])
        sample_count = 0
        vectors_aligned = True
        malformed_layers: list[int] = []
        for layer_id in layer_ids:
            lengths = _dataset_lengths(xypt[str(layer_id)], ("X", "Y", "P", "T"))
            sample_count += lengths["P"]
            if len(set(lengths.values())) != 1:
                vectors_aligned = False
                malformed_layers.append(layer_id)

    trigger_layer_time_ready = (
        contiguous_layers
        and digital_rate > 0.0
        and trigger_bit2.strip().lower() == "staringcamera"
        and x_units.strip().lower() == "mm"
        and y_units.strip().lower() == "mm"
        and vectors_aligned
    )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "digital_rate_hz": digital_rate,
        "seconds_per_command": 1.0 / digital_rate if digital_rate > 0.0 else None,
        "trigger_bit2": trigger_bit2,
        "layer_ids": layer_ids,
        "layer_count": len(layer_ids),
        "layer_ids_contiguous_from_one": contiguous_layers,
        "layer_height_um": layer_height_um,
        "x_units": x_units,
        "y_units": y_units,
        "xypt_sample_count": sample_count,
        "xypt_vectors_aligned": vectors_aligned,
        "malformed_layers": malformed_layers,
        "trigger_layer_time_ready": trigger_layer_time_ready,
        "interpretation": (
            "XYPT command index supplies a per-layer 1/digital_rate time coordinate, and bit 2 "
            "is the StaringCamera trigger."
            if trigger_layer_time_ready
            else "XYPT trigger/layer evidence is incomplete or internally inconsistent."
        ),
    }


def _target_relative_path(build_id: str, target_name: str) -> str:
    return f"Staring_Thermography/AMB2022-01-718-AMMT-{build_id}-StaringCamera_{target_name}.h5"


def inspect_target(
    *,
    path: Path,
    build_id: str,
    target_name: str,
    expected: dict[str, Any] | None,
    expected_layer_count: int,
) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("h5py is required for the trigger-layer-space gate") from exc

    result: dict[str, Any] = {
        "build_id": build_id,
        "target_name": target_name,
        "path": str(path),
        "exists": path.exists(),
        "manifest_entry_present": expected is not None,
    }
    if not path.exists():
        result["ready"] = False
        result["blockers"] = ["target_file_missing"]
        return result

    actual_size = path.stat().st_size
    actual_sha256 = _sha256(path)
    expected_size = int(expected["byte_size"]) if expected else None
    expected_sha256 = str(expected["sha256"]) if expected else None
    size_match = expected_size is not None and actual_size == expected_size
    checksum_match = expected_sha256 is not None and actual_sha256 == expected_sha256
    result.update(
        {
            "byte_size": actual_size,
            "sha256": actual_sha256,
            "expected_byte_size": expected_size,
            "expected_sha256": expected_sha256,
            "byte_size_match": size_match,
            "sha256_match": checksum_match,
            "source_url": expected.get("source_url") if expected else None,
        }
    )

    blockers: list[str] = []
    with h5py.File(path, "r") as handle:
        build_group_name = f"AMB2022-01-718-AMMT-{build_id}"
        if build_group_name not in handle:
            raise KeyError(f"Missing build group {build_group_name} in {path}")
        build_group = handle[build_group_name]
        registration = handle["Calibration/Registration"]
        thermal = handle["ThermalData"]
        data_name = target_name
        if data_name not in thermal:
            raise KeyError(f"Missing ThermalData/{data_name} in {path}")

        scan_reference = _attr_text(build_group.attrs.get("ScanStrategy_ref", ""))
        registration_type = _attr_text(registration.attrs.get("Registration_type", ""))
        x_units = _attr_text(handle["Calibration/Registration/Xgrid_v"].attrs.get("Xgrid_unit", ""))
        y_units = _attr_text(handle["Calibration/Registration/Ygrid_v"].attrs.get("Ygrid_unit", ""))
        z_units = _attr_text(handle["Calibration/Registration/Zgrid_v"].attrs.get("Zgrid_unit", ""))
        x_grid = handle["Calibration/Registration/Xgrid_v"][...]
        y_grid = handle["Calibration/Registration/Ygrid_v"][...]
        z_grid = handle["Calibration/Registration/Zgrid_v"][...]
        target_shape = tuple(int(value) for value in thermal[data_name].shape)
        layer_height_um = _attr_float(build_group.attrs.get("layer_height", 0.0))
        frame_rate_hz = _attr_float(thermal.attrs.get("frame_rate", 0.0))

    grid_shape_match = (
        len(target_shape) == 3
        and target_shape[0] == expected_layer_count
        and target_shape[1] == len(y_grid)
        and target_shape[2] == len(x_grid)
        and len(z_grid) == expected_layer_count
    )
    z_spacing_mm = float(z_grid[1] - z_grid[0]) if len(z_grid) > 1 else None
    z_spacing_match = z_spacing_mm is not None and abs(z_spacing_mm - 0.04) <= 1e-6
    scan_reference_match = SCAN_REFERENCE_MARKER in scan_reference
    spatial_registration_ready = (
        registration_type.lower() == "rigid2d"
        and x_units.lower() == "mm"
        and y_units.lower() == "mm"
        and z_units.lower() == "mm"
        and grid_shape_match
        and z_spacing_match
    )
    if not size_match:
        blockers.append("target_byte_size_mismatch")
    if not checksum_match:
        blockers.append("target_sha256_mismatch")
    if not scan_reference_match:
        blockers.append("target_scan_strategy_reference_mismatch")
    if not spatial_registration_ready:
        blockers.append("target_spatial_registration_incomplete")
    result.update(
        {
            "scan_strategy_ref": scan_reference,
            "scan_strategy_ref_matches_mds2_2607": scan_reference_match,
            "registration_type": registration_type,
            "grid_units": {"x": x_units, "y": y_units, "z": z_units},
            "grid_ranges_mm": {
                "x": [float(x_grid.min()), float(x_grid.max())],
                "y": [float(y_grid.min()), float(y_grid.max())],
                "z": [float(z_grid.min()), float(z_grid.max())],
            },
            "target_shape": list(target_shape),
            "grid_shape_match": grid_shape_match,
            "z_spacing_mm": z_spacing_mm,
            "z_spacing_matches_40um_layer_height": z_spacing_match,
            "build_layer_height_um": layer_height_um,
            "camera_frame_rate_hz": frame_rate_hz,
            "spatial_registration_ready": spatial_registration_ready,
            "blockers": blockers,
            "ready": not blockers,
        }
    )
    return result


def build_gate(
    *,
    scan_summary: dict[str, Any],
    target_summaries: list[dict[str, Any]],
    required_build_ids: Iterable[str] = BUILD_IDS,
) -> dict[str, Any]:
    required_builds = tuple(required_build_ids)
    by_build: dict[str, dict[str, dict[str, Any]]] = {build_id: {} for build_id in required_builds}
    for summary in target_summaries:
        build_id = str(summary["build_id"])
        target_name = str(summary["target_name"])
        if build_id in by_build:
            by_build[build_id][target_name] = summary

    blockers: list[str] = []
    if not bool(scan_summary.get("trigger_layer_time_ready")):
        blockers.append("xypt_trigger_layer_time_not_auditable")
    for build_id in required_builds:
        for target_name in TARGET_NAMES:
            summary = by_build[build_id].get(target_name)
            if summary is None:
                blockers.append(f"{build_id}_{target_name.lower()}_summary_missing")
            elif not bool(summary.get("ready")):
                blockers.append(f"{build_id}_{target_name.lower()}_not_registered")

    ready = not blockers
    return {
        "status": (
            "phase181_trigger_layer_space_registration_ready_phase182_dataset_construction"
            if ready
            else "phase181_trigger_layer_space_registration_blocked"
        ),
        "scope": "per-layer XYPT trigger and machine-coordinate TAM/SCR supervision",
        "required_build_ids": list(required_builds),
        "trigger_layer_space_registration_ready": ready,
        "phase182_dataset_construction_allowed": ready,
        "model_training_allowed": False,
        "raw_frame_event_causal_training_allowed": False,
        "absolute_wall_clock_trajectory_claim_allowed": False,
        "blocking_audits": blockers,
        "next_action": (
            "construct leakage-controlled B6/B7/B8 layer-space datasets and controls; retain the "
            "Phase 180 absolute-clock block for raw-frame causal claims"
            if ready
            else "complete checksum-verified registered TAM/SCR target files before dataset construction"
        ),
    }


def inspect(
    *,
    scan_root: Path,
    thermography_root: Path,
    file_list: Path,
    build_ids: Iterable[str] = BUILD_IDS,
) -> dict[str, Any]:
    build_ids = tuple(build_ids)
    scan_path = scan_root / "Scan_Strategy" / SCAN_FILENAME
    scan_summary = inspect_scan_strategy(scan_path)
    records = read_pdr_file_list(file_list)
    target_summaries: list[dict[str, Any]] = []
    for build_id in build_ids:
        for target_name in TARGET_NAMES:
            relative_path = _target_relative_path(build_id, target_name)
            target_summaries.append(
                inspect_target(
                    path=thermography_root / relative_path,
                    build_id=build_id,
                    target_name=target_name,
                    expected=records.get(relative_path),
                    expected_layer_count=int(scan_summary["layer_count"]),
                )
            )
    gate = build_gate(
        scan_summary=scan_summary,
        target_summaries=target_summaries,
        required_build_ids=build_ids,
    )
    return {
        "phase": 181,
        "scan_strategy": scan_summary,
        "nist_pdr_file_list": str(file_list),
        "targets": target_summaries,
        "gate": gate,
    }


def build_markdown(payload: dict[str, Any]) -> str:
    gate = payload["gate"]
    scan = payload["scan_strategy"]
    lines = [
        "# Phase 181: Trigger-Layer-Space Registration Gate",
        "",
        f"- Status: `{gate['status']}`",
        f"- XYPT digital rate: `{scan['digital_rate_hz']:.0f} Hz`",
        f"- XYPT StaringCamera trigger: `{scan['trigger_bit2']}`",
        f"- XYPT layers: `{scan['layer_count']}`",
        f"- Dataset construction allowed: `{gate['phase182_dataset_construction_allowed']}`",
        f"- Raw-frame causal training allowed: `{gate['raw_frame_event_causal_training_allowed']}`",
        f"- Absolute wall-clock trajectory claim allowed: `{gate['absolute_wall_clock_trajectory_claim_allowed']}`",
        "",
        "| Build | Target | Checksum | Scan ref | Spatial registration | Ready |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for target in payload["targets"]:
        lines.append(
            "| {build_id} | {target_name} | {checksum} | {scan_ref} | {spatial} | {ready} |".format(
                build_id=target["build_id"],
                target_name=target["target_name"],
                checksum=target.get("sha256_match", False),
                scan_ref=target.get("scan_strategy_ref_matches_mds2_2607", False),
                spatial=target.get("spatial_registration_ready", False),
                ready=target.get("ready", False),
            )
        )
    if gate["blocking_audits"]:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- `{item}`" for item in gate["blocking_audits"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", type=Path, default=DEFAULT_SCAN_ROOT)
    parser.add_argument("--thermography-root", type=Path, default=DEFAULT_THERMOGRAPHY_ROOT)
    parser.add_argument("--file-list", type=Path, default=DEFAULT_FILE_LIST)
    parser.add_argument("--build-id", dest="build_ids", action="append", choices=BUILD_IDS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args(argv)
    payload = inspect(
        scan_root=args.scan_root,
        thermography_root=args.thermography_root,
        file_list=args.file_list,
        build_ids=tuple(args.build_ids or BUILD_IDS),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(build_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
