#!/usr/bin/env python3
"""Materialize a registered, leakage-controlled AMB2022-01 layer-space dataset.

This converter consumes only Phase-181-admitted sources: the public XYPT command
stream and B6/B7/B8 TAM/SCR fields.  It deliberately creates layer-space targets,
not raw-frame trajectories.  The output contains no build identity as a model
feature; B6, B7, and B8 are reserved for train, validation, and test replication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


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
DEFAULT_PHASE181_GATE = Path(
    os.environ.get(
        "AMB2022_01_PHASE181_GATE",
        "/root/matsci-gnn-pinn-ops/phase181_trigger_layer_space_registration_gate.json",
    )
)
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get(
        "AMB2022_01_PHASE182_OUTPUT_ROOT",
        "/root/matsci-gnn-pinn-data/derived/ambench/2022_3d_build/AMB2022-01/phase182",
    )
)

SCAN_FILENAME = "AMB2022-01-AMMT-XYPT_v1.h5"
BUILD_IDS = ("B6", "B7", "B8")
SPLIT_CODES = {"B6": 0, "B7": 1, "B8": 2}
SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}
SCR_CANONICAL_UNITS = "C/s"
FEATURE_NAMES = (
    "x_mm",
    "y_mm",
    "z_mm",
    "layer_fraction",
    "block_area_mm2",
    "laser_active",
    "laser_dwell_s",
    "laser_energy_J",
    "laser_energy_density_J_mm2",
    "mean_power_W",
    "max_power_W",
    "first_laser_time_s",
    "last_laser_time_s",
    "time_since_last_laser_s",
    "energy_weighted_progress",
    "laser_progress_span",
    "staring_trigger_count",
)


def _h5py() -> Any:
    try:
        import h5py
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("h5py is required for Phase 182 dataset construction") from exc
    return h5py


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attr_text(value: Any) -> str:
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
            raise ValueError(f"Expected scalar attribute, got {value!r}")
        value = value[0]
    return float(value)


def _target_filename(build_id: str, target_name: str) -> str:
    return f"AMB2022-01-718-AMMT-{build_id}-StaringCamera_{target_name}.h5"


def resolve_scr_units(declared_units: str) -> dict[str, Any]:
    """Preserve a known HDF5 metadata defect while exposing the documented CR unit."""
    declared_units = declared_units.strip()
    if declared_units not in {"s", SCR_CANONICAL_UNITS}:
        raise ValueError(f"Unexpected SCR units attribute: {declared_units!r}")
    return {
        "declared_units": declared_units,
        "canonical_units": SCR_CANONICAL_UNITS,
        "metadata_correction_applied": declared_units != SCR_CANONICAL_UNITS,
        "evidence": "NIST CR_v1.m computes DT./Dt and labels the result [oC/s]",
    }


def _read_phase181_gate(path: Path, build_ids: Iterable[str]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    gate = payload.get("gate", {})
    expected_status = "phase181_trigger_layer_space_registration_ready_phase182_dataset_construction"
    if gate.get("status") != expected_status or not gate.get("phase182_dataset_construction_allowed"):
        raise ValueError(f"Phase 181 does not permit dataset construction: {gate.get('status')!r}")
    required = set(build_ids)
    targets = {
        (str(item.get("build_id")), str(item.get("target_name"))): item
        for item in payload.get("targets", [])
    }
    missing = [
        f"{build_id}_{target_name}"
        for build_id in required
        for target_name in ("TAM", "SCR")
        if not targets.get((build_id, target_name), {}).get("ready")
    ]
    if missing:
        raise ValueError(f"Phase 181 target evidence missing or not ready: {missing}")
    return payload


def _coordinate_edges(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 1 or len(coords) < 2 or not np.all(np.diff(coords) > 0.0):
        raise ValueError("Coordinate vector must be strictly increasing with at least two entries")
    edges = np.empty(len(coords) + 1, dtype=float)
    edges[1:-1] = 0.5 * (coords[:-1] + coords[1:])
    edges[0] = coords[0] - 0.5 * (coords[1] - coords[0])
    edges[-1] = coords[-1] + 0.5 * (coords[-1] - coords[-2])
    return edges


def _block_geometry(x_coords: np.ndarray, y_coords: np.ndarray, block_size: int) -> dict[str, np.ndarray]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if len(x_coords) % block_size or len(y_coords) % block_size:
        raise ValueError(
            f"Grid shape {(len(y_coords), len(x_coords))} is not divisible by block_size={block_size}"
        )
    n_block_cols = len(x_coords) // block_size
    n_block_rows = len(y_coords) // block_size
    x_centers = np.asarray(x_coords, dtype=float).reshape(n_block_cols, block_size).mean(axis=1)
    y_centers = np.asarray(y_coords, dtype=float).reshape(n_block_rows, block_size).mean(axis=1)
    x_edges = _coordinate_edges(x_coords)
    y_edges = _coordinate_edges(y_coords)
    x_widths = np.asarray(
        [x_edges[(index + 1) * block_size] - x_edges[index * block_size] for index in range(n_block_cols)],
        dtype=float,
    )
    y_widths = np.asarray(
        [y_edges[(index + 1) * block_size] - y_edges[index * block_size] for index in range(n_block_rows)],
        dtype=float,
    )
    return {
        "n_block_rows": np.asarray(n_block_rows),
        "n_block_cols": np.asarray(n_block_cols),
        "x_centers": np.tile(x_centers, n_block_rows),
        "y_centers": np.repeat(y_centers, n_block_cols),
        "block_area_mm2": np.repeat(y_widths, n_block_cols) * np.tile(x_widths, n_block_rows),
        "block_rows": np.repeat(np.arange(n_block_rows, dtype=np.int16), n_block_cols),
        "block_cols": np.tile(np.arange(n_block_cols, dtype=np.int16), n_block_rows),
    }


def _nearest_coordinate_index(coords: np.ndarray, values: np.ndarray) -> np.ndarray:
    insertion = np.searchsorted(coords, values, side="left")
    insertion = np.clip(insertion, 1, len(coords) - 1)
    left = insertion - 1
    use_left = np.abs(values - coords[left]) <= np.abs(values - coords[insertion])
    return np.where(use_left, left, insertion).astype(np.int64, copy=False)


def _block_nanmean(values: np.ndarray, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D target plane, got shape {values.shape}")
    rows, cols = values.shape
    if rows % block_size or cols % block_size:
        raise ValueError(f"Target plane shape {values.shape} is not divisible by block size {block_size}")
    grouped = values.reshape(rows // block_size, block_size, cols // block_size, block_size)
    finite = np.isfinite(grouped)
    count = finite.sum(axis=(1, 3))
    totals = np.where(finite, grouped, 0.0).sum(axis=(1, 3), dtype=np.float64)
    means = np.divide(
        totals,
        count,
        out=np.full_like(totals, np.nan, dtype=np.float64),
        where=count > 0,
    )
    fractions = count.astype(np.float32) / float(block_size * block_size)
    return means.astype(np.float32, copy=False).ravel(), fractions.ravel()


def scan_block_features(
    *,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    x_commands: np.ndarray,
    y_commands: np.ndarray,
    power_commands: np.ndarray,
    trigger_commands: np.ndarray,
    digital_rate_hz: float,
    block_size: int,
) -> dict[str, np.ndarray]:
    """Aggregate XYPT laser commands into physical camera-grid blocks."""
    geometry = _block_geometry(x_coords, y_coords, block_size)
    n_block_rows = int(geometry["n_block_rows"])
    n_block_cols = int(geometry["n_block_cols"])
    n_blocks = n_block_rows * n_block_cols
    x_commands = np.asarray(x_commands, dtype=float).reshape(-1)
    y_commands = np.asarray(y_commands, dtype=float).reshape(-1)
    power_commands = np.asarray(power_commands, dtype=float).reshape(-1)
    trigger_commands = np.asarray(trigger_commands).reshape(-1)
    if not (len(x_commands) == len(y_commands) == len(power_commands) == len(trigger_commands)):
        raise ValueError("XYPT X/Y/P/T vectors must have the same flattened length")
    if digital_rate_hz <= 0.0:
        raise ValueError("digital_rate_hz must be positive")

    command_index = np.arange(len(power_commands), dtype=np.int64)
    active = power_commands > 0.0
    active &= x_commands >= float(x_coords[0])
    active &= x_commands <= float(x_coords[-1])
    active &= y_commands >= float(y_coords[0])
    active &= y_commands <= float(y_coords[-1])
    selected_x = x_commands[active]
    selected_y = y_commands[active]
    selected_power = power_commands[active]
    selected_trigger = trigger_commands[active]
    selected_index = command_index[active]
    x_pixel = _nearest_coordinate_index(x_coords, selected_x)
    y_pixel = _nearest_coordinate_index(y_coords, selected_y)
    cell_index = (y_pixel // block_size) * n_block_cols + (x_pixel // block_size)
    dt = 1.0 / digital_rate_hz
    layer_duration_s = len(power_commands) * dt

    dwell_count = np.bincount(cell_index, minlength=n_blocks).astype(np.float64)
    dwell_s = dwell_count * dt
    energy_j = np.bincount(cell_index, weights=selected_power * dt, minlength=n_blocks).astype(np.float64)
    power_sum = np.bincount(cell_index, weights=selected_power, minlength=n_blocks).astype(np.float64)
    progress = selected_index.astype(np.float64) / max(1, len(power_commands) - 1)
    weighted_progress = np.bincount(
        cell_index,
        weights=progress * selected_power * dt,
        minlength=n_blocks,
    ).astype(np.float64)
    first_time = np.full(n_blocks, np.inf, dtype=np.float64)
    last_time = np.full(n_blocks, -np.inf, dtype=np.float64)
    time_values = selected_index.astype(np.float64) * dt
    np.minimum.at(first_time, cell_index, time_values)
    np.maximum.at(last_time, cell_index, time_values)
    max_power = np.zeros(n_blocks, dtype=np.float64)
    np.maximum.at(max_power, cell_index, selected_power)
    trigger_values = ((selected_trigger.astype(np.uint8) & np.uint8(4)) > 0).astype(np.float64)
    trigger_count = np.bincount(cell_index, weights=trigger_values, minlength=n_blocks).astype(np.float64)

    active_block = dwell_count > 0.0
    mean_power = np.divide(power_sum, dwell_count, out=np.zeros_like(power_sum), where=active_block)
    weighted_progress = np.divide(
        weighted_progress,
        energy_j,
        out=np.zeros_like(weighted_progress),
        where=energy_j > 0.0,
    )
    first_time = np.where(active_block, first_time, 0.0)
    last_time = np.where(active_block, last_time, 0.0)
    progress_span = np.where(active_block, (last_time - first_time) / max(layer_duration_s, dt), 0.0)
    time_since_last = np.where(active_block, layer_duration_s - last_time, layer_duration_s)
    energy_density = np.divide(
        energy_j,
        geometry["block_area_mm2"],
        out=np.zeros_like(energy_j),
        where=geometry["block_area_mm2"] > 0.0,
    )
    return {
        **geometry,
        "layer_duration_s": np.asarray(layer_duration_s),
        "laser_active": active_block.astype(np.float32),
        "laser_dwell_s": dwell_s.astype(np.float32),
        "laser_energy_J": energy_j.astype(np.float32),
        "laser_energy_density_J_mm2": energy_density.astype(np.float32),
        "mean_power_W": mean_power.astype(np.float32),
        "max_power_W": max_power.astype(np.float32),
        "first_laser_time_s": first_time.astype(np.float32),
        "last_laser_time_s": last_time.astype(np.float32),
        "time_since_last_laser_s": time_since_last.astype(np.float32),
        "energy_weighted_progress": weighted_progress.astype(np.float32),
        "laser_progress_span": progress_span.astype(np.float32),
        "staring_trigger_count": trigger_count.astype(np.float32),
    }


def _feature_matrix(
    features: dict[str, np.ndarray],
    *,
    z_mm: float,
    layer_index: int,
    layer_count: int,
) -> np.ndarray:
    n_rows = len(features["x_centers"])
    columns = [
        features["x_centers"],
        features["y_centers"],
        np.full(n_rows, z_mm, dtype=np.float32),
        np.full(n_rows, layer_index / max(1, layer_count - 1), dtype=np.float32),
        features["block_area_mm2"],
        features["laser_active"],
        features["laser_dwell_s"],
        features["laser_energy_J"],
        features["laser_energy_density_J_mm2"],
        features["mean_power_W"],
        features["max_power_W"],
        features["first_laser_time_s"],
        features["last_laser_time_s"],
        features["time_since_last_laser_s"],
        features["energy_weighted_progress"],
        features["laser_progress_span"],
        features["staring_trigger_count"],
    ]
    matrix = np.column_stack(columns).astype(np.float32, copy=False)
    if matrix.shape[1] != len(FEATURE_NAMES) or not np.isfinite(matrix).all():
        raise ValueError("Feature construction produced non-finite or malformed values")
    return matrix


def _validate_grid(
    *,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    target_shape: tuple[int, ...],
) -> None:
    if target_shape != (len(z_coords), len(y_coords), len(x_coords)):
        raise ValueError(
            f"Target shape {target_shape} does not match Z/Y/X grid lengths "
            f"{(len(z_coords), len(y_coords), len(x_coords))}"
        )
    if len(z_coords) < 2 or not np.allclose(np.diff(z_coords), 0.04, atol=1e-6, rtol=0.0):
        raise ValueError("Expected a 40 um evenly spaced Z grid")


def build_dataset(
    *,
    scan_root: Path,
    thermography_root: Path,
    phase181_gate: Path,
    output_hdf5: Path,
    manifest_path: Path,
    build_ids: Iterable[str] = BUILD_IDS,
    block_size: int = 16,
    min_valid_fraction: float = 0.5,
    max_layers: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    h5py = _h5py()
    build_ids = tuple(build_ids)
    if not build_ids:
        raise ValueError("At least one build ID is required")
    if any(build_id not in SPLIT_CODES for build_id in build_ids):
        raise ValueError(f"Unsupported build IDs: {build_ids}")
    if not 0.0 < min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be in (0, 1]")
    if output_hdf5.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_hdf5}; pass --overwrite to replace it")

    gate_payload = _read_phase181_gate(phase181_gate, build_ids)
    scan_summary = gate_payload["scan_strategy"]
    scan_path = scan_root / "Scan_Strategy" / SCAN_FILENAME
    scan_sha256 = _sha256(scan_path)
    if scan_sha256 != scan_summary.get("sha256"):
        raise ValueError("XYPT SHA-256 differs from the Phase 181-admitted source")
    digital_rate_hz = float(scan_summary["digital_rate_hz"])

    target_summary = {
        (item["build_id"], item["target_name"]): item for item in gate_payload["targets"]
    }
    first_tam_path = thermography_root / "Staring_Thermography" / _target_filename(build_ids[0], "TAM")
    with h5py.File(first_tam_path, "r") as first:
        x_coords = np.asarray(first["Calibration/Registration/Xgrid_v"][...], dtype=np.float32)
        y_coords = np.asarray(first["Calibration/Registration/Ygrid_v"][...], dtype=np.float32)
        z_coords = np.asarray(first["Calibration/Registration/Zgrid_v"][...], dtype=np.float32)
        _validate_grid(
            x_coords=x_coords,
            y_coords=y_coords,
            z_coords=z_coords,
            target_shape=tuple(int(value) for value in first["ThermalData/TAM"].shape),
        )
    geometry = _block_geometry(x_coords, y_coords, block_size)
    n_blocks = int(geometry["n_block_rows"]) * int(geometry["n_block_cols"])
    layer_indices = list(range(len(z_coords)))
    if max_layers is not None:
        if max_layers <= 0:
            raise ValueError("max_layers must be positive")
        layer_indices = layer_indices[:max_layers]
    total_rows = len(build_ids) * len(layer_indices) * n_blocks
    output_hdf5.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_hdf5.with_suffix(output_hdf5.suffix + ".tmp")
    if temp_output.exists():
        temp_output.unlink()

    source_records: list[dict[str, Any]] = []
    build_quality: dict[str, Any] = {}
    offset = 0
    row_chunk = min(8192, total_rows)
    vector_compression = {
        "compression": "gzip",
        "compression_opts": 4,
        "shuffle": True,
        "chunks": (row_chunk,),
    }
    matrix_compression = {**vector_compression, "chunks": (row_chunk, len(FEATURE_NAMES))}
    mask_compression = {**vector_compression, "chunks": (row_chunk, 2)}
    try:
        with h5py.File(scan_path, "r") as scan_handle, h5py.File(temp_output, "w") as output:
            features_ds = output.create_dataset(
                "features", shape=(total_rows, len(FEATURE_NAMES)), dtype="f4", **matrix_compression
            )
            tam_ds = output.create_dataset("target_tam_s", shape=(total_rows,), dtype="f4", **vector_compression)
            scr_ds = output.create_dataset("target_scr_C_per_s", shape=(total_rows,), dtype="f4", **vector_compression)
            scr_ds.attrs["units"] = SCR_CANONICAL_UNITS
            valid_ds = output.create_dataset("target_valid_mask", shape=(total_rows, 2), dtype="u1", **mask_compression)
            tam_fraction_ds = output.create_dataset("tam_valid_fraction", shape=(total_rows,), dtype="f4", **vector_compression)
            scr_fraction_ds = output.create_dataset("scr_valid_fraction", shape=(total_rows,), dtype="f4", **vector_compression)
            build_ds = output.create_dataset("build_index", shape=(total_rows,), dtype="u1", **vector_compression)
            layer_ds = output.create_dataset("layer_index", shape=(total_rows,), dtype="u2", **vector_compression)
            block_row_ds = output.create_dataset("block_row", shape=(total_rows,), dtype="u2", **vector_compression)
            block_col_ds = output.create_dataset("block_col", shape=(total_rows,), dtype="u2", **vector_compression)
            split_ds = output.create_dataset("primary_split", shape=(total_rows,), dtype="u1", **vector_compression)
            output.attrs["feature_names"] = json.dumps(FEATURE_NAMES)
            output.attrs["build_index_map"] = json.dumps({index: build_id for index, build_id in enumerate(build_ids)})
            output.attrs["primary_split_map"] = json.dumps(SPLIT_NAMES)
            output.attrs["scope"] = "registered_layer_space_tam_scr_not_raw_frame_trajectory"
            output.attrs["block_size_pixels"] = block_size
            output.attrs["phase181_gate"] = str(phase181_gate)

            xypt = scan_handle["XYPT"]
            for build_index, build_id in enumerate(build_ids):
                target_paths = {
                    name: thermography_root / "Staring_Thermography" / _target_filename(build_id, name)
                    for name in ("TAM", "SCR")
                }
                expected_summaries = {
                    name: target_summary[(build_id, name)] for name in ("TAM", "SCR")
                }
                actual_sources: dict[str, dict[str, Any]] = {}
                for name, path in target_paths.items():
                    actual_sha = _sha256(path)
                    expected_sha = expected_summaries[name]["sha256"]
                    if actual_sha != expected_sha:
                        raise ValueError(f"{build_id} {name} SHA-256 differs from Phase 181 evidence")
                    actual_sources[name] = {
                        "path": str(path),
                        "sha256": actual_sha,
                        "source_units": None,
                    }
                with h5py.File(target_paths["TAM"], "r") as tam_handle, h5py.File(target_paths["SCR"], "r") as scr_handle:
                    tam_grid_x = np.asarray(tam_handle["Calibration/Registration/Xgrid_v"][...], dtype=np.float32)
                    tam_grid_y = np.asarray(tam_handle["Calibration/Registration/Ygrid_v"][...], dtype=np.float32)
                    tam_grid_z = np.asarray(tam_handle["Calibration/Registration/Zgrid_v"][...], dtype=np.float32)
                    for name, handle in (("TAM", tam_handle), ("SCR", scr_handle)):
                        dataset = handle[f"ThermalData/{name}"]
                        _validate_grid(
                            x_coords=tam_grid_x,
                            y_coords=tam_grid_y,
                            z_coords=tam_grid_z,
                            target_shape=tuple(int(value) for value in dataset.shape),
                        )
                        if not (np.allclose(tam_grid_x, x_coords) and np.allclose(tam_grid_y, y_coords) and np.allclose(tam_grid_z, z_coords)):
                            raise ValueError(f"{build_id} {name} registration grid differs from the admitted reference grid")
                        declared_units = _attr_text(dataset.attrs.get("units", ""))
                        actual_sources[name]["source_units_declared"] = declared_units
                        if name == "SCR":
                            actual_sources[name]["unit_resolution"] = resolve_scr_units(declared_units)

                    valid_both_count = 0
                    for layer_index in layer_indices:
                        xypt_layer = xypt[str(layer_index + 1)]
                        layer_features = scan_block_features(
                            x_coords=x_coords,
                            y_coords=y_coords,
                            x_commands=xypt_layer["X"][...],
                            y_commands=xypt_layer["Y"][...],
                            power_commands=xypt_layer["P"][...],
                            trigger_commands=xypt_layer["T"][...],
                            digital_rate_hz=digital_rate_hz,
                            block_size=block_size,
                        )
                        matrix = _feature_matrix(
                            layer_features,
                            z_mm=float(z_coords[layer_index]),
                            layer_index=layer_index,
                            layer_count=len(z_coords),
                        )
                        tam_values, tam_fractions = _block_nanmean(tam_handle["ThermalData/TAM"][layer_index, ...], block_size)
                        scr_values, scr_fractions = _block_nanmean(scr_handle["ThermalData/SCR"][layer_index, ...], block_size)
                        tam_valid = tam_fractions >= min_valid_fraction
                        scr_valid = scr_fractions >= min_valid_fraction
                        start, stop = offset, offset + n_blocks
                        features_ds[start:stop, :] = matrix
                        tam_ds[start:stop] = tam_values
                        scr_ds[start:stop] = scr_values
                        valid_ds[start:stop, :] = np.column_stack([tam_valid, scr_valid]).astype(np.uint8)
                        tam_fraction_ds[start:stop] = tam_fractions
                        scr_fraction_ds[start:stop] = scr_fractions
                        build_ds[start:stop] = build_index
                        layer_ds[start:stop] = layer_index
                        block_row_ds[start:stop] = layer_features["block_rows"]
                        block_col_ds[start:stop] = layer_features["block_cols"]
                        split_ds[start:stop] = SPLIT_CODES[build_id]
                        valid_both_count += int(np.sum(tam_valid & scr_valid))
                        offset = stop
                    build_quality[build_id] = {
                        "rows": len(layer_indices) * n_blocks,
                        "target_valid_both_rows": valid_both_count,
                        "target_valid_both_fraction": valid_both_count / max(1, len(layer_indices) * n_blocks),
                        "source_units": {
                            name: source.get("source_units_declared") for name, source in actual_sources.items()
                        },
                        "scr_unit_resolution": actual_sources["SCR"]["unit_resolution"],
                    }
                source_records.append({"build_id": build_id, "targets": actual_sources})
    except Exception:
        if temp_output.exists():
            temp_output.unlink()
        raise
    if offset != total_rows:
        temp_output.unlink(missing_ok=True)
        raise RuntimeError(f"Wrote {offset} rows but expected {total_rows}")
    temp_output.replace(output_hdf5)

    split_counts = {
        SPLIT_NAMES[SPLIT_CODES[build_id]]: len(layer_indices) * n_blocks for build_id in build_ids
    }
    manifest = {
        "phase": 182,
        "objective": "registered_layer_space_tam_scr_dataset_construction",
        "scope": "per-layer XYPT command features and spatially registered TAM/SCR targets",
        "excluded_claims": [
            "raw_frame_level_causal_history",
            "global_absolute_wall_clock_trajectory",
            "build_identity_as_model_feature",
        ],
        "phase181_gate": str(phase181_gate),
        "scan_source": {
            "path": str(scan_path),
            "sha256": scan_sha256,
            "digital_rate_hz": digital_rate_hz,
            "trigger_bit2": scan_summary.get("trigger_bit2"),
        },
        "target_sources": source_records,
        "output_hdf5": str(output_hdf5),
        "row_count": total_rows,
        "feature_names": list(FEATURE_NAMES),
        "block_size_pixels": block_size,
        "coarse_grid_shape": [int(geometry["n_block_rows"]), int(geometry["n_block_cols"])],
        "source_grid_shape": [len(y_coords), len(x_coords)],
        "layer_indices": [int(index + 1) for index in layer_indices],
        "min_valid_fraction": min_valid_fraction,
        "primary_split": {
            "strategy": "build_holdout_replication",
            "build_to_split": {build_id: SPLIT_NAMES[SPLIT_CODES[build_id]] for build_id in build_ids},
            "row_counts": split_counts,
            "build_identity_in_feature_matrix": False,
        },
        "quality": build_quality,
        "scr_unit_resolution": {
            "canonical_units": SCR_CANONICAL_UNITS,
            "metadata_defect": "B6/B8 HDF5 declares s while B7 declares C/s",
            "documented_algorithm_evidence": "NIST CR_v1.m line 61 labels DT./Dt as [oC/s]",
            "source_metadata_retained_per_build": True,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", type=Path, default=DEFAULT_SCAN_ROOT)
    parser.add_argument("--thermography-root", type=Path, default=DEFAULT_THERMOGRAPHY_ROOT)
    parser.add_argument("--phase181-gate", type=Path, default=DEFAULT_PHASE181_GATE)
    parser.add_argument("--output-hdf5", type=Path, default=DEFAULT_OUTPUT_ROOT / "phase182_layer_space_dataset.h5")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT_ROOT / "phase182_layer_space_dataset_manifest.json")
    parser.add_argument("--build-id", dest="build_ids", action="append", choices=BUILD_IDS)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--min-valid-fraction", type=float, default=0.5)
    parser.add_argument("--max-layers", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_dataset(
        scan_root=args.scan_root,
        thermography_root=args.thermography_root,
        phase181_gate=args.phase181_gate,
        output_hdf5=args.output_hdf5,
        manifest_path=args.manifest,
        build_ids=tuple(args.build_ids or BUILD_IDS),
        block_size=args.block_size,
        min_valid_fraction=args.min_valid_fraction,
        max_layers=args.max_layers,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
