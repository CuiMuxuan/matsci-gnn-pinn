from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_probe_module():
    module_path = Path("scripts/server/phase53_source_path_data_pivot_gate.py")
    module_spec = importlib.util.spec_from_file_location("phase53_probe", module_path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def _write_thermal_file(path: Path, *, registration_attr: bool = False) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        thermal = handle.create_group("ThermalData")
        for name, frames in [("Line_0_1", 4), ("X_pad1", 5), ("Y_pad1", 6)]:
            group = thermal.create_group(name)
            group.attrs["laser_power"] = [285.0]
            group.attrs["scan_speed"] = [960.0]
            group.attrs["spot_size"] = [67.0]
            if registration_attr and name == "X_pad1":
                group.attrs["camera_to_galvo_affine"] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            group.create_dataset("Signal", data=[[[1, 2], [3, 4]]] * frames)


def _write_scan_file(path: Path, *, include_line: bool = False) -> None:
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for group_name in ["Xpad", "Ypad"]:
            group = handle.create_group(f"XYPT/{group_name}")
            group.create_dataset("X", data=[[0.0, 1.0]])
            group.create_dataset("Y", data=[[0.0, 1.0]])
            group.create_dataset("P", data=[[0.0, 285.0]])
            group.create_dataset("T", data=[[0, 4]])
        if include_line:
            line = handle.create_group("XYPT/Line_0_1")
            line.create_dataset("X", data=[[0.0, 1.0]])
            line.create_dataset("Y", data=[[0.0, 0.0]])


def test_phase53_blocks_pad_only_without_registration_metadata(tmp_path: Path):
    probe = _load_probe_module()
    thermal = tmp_path / "thermal.h5"
    scan_root = tmp_path / "ScanStrategy"
    output = tmp_path / "summary.json"
    _write_thermal_file(thermal)
    _write_scan_file(scan_root / "pad_xypt.h5")

    status = probe.main(
        [
            "--dataset-root",
            str(tmp_path),
            "--thermal-hdf5",
            str(thermal),
            "--scan-root",
            str(scan_root),
            "--json-output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert payload["decision"]["status"] == "negative"
    assert payload["decision"]["pad_registered_diagnostic_available"] is True
    assert payload["decision"]["paper_facing_source_path_ready"] is False
    assert payload["decision"]["source_inversion_broad12_broad21_blocked"] is True
    assert payload["thermal"]["categories"] == {"line": 1, "pad": 2, "other": 0}


def test_phase53_allows_single_track_scan_path(tmp_path: Path):
    probe = _load_probe_module()
    thermal = tmp_path / "thermal.h5"
    scan_root = tmp_path / "ScanStrategy"
    _write_thermal_file(thermal)
    _write_scan_file(scan_root / "scan_with_line.h5", include_line=True)

    payload = probe.run(
        probe.build_parser().parse_args(
            [
                "--dataset-root",
                str(tmp_path),
                "--thermal-hdf5",
                str(thermal),
                "--scan-root",
                str(scan_root),
            ]
        )
    )

    assert payload["decision"]["status"] == "positive"
    assert payload["decision"]["single_track_scan_path_available"] is True
    assert payload["decision"]["source_inversion_broad12_broad21_blocked"] is False


def test_phase53_reports_pad_matches_and_registration_attrs(tmp_path: Path):
    probe = _load_probe_module()
    thermal = tmp_path / "thermal.h5"
    scan_root = tmp_path / "ScanStrategy"
    _write_thermal_file(thermal, registration_attr=True)
    _write_scan_file(scan_root / "pad_xypt.h5")

    payload = probe.run(
        probe.build_parser().parse_args(
            [
                "--dataset-root",
                str(tmp_path),
                "--thermal-hdf5",
                str(thermal),
                "--scan-root",
                str(scan_root),
            ]
        )
    )

    assert payload["decision"]["status"] == "positive"
    assert payload["decision"]["pad_registered_diagnostic_available"] is True
    assert payload["decision"]["hdf5_registration_metadata_available"] is True
    assert {match["xypt_group"] for match in payload["pad_xypt_matches"]} == {"XYPT/Xpad", "XYPT/Ypad"}
