from __future__ import annotations

import csv
import importlib.util
import struct
import zipfile
from pathlib import Path


def _load_module():
    script = Path("scripts/server/phase103_nist_ammt_deep_registration_probe.py")
    spec = importlib.util.spec_from_file_location("phase103_deep_probe", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bmp_header(width: int = 64, height: int = 32, bits_per_pixel: int = 24) -> bytes:
    row_bytes = ((width * bits_per_pixel + 31) // 32) * 4
    pixel_bytes = row_bytes * height
    file_size = 54 + pixel_bytes
    header = bytearray(54)
    header[0:2] = b"BM"
    struct.pack_into("<I", header, 2, file_size)
    struct.pack_into("<I", header, 10, 54)
    struct.pack_into("<I", header, 14, 40)
    struct.pack_into("<i", header, 18, width)
    struct.pack_into("<i", header, 22, height)
    struct.pack_into("<H", header, 26, 1)
    struct.pack_into("<H", header, 28, bits_per_pixel)
    struct.pack_into("<I", header, 34, pixel_bytes)
    return bytes(header) + b"\x00" * 16


def _write_zip(path: Path, members: dict[str, bytes | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def test_deep_probe_recognizes_binary_target_schema_but_keeps_timing_blocked(
    tmp_path: Path,
):
    module = _load_module()
    data_root = tmp_path / "data"
    _write_zip(data_root / "Metadata.zip", {"Metadata/grid.txt": "grid"})
    _write_zip(data_root / "Build Command Data.zip", {"Commands/path.gcode": "G1 X0 Y0\n"})
    _write_zip(
        data_root / "In-situ Meas Data.zip",
        {
            "In-situ Meas Data/Layer Camera/A0001.bmp": _bmp_header(),
            "In-situ Meas Data/Layer Camera/A0002.bmp": _bmp_header(),
            "In-situ Meas Data/Layer Camera/A0003.bmp": _bmp_header(),
        },
    )

    manifest = module.build_package(
        root=Path(".").resolve(),
        data_root=data_root,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert gate["status"] == "target_binary_schema_ready_trigger_timing_missing"
    assert gate["target_observation_binary_schema_ready"] is True
    assert gate["implicit_sequence_index_ready"] is True
    assert gate["explicit_trigger_timing_ready"] is False
    assert gate["missing_or_blocked_required_roles"] == ["trigger_timing"]
    assert gate["phase104_baseline_smoke_allowed"] is False
    assert gate["a100_training_allowed_now"] is False

    with (tmp_path / "out/phase103_nist_ammt_deep_target_binary_samples.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert rows[0]["format"] == "bmp"
    assert rows[0]["width"] == "64"
    assert rows[0]["height"] == "32"
    assert rows[0]["target_observation_binary_schema_ready"] == "true"


def test_deep_probe_records_timing_evidence_without_opening_training(tmp_path: Path):
    module = _load_module()
    data_root = tmp_path / "data"
    _write_zip(data_root / "Metadata.zip", {"Metadata/grid.txt": "grid"})
    _write_zip(data_root / "Build Command Data.zip", {"Commands/path.gcode": "G1 X0 Y0\n"})
    _write_zip(
        data_root / "In-situ Meas Data.zip",
        {
            "In-situ Meas Data/Layer Camera/A0001.bmp": _bmp_header(),
            "In-situ Meas Data/Layer Camera/A0002.bmp": _bmp_header(),
            "In-situ Meas Data/frame_timestamp_table.csv": "frame,timestamp\n1,0.0\n",
        },
    )

    manifest = module.build_package(
        root=Path(".").resolve(),
        data_root=data_root,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert gate["status"] == "deep_probe_ready_manual_registration_required"
    assert gate["explicit_trigger_timing_ready"] is True
    assert gate["missing_or_blocked_required_roles"] == []
    assert gate["phase104_baseline_smoke_allowed"] is False
    assert gate["phase105_model_mechanism_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
