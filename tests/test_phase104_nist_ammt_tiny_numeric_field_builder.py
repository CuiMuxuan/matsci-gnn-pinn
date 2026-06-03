from __future__ import annotations

import csv
import importlib.util
import json
import struct
import zipfile
from pathlib import Path


def _load_module():
    script = Path("scripts/server/phase104_nist_ammt_tiny_numeric_field_builder.py")
    spec = importlib.util.spec_from_file_location("phase104_numeric_builder", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bmp(width: int, height: int, pixels: list[int]) -> bytes:
    row_stride = ((width * 8 + 31) // 32) * 4
    pixel_offset = 14 + 40 + 256 * 4
    image_size = row_stride * height
    file_size = pixel_offset + image_size
    header = bytearray()
    header.extend(b"BM")
    header.extend(struct.pack("<IHHI", file_size, 0, 0, pixel_offset))
    header.extend(struct.pack("<IiiHHIIiiII", 40, width, height, 1, 8, 0, image_size, 0, 0, 256, 0))
    for value in range(256):
        header.extend(bytes([value, value, value, 0]))
    body = bytearray()
    for row in range(height):
        start = row * width
        body.extend(bytes(pixels[start : start + width]))
        body.extend(b"\x00" * (row_stride - width))
    return bytes(header + body)


def _write_tiny_table(path: Path) -> Path:
    rows = [
        {
            "row_id": "A::0002",
            "source_layer_index": "2",
            "split_name": "train",
            "source_file_name": "Build Command Data.zip",
            "source_group_key": "source{index}",
            "source_member_name": "Build Command Data/XYPT Commands/layer0002.csv",
            "target_file_name": "In-situ Meas Data.zip",
            "target_group_key": "target{index}",
            "target_member_name": "In-situ Meas Data/Layer Camera/A0002.bmp",
            "target_type": "layer_camera",
            "target_layer_index": "2",
            "join_offset": "0",
            "source_coverage": "1.0",
            "target_coverage": "1.0",
            "target_width": "2",
            "target_height": "2",
            "target_bits_per_pixel": "8",
            "target_channels": "1",
        },
        {
            "row_id": "B::0001",
            "source_layer_index": "2",
            "split_name": "train",
            "source_file_name": "Build Command Data.zip",
            "source_group_key": "source{index}",
            "source_member_name": "Build Command Data/XYPT Commands/layer0002.csv",
            "target_file_name": "In-situ Meas Data.zip",
            "target_group_key": "target{index}",
            "target_member_name": "In-situ Meas Data/Layer Camera/B0001.bmp",
            "target_type": "layer_camera",
            "target_layer_index": "1",
            "join_offset": "1",
            "source_coverage": "1.0",
            "target_coverage": "1.0",
            "target_width": "2",
            "target_height": "2",
            "target_bits_per_pixel": "8",
            "target_channels": "1",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_zips(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(root / "Build Command Data.zip", "w") as archive:
        archive.writestr(
            "Build Command Data/XYPT Commands/layer0002.csv",
            "0,0,0,0\n1,2,10,0.5\n2,4,20,1.0\n",
        )
    with zipfile.ZipFile(root / "In-situ Meas Data.zip", "w") as archive:
        archive.writestr("In-situ Meas Data/Layer Camera/A0002.bmp", _bmp(2, 2, [1, 2, 3, 4]))
        archive.writestr("In-situ Meas Data/Layer Camera/B0001.bmp", _bmp(2, 2, [10, 20, 30, 40]))


def test_phase104_numeric_builder_writes_field_table_and_split(tmp_path: Path):
    module = _load_module()
    data_root = tmp_path / "data"
    _write_zips(data_root)
    tiny_table = _write_tiny_table(tmp_path / "tiny.csv")

    manifest = module.build_package(
        root=Path(".").resolve(),
        data_root=data_root,
        tiny_table_csv=tiny_table,
        output_dir=tmp_path / "out",
        max_source_rows=None,
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase104_numeric_field_table_ready_baseline_pending"
    assert gate["baseline_smoke_ready"] is True
    assert gate["phase105_model_mechanism_allowed"] is False
    assert gate["a100_training_allowed_now"] is False

    with (tmp_path / "out/phase104_nist_ammt_tiny_numeric_field_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["target_intensity_mean"] == "2.500000"
    assert rows[1]["target_camera_code"] == "1"
    assert rows[0]["source_p_nonzero_fraction"] == "0.666667"
    split = json.loads(
        (tmp_path / "out/phase104_nist_ammt_tiny_numeric_split_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert split["splits"]["train"] == [0, 1]
    assert split["leakage_safe"] is True


def test_phase104_numeric_builder_records_truncated_source_rows(tmp_path: Path):
    module = _load_module()
    data_root = tmp_path / "data"
    _write_zips(data_root)
    tiny_table = _write_tiny_table(tmp_path / "tiny.csv")

    module.build_package(
        root=Path(".").resolve(),
        data_root=data_root,
        tiny_table_csv=tiny_table,
        output_dir=tmp_path / "out",
        max_source_rows=1,
    )

    with (tmp_path / "out/phase104_nist_ammt_tiny_numeric_field_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["source_rows_read"] == "1"
    assert rows[0]["source_rows_truncated"] == "true"
