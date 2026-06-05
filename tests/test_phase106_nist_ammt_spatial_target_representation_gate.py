from __future__ import annotations

import csv
import importlib.util
import json
import struct
import zipfile
from pathlib import Path

import pytest


def _load_module():
    script = Path("scripts/server/phase106_nist_ammt_spatial_target_representation_gate.py")
    spec = importlib.util.spec_from_file_location("phase106_spatial_gate", script)
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


def _write_zips(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(root / "In-situ Meas Data.zip", "w") as archive:
        for layer in range(1, 8):
            base = layer * 10
            archive.writestr(
                f"In-situ Meas Data/Layer Camera/A{layer:04d}.bmp",
                _bmp(
                    4,
                    4,
                    [
                        base,
                        base,
                        base + 5,
                        base + 5,
                        base,
                        base + 10,
                        base + 10,
                        base + 5,
                        base + 1,
                        base + 12,
                        base + 12,
                        base + 3,
                        base + 1,
                        base + 1,
                        base + 4,
                        base + 4,
                    ],
                ),
            )
            archive.writestr(
                f"In-situ Meas Data/Layer Camera/B{layer:04d}.bmp",
                _bmp(
                    4,
                    4,
                    [
                        base + 20,
                        base + 21,
                        base + 23,
                        base + 24,
                        base + 20,
                        base + 25,
                        base + 26,
                        base + 24,
                        base + 19,
                        base + 27,
                        base + 28,
                        base + 22,
                        base + 18,
                        base + 19,
                        base + 21,
                        base + 22,
                    ],
                ),
            )


def _write_numeric_table(path: Path) -> Path:
    rows = []
    split_names = ["train", "train", "train", "val", "val", "test", "test"]
    for index, split_name in enumerate(split_names, start=1):
        camera = "A" if index % 2 else "B"
        rows.append(
            {
                "x": str(index),
                "y": "0",
                "t": str(index),
                "target_intensity_mean": str(index * 2),
                "target_intensity_std": str(index),
                "source_layer_index": str(index),
                "target_layer_index": str(index),
                "target_camera_code": "0" if camera == "A" else "1",
                "source_p_mean": str(index + 1),
                "source_p_nonzero_fraction": "1",
                "source_x_range": "1",
                "source_y_range": "1",
                "split_name": split_name,
                "row_id": f"{camera}::{index:04d}",
                "target_file_name": "In-situ Meas Data.zip",
                "target_member_name": f"In-situ Meas Data/Layer Camera/{camera}{index:04d}.bmp",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_split(path: Path) -> Path:
    path.write_text(
        json.dumps({"splits": {"train": [0, 1, 2], "val": [3, 4], "test": [5, 6]}}) + "\n",
        encoding="utf-8",
    )
    return path


def _write_baseline_gate(path: Path, *, ready: bool = True) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": "phase104_baseline_smoke_complete_mechanisms_review_required",
                "baseline_smoke_completed": ready,
                "sample_size_sufficient_for_phase105": ready,
                "a100_training_allowed_now": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_phase105_gate(path: Path, *, closed: bool = True) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": "phase105_source_path_feature_gate_blocked_no_hgb_gain"
                if closed
                else "phase105_source_path_feature_gate_ready_for_cpu_smoke",
                "phase105_cpu_smoke_allowed": not closed,
                "phase105_model_training_allowed": False,
                "a100_training_allowed_now": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_spatial_target_gate_writes_table_and_keeps_training_locked(tmp_path: Path):
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn is not installed in this environment")
    module = _load_module()
    data_root = tmp_path / "data"
    _write_zips(data_root)

    manifest = module.build_package(
        root=Path(".").resolve(),
        data_root=data_root,
        numeric_field_table=_write_numeric_table(tmp_path / "field.csv"),
        split_manifest=_write_split(tmp_path / "split.json"),
        baseline_gate_path=_write_baseline_gate(tmp_path / "baseline_gate.json", ready=True),
        phase105_gate_path=_write_phase105_gate(tmp_path / "phase105_gate.json", closed=True),
        output_dir=tmp_path / "out",
        grid_size=2,
        max_pixels_per_target=64,
        target_columns=(
            "target_center_periphery_contrast",
            "target_grid_mean_range",
            "target_camera_pair_gradient_delta",
        ),
        target_priority=("target_camera_pair_gradient_delta", "target_grid_mean_range"),
        min_validation_relative_improvement=0.0,
        min_unsolved_validation_normalized_rmse=0.0,
        n_neighbors=1,
        n_estimators=5,
    )

    gate = manifest["gate"]
    assert gate["target_representation"] == "registered_layer_camera_spatial_statistics_v1"
    assert gate["phase106_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False

    with (tmp_path / "out/phase106_nist_ammt_spatial_target_field_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 7
    assert "target_center_periphery_contrast" in rows[0]
    assert "target_local_variance_mean" in rows[0]
    assert "target_gradient_q90" in rows[0]
    assert "target_camera_pair_gradient_delta" in rows[0]

    with (tmp_path / "out/phase106_nist_ammt_spatial_target_metric_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        metric_rows = list(csv.DictReader(handle))
    assert len(metric_rows) == 3 * len(module.METHODS) * 3


def test_spatial_target_gate_blocks_when_phase105_not_closed(tmp_path: Path):
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn is not installed in this environment")
    module = _load_module()
    data_root = tmp_path / "data"
    _write_zips(data_root)

    manifest = module.build_package(
        root=Path(".").resolve(),
        data_root=data_root,
        numeric_field_table=_write_numeric_table(tmp_path / "field.csv"),
        split_manifest=_write_split(tmp_path / "split.json"),
        baseline_gate_path=_write_baseline_gate(tmp_path / "baseline_gate.json", ready=True),
        phase105_gate_path=_write_phase105_gate(tmp_path / "phase105_gate.json", closed=False),
        output_dir=tmp_path / "out",
        grid_size=2,
        max_pixels_per_target=64,
        target_columns=("target_grid_mean_range",),
        target_priority=("target_grid_mean_range",),
        min_validation_relative_improvement=0.0,
        min_unsolved_validation_normalized_rmse=0.0,
        n_neighbors=1,
        n_estimators=5,
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase106_spatial_target_gate_blocked_by_phase105_boundary"
    assert gate["phase106_seed7_focused_validation_allowed"] is False
    assert gate["phase106_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
