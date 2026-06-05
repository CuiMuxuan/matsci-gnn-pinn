from __future__ import annotations

import csv
import importlib.util
import json
import struct
import zipfile
from pathlib import Path

import pytest


def _load_module():
    script = Path("scripts/server/phase112_nist_ammt_melt_pool_target_gate.py")
    spec = importlib.util.spec_from_file_location("phase112_melt_pool_gate", script)
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


def _write_data_zip(root: Path, *, layers: int = 10, frames: int = 3) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(root / "In-situ Meas Data.zip", "w") as archive:
        for layer in range(1, layers + 1):
            for frame in range(1, frames + 1):
                value = 20 + layer * 5 + frame
                archive.writestr(
                    f"In-situ Meas Data/Melt Pool Camera/MIA_L{layer:04d}/frame{frame:05d}.bmp",
                    _bmp(4, 4, [value for _ in range(16)]),
                )


def _write_numeric_table(path: Path, *, layers: int = 10) -> Path:
    rows = []
    for source_layer in range(2, layers + 2):
        rows.append(
            {
                "x": str(source_layer * 0.5),
                "y": str(source_layer * -0.25),
                "t": str(source_layer),
                "source_layer_index": str(source_layer),
                "target_layer_index": str(source_layer),
                "target_camera_code": "0",
                "source_p_mean": str(100 + source_layer * 2),
                "source_p_nonzero_fraction": "0.8",
                "source_x_range": "4.0",
                "source_y_range": "2.0",
                "source_p_range": "10.0",
                "source_t_range": "1.5",
                "source_rows_read": "1000",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_join_candidates(path: Path, *, layers: int = 10) -> Path:
    row = {
        "source_group_key": "Build Command Data/XYPT Commands/T500_3D_Scan_Strategies_fused_layer{index}.csv",
        "target_group_key": "In-situ Meas Data/Melt Pool Camera/MIA_L{layer}/frame{index}.bmp",
        "target_type": "melt_pool_camera_layer_directory",
        "source_first_index": "2",
        "source_last_index": str(layers + 1),
        "source_count": str(layers),
        "target_first_index": "1",
        "target_last_index": str(layers),
        "target_count": str(layers),
        "best_source_minus_target_offset": "1",
        "matched_pairs": str(layers),
        "source_coverage": "1.0",
        "target_coverage": "1.0",
        "first_pair": "source_layer=2;target_index=1",
        "last_pair": f"source_layer={layers + 1};target_index={layers}",
        "join_evidence_status": "source_target_layer_join_ready",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    return path


def _write_join_gate(path: Path, *, ready: bool = True) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": "source_target_layer_join_ready_timing_not_absolute"
                if ready
                else "source_target_join_incomplete",
                "melt_pool_layer_join_ready": ready,
                "a100_training_allowed_now": False,
                "a100_80gb_request_now": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_sequence_groups(path: Path, *, layers: int = 10, frames: int = 3) -> Path:
    rows = []
    for layer in range(1, layers + 1):
        rows.append(
            {
                "file_name": "In-situ Meas Data.zip",
                "group_key": f"In-situ Meas Data/Melt Pool Camera/MIA_L{layer:04d}/frame{{index}}.bmp",
                "directory": f"In-situ Meas Data/Melt Pool Camera/MIA_L{layer:04d}",
                "extension": ".bmp",
                "count": str(frames),
                "first_index": "1",
                "last_index": str(frames),
                "zero_padded_width": "5",
                "min_member_size": "0",
                "max_member_size": "0",
                "example_first": f"In-situ Meas Data/Melt Pool Camera/MIA_L{layer:04d}/frame00001.bmp",
                "example_last": f"In-situ Meas Data/Melt Pool Camera/MIA_L{layer:04d}/frame{frames:05d}.bmp",
                "target_observation_candidate": "true",
                "implicit_sequence_index_ready": "true",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_melt_pool_gate_writes_table_and_keeps_training_locked(tmp_path: Path):
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn is not installed in this environment")
    module = _load_module()
    data_root = tmp_path / "data"
    _write_data_zip(data_root)

    manifest = module.build_package(
        root=Path(".").resolve(),
        data_root=data_root,
        numeric_field_table=_write_numeric_table(tmp_path / "numeric.csv"),
        join_candidates_csv=_write_join_candidates(tmp_path / "join.csv"),
        join_gate_path=_write_join_gate(tmp_path / "join_gate.json"),
        sequence_groups_csv=_write_sequence_groups(tmp_path / "sequence.csv"),
        output_dir=tmp_path / "out",
        discover_from_zip=False,
        max_frames_per_layer=3,
        target_columns=("target_mp_q90_mean", "target_mp_temporal_mean_range"),
        target_priority=("target_mp_q90_mean", "target_mp_temporal_mean_range"),
        min_validation_relative_improvement=0.0,
        min_unsolved_validation_normalized_rmse=0.0,
        min_shortcut_val_rmse_delta=1e9,
        min_rows_for_review=5,
        n_neighbors=1,
        n_estimators=5,
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase112_melt_pool_target_gate_closed_layer_time_shortcut"
    assert gate["melt_pool_layer_join_ready"] is True
    assert gate["row_count"] == 10
    assert gate["leakage_safe"] is True
    assert gate["phase112_focused_review_allowed"] is False
    assert gate["phase112_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False

    with (tmp_path / "out/phase112_nist_ammt_melt_pool_target_field_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert rows[0]["sampled_frame_count"] == "3"
    assert "target_mp_q90_mean" in rows[0]

    with (tmp_path / "out/phase112_nist_ammt_melt_pool_target_metric_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        metric_rows = list(csv.DictReader(handle))
    assert len(metric_rows) == 2 * (1 + 4 * 3) * 3


def test_melt_pool_gate_closes_when_sample_size_limited(tmp_path: Path):
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn is not installed in this environment")
    module = _load_module()
    data_root = tmp_path / "data"
    _write_data_zip(data_root, layers=4, frames=3)

    manifest = module.build_package(
        root=Path(".").resolve(),
        data_root=data_root,
        numeric_field_table=_write_numeric_table(tmp_path / "numeric.csv", layers=4),
        join_candidates_csv=_write_join_candidates(tmp_path / "join.csv", layers=4),
        join_gate_path=_write_join_gate(tmp_path / "join_gate.json"),
        sequence_groups_csv=_write_sequence_groups(tmp_path / "sequence.csv", layers=4, frames=3),
        output_dir=tmp_path / "out",
        discover_from_zip=True,
        max_frames_per_layer=3,
        target_columns=("target_mp_q90_mean",),
        target_priority=("target_mp_q90_mean",),
        min_validation_relative_improvement=0.0,
        min_unsolved_validation_normalized_rmse=0.0,
        min_shortcut_val_rmse_delta=1e9,
        min_rows_for_review=10,
        n_neighbors=1,
        n_estimators=5,
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase112_melt_pool_target_gate_closed_sample_size_limited"
    assert gate["row_count"] == 4
    assert gate["phase112_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False


def test_melt_pool_gate_blocks_without_join(tmp_path: Path):
    module = _load_module()
    data_root = tmp_path / "data"
    _write_data_zip(data_root, layers=2, frames=2)
    _write_join_candidates(tmp_path / "join.csv", layers=2)

    manifest = module.build_package(
        root=Path(".").resolve(),
        data_root=data_root,
        numeric_field_table=_write_numeric_table(tmp_path / "numeric.csv", layers=2),
        join_candidates_csv=tmp_path / "join.csv",
        join_gate_path=_write_join_gate(tmp_path / "join_gate.json", ready=False),
        sequence_groups_csv=_write_sequence_groups(tmp_path / "sequence.csv", layers=2, frames=2),
        output_dir=tmp_path / "out",
        discover_from_zip=False,
        max_frames_per_layer=2,
        target_columns=("target_mp_q90_mean",),
        target_priority=("target_mp_q90_mean",),
        min_validation_relative_improvement=0.0,
        min_unsolved_validation_normalized_rmse=0.0,
        min_shortcut_val_rmse_delta=1e9,
        min_rows_for_review=1,
        n_neighbors=1,
        n_estimators=5,
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase112_melt_pool_target_gate_blocked_no_join"
    assert gate["melt_pool_layer_join_ready"] is False
    assert gate["phase112_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
