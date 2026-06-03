from __future__ import annotations

import csv
import importlib.util
import zipfile
from pathlib import Path


def _load_module():
    script = Path("scripts/server/phase103_nist_ammt_member_schema_sampler.py")
    spec = importlib.util.spec_from_file_location("phase103_member_sampler", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_zip(path: Path, members: dict[str, bytes | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def _write_candidates(path: Path) -> None:
    rows = [
        {
            "file_id": "metadata",
            "file_name": "Metadata.zip",
            "member_name": "Metadata/pixel_coordinate_transform.csv",
            "member_size": "24",
            "extension": ".csv",
            "roles": "coordinate_transform",
            "coordinate_transform": "true",
            "trigger_timing": "false",
            "source_command_path": "false",
            "target_observation": "false",
            "split_key": "false",
        },
        {
            "file_id": "commands",
            "file_name": "Build Command Data.zip",
            "member_name": "Commands/laser_scan_path_XYPT.csv",
            "member_size": "22",
            "extension": ".csv",
            "roles": "source_command_path",
            "coordinate_transform": "false",
            "trigger_timing": "false",
            "source_command_path": "true",
            "target_observation": "false",
            "split_key": "false",
        },
        {
            "file_id": "insitu",
            "file_name": "In-situ Meas Data.zip",
            "member_name": "InSitu/trigger_timestamp_frame_table.csv",
            "member_size": "18",
            "extension": ".csv",
            "roles": "trigger_timing;target_observation",
            "coordinate_transform": "false",
            "trigger_timing": "true",
            "source_command_path": "false",
            "target_observation": "true",
            "split_key": "false",
        },
        {
            "file_id": "insitu",
            "file_name": "In-situ Meas Data.zip",
            "member_name": "InSitu/melt_pool_camera_frame.bmp",
            "member_size": "16",
            "extension": ".bmp",
            "roles": "target_observation",
            "coordinate_transform": "false",
            "trigger_timing": "false",
            "source_command_path": "false",
            "target_observation": "true",
            "split_key": "false",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_member_schema_sampler_reads_small_text_prefixes_and_keeps_gates_closed(tmp_path: Path):
    module = _load_module()
    data_root = tmp_path / "data"
    _write_zip(
        data_root / "Metadata.zip",
        {"Metadata/pixel_coordinate_transform.csv": "pixel_x,pixel_y,ammt_x,ammt_y\n1,2,3,4\n"},
    )
    _write_zip(
        data_root / "Build Command Data.zip",
        {"Commands/laser_scan_path_XYPT.csv": "x,y,p,t\n1,2,100,0.1\n"},
    )
    _write_zip(
        data_root / "In-situ Meas Data.zip",
        {
            "InSitu/trigger_timestamp_frame_table.csv": "frame,timestamp\n0,0.0\n",
            "InSitu/melt_pool_camera_frame.bmp": b"BM\x00\x00binary-preview",
        },
    )
    candidates = tmp_path / "candidates.csv"
    _write_candidates(candidates)

    manifest = module.build_package(
        root=Path(".").resolve(),
        data_root=data_root,
        candidates_csv=candidates,
        output_dir=tmp_path / "out",
        max_bytes=64,
    )

    gate = manifest["gate"]
    assert gate["status"] == "member_schema_samples_ready_manual_registration_required"
    assert gate["phase104_baseline_smoke_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    with (tmp_path / "out/phase103_nist_ammt_member_schema_samples.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert any(row["header_line"] == "pixel_x,pixel_y,ammt_x,ammt_y" for row in rows)
    assert any(row["header_line"] == "x,y,p,t" for row in rows)
    assert any(row["sample_status"] == "non_text_preview_skipped" for row in rows)


def test_member_schema_sampler_requires_scout_candidates(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        root=Path(".").resolve(),
        data_root=tmp_path / "data",
        candidates_csv=tmp_path / "missing_candidates.csv",
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert gate["status"] == "schema_scout_candidates_required"
    assert gate["sample_rows"] == 0
    assert gate["phase104_baseline_smoke_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
