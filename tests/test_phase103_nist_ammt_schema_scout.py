from __future__ import annotations

import csv
import importlib.util
import zipfile
from pathlib import Path


def _load_module():
    script = Path("scripts/server/phase103_nist_ammt_schema_scout.py")
    spec = importlib.util.spec_from_file_location("phase103_schema_scout", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_zip(path: Path, members: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in members.items():
            archive.writestr(name, text)


def _write_data_card(tmp_path: Path, data_root: Path) -> Path:
    metadata = data_root / "Metadata.zip"
    command = data_root / "Build Command Data.zip"
    insitu = data_root / "In-situ Meas Data.zip"
    card = tmp_path / "phase102_nist_ammt_data_card.json"
    card.write_text(
        (
            '{"files":['
            '{"file_id":"metadata","file_name":"Metadata.zip",'
            f'"expected_bytes":{metadata.stat().st_size},'
            '"url":"file:///unused/Metadata.zip"},'
            '{"file_id":"commands","file_name":"Build Command Data.zip",'
            f'"expected_bytes":{command.stat().st_size},'
            '"url":"file:///unused/Build%20Command%20Data.zip"},'
            '{"file_id":"insitu","file_name":"In-situ Meas Data.zip",'
            f'"expected_bytes":{insitu.stat().st_size},'
            '"url":"file:///unused/In-situ%20Meas%20Data.zip"},'
            '{"file_id":"movies","file_name":"Movies.zip",'
            '"expected_bytes":999,'
            '"url":"file:///unused/Movies.zip"}]}'
        ),
        encoding="utf-8",
    )
    return card


def test_schema_scout_finds_all_required_candidate_roles(tmp_path: Path):
    module = _load_module()
    data_root = tmp_path / "data"
    _write_zip(
        data_root / "Metadata.zip",
        {
            "Metadata/AMMT_pixel_coordinate_transform.csv": "x,y",
            "Metadata/dotgrid_calibration_notes.txt": "grid",
        },
    )
    _write_zip(
        data_root / "Build Command Data.zip",
        {
            "BuildCommands/laser_scan_path_XYPT_commands.csv": "x,y,p,t",
            "BuildCommands/layer_index_table.csv": "layer,scan",
        },
    )
    _write_zip(
        data_root / "In-situ Meas Data.zip",
        {
            "InSitu/trigger_timestamp_frame_table.csv": "frame,t",
            "InSitu/melt_pool_monitoring_camera_frames.csv": "frame,path",
        },
    )
    data_card = _write_data_card(tmp_path, data_root)

    manifest = module.build_package(
        root=Path(".").resolve(),
        data_card_path=data_card,
        data_root=data_root,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert gate["status"] == "schema_candidates_ready_manual_sampling_required"
    assert gate["missing_required_roles"] == []
    assert gate["phase104_baseline_smoke_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["role_hits"]["coordinate_transform"] >= 1
    assert gate["role_hits"]["trigger_timing"] >= 1
    assert gate["role_hits"]["source_command_path"] >= 1
    assert gate["role_hits"]["target_observation"] >= 1

    with (tmp_path / "out/phase103_nist_ammt_schema_scout_candidates.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert {row["file_name"] for row in rows} == {
        "Metadata.zip",
        "Build Command Data.zip",
        "In-situ Meas Data.zip",
    }


def test_schema_scout_keeps_large_intake_incomplete_when_required_zip_missing(tmp_path: Path):
    module = _load_module()
    data_root = tmp_path / "data"
    _write_zip(data_root / "Metadata.zip", {"Metadata/AMMT_pixel_coordinate_transform.csv": "x,y"})
    _write_zip(data_root / "Build Command Data.zip", {"BuildCommands/laser_scan_path.csv": "x,y"})
    missing_insitu = data_root / "In-situ Meas Data.zip"
    missing_insitu.parent.mkdir(parents=True, exist_ok=True)
    _write_zip(missing_insitu, {"temporary/member.txt": "placeholder"})
    data_card = _write_data_card(tmp_path, data_root)
    missing_insitu.unlink()

    manifest = module.build_package(
        root=Path(".").resolve(),
        data_card_path=data_card,
        data_root=data_root,
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert gate["status"] == "large_intake_incomplete"
    assert gate["missing_or_invalid_required_rows"] == 1
    assert gate["phase104_baseline_smoke_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
