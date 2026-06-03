from __future__ import annotations

import csv
import importlib.util
import zipfile
from pathlib import Path


def _load_module():
    script = Path("scripts/server/phase103_nist_ammt_registered_intake_audit.py")
    spec = importlib.util.spec_from_file_location("phase103_intake", script)
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


def _phase102_inputs(tmp_path: Path, metadata_bytes: int) -> dict[str, Path]:
    gate = tmp_path / "phase102_gate.json"
    card = tmp_path / "phase102_card.json"
    gate.write_text(
        '{"status":"source_manifest_ready_phase103_intake_allowed","phase103_intake_allowed":true}\n',
        encoding="utf-8",
    )
    card.write_text(
        (
            '{"files":['
            '{"file_id":"p102_nist_metadata_zip","file_name":"Metadata.zip",'
            f'"expected_bytes":{metadata_bytes},'
            '"url":"file:///unused/Metadata.zip",'
            '"required_for_phase103":true,'
            '"download_scope":"minimal_registration_metadata_intake"},'
            '{"file_id":"p102_nist_build_command_data_zip","file_name":"Build Command Data.zip",'
            '"expected_bytes":999,'
            '"url":"file:///unused/Build%20Command%20Data.zip",'
            '"required_for_phase103":false,'
            '"download_scope":"long_running_server_download_after_metadata_pass"}]}'
        ),
        encoding="utf-8",
    )
    return {"phase102_gate": gate, "phase102_data_card": card}


def test_phase103_metadata_zip_audit_finds_registration_candidates(tmp_path: Path):
    module = _load_module()
    data_root = tmp_path / "data"
    metadata = data_root / "Metadata.zip"
    _write_zip(
        metadata,
        {
            "calibration/pixel_to_AMMT_coordinate_transform.csv": "x,y",
            "timing/trigger_timestamp_table.csv": "t,frame",
            "commands/XYPT_scan_path_commands.csv": "x,y,p,t",
            "monitoring/melt_pool_frame_index.csv": "frame,path",
        },
    )
    paths = _phase102_inputs(tmp_path, metadata.stat().st_size)

    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        data_root=data_root,
        paths=paths,
        download=False,
    )

    gate = manifest["gate"]
    assert gate["status"] == "metadata_ready_registration_schema_candidate"
    assert gate["metadata_ready"] is True
    assert gate["registration_keyword_hits"] >= 1
    assert gate["timing_keyword_hits"] >= 1
    assert gate["command_keyword_hits"] >= 1
    assert gate["target_keyword_hits"] >= 1
    assert gate["phase104_baseline_smoke_allowed"] is False
    assert gate["a100_training_allowed_now"] is False

    with (tmp_path / "out/phase103_nist_ammt_file_audit.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    metadata_row = next(row for row in rows if row["file_name"] == "Metadata.zip")
    assert metadata_row["size_ok"] == "true"
    assert metadata_row["zip_status"] == "valid_zip"
    assert metadata_row["status"] == "ready_for_schema_audit"


def test_phase103_missing_metadata_keeps_intake_incomplete(tmp_path: Path):
    module = _load_module()
    paths = _phase102_inputs(tmp_path, metadata_bytes=100)

    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        data_root=tmp_path / "data",
        paths=paths,
        download=False,
    )

    gate = manifest["gate"]
    assert gate["status"] == "metadata_intake_incomplete"
    assert gate["metadata_ready"] is False
    assert gate["required_missing_rows"] == 1
    assert gate["phase104_baseline_smoke_allowed"] is False
    assert gate["a100_training_allowed_now"] is False


def test_phase103_size_mismatch_is_not_ready(tmp_path: Path):
    module = _load_module()
    data_root = tmp_path / "data"
    metadata = data_root / "Metadata.zip"
    _write_zip(metadata, {"metadata/readme.txt": "small"})
    paths = _phase102_inputs(tmp_path, metadata_bytes=metadata.stat().st_size + 1)

    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        data_root=data_root,
        paths=paths,
        download=False,
    )

    gate = manifest["gate"]
    assert gate["status"] == "metadata_intake_incomplete"
    assert gate["metadata_ready"] is False
    assert gate["size_mismatch_rows"] == 1


def test_phase103_existing_partial_large_file_is_resumed(tmp_path: Path, monkeypatch):
    module = _load_module()
    data_root = tmp_path / "data"
    metadata = data_root / "Metadata.zip"
    _write_zip(
        metadata,
        {
            "calibration/pixel_to_AMMT_coordinate_transform.csv": "x,y",
            "timing/trigger_timestamp_table.csv": "t,frame",
        },
    )
    full_build = tmp_path / "full_build.zip"
    _write_zip(full_build, {"commands/XYPT_scan_path_commands.csv": "x,y,p,t"})
    partial_build = data_root / "Build Command Data.zip"
    partial_build.write_bytes(b"partial")

    gate = tmp_path / "phase102_gate.json"
    card = tmp_path / "phase102_card.json"
    gate.write_text(
        '{"status":"source_manifest_ready_phase103_intake_allowed","phase103_intake_allowed":true}\n',
        encoding="utf-8",
    )
    card.write_text(
        (
            '{"files":['
            '{"file_id":"p102_nist_metadata_zip","file_name":"Metadata.zip",'
            f'"expected_bytes":{metadata.stat().st_size},'
            '"url":"file:///unused/Metadata.zip",'
            '"required_for_phase103":true,'
            '"download_scope":"minimal_registration_metadata_intake"},'
            '{"file_id":"p102_nist_build_command_data_zip","file_name":"Build Command Data.zip",'
            f'"expected_bytes":{full_build.stat().st_size},'
            '"url":"file:///unused/Build%20Command%20Data.zip",'
            '"required_for_phase103":false,'
            '"download_scope":"long_running_server_download_after_metadata_pass"}]}'
        ),
        encoding="utf-8",
    )

    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        assert kwargs["output"] == partial_build
        assert kwargs["resume"] is True
        partial_build.write_bytes(full_build.read_bytes())
        return "downloaded_wget"

    monkeypatch.setattr(module, "_download_with_external", fake_download)

    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        data_root=data_root,
        paths={"phase102_gate": gate, "phase102_data_card": card},
        download=True,
        large_downloads=True,
        backend="wget",
    )

    assert len(calls) == 1
    with (tmp_path / "out/phase103_nist_ammt_file_audit.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    build_row = next(row for row in rows if row["file_name"] == "Build Command Data.zip")
    assert build_row["download_status"] == "downloaded_wget"
    assert build_row["size_ok"] == "true"
    assert build_row["status"] == "ready_for_schema_audit"
    assert manifest["gate"]["phase104_baseline_smoke_allowed"] is False
