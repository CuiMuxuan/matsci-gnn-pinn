from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/phase103_nist_ammt_tiny_table_feasibility_gate.py")
    spec = importlib.util.spec_from_file_location("phase103_tiny_table_gate", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_samples(path: Path, *, include_target: bool = True) -> Path:
    rows = [
        {
            "file_id": "metadata",
            "file_name": "Metadata.zip",
            "member_name": "Metadata/pixel_coordinate_transform.csv",
            "member_size": "24",
            "extension": ".csv",
            "roles": "coordinate_transform",
            "sample_status": "sampled_text",
            "bytes_read": "24",
            "line_count": "2",
            "header_line": "pixel_x,pixel_y,ammt_x,ammt_y",
            "preview_text": "pixel_x,pixel_y,ammt_x,ammt_y\n1,2,3,4",
        },
        {
            "file_id": "commands",
            "file_name": "Build Command Data.zip",
            "member_name": "Commands/laser_scan_path_XYPT.csv",
            "member_size": "22",
            "extension": ".csv",
            "roles": "source_command_path",
            "sample_status": "sampled_text",
            "bytes_read": "22",
            "line_count": "2",
            "header_line": "x,y,p,t",
            "preview_text": "x,y,p,t\n1,2,100,0.1",
        },
        {
            "file_id": "insitu",
            "file_name": "In-situ Meas Data.zip",
            "member_name": "InSitu/trigger_timestamp_frame_table.csv",
            "member_size": "18",
            "extension": ".csv",
            "roles": "trigger_timing",
            "sample_status": "sampled_text",
            "bytes_read": "18",
            "line_count": "2",
            "header_line": "frame,timestamp",
            "preview_text": "frame,timestamp\n0,0.0",
        },
    ]
    if include_target:
        rows.append(
            {
                "file_id": "insitu",
                "file_name": "In-situ Meas Data.zip",
                "member_name": "InSitu/melt_pool_monitoring_camera_frames.csv",
                "member_size": "24",
                "extension": ".csv",
                "roles": "target_observation",
                "sample_status": "sampled_text",
                "bytes_read": "24",
                "line_count": "2",
                "header_line": "frame,image_path,intensity",
                "preview_text": "frame,image_path,intensity\n0,a.bmp,1",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_ready_gates(tmp_path: Path) -> dict[str, Path]:
    intake = _write_json(
        tmp_path / "intake_gate.json",
        {"status": "metadata_ready_registration_evidence_pending", "metadata_ready": True},
    )
    scout = _write_json(
        tmp_path / "scout_gate.json",
        {
            "status": "schema_candidates_ready_manual_sampling_required",
            "role_hits": {
                "coordinate_transform": 2,
                "trigger_timing": 1,
                "source_command_path": 1,
                "target_observation": 1,
                "split_key": 1,
            },
        },
    )
    sampler = _write_json(
        tmp_path / "sampler_gate.json",
        {"status": "member_schema_samples_ready_manual_registration_required"},
    )
    deep_probe = _write_json(
        tmp_path / "deep_probe_gate.json",
        {
            "status": "deep_probe_ready_manual_registration_required",
            "target_observation_binary_schema_ready": True,
            "explicit_trigger_timing_ready": True,
        },
    )
    samples = _write_samples(tmp_path / "samples.csv")
    return {
        "intake": intake,
        "scout": scout,
        "sampler": sampler,
        "deep_probe": deep_probe,
        "samples": samples,
    }


def test_tiny_table_feasibility_allows_manual_construction_but_keeps_training_locked(
    tmp_path: Path,
):
    module = _load_module()
    paths = _write_ready_gates(tmp_path)

    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        intake_gate_path=paths["intake"],
        scout_gate_path=paths["scout"],
        sampler_gate_path=paths["sampler"],
        deep_probe_gate_path=paths["deep_probe"],
        samples_csv_path=paths["samples"],
    )

    gate = manifest["gate"]
    assert gate["status"] == "tiny_registered_table_construction_allowed_training_locked"
    assert gate["tiny_registered_table_construction_allowed"] is True
    assert gate["tiny_registered_table_ready"] is False
    assert gate["leakage_safe_split_manifest_ready"] is False
    assert gate["phase104_baseline_smoke_allowed"] is False
    assert gate["phase105_model_mechanism_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    with (tmp_path / "out/phase103_nist_ammt_tiny_table_feasibility_roles.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert {row["role"] for row in rows if row["required"] == "true"} == {
        "coordinate_transform",
        "trigger_timing",
        "source_command_path",
        "target_observation",
    }
    assert all(
        row["status"] == "ready_for_manual_join_review"
        for row in rows
        if row["required"] == "true"
    )


def test_tiny_table_feasibility_blocks_when_sampler_gate_missing(tmp_path: Path):
    module = _load_module()
    paths = _write_ready_gates(tmp_path)

    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        intake_gate_path=paths["intake"],
        scout_gate_path=paths["scout"],
        sampler_gate_path=tmp_path / "missing_sampler_gate.json",
        deep_probe_gate_path=paths["deep_probe"],
        samples_csv_path=paths["samples"],
    )

    gate = manifest["gate"]
    assert gate["status"] == "member_schema_sampler_gate_required"
    assert gate["tiny_registered_table_construction_allowed"] is False
    assert gate["phase104_baseline_smoke_allowed"] is False
    assert gate["a100_training_allowed_now"] is False


def test_tiny_table_feasibility_blocks_when_required_role_lacks_text_sample(tmp_path: Path):
    module = _load_module()
    paths = _write_ready_gates(tmp_path)
    samples = _write_samples(tmp_path / "samples_missing_target.csv", include_target=False)

    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        intake_gate_path=paths["intake"],
        scout_gate_path=paths["scout"],
        sampler_gate_path=paths["sampler"],
        deep_probe_gate_path=tmp_path / "missing_deep_probe_gate.json",
        samples_csv_path=samples,
    )

    gate = manifest["gate"]
    assert gate["status"] == "tiny_registered_table_feasibility_blocked"
    assert gate["missing_or_blocked_required_roles"] == ["target_observation"]
    assert gate["phase104_baseline_smoke_allowed"] is False
    assert gate["a100_training_allowed_now"] is False


def test_tiny_table_feasibility_accepts_deep_binary_target_schema_but_keeps_training_locked(
    tmp_path: Path,
):
    module = _load_module()
    paths = _write_ready_gates(tmp_path)
    samples = _write_samples(tmp_path / "samples_missing_target.csv", include_target=False)

    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        intake_gate_path=paths["intake"],
        scout_gate_path=paths["scout"],
        sampler_gate_path=paths["sampler"],
        deep_probe_gate_path=paths["deep_probe"],
        samples_csv_path=samples,
    )

    gate = manifest["gate"]
    assert gate["status"] == "tiny_registered_table_construction_allowed_training_locked"
    assert gate["role_statuses"]["target_observation"] == (
        "ready_for_manual_join_review_binary_schema"
    )
    assert gate["tiny_registered_table_construction_allowed"] is True
    assert gate["phase104_baseline_smoke_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
