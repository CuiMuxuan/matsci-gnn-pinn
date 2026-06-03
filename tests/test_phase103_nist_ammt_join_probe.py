from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


def _load_module():
    script = Path("scripts/server/phase103_nist_ammt_join_probe.py")
    spec = importlib.util.spec_from_file_location("phase103_join_probe", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_sequence_groups(path: Path) -> Path:
    rows = [
        {
            "file_name": "Build Command Data.zip",
            "group_key": "Build Command Data/XYPT Commands/T500_layer{index}.csv",
            "directory": "Build Command Data/XYPT Commands",
            "extension": ".csv",
            "count": "249",
            "first_index": "2",
            "last_index": "250",
            "zero_padded_width": "4",
            "min_member_size": "100",
            "max_member_size": "200",
            "example_first": "T500_layer0002.csv",
            "example_last": "T500_layer0250.csv",
            "target_observation_candidate": "false",
            "implicit_sequence_index_ready": "true",
        },
        {
            "file_name": "In-situ Meas Data.zip",
            "group_key": "In-situ Meas Data/Layer Camera/A{index}.bmp",
            "directory": "In-situ Meas Data/Layer Camera",
            "extension": ".bmp",
            "count": "246",
            "first_index": "2",
            "last_index": "247",
            "zero_padded_width": "4",
            "min_member_size": "4001078",
            "max_member_size": "4001078",
            "example_first": "A0002.bmp",
            "example_last": "A0247.bmp",
            "target_observation_candidate": "true",
            "implicit_sequence_index_ready": "true",
        },
        {
            "file_name": "In-situ Meas Data.zip",
            "group_key": "In-situ Meas Data/Layer Camera/B{index}.bmp",
            "directory": "In-situ Meas Data/Layer Camera",
            "extension": ".bmp",
            "count": "246",
            "first_index": "1",
            "last_index": "246",
            "zero_padded_width": "4",
            "min_member_size": "4001078",
            "max_member_size": "4001078",
            "example_first": "B0001.bmp",
            "example_last": "B0246.bmp",
            "target_observation_candidate": "true",
            "implicit_sequence_index_ready": "true",
        },
        {
            "file_name": "In-situ Meas Data.zip",
            "group_key": "In-situ Meas Data/Melt Pool Camera/MIA_L0001/frame{index}.bmp",
            "directory": "In-situ Meas Data/Melt Pool Camera/MIA_L0001",
            "extension": ".bmp",
            "count": "20",
            "first_index": "1",
            "last_index": "20",
            "zero_padded_width": "5",
            "min_member_size": "16438",
            "max_member_size": "16438",
            "example_first": "frame00001.bmp",
            "example_last": "frame00020.bmp",
            "target_observation_candidate": "true",
            "implicit_sequence_index_ready": "true",
        },
        {
            "file_name": "In-situ Meas Data.zip",
            "group_key": "In-situ Meas Data/Melt Pool Camera/MIA_L0002/frame{index}.bmp",
            "directory": "In-situ Meas Data/Melt Pool Camera/MIA_L0002",
            "extension": ".bmp",
            "count": "21",
            "first_index": "1",
            "last_index": "21",
            "zero_padded_width": "5",
            "min_member_size": "16438",
            "max_member_size": "16438",
            "example_first": "frame00001.bmp",
            "example_last": "frame00021.bmp",
            "target_observation_candidate": "true",
            "implicit_sequence_index_ready": "true",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_join_probe_finds_layer_camera_offset_and_keeps_training_locked(tmp_path: Path):
    module = _load_module()
    sequence_groups = _write_sequence_groups(tmp_path / "sequence_groups.csv")

    manifest = module.build_package(
        root=Path(".").resolve(),
        sequence_groups_csv=sequence_groups,
        output_dir=tmp_path / "out",
        min_target_coverage=0.95,
        min_layer_pairs=20,
        min_melt_pool_pairs=2,
    )

    gate = manifest["gate"]
    assert gate["status"] == "source_target_layer_join_ready_timing_not_absolute"
    assert gate["source_target_join_ready"] is True
    assert gate["explicit_absolute_timing_ready"] is False
    assert gate["layer_camera_join_ready"] is True
    assert gate["phase104_baseline_smoke_allowed"] is False
    assert gate["a100_training_allowed_now"] is False

    with (tmp_path / "out/phase103_nist_ammt_source_target_join_candidates.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    a_row = next(row for row in rows if row["target_group_key"].endswith("A{index}.bmp"))
    b_row = next(row for row in rows if row["target_group_key"].endswith("B{index}.bmp"))
    assert a_row["best_source_minus_target_offset"] == "0"
    assert b_row["best_source_minus_target_offset"] == "1"
    assert a_row["join_evidence_status"] == "source_target_layer_join_ready"
    assert b_row["join_evidence_status"] == "source_target_layer_join_ready"


def test_join_probe_blocks_when_no_sequence_groups_exist(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        root=Path(".").resolve(),
        sequence_groups_csv=tmp_path / "missing.csv",
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert gate["status"] == "source_target_join_incomplete"
    assert gate["source_target_join_ready"] is False
    assert gate["phase104_baseline_smoke_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
