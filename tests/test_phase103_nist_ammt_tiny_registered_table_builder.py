from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/phase103_nist_ammt_tiny_registered_table_builder.py")
    spec = importlib.util.spec_from_file_location("phase103_tiny_builder", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_sequence_groups(path: Path) -> Path:
    rows = [
        {
            "file_name": "Build Command Data.zip",
            "group_key": "Build Command Data/XYPT Commands/T500_3D_Scan_Strategies_fused_layer{index}.csv",
            "directory": "Build Command Data/XYPT Commands",
            "extension": ".csv",
            "count": "249",
            "first_index": "2",
            "last_index": "250",
            "zero_padded_width": "4",
            "min_member_size": "100",
            "max_member_size": "200",
            "example_first": "Build Command Data/XYPT Commands/T500_3D_Scan_Strategies_fused_layer0002.csv",
            "example_last": "Build Command Data/XYPT Commands/T500_3D_Scan_Strategies_fused_layer0250.csv",
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
            "example_first": "In-situ Meas Data/Layer Camera/A0002.bmp",
            "example_last": "In-situ Meas Data/Layer Camera/A0247.bmp",
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
            "example_first": "In-situ Meas Data/Layer Camera/B0001.bmp",
            "example_last": "In-situ Meas Data/Layer Camera/B0246.bmp",
            "target_observation_candidate": "true",
            "implicit_sequence_index_ready": "true",
        },
    ]
    return _write_csv(path, rows)


def _write_join_candidates(path: Path) -> Path:
    rows = [
        {
            "source_group_key": "Build Command Data/XYPT Commands/T500_3D_Scan_Strategies_fused_layer{index}.csv",
            "target_group_key": "In-situ Meas Data/Layer Camera/A{index}.bmp",
            "target_type": "layer_camera",
            "source_first_index": "2",
            "source_last_index": "250",
            "source_count": "249",
            "target_first_index": "2",
            "target_last_index": "247",
            "target_count": "246",
            "best_source_minus_target_offset": "0",
            "matched_pairs": "246",
            "source_coverage": "0.987952",
            "target_coverage": "1.000000",
            "first_pair": "source_layer=2;target_index=2",
            "last_pair": "source_layer=247;target_index=247",
            "join_evidence_status": "source_target_layer_join_ready",
        },
        {
            "source_group_key": "Build Command Data/XYPT Commands/T500_3D_Scan_Strategies_fused_layer{index}.csv",
            "target_group_key": "In-situ Meas Data/Layer Camera/B{index}.bmp",
            "target_type": "layer_camera",
            "source_first_index": "2",
            "source_last_index": "250",
            "source_count": "249",
            "target_first_index": "1",
            "target_last_index": "246",
            "target_count": "246",
            "best_source_minus_target_offset": "1",
            "matched_pairs": "246",
            "source_coverage": "0.987952",
            "target_coverage": "1.000000",
            "first_pair": "source_layer=2;target_index=1",
            "last_pair": "source_layer=247;target_index=246",
            "join_evidence_status": "source_target_layer_join_ready",
        },
    ]
    return _write_csv(path, rows)


def _write_target_samples(path: Path, *, include_b: bool = True) -> Path:
    rows = [
        {
            "file_name": "In-situ Meas Data.zip",
            "member_name": "In-situ Meas Data/Layer Camera/A0002.bmp",
            "member_size": "4001078",
            "extension": ".bmp",
            "format": "bmp",
            "width": "2000",
            "height": "2000",
            "bits_per_pixel": "8",
            "channels": "1",
            "header_status": "parsed_binary_header",
            "target_observation_binary_schema_ready": "true",
        },
    ]
    if include_b:
        rows.append(
            {
                "file_name": "In-situ Meas Data.zip",
                "member_name": "In-situ Meas Data/Layer Camera/B0001.bmp",
                "member_size": "4001078",
                "extension": ".bmp",
                "format": "bmp",
                "width": "2000",
                "height": "2000",
                "bits_per_pixel": "8",
                "channels": "1",
                "header_status": "parsed_binary_header",
                "target_observation_binary_schema_ready": "true",
            }
        )
    return _write_csv(path, rows)


def _write_inputs(tmp_path: Path, *, include_b_sample: bool = True) -> dict[str, Path]:
    out = tmp_path / "out"
    return {
        "sequence_groups": _write_sequence_groups(tmp_path / "sequence_groups.csv"),
        "join_gate": _write_json(
            tmp_path / "join_gate.json",
            {
                "status": "source_target_layer_join_ready_timing_not_absolute",
                "source_target_join_ready": True,
                "explicit_absolute_timing_ready": False,
            },
        ),
        "join_candidates": _write_join_candidates(
            out / "phase103_nist_ammt_source_target_join_candidates.csv"
        ),
        "target_samples": _write_target_samples(
            tmp_path / "target_samples.csv",
            include_b=include_b_sample,
        ),
        "out": out,
    }


def test_tiny_registered_table_builder_writes_layer_join_table_and_split_manifest(
    tmp_path: Path,
):
    module = _load_module()
    paths = _write_inputs(tmp_path)

    manifest = module.build_package(
        root=Path(".").resolve(),
        sequence_groups_csv=paths["sequence_groups"],
        join_probe_gate_path=paths["join_gate"],
        join_candidates_csv=paths["join_candidates"],
        target_binary_samples_csv=paths["target_samples"],
        output_dir=paths["out"],
        rows_per_target_type=3,
    )

    gate = manifest["gate"]
    assert gate["status"] == "tiny_registered_table_ready_manual_baseline_pending"
    assert gate["tiny_registered_table_ready"] is True
    assert gate["leakage_safe_split_manifest_ready"] is True
    assert gate["phase104_baseline_smoke_allowed"] is False
    assert gate["phase105_model_mechanism_allowed"] is False
    assert gate["a100_training_allowed_now"] is False

    with (paths["out"] / "phase103_nist_ammt_tiny_registered_source_target_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert len({row["row_id"] for row in rows}) == 6
    assert rows[0]["source_member_name"].endswith(
        "T500_3D_Scan_Strategies_fused_layer0002.csv"
    )
    assert rows[0]["target_member_name"] == "In-situ Meas Data/Layer Camera/A0002.bmp"
    assert rows[3]["source_member_name"].endswith(
        "T500_3D_Scan_Strategies_fused_layer0002.csv"
    )
    assert rows[3]["target_member_name"] == "In-situ Meas Data/Layer Camera/B0001.bmp"
    splits_by_source: dict[str, set[str]] = {}
    for row in rows:
        splits_by_source.setdefault(row["source_layer_index"], set()).add(row["split_name"])
    assert all(len(splits) == 1 for splits in splits_by_source.values())

    split_manifest = json.loads(
        (paths["out"] / "phase103_nist_ammt_tiny_registered_split_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert split_manifest["leakage_group"] == "source_layer_index"
    assert split_manifest["leakage_safe"] is True
    assert split_manifest["split_counts"] == {"test": 2, "train": 2, "val": 2}
    assert (
        paths["out"] / "phase103_nist_ammt_tiny_registered_table_summary.md"
    ).exists()


def test_tiny_registered_table_builder_blocks_when_binary_schema_sample_is_missing(
    tmp_path: Path,
):
    module = _load_module()
    paths = _write_inputs(tmp_path, include_b_sample=False)

    manifest = module.build_package(
        root=Path(".").resolve(),
        sequence_groups_csv=paths["sequence_groups"],
        join_probe_gate_path=paths["join_gate"],
        join_candidates_csv=paths["join_candidates"],
        target_binary_samples_csv=paths["target_samples"],
        output_dir=paths["out"],
        rows_per_target_type=3,
    )

    gate = manifest["gate"]
    assert gate["status"] == "tiny_registered_table_ready_manual_baseline_pending"
    assert gate["tiny_registered_table_ready"] is True
    assert gate["phase104_baseline_smoke_allowed"] is False

    with (paths["out"] / "phase103_nist_ammt_tiny_registered_source_target_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert {row["target_group_key"] for row in rows} == {
        "In-situ Meas Data/Layer Camera/A{index}.bmp"
    }
