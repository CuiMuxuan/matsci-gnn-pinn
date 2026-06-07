from __future__ import annotations

import csv
import importlib.util
import json
import struct
import zipfile
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase150_dense_tensorization_inventory_gate.py")
    spec = importlib.util.spec_from_file_location("phase150_dense_inventory", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def _write_positive_floor(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["floor_id", "split", "route", "manuscript_use"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "floor_id": "P116-FLOOR-001",
                "split": "spot_size",
                "route": "broad_process_v1",
                "manuscript_use": "current_main_text_floor",
            }
        )
    return path


def _bmp_header(width: int = 4, height: int = 3) -> bytes:
    payload = bytearray(64)
    payload[0:2] = b"BM"
    struct.pack_into("<I", payload, 2, len(payload))
    struct.pack_into("<I", payload, 10, 54)
    struct.pack_into("<I", payload, 14, 40)
    struct.pack_into("<i", payload, 18, width)
    struct.pack_into("<i", payload, 22, height)
    struct.pack_into("<H", payload, 26, 1)
    struct.pack_into("<H", payload, 28, 8)
    return bytes(payload)


def _write_zip(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as handle:
        handle.writestr("Layer Camera/layer_0001/camera_A.bmp", _bmp_header())
        handle.writestr("Melt Pool Camera/layer_0001/frame_0001.bmp", _bmp_header(2, 2))
    return path


def _write_dense_csv(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "x",
                "y",
                "t",
                "temperature_C",
                "frame_index",
                "row_index",
                "col_index",
                "line_id",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for frame in range(2):
            for row in range(2):
                for col in range(3):
                    writer.writerow(
                        {
                            "x": col,
                            "y": row,
                            "t": frame * 0.1,
                            "temperature_C": 100 + frame + row + col,
                            "frame_index": frame,
                            "row_index": row,
                            "col_index": col,
                            "line_id": "Line_0_1",
                        }
                    )
    return path


def _phase_inputs(tmp_path: Path) -> dict[str, Path]:
    return {
        "phase149_gate": _write_json(
            tmp_path / "phase149_gate.json",
            {
                "status": "phase149_neural_operator_readiness_closed_not_ready_for_operator_training",
                "phase150_dense_tensorization_inventory_allowed": True,
                "operator_training_allowed_now": False,
                "phase149_model_training_allowed": False,
                "a100_training_allowed_now": False,
                "a100_80gb_request_now": False,
            },
        ),
        "phase116_gate": _write_json(
            tmp_path / "phase116_gate.json",
            {
                "status": "phase116_paper_evidence_consolidation_ready",
                "a100_training_allowed_now": False,
            },
        ),
        "phase116_positive_floor": _write_positive_floor(tmp_path / "phase116_floor.csv"),
        "phase106_gate": _write_json(
            tmp_path / "phase106_gate.json",
            {"status": "phase106_spatial_target_gap_ready_focused_no_training_validation"},
        ),
        "phase148_gate": _write_json(
            tmp_path / "phase148_gate.json",
            {
                "status": "phase148_path_contact_graph_audit_closed_no_guarded_graph_gap",
                "a100_training_allowed_now": False,
            },
        ),
        "phase53_pivot": _write_text(
            tmp_path / "phase53.md",
            "no HDF5 camera-pixel to galvo-mm registration metadata was found",
        ),
    }


def test_phase150_inventories_dense_candidates_without_training(tmp_path: Path):
    module = _load_module()
    zip_path = _write_zip(tmp_path / "raw" / "In-situ Meas Data.zip")
    csv_path = _write_dense_csv(tmp_path / "dense.csv")

    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_phase_inputs(tmp_path),
        candidate_sources=[
            {
                "candidate_id": "toy_layer_zip",
                "source_kind": "zip_bmp_members",
                "path": zip_path,
                "member_needles": ("layer camera",),
                "target_family": "toy layer camera",
            },
            {
                "candidate_id": "toy_dense_csv",
                "source_kind": "dense_csv",
                "path": csv_path,
                "target_column": "temperature_C",
                "target_family": "toy indexed dense csv",
            },
            {
                "candidate_id": "missing_hdf5",
                "source_kind": "hdf5_dataset",
                "path": tmp_path / "missing.h5",
                "dataset_regex": "ThermalData/.*/Signal$",
                "target_family": "missing hdf5",
            },
        ],
        max_preview_rows=100,
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 150
    assert gate["status"] == "phase150_dense_tensorization_inventory_ready_phase151_fixed_grid_baseline_review"
    assert gate["tensorizable_candidate_rows"] >= 2
    assert gate["operator_gap_ready_rows"] == 0
    assert gate["phase151_fixed_grid_baseline_review_allowed"] is True
    assert gate["operator_training_allowed_now"] is False
    assert gate["phase150_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False

    with (tmp_path / "out/phase150_dense_source_inventory_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["candidate_id"]: row for row in rows}
    assert by_id["toy_layer_zip"]["metadata_count"] == "1"
    assert by_id["toy_dense_csv"]["grid_index_columns"] == "frame_index,row_index,col_index"
    assert "candidate_indexed_dense_csv" in by_id["toy_dense_csv"]["tensorization_status"]
    assert by_id["missing_hdf5"]["present"] == "false"

    markdown = (tmp_path / "out/phase150_dense_tensorization_inventory_gate.md").read_text(
        encoding="utf-8"
    )
    assert "Operator training allowed now: `false`" in markdown
    assert "Phase 151 fixed-grid baseline review allowed: `true`" in markdown
