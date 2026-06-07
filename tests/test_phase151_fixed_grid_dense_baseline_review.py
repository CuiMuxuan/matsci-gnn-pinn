from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase151_fixed_grid_dense_baseline_review.py")
    spec = importlib.util.spec_from_file_location("phase151_dense_review", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _write_dense_csv(path: Path) -> Path:
    rows: list[dict[str, object]] = []
    for line_index in range(4):
        line_id = f"Line_{line_index}_1"
        for frame in range(6):
            for row_index in range(2):
                for col_index in range(3):
                    rows.append(
                        {
                            "x": col_index,
                            "y": row_index,
                            "z": 0.0,
                            "t": frame * 0.1,
                            "temperature_C": 100 + line_index * 10 + frame * 2 + row_index + col_index,
                            "frame_index": frame,
                            "row_index": row_index,
                            "col_index": col_index,
                            "dataset_path": f"ThermalData/{line_id}/Signal",
                            "line_id": line_id,
                            "line_index": line_index,
                            "laser_power_W": 280 + line_index,
                            "scan_speed_mm_s": 900 + line_index * 5,
                            "spot_size_um": 60,
                        }
                    )
    return _write_csv(
        path,
        rows,
        [
            "x",
            "y",
            "z",
            "t",
            "temperature_C",
            "frame_index",
            "row_index",
            "col_index",
            "dataset_path",
            "line_id",
            "line_index",
            "laser_power_W",
            "scan_speed_mm_s",
            "spot_size_um",
        ],
    )


def _phase_inputs(tmp_path: Path) -> dict[str, Path]:
    phase150_inventory = _write_csv(
        tmp_path / "phase150_inventory.csv",
        [
            {
                "candidate_id": "toy_dense_csv",
                "source_kind": "dense_csv",
                "path": str(tmp_path / "dense.csv"),
                "present": "true",
                "target_column": "temperature_C",
                "tensorization_status": "candidate_indexed_dense_csv_needs_split_and_operator_baseline",
            }
        ],
        [
            "candidate_id",
            "source_kind",
            "path",
            "present",
            "target_column",
            "tensorization_status",
        ],
    )
    return {
        "phase150_gate": _write_json(
            tmp_path / "phase150_gate.json",
            {
                "status": "phase150_dense_tensorization_inventory_ready_phase151_fixed_grid_baseline_review",
                "operator_training_allowed_now": False,
                "phase150_model_training_allowed": False,
                "a100_training_allowed_now": False,
                "a100_80gb_request_now": False,
            },
        ),
        "phase150_inventory": phase150_inventory,
        "phase149_gate": _write_json(
            tmp_path / "phase149_gate.json",
            {
                "status": "phase149_neural_operator_readiness_closed_not_ready_for_operator_training",
                "operator_training_allowed_now": False,
            },
        ),
        "phase148_gate": _write_json(
            tmp_path / "phase148_gate.json",
            {
                "status": "phase148_path_contact_graph_audit_closed_no_guarded_graph_gap",
                "a100_training_allowed_now": False,
            },
        ),
    }


def test_phase151_builds_fixed_grid_split_and_non_neural_review(tmp_path: Path):
    module = _load_module()
    dense_csv = _write_dense_csv(tmp_path / "dense.csv")

    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_phase_inputs(tmp_path),
        dense_candidates=[
            {
                "candidate_id": "toy_dense_csv",
                "source_path": dense_csv,
                "target_column": "temperature_C",
            }
        ],
        min_points_per_frame=1,
        min_summary_rows=6,
        n_estimators=8,
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 151
    assert gate["status"] in {
        "phase151_fixed_grid_dense_baseline_ready_low_capacity_design",
        "phase151_fixed_grid_dense_baseline_closed_no_operator_gap",
    }
    assert gate["leakage_safe_source_rows"] == 1
    assert gate["phase151_model_training_allowed"] is False
    assert gate["operator_training_allowed_now"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False

    with (tmp_path / "out/phase151_split_contract_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        split_rows = list(csv.DictReader(handle))
    assert split_rows[0]["split_contract_status"] == "leakage_safe_line_group_split"
    assert split_rows[0]["leakage_safe_split"] == "true"

    with (tmp_path / "out/phase151_dense_baseline_metric_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        metric_rows = list(csv.DictReader(handle))
    assert {row["method"] for row in metric_rows} >= {
        "mean",
        "knn",
        "extra_trees",
        "hist_gradient_boosting",
    }

    markdown = (tmp_path / "out/phase151_fixed_grid_dense_baseline_review.md").read_text(
        encoding="utf-8"
    )
    assert "Phase 151 model training allowed: `false`" in markdown
    assert "Operator training allowed now: `false`" in markdown
