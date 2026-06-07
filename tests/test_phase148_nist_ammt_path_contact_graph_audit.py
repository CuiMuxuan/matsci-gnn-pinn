from __future__ import annotations

import csv
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest


def _load_module():
    script = Path("scripts/server/phase148_nist_ammt_path_contact_graph_audit.py")
    spec = importlib.util.spec_from_file_location("phase148_path_contact_graph", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_csv(layer: int) -> str:
    rows = []
    for index in range(24):
        if layer % 2:
            x = index % 6
            y = index // 6
        else:
            x = index // 4
            y = index % 4
        if layer >= 4 and index % 6 in {0, 1}:
            x = 1
            y = index % 4
        power = 10.0 + layer + (index % 3) * 0.5
        rows.append(f"{x},{y},{power},{index * 0.01}")
    return "\n".join(rows) + "\n"


def _write_zips(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(root / "Build Command Data.zip", "w") as archive:
        for layer in range(1, 8):
            archive.writestr(
                f"Build Command Data/XYPT Commands/layer{layer:04d}.csv",
                _source_csv(layer),
            )


def _write_spatial_field_table(path: Path) -> Path:
    rows = []
    split_names = ["train", "train", "train", "val", "val", "test", "test"]
    targets = [0.2, 0.4, 0.3, 2.0, 2.2, 2.4, 2.6]
    for layer, split_name in enumerate(split_names, start=1):
        rows.append(
            {
                "x": str(layer),
                "y": "0",
                "t": str(layer),
                "target_center_periphery_contrast": str(targets[layer - 1]),
                "source_layer_index": str(layer),
                "target_camera_code": str(layer % 2),
                "source_p_mean": "10",
                "source_p_nonzero_fraction": "1",
                "source_x_range": "5",
                "source_y_range": "3",
                "split_name": split_name,
                "row_id": f"row::{layer:04d}",
                "source_file_name": "Build Command Data.zip",
                "source_member_name": f"Build Command Data/XYPT Commands/layer{layer:04d}.csv",
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


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_phase106_gate(path: Path) -> Path:
    return _write_json(
        path,
        {
            "status": "phase106_spatial_target_gap_ready_focused_no_training_validation",
            "selected_target": "target_center_periphery_contrast",
            "selected_validation_method": "hist_gradient_boosting",
            "selected_validation_rmse": 100.0,
            "selected_test_rmse": 100.0,
            "phase106_model_training_allowed": False,
            "a100_training_allowed_now": False,
        },
    )


def _write_phase114_gate(path: Path, *, closed: bool = True) -> Path:
    return _write_json(
        path,
        {
            "status": "phase114_gcode_strategy_source_gate_closed_no_guarded_baseline_gap"
            if closed
            else "phase114_gcode_strategy_source_gate_ready_focused_review",
            "phase114_model_training_allowed": False,
            "a100_training_allowed_now": False,
            "a100_80gb_request_now": False,
        },
    )


def _write_phase147_gate(path: Path, *, ready: bool = True) -> Path:
    return _write_json(
        path,
        {
            "status": "phase147_literature_guided_model_roadmap_ready_phase148_no_training_design"
            if ready
            else "phase147_literature_guided_model_roadmap_incomplete",
            "phase148_no_training_design_allowed": ready,
            "phase147_model_training_allowed": False,
            "a100_training_allowed_now": False,
            "a100_80gb_request_now": False,
        },
    )


def test_path_contact_graph_audit_writes_features_and_keeps_training_locked(tmp_path: Path):
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn is not installed in this environment")
    module = _load_module()
    data_root = tmp_path / "data"
    _write_zips(data_root)

    manifest = module.build_package(
        root=Path(".").resolve(),
        data_root=data_root,
        spatial_field_table=_write_spatial_field_table(tmp_path / "spatial.csv"),
        split_manifest=_write_split(tmp_path / "split.json"),
        phase106_gate_path=_write_phase106_gate(tmp_path / "phase106_gate.json"),
        phase114_gate_path=_write_phase114_gate(tmp_path / "phase114_gate.json"),
        phase147_gate_path=_write_phase147_gate(tmp_path / "phase147_gate.json"),
        output_dir=tmp_path / "out",
        min_validation_improvement=-100.0,
        min_control_margin=-100.0,
        max_source_rows=None,
        max_graph_points=24,
        contact_radius_fraction=0.25,
        n_neighbors=1,
        n_estimators=5,
    )

    gate = manifest["gate"]
    assert gate["target"] == "target_center_periphery_contrast"
    assert gate["phase147_gate_status"] == "phase147_literature_guided_model_roadmap_ready_phase148_no_training_design"
    assert gate["phase148_model_mechanism_allowed"] is False
    assert gate["phase148_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False

    with (tmp_path / "out/phase148_nist_ammt_path_contact_graph_augmented_field_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 7
    assert "path_contact_degree_mean" in rows[0]
    assert "path_reheat_count_mean" in rows[0]
    assert "path_reheat_count_mean_shuffled" in rows[0]
    assert "path_reheat_count_mean_ordered_minus_shuffled" in rows[0]

    with (tmp_path / "out/phase148_nist_ammt_path_contact_graph_metric_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        metric_rows = list(csv.DictReader(handle))
    assert len(metric_rows) == len(module.FEATURE_PROFILES) * len(module.METHODS) * 3


def test_path_contact_graph_audit_blocks_when_phase147_not_ready(tmp_path: Path):
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn is not installed in this environment")
    module = _load_module()
    data_root = tmp_path / "data"
    _write_zips(data_root)

    manifest = module.build_package(
        root=Path(".").resolve(),
        data_root=data_root,
        spatial_field_table=_write_spatial_field_table(tmp_path / "spatial.csv"),
        split_manifest=_write_split(tmp_path / "split.json"),
        phase106_gate_path=_write_phase106_gate(tmp_path / "phase106_gate.json"),
        phase114_gate_path=_write_phase114_gate(tmp_path / "phase114_gate.json"),
        phase147_gate_path=_write_phase147_gate(tmp_path / "phase147_gate.json", ready=False),
        output_dir=tmp_path / "out",
        min_validation_improvement=-100.0,
        min_control_margin=-100.0,
        max_source_rows=None,
        max_graph_points=24,
        contact_radius_fraction=0.25,
        n_neighbors=1,
        n_estimators=5,
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase148_path_contact_graph_audit_blocked_by_phase147"
    assert gate["phase148_focused_review_allowed"] is False
    assert gate["phase148_model_training_allowed"] is False
