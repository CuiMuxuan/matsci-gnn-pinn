from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    script = Path("scripts/server/phase108_nist_ammt_sequence_target_gate.py")
    spec = importlib.util.spec_from_file_location("phase108_sequence_gate", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_spatial_field_table(path: Path) -> Path:
    rows = []
    split_by_layer = {1: "train", 2: "train", 3: "train", 4: "val", 5: "val", 6: "test", 7: "test"}
    for layer in range(1, 8):
        for camera in (0, 1):
            value = layer * 2.0 + (1.5 if camera == 1 else -0.5)
            rows.append(
                {
                    "x": str(layer),
                    "y": "0",
                    "t": str(layer),
                    "target_center_periphery_contrast": str(value),
                    "source_layer_index": str(layer),
                    "target_camera_code": str(camera),
                    "source_p_mean": str(layer + 1),
                    "source_p_nonzero_fraction": "1",
                    "source_x_range": "4",
                    "source_y_range": "3",
                    "split_name": split_by_layer[layer],
                    "row_id": f"{camera}::{layer:04d}",
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
        json.dumps(
            {
                "splits": {
                    "train": [0, 1, 2, 3, 4, 5],
                    "val": [6, 7, 8, 9],
                    "test": [10, 11, 12, 13],
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_phase106_gate(path: Path, *, ready: bool = True) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": "phase106_spatial_target_gap_ready_focused_no_training_validation"
                if ready
                else "phase106_spatial_target_gate_closed_no_baseline_gap",
                "selected_target": "target_center_periphery_contrast",
                "phase106_seed7_focused_validation_allowed": ready,
                "phase106_model_training_allowed": False,
                "a100_training_allowed_now": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_phase107_gate(path: Path, *, closed: bool = True) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": "phase107_source_region_feature_gate_blocked_no_phase106_gain"
                if closed
                else "phase107_source_region_feature_gate_ready_focused_review",
                "phase107_focused_review_allowed": not closed,
                "phase107_model_training_allowed": False,
                "a100_training_allowed_now": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_sequence_gate_writes_targets_and_keeps_training_locked(tmp_path: Path):
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn is not installed in this environment")
    module = _load_module()

    manifest = module.build_package(
        root=Path(".").resolve(),
        spatial_field_table=_write_spatial_field_table(tmp_path / "spatial.csv"),
        split_manifest=_write_split(tmp_path / "split.json"),
        phase106_gate_path=_write_phase106_gate(tmp_path / "phase106_gate.json", ready=True),
        phase107_gate_path=_write_phase107_gate(tmp_path / "phase107_gate.json", closed=True),
        output_dir=tmp_path / "out",
        target_columns=(
            "target_cp_camera_pair_delta",
            "target_cp_prev_same_camera_delta",
            "target_cp_layer_camera_range",
        ),
        target_priority=("target_cp_camera_pair_delta",),
        min_validation_relative_improvement=0.0,
        min_unsolved_validation_normalized_rmse=0.0,
        n_neighbors=1,
        n_estimators=5,
    )

    gate = manifest["gate"]
    assert gate["base_target"] == "target_center_periphery_contrast"
    assert gate["phase108_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False

    with (tmp_path / "out/phase108_nist_ammt_sequence_target_field_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 14
    assert float(rows[0]["target_cp_camera_pair_delta"]) == pytest.approx(-2.0)
    assert float(rows[1]["target_cp_camera_pair_delta"]) == pytest.approx(2.0)
    assert "target_cp_prev_same_camera_delta" in rows[0]

    with (tmp_path / "out/phase108_nist_ammt_sequence_target_metric_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        metric_rows = list(csv.DictReader(handle))
    assert len(metric_rows) == 3 * len(module.METHODS) * 3


def test_sequence_gate_blocks_when_phase107_not_closed(tmp_path: Path):
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn is not installed in this environment")
    module = _load_module()

    manifest = module.build_package(
        root=Path(".").resolve(),
        spatial_field_table=_write_spatial_field_table(tmp_path / "spatial.csv"),
        split_manifest=_write_split(tmp_path / "split.json"),
        phase106_gate_path=_write_phase106_gate(tmp_path / "phase106_gate.json", ready=True),
        phase107_gate_path=_write_phase107_gate(tmp_path / "phase107_gate.json", closed=False),
        output_dir=tmp_path / "out",
        target_columns=("target_cp_camera_pair_delta",),
        target_priority=("target_cp_camera_pair_delta",),
        min_validation_relative_improvement=0.0,
        min_unsolved_validation_normalized_rmse=0.0,
        n_neighbors=1,
        n_estimators=5,
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase108_sequence_target_gate_blocked_by_phase107"
    assert gate["phase108_focused_review_allowed"] is False
    assert gate["phase108_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
