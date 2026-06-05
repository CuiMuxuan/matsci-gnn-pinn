from __future__ import annotations

import csv
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest


def _load_module():
    script = Path("scripts/server/phase107_nist_ammt_source_region_feature_gate.py")
    spec = importlib.util.spec_from_file_location("phase107_source_region_gate", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_csv(layer: int) -> str:
    rows = []
    for index in range(20):
        x = index % 5
        y = index // 5
        center = 1 if 1 <= x <= 3 and 1 <= y <= 2 else 0
        power = layer + 5 * center + index * 0.1
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
    targets = [0, 1, 2, 3, 4, 5, 6]
    for layer, split_name in enumerate(split_names, start=1):
        rows.append(
            {
                "x": str(layer),
                "y": "0",
                "t": str(layer),
                "target_center_periphery_contrast": str(targets[layer - 1]),
                "source_layer_index": str(layer),
                "target_camera_code": str(layer % 2),
                "source_p_mean": str(layer + 1),
                "source_p_nonzero_fraction": "1",
                "source_x_range": "4",
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


def _write_phase106_gate(path: Path, *, ready: bool = True) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": "phase106_spatial_target_gap_ready_focused_no_training_validation"
                if ready
                else "phase106_spatial_target_gate_closed_no_baseline_gap",
                "selected_target": "target_center_periphery_contrast",
                "selected_validation_method": "hist_gradient_boosting",
                "selected_validation_rmse": 100.0,
                "selected_test_rmse": 100.0,
                "phase106_seed7_focused_validation_allowed": ready,
                "phase106_model_training_allowed": False,
                "a100_training_allowed_now": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_source_region_gate_writes_augmented_features_and_keeps_training_locked(
    tmp_path: Path,
):
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
        phase106_gate_path=_write_phase106_gate(tmp_path / "phase106_gate.json", ready=True),
        output_dir=tmp_path / "out",
        min_validation_improvement=0.0,
        max_source_rows=None,
        grid_size=2,
        n_neighbors=1,
        n_estimators=5,
    )

    gate = manifest["gate"]
    assert gate["target"] == "target_center_periphery_contrast"
    assert gate["phase107_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False

    with (tmp_path / "out/phase107_nist_ammt_source_region_augmented_field_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 7
    assert "source_center_periphery_power_balance" in rows[0]
    assert "source_grid_power_fraction_range" in rows[0]
    assert "source_temporal_x_drift_norm" in rows[0]

    with (tmp_path / "out/phase107_nist_ammt_source_region_metric_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        metric_rows = list(csv.DictReader(handle))
    assert len(metric_rows) == len(module.FEATURE_PROFILES) * len(module.METHODS) * 3


def test_source_region_gate_blocks_when_phase106_not_ready(tmp_path: Path):
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
        phase106_gate_path=_write_phase106_gate(tmp_path / "phase106_gate.json", ready=False),
        output_dir=tmp_path / "out",
        min_validation_improvement=0.0,
        max_source_rows=None,
        grid_size=2,
        n_neighbors=1,
        n_estimators=5,
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase107_source_region_gate_blocked_by_phase106"
    assert gate["phase107_focused_review_allowed"] is False
    assert gate["phase107_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
