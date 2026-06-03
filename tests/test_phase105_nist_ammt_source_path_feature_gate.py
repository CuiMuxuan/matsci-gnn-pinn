from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    script = Path("scripts/server/phase105_nist_ammt_source_path_feature_gate.py")
    spec = importlib.util.spec_from_file_location("phase105_source_path_gate", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_field_table(path: Path) -> Path:
    rows = []
    split_names = ["train", "train", "train", "val", "val", "test", "test"]
    target_values = [0, 1, 2, 3, 4, 5, 6]
    for index, split_name in enumerate(split_names):
        rows.append(
            {
                "x": str(index),
                "y": "0",
                "t": str(index),
                "target_intensity_std": str(target_values[index]),
                "source_layer_index": str(index + 1),
                "source_p_mean": str(index + 1),
                "source_p_nonzero_fraction": "1",
                "source_p_range": "0.5",
                "source_t_range": "2",
                "source_x_range": "1",
                "source_y_range": "1",
                "target_camera_code": "0",
                "split_name": split_name,
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


def _write_gate(path: Path, *, allowed: bool = True) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": "phase104_target_hardness_review_ready_phase105_design",
                "selected_target": "target_intensity_std",
                "selected_validation_method": "hist_gradient_boosting",
                "selected_validation_rmse": 100.0 if allowed else 0.01,
                "selected_test_rmse": 100.0 if allowed else 0.01,
                "phase105_model_mechanism_allowed": allowed,
                "a100_training_allowed_now": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_source_path_feature_gate_writes_augmented_features_and_keeps_training_locked(
    tmp_path: Path,
):
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn is not installed in this environment")
    module = _load_module()
    manifest = module.build_package(
        root=Path(".").resolve(),
        field_table=_write_field_table(tmp_path / "field.csv"),
        split_manifest=_write_split(tmp_path / "split.json"),
        target_hardness_gate_path=_write_gate(tmp_path / "gate.json", allowed=True),
        output_dir=tmp_path / "out",
        min_validation_improvement=0.0,
        n_neighbors=1,
        n_estimators=5,
    )

    gate = manifest["gate"]
    assert gate["target"] == "target_intensity_std"
    assert gate["phase105_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False

    augmented = tmp_path / "out/phase105_nist_ammt_source_path_augmented_field_table.csv"
    with augmented.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 7
    assert float(rows[0]["source_energy_proxy"]) == pytest.approx(2.0)
    assert "source_green_log_proxy" in rows[0]

    with (tmp_path / "out/phase105_nist_ammt_source_path_metric_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        metric_rows = list(csv.DictReader(handle))
    assert len(metric_rows) == len(module.FEATURE_PROFILES) * len(module.METHODS) * 3


def test_source_path_feature_gate_blocks_when_phase104_gate_is_not_open(tmp_path: Path):
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn is not installed in this environment")
    module = _load_module()
    manifest = module.build_package(
        root=Path(".").resolve(),
        field_table=_write_field_table(tmp_path / "field.csv"),
        split_manifest=_write_split(tmp_path / "split.json"),
        target_hardness_gate_path=_write_gate(tmp_path / "gate.json", allowed=False),
        output_dir=tmp_path / "out",
        min_validation_improvement=0.0,
        n_neighbors=1,
        n_estimators=5,
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase105_source_path_gate_blocked_by_phase104_target_review"
    assert gate["phase105_cpu_smoke_allowed"] is False
    assert gate["phase105_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False


def test_source_path_feature_gate_requires_strict_validation_gain(tmp_path: Path):
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn is not installed in this environment")
    module = _load_module()
    manifest = module.build_package(
        root=Path(".").resolve(),
        field_table=_write_field_table(tmp_path / "field.csv"),
        split_manifest=_write_split(tmp_path / "split.json"),
        target_hardness_gate_path=_write_gate(tmp_path / "gate.json", allowed=True),
        output_dir=tmp_path / "out",
        min_validation_improvement=1000.0,
        n_neighbors=1,
        n_estimators=5,
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase105_source_path_feature_gate_blocked_no_hgb_gain"
    assert gate["candidate_feature_profiles"] == []
    assert gate["phase105_cpu_smoke_allowed"] is False
    assert gate["phase105_model_training_allowed"] is False
