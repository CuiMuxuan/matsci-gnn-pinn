from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    script = Path("scripts/server/phase104_nist_ammt_target_hardness_review.py")
    spec = importlib.util.spec_from_file_location("phase104_target_review", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_field_table(path: Path) -> Path:
    rows = []
    split_names = ["train", "train", "train", "val", "val", "test", "test"]
    hard_values = [0, 1, 2, 3, 4, 5, 6]
    mean_like_values = [10, 10, 10, 10, 10, 10, 10]
    for index, split_name in enumerate(split_names):
        rows.append(
            {
                "x": str(index),
                "y": "0",
                "t": str(index),
                "source_p_mean": str(hard_values[index]),
                "source_p_nonzero_fraction": "1",
                "source_x_range": "1",
                "source_y_range": "1",
                "target_camera_code": "0",
                "target_intensity_mean": str(mean_like_values[index]),
                "target_intensity_std": str(hard_values[index]),
                "target_intensity_min": str(hard_values[index]),
                "target_intensity_max": "255",
                "target_intensity_q90": str(mean_like_values[index]),
                "split_name": split_name,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_target_hardness_review_selects_validation_positive_target_and_keeps_a100_locked(
    tmp_path: Path,
):
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn is not installed in this environment")
    module = _load_module()
    field_table = _write_field_table(tmp_path / "field.csv")
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps({"splits": {"train": [0, 1, 2], "val": [3, 4], "test": [5, 6]}}) + "\n",
        encoding="utf-8",
    )
    baseline_gate = tmp_path / "baseline_gate.json"
    baseline_gate.write_text(
        json.dumps(
            {
                "status": "phase104_baseline_smoke_complete_mechanisms_review_required",
                "baseline_smoke_completed": True,
                "sample_size_sufficient_for_phase105": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = module.build_package(
        root=Path(".").resolve(),
        field_table=field_table,
        split_manifest=split,
        baseline_gate_path=baseline_gate,
        output_dir=tmp_path / "out",
        target_columns=(
            "target_intensity_mean",
            "target_intensity_std",
            "target_intensity_max",
        ),
        target_priority=("target_intensity_std", "target_intensity_mean"),
        min_validation_relative_improvement=0.05,
        n_neighbors=1,
        n_estimators=5,
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase104_target_hardness_review_ready_phase105_design"
    assert gate["selected_target"] == "target_intensity_std"
    assert "hist_gradient_boosting" in module.METHODS
    assert gate["selected_validation_method"] in module.METHODS
    assert gate["selected_validation_method"] != "mean"
    assert gate["phase105_model_mechanism_allowed"] is True
    assert gate["phase105_low_capacity_design_only"] is True
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False

    with (tmp_path / "out/phase104_nist_ammt_target_hardness_review_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = {row["target"]: row for row in csv.DictReader(handle)}
    assert rows["target_intensity_std"]["status"] == "candidate_target_ready_for_phase105_design"
    assert rows["target_intensity_mean"]["status"] == "blocked_zero_variance_target"
    assert rows["target_intensity_max"]["status"] == "blocked_zero_variance_target"


def test_target_hardness_review_blocks_when_baseline_gate_is_not_ready(tmp_path: Path):
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn is not installed in this environment")
    module = _load_module()
    field_table = _write_field_table(tmp_path / "field.csv")
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps({"splits": {"train": [0, 1, 2], "val": [3, 4], "test": [5, 6]}}) + "\n",
        encoding="utf-8",
    )
    baseline_gate = tmp_path / "baseline_gate.json"
    baseline_gate.write_text(
        json.dumps(
            {
                "status": "phase104_baseline_smoke_boundary_tiny_sample_mechanisms_locked",
                "baseline_smoke_completed": True,
                "sample_size_sufficient_for_phase105": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = module.build_package(
        root=Path(".").resolve(),
        field_table=field_table,
        split_manifest=split,
        baseline_gate_path=baseline_gate,
        output_dir=tmp_path / "out",
        target_columns=("target_intensity_std",),
        target_priority=("target_intensity_std",),
        min_validation_relative_improvement=0.05,
        n_neighbors=1,
        n_estimators=5,
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase104_target_hardness_blocked_by_baseline_gate"
    assert gate["phase105_model_mechanism_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
