from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase113_nist_ammt_melt_pool_focused_review.py")
    spec = importlib.util.spec_from_file_location("phase113_melt_pool_review", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_gate(path: Path, *, ready: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "phase112_melt_pool_target_gap_ready_focused_review"
                if ready
                else "phase112_melt_pool_target_gate_closed_no_baseline_gap",
                "selected_target": "target_mp_temporal_mean_range",
                "selected_validation_rmse": 1.0,
                "selected_test_rmse": 2.0,
                "phase112_model_training_allowed": False,
                "a100_training_allowed_now": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_review(path: Path, *, positive: bool = False) -> Path:
    rows = [
        {
            "target": "target_mp_temporal_mean_range",
            "target_min": "0",
            "target_max": "10",
            "target_range": "10",
            "selected_feature_profile": "source_geometry",
            "selected_validation_method": "knn",
            "selected_validation_rmse": "1.0",
            "selected_validation_normalized_rmse": "0.2",
            "selected_test_rmse": "2.0",
            "selected_test_normalized_rmse": "0.3",
            "mean_validation_rmse": "2.0",
            "mean_validation_normalized_rmse": "0.4",
            "mean_test_rmse": "1.5" if not positive else "3.0",
            "mean_test_normalized_rmse": "0.35",
            "layer_time_validation_rmse": "1.5",
            "layer_time_validation_normalized_rmse": "0.3",
            "layer_time_shortcut_detected": "false",
            "validation_relative_improvement_over_mean": "0.5",
            "test_relative_improvement_over_mean": "-0.333333" if not positive else "0.333333",
            "baseline_visible_gap": "true",
            "strong_baseline_solved": "false",
            "zero_variance_target": "false",
            "physical_priority": "0",
            "status": "candidate_melt_pool_target_gap_ready_for_focused_review",
            "phase112_candidate": "true",
        },
        {
            "target": "target_mp_peak_frame_position",
            "target_min": "0",
            "target_max": "1",
            "target_range": "1",
            "selected_feature_profile": "source_geometry",
            "selected_validation_method": "hist_gradient_boosting",
            "selected_validation_rmse": "0.2",
            "selected_validation_normalized_rmse": "0.2",
            "selected_test_rmse": "0.3",
            "selected_test_normalized_rmse": "0.3",
            "mean_validation_rmse": "0.2",
            "mean_validation_normalized_rmse": "0.2",
            "mean_test_rmse": "0.3",
            "mean_test_normalized_rmse": "0.3",
            "layer_time_validation_rmse": "0.2",
            "layer_time_validation_normalized_rmse": "0.2",
            "layer_time_shortcut_detected": "true",
            "validation_relative_improvement_over_mean": "0.0",
            "test_relative_improvement_over_mean": "0.0",
            "baseline_visible_gap": "false",
            "strong_baseline_solved": "false",
            "zero_variance_target": "false",
            "physical_priority": "5",
            "status": "blocked_no_baseline_visible_gap",
            "phase112_candidate": "false",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_phase113_closes_melt_pool_on_validation_test_reversal(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase112_gate_path=_write_gate(tmp_path / "phase112_gate.json"),
        phase112_review_table=_write_review(tmp_path / "phase112_review.csv"),
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase113_melt_pool_focused_review_closed_validation_test_reversal"
    assert gate["mechanism_allowed_targets"] == []
    assert gate["validation_test_reversal_target_count"] == 1
    assert gate["phase113_model_mechanism_allowed"] is False
    assert gate["phase113_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False

    with (tmp_path / "out/phase113_nist_ammt_melt_pool_focused_review_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = {row["target"]: row for row in csv.DictReader(handle)}
    assert rows["target_mp_temporal_mean_range"]["focused_review_status"] == (
        "blocked_validation_test_reversal"
    )
    assert rows["target_mp_peak_frame_position"]["focused_review_status"] == "not_phase112_candidate"


def test_phase113_can_leave_mechanism_design_open_without_training(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase112_gate_path=_write_gate(tmp_path / "phase112_gate.json"),
        phase112_review_table=_write_review(tmp_path / "phase112_review.csv", positive=True),
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase113_melt_pool_focused_review_ready_mechanism_design"
    assert gate["mechanism_allowed_targets"] == ["target_mp_temporal_mean_range"]
    assert gate["phase113_model_mechanism_allowed"] is True
    assert gate["phase113_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False


def test_phase113_blocks_when_phase112_not_ready(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        root=Path(".").resolve(),
        phase112_gate_path=_write_gate(tmp_path / "phase112_gate.json", ready=False),
        phase112_review_table=_write_review(tmp_path / "phase112_review.csv", positive=True),
        output_dir=tmp_path / "out",
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase113_melt_pool_review_blocked_by_phase112"
    assert gate["phase113_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
