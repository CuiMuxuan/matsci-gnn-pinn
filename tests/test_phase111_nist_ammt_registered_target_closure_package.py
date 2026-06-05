from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase111_nist_ammt_registered_target_closure_package.py")
    spec = importlib.util.spec_from_file_location("phase111_closure_package", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _synthetic_phase_inputs(tmp_path: Path, *, phase109_closed: bool = True) -> dict[str, Path]:
    return {
        "phase106": _write_json(
            tmp_path / "phase106_gate.json",
            {
                "status": "phase106_spatial_target_gap_ready_focused_no_training_validation",
                "selected_target": "target_center_periphery_contrast",
                "selected_validation_rmse": 1.174314337940004,
                "selected_test_rmse": 1.3827508688560513,
                "phase106_model_training_allowed": False,
                "a100_training_allowed_now": False,
            },
        ),
        "phase107": _write_json(
            tmp_path / "phase107_gate.json",
            {
                "status": "phase107_source_region_feature_gate_blocked_no_phase106_gain",
                "selected_feature_profile": None,
                "selected_validation_rmse": None,
                "selected_test_rmse": None,
                "next_action": "close sampled source-region path features as diagnostic; do not train",
                "phase107_model_training_allowed": False,
                "a100_training_allowed_now": False,
            },
        ),
        "phase108": _write_json(
            tmp_path / "phase108_gate.json",
            {
                "status": "phase108_sequence_target_gap_ready_focused_review",
                "selected_target": "target_cp_camera_pair_delta",
                "selected_validation_rmse": 1.80945034831816,
                "selected_test_rmse": 1.8099898753521704,
                "phase108_model_training_allowed": False,
                "a100_training_allowed_now": False,
            },
        ),
        "phase109": _write_json(
            tmp_path / "phase109_gate.json",
            {
                "status": "phase109_sequence_target_focused_review_closed_camera_shortcut"
                if phase109_closed
                else "phase109_sequence_target_focused_review_ready_mechanism_design",
                "reviewed_target": "target_cp_camera_pair_delta",
                "full_phase108_validation_rmse": 1.80945034831816,
                "full_phase108_test_rmse": 1.8099898753521704,
                "phase109_model_training_allowed": False,
                "a100_training_allowed_now": False,
            },
        ),
        "phase110": _write_json(
            tmp_path / "phase110_gate.json",
            {
                "status": "phase110_layer_mean_target_review_closed_layer_time_shortcut",
                "reviewed_target": "target_cp_layer_mean",
                "full_phase108_validation_rmse": 0.9400978516652343,
                "full_phase108_test_rmse": 1.0606914883814935,
                "phase110_model_training_allowed": False,
                "a100_training_allowed_now": False,
            },
        ),
    }


def test_phase111_closes_sequence_branch_and_keeps_training_locked(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        phase_inputs=_synthetic_phase_inputs(tmp_path),
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 111
    assert gate["status"] == "phase111_registered_target_closure_package_ready_sequence_branch_closed"
    assert gate["nist_ammt_sequence_branch_closed"] is True
    assert gate["appendix_diagnostic_package_ready"] is True
    assert gate["main_paper_new_nist_ammt_claim_ready"] is False
    assert gate["main_paper_floor"] == "Phase 55/60/74 broad_process_v1 fixed-sampling spot_size"
    assert gate["phase111_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert manifest["counts"]["evidence_rows"] == 5
    assert manifest["counts"]["claim_rows"] == 4
    assert manifest["counts"]["boundary_rows"] == 5
    assert manifest["counts"]["training_allowed_boundary_rows"] == 0
    assert manifest["counts"]["a100_80gb_allowed_boundary_rows"] == 0
    assert manifest["source_gates"]["phase110"] == (
        "phase110_layer_mean_target_review_closed_layer_time_shortcut"
    )

    with (tmp_path / "out/phase111_nist_ammt_registered_target_evidence_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        evidence_rows = list(csv.DictReader(handle))
    assert [row["phase"] for row in evidence_rows] == ["106", "107", "108", "109", "110"]
    assert evidence_rows[0]["target_or_profile"] == "target_center_periphery_contrast"
    assert evidence_rows[-1]["interpretation"] == (
        "alternate layer-mean target closed as layer/time shortcut"
    )

    with (tmp_path / "out/phase111_nist_ammt_registered_target_claim_use_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        claim_rows = list(csv.DictReader(handle))
    main_floor = next(row for row in claim_rows if row["claim_id"] == "main_paper_floor_remains_phase55_60_74")
    assert main_floor["claim_use"] == "main_text_floor"
    assert main_floor["evidence_status"] == "unchanged_positive_floor"

    with (tmp_path / "out/phase111_nist_ammt_registered_target_boundary_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        boundary_rows = list(csv.DictReader(handle))
    assert {row["training_allowed"] for row in boundary_rows} == {"false"}
    assert {row["a100_80gb_request_allowed"] for row in boundary_rows} == {"false"}
    assert "do not train" in (
        tmp_path / "out/phase111_nist_ammt_registered_target_closure_package.md"
    ).read_text(encoding="utf-8")


def test_phase111_gate_incomplete_when_shortcut_review_not_closed(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        phase_inputs=_synthetic_phase_inputs(tmp_path, phase109_closed=False),
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase111_registered_target_closure_package_incomplete"
    assert gate["nist_ammt_sequence_branch_closed"] is False
    assert gate["appendix_diagnostic_package_ready"] is False
    assert gate["phase111_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False


def test_phase111_real_artifact_contract(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        phase_inputs=module.PHASE_INPUTS,
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase111_registered_target_closure_package_ready_sequence_branch_closed"
    assert gate["phase106_status"] == "phase106_spatial_target_gap_ready_focused_no_training_validation"
    assert gate["phase107_status"] == "phase107_source_region_feature_gate_blocked_no_phase106_gain"
    assert gate["phase108_status"] == "phase108_sequence_target_gap_ready_focused_review"
    assert gate["phase109_status"] == "phase109_sequence_target_focused_review_closed_camera_shortcut"
    assert gate["phase110_status"] == "phase110_layer_mean_target_review_closed_layer_time_shortcut"
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
