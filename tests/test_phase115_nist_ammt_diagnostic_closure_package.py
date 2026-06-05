from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase115_nist_ammt_diagnostic_closure_package.py")
    spec = importlib.util.spec_from_file_location("phase115_closure_package", script)
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


def _synthetic_phase_inputs(tmp_path: Path, *, phase114_closed: bool = True) -> dict[str, Path]:
    return {
        "phase111_gate": _write_json(
            tmp_path / "phase111_gate.json",
            {
                "status": "phase111_registered_target_closure_package_ready_sequence_branch_closed",
                "phase111_model_training_allowed": False,
                "a100_training_allowed_now": False,
                "a100_80gb_request_now": False,
            },
        ),
        "phase112_gate": _write_json(
            tmp_path / "phase112_gate.json",
            {
                "status": "phase112_melt_pool_target_gap_ready_focused_review",
                "row_count": 63,
                "selected_target": "target_mp_temporal_mean_range",
                "selected_validation_rmse": 1.6666052321034104,
                "selected_test_rmse": 2.8227071700279938,
                "phase112_model_training_allowed": False,
                "a100_training_allowed_now": False,
                "a100_80gb_request_now": False,
            },
        ),
        "phase113_gate": _write_json(
            tmp_path / "phase113_gate.json",
            {
                "status": "phase113_melt_pool_focused_review_closed_validation_test_reversal",
                "phase112_selected_target": "target_mp_temporal_mean_range",
                "phase112_selected_validation_rmse": 1.6666052321034104,
                "phase112_selected_test_rmse": 2.8227071700279938,
                "phase113_model_training_allowed": False,
                "a100_training_allowed_now": False,
                "a100_80gb_request_now": False,
            },
        ),
        "phase113_review_table": _write_csv(
            tmp_path / "phase113_review.csv",
            [
                {
                    "target": "target_mp_temporal_mean_range",
                    "focused_review_status": "blocked_validation_test_reversal",
                    "mechanism_allowed": "false",
                    "reason": "validation-selected profile is worse than mean guard on test",
                }
            ],
        ),
        "phase114_gate": _write_json(
            tmp_path / "phase114_gate.json",
            {
                "status": "phase114_gcode_strategy_source_gate_closed_no_guarded_baseline_gap"
                if phase114_closed
                else "phase114_gcode_strategy_source_gap_ready_focused_review",
                "row_count": 128,
                "selected_target": None,
                "selected_validation_rmse": None,
                "selected_test_rmse": None,
                "phase114_model_training_allowed": False,
                "a100_training_allowed_now": False,
                "a100_80gb_request_now": False,
            },
        ),
        "phase114_review_table": _write_csv(
            tmp_path / "phase114_review.csv",
            [
                {
                    "target": "target_intensity_std",
                    "status": "blocked_layer_time_strategy_shortcut",
                    "phase114_candidate": "false",
                },
                {
                    "target": "target_center_periphery_contrast",
                    "status": "blocked_no_gain_over_xypt_guard",
                    "phase114_candidate": "false",
                },
            ],
        ),
    }


def test_phase115_closes_nist_ammt_diagnostics_and_keeps_training_locked(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        phase_inputs=_synthetic_phase_inputs(tmp_path),
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 115
    assert gate["status"] == "phase115_nist_ammt_diagnostic_closure_package_ready_all_new_branches_closed"
    assert gate["phase111_registered_target_branch_closed"] is True
    assert gate["phase113_melt_pool_branch_closed"] is True
    assert gate["phase114_gcode_strategy_branch_closed"] is True
    assert gate["all_training_locks_verified"] is True
    assert gate["phase115_model_mechanism_allowed"] is False
    assert gate["phase115_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert manifest["counts"]["evidence_rows"] == 4
    assert manifest["counts"]["claim_rows"] == 4
    assert manifest["counts"]["boundary_rows"] == 4
    assert manifest["counts"]["training_allowed_boundary_rows"] == 0
    assert manifest["counts"]["a100_80gb_allowed_boundary_rows"] == 0

    with (tmp_path / "out/phase115_nist_ammt_diagnostic_boundary_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        boundary_rows = list(csv.DictReader(handle))
    assert {row["model_training_allowed"] for row in boundary_rows} == {"false"}
    assert {row["a100_80gb_request_now"] for row in boundary_rows} == {"false"}

    with (tmp_path / "out/phase115_nist_ammt_diagnostic_claim_use_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        claims = list(csv.DictReader(handle))
    floor = next(row for row in claims if row["claim_id"] == "main_paper_floor_unchanged")
    assert floor["claim_use"] == "main_text_floor"
    assert floor["evidence_status"] == "unchanged_positive_floor"


def test_phase115_incomplete_when_phase114_not_closed(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        phase_inputs=_synthetic_phase_inputs(tmp_path, phase114_closed=False),
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase115_nist_ammt_diagnostic_closure_package_incomplete"
    assert gate["phase114_gcode_strategy_branch_closed"] is False
    assert gate["phase115_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False


def test_phase115_real_artifact_contract(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        phase_inputs=module.PHASE_INPUTS,
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase115_nist_ammt_diagnostic_closure_package_ready_all_new_branches_closed"
    assert manifest["source_statuses"]["phase111"] == (
        "phase111_registered_target_closure_package_ready_sequence_branch_closed"
    )
    assert manifest["source_statuses"]["phase113"] == (
        "phase113_melt_pool_focused_review_closed_validation_test_reversal"
    )
    assert manifest["source_statuses"]["phase114"] == (
        "phase114_gcode_strategy_source_gate_closed_no_guarded_baseline_gap"
    )
    assert gate["all_training_locks_verified"] is True
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
