from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase116_paper_evidence_consolidation.py")
    spec = importlib.util.spec_from_file_location("phase116_consolidation", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _phase91_main_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in ["broad12", "broad21"]:
        for metric in ["Test RMSE", "Hot q90 RMSE", "Gradient q90 RMSE"]:
            rows.append(
                {
                    "freeze_id": f"{dataset}-{metric}",
                    "dataset": dataset,
                    "split": "spot_size",
                    "route": "film/global_standard",
                    "metric": metric,
                    "broad_process_v1_mean": "1.0",
                    "broad_process_v1_std": "0.1",
                    "best_strong_baseline": "1.5",
                    "delta_vs_best_strong": "-0.5",
                    "n_seeds": "3",
                    "claim_anchor": "C61-MAIN-001",
                    "freeze_status": "frozen_main_claim_row",
                }
            )
    return rows


def _phase91_appendix_rows() -> list[dict[str, object]]:
    return [
        {
            "freeze_id": f"P91-APPX-{index:03d}",
            "appendix_id": f"P88-APPX-{index:03d}",
            "phase": str(30 + index),
            "branch": f"diagnostic-{index}",
            "status": "negative",
            "artifact": f"artifact-{index}.md",
            "manuscript_use": "appendix diagnostic",
            "reason": "closed by gate",
            "freeze_status": "frozen_appendix_diagnostic_row",
        }
        for index in range(1, 4)
    ]


def _phase74_claim_rows() -> list[dict[str, object]]:
    return [
        {
            "claim_id": "C61-MAIN-001",
            "claim_location": "Results",
            "claim_summary": "fixed spot_size floor",
            "source_anchor": "C61-MAIN-001",
            "support_type": "result",
            "evidence_locator": "phase60_main.csv",
            "audit_status": "supported_for_v0",
            "allowed_in_v0": "yes",
            "claim_strength": "moderate",
            "required_wording_guard": "limited to spot_size",
        }
    ]


def _phase74_boundary_rows() -> list[dict[str, object]]:
    return [
        {
            "boundary_id": "C74-EXCL-001",
            "candidate_or_scope": "density-invariant robustness",
            "status": "excluded_from_main_claim",
            "main_text_treatment": "state as limitation",
            "appendix_treatment": "map to appendix",
            "evidence_locator": "phase60_manifest.json",
            "reason": "excluded by claim boundary",
        }
    ]


def _phase92_manual_rows() -> list[dict[str, object]]:
    return [
        {
            "queue_id": "P92-MANUAL-001",
            "priority": "P0",
            "needed_input": "target venue or author guide",
            "minimum_acceptance": "one named venue",
            "reason": "venue dependent",
            "blocks_submission": "true",
            "blocks_model_training": "false",
            "suggested_user_action": "provide target venue",
        },
        {
            "queue_id": "P92-MANUAL-002",
            "priority": "P0",
            "needed_input": "benchmark papers",
            "minimum_acceptance": "at least 3 usable papers",
            "reason": "review dependent",
            "blocks_submission": "true",
            "blocks_model_training": "false",
            "suggested_user_action": "provide benchmark papers",
        },
    ]


def _phase115_claim_rows() -> list[dict[str, object]]:
    return [
        {
            "claim_id": "main_paper_floor_unchanged",
            "branch": "broad_process_v1_spot_size",
            "claim_text": "floor unchanged",
            "claim_use": "main_text_floor",
            "evidence_status": "unchanged_positive_floor",
        },
        {
            "claim_id": "gcode_strategy_source_branch_closed",
            "branch": "gcode_strategy_source",
            "claim_text": "closed",
            "claim_use": "appendix_negative_result",
            "evidence_status": "closed_negative",
        },
    ]


def _phase115_boundary_rows(*, training_allowed: str = "false") -> list[dict[str, object]]:
    return [
        {
            "boundary_id": "phase115_no_training_on_phase114_gcode_features",
            "branch": "gcode_strategy_source",
            "blocked_item": "G-code strategy features",
            "reason": "no guarded baseline gap",
            "model_mechanism_allowed": "false",
            "model_training_allowed": training_allowed,
            "a100_training_allowed_now": "false",
            "a100_80gb_request_now": "false",
        },
        {
            "boundary_id": "phase115_no_a100_80gb_request",
            "branch": "compute",
            "blocked_item": "A100-SXM4-80GB",
            "reason": "no seed-positive blockage",
            "model_mechanism_allowed": "false",
            "model_training_allowed": "false",
            "a100_training_allowed_now": "false",
            "a100_80gb_request_now": "false",
        },
    ]


def _paths(tmp_path: Path, *, training_allowed: str = "false") -> dict[str, Path]:
    return {
        "phase60_manifest": _write_json(
            tmp_path / "phase60_manifest.json",
            {"claim_boundary": {"main_claim": "fixed-sampling broad12/broad21 spot_size"}},
        ),
        "phase74_claim_audit": _write_csv(tmp_path / "phase74_claim.csv", _phase74_claim_rows()),
        "phase74_boundary_register": _write_csv(
            tmp_path / "phase74_boundary.csv", _phase74_boundary_rows()
        ),
        "phase74_manifest": _write_json(
            tmp_path / "phase74_manifest.json",
            {"writing_stage_gate": {"status": "ready_for_internal_manuscript_review", "main_claim_locked": True}},
        ),
        "phase91_gate": _write_json(
            tmp_path / "phase91_gate.json",
            {
                "status": "table_figure_appendix_frozen_venue_unresolved",
                "table_figure_appendix_frozen": True,
                "a100_training_allowed_now": False,
                "a100_80gb_request_now": False,
            },
        ),
        "phase91_main_table": _write_csv(tmp_path / "phase91_main.csv", _phase91_main_rows()),
        "phase91_appendix": _write_csv(tmp_path / "phase91_appendix.csv", _phase91_appendix_rows()),
        "phase92_gate": _write_json(
            tmp_path / "phase92_gate.json",
            {
                "status": "blocked_missing_target_benchmarks",
                "benchmark_review_ready": False,
                "a100_training_allowed_now": False,
                "a100_80gb_request_now": False,
            },
        ),
        "phase92_manual_queue": _write_csv(tmp_path / "phase92_manual.csv", _phase92_manual_rows()),
        "phase115_gate": _write_json(
            tmp_path / "phase115_gate.json",
            {
                "status": "phase115_nist_ammt_diagnostic_closure_package_ready_all_new_branches_closed",
                "main_paper_new_nist_ammt_claim_ready": False,
                "all_training_locks_verified": True,
                "phase115_model_training_allowed": False,
                "a100_training_allowed_now": False,
                "a100_80gb_request_now": False,
                "next_action": "do not train on closed branches",
            },
        ),
        "phase115_claim_use": _write_csv(tmp_path / "phase115_claim.csv", _phase115_claim_rows()),
        "phase115_boundary": _write_csv(
            tmp_path / "phase115_boundary.csv",
            _phase115_boundary_rows(training_allowed=training_allowed),
        ),
    }


def test_phase116_consolidates_floor_nist_closure_and_submission_blockers(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path),
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 116
    assert gate["status"] == "phase116_paper_evidence_consolidation_ready_venue_unresolved"
    assert gate["paper_evidence_consolidated"] is True
    assert gate["positive_floor_ready"] is True
    assert gate["phase115_nist_ammt_closed"] is True
    assert gate["main_paper_new_nist_ammt_claim_ready"] is False
    assert gate["phase116_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert manifest["counts"]["positive_floor_rows"] == 6
    assert manifest["counts"]["phase115_boundary_rows"] == 2
    assert manifest["counts"]["training_allowed_negative_rows"] == 0

    with (tmp_path / "out/phase116_negative_diagnostic_addendum.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        negative_rows = list(csv.DictReader(handle))
    assert any(row["diagnostic_id"].startswith("P116-NIST") for row in negative_rows)
    assert {row["a100_80gb_request_now"] for row in negative_rows} == {"false"}

    with (tmp_path / "out/phase116_remaining_blocker_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        blockers = list(csv.DictReader(handle))
    assert any(row["blocker_id"] == "P116-NIST-AMMT-TRAINING-LOCK" for row in blockers)
    assert any(row["blocks_submission"] == "true" for row in blockers)


def test_phase116_incomplete_if_negative_rows_unlock_training(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path, training_allowed="true"),
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase116_paper_evidence_consolidation_incomplete"
    assert gate["all_training_locks_verified"] is False
    assert gate["phase116_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert manifest["counts"]["training_allowed_negative_rows"] == 1


def test_phase116_real_artifact_contract(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        phase_inputs=module.PHASE_INPUTS,
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase116_paper_evidence_consolidation_ready_venue_unresolved"
    assert gate["paper_evidence_consolidated"] is True
    assert gate["positive_floor_rows"] == 6
    assert gate["negative_diagnostic_rows"] >= 22
    assert gate["phase92_status"] == "blocked_missing_target_benchmarks"
    assert gate["phase115_nist_ammt_closed"] is True
    assert gate["submission_ready"] is False
    assert gate["phase116_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert manifest["source_statuses"]["phase115"] == (
        "phase115_nist_ammt_diagnostic_closure_package_ready_all_new_branches_closed"
    )
