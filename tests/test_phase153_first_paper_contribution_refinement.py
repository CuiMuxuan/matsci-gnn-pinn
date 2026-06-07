from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase153_first_paper_contribution_refinement.py")
    spec = importlib.util.spec_from_file_location("phase153_contribution_refinement", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _paths(tmp_path: Path, *, phase152_status: str | None = None) -> dict[str, Path]:
    main_rows = [
        {
            "dataset": "broad12",
            "split": "spot_size",
            "route": "film/global_standard",
            "metric": "Test RMSE",
            "broad_process_v1_mean": "136.38",
            "best_strong_baseline": "151.85",
        },
        {
            "dataset": "broad21",
            "split": "spot_size",
            "route": "film/global_standard",
            "metric": "Test RMSE",
            "broad_process_v1_mean": "146.00",
            "best_strong_baseline": "149.18",
        },
    ]
    phase152_gate = {
        "status": phase152_status
        or "phase152_paper_evidence_refresh_ready_first_paper_narrow_claims_neural_operator_closed",
        "first_paper_draft_allowed_now": True,
        "main_paper_floor": (
            "fixed-sampling broad12/broad21 spot_size under broad_process_v1, seeds 7/1/2"
        ),
        "phase152_model_training_allowed": False,
        "operator_training_allowed_now": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
    }
    return {
        "phase60_main_table": _write_csv(tmp_path / "p60/main.csv", main_rows),
        "phase60_route_guard_table": _write_csv(
            tmp_path / "p60/route.csv",
            [
                {
                    "dataset": "broad12",
                    "split": "laser_power",
                    "classification": "route_guard_positive",
                    "claim_use": "route-guard-only boundary evidence",
                }
            ],
        ),
        "phase88_claim_lock_table": _write_csv(
            tmp_path / "p88/locks.csv",
            [
                {
                    "lock_id": "P88-MAIN-LOCK",
                    "claim_scope": "spot_size",
                    "status": "locked_experimental_main_claim",
                }
            ],
        ),
        "phase116_positive_floor_table": _write_csv(
            tmp_path / "p116/floor.csv",
            [{"floor_id": "P116-FLOOR-001", "dataset": "broad12", "split": "spot_size"}],
        ),
        "phase116_claim_status_table": _write_csv(
            tmp_path / "p116/claims.csv",
            [{"claim_id": "C61-MAIN-001", "status": "supported_for_v0"}],
        ),
        "phase152_gate": _write_json(tmp_path / "p152/gate.json", phase152_gate),
        "phase152_claim_boundary_table": _write_csv(
            tmp_path / "p152/claim_boundary.csv",
            [
                {
                    "claim_id": "P152-CLAIM-002",
                    "claim_area": "neural_operator_or_fno",
                    "claim_status": "blocked_success_claim",
                }
            ],
        ),
    }


def test_phase153_builds_contribution_refinement_package(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path),
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase153_first_paper_contribution_refinement_ready_narrow_claims"
    assert gate["contribution_refinement_ready"] is True
    assert gate["first_paper_draft_allowed_now"] is True
    assert gate["first_paper_submission_ready"] is False
    assert gate["new_model_claim_ready"] is False
    assert gate["phase153_model_training_allowed"] is False
    assert gate["operator_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert manifest["counts"]["contribution_rows"] == 5
    assert manifest["counts"]["phrasing_guard_rows"] == 7
    assert manifest["counts"]["submission_blocker_rows"] == 2

    markdown = (tmp_path / "out/phase153_first_paper_contribution_refinement.md").read_text(
        encoding="utf-8"
    )
    assert "route-guarded process-conditioned Macro PINN" in markdown
    assert "neural-operator/FNO success" in markdown
    assert "P153-PHRASE-001" in markdown
    assert "|  |  |  |" not in markdown


def test_phase153_incomplete_if_phase152_route_closure_not_ready(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path, phase152_status="phase152_incomplete"),
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase153_first_paper_contribution_refinement_incomplete"
    assert gate["contribution_refinement_ready"] is False
    assert gate["phase153_model_training_allowed"] is False
    assert gate["operator_training_allowed_now"] is False
