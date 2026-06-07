from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase154_route_coverage_and_remaining_scheme_audit.py")
    spec = importlib.util.spec_from_file_location("phase154_route_coverage", script)
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


def _paths(tmp_path: Path, *, phase153_status: str | None = None) -> dict[str, Path]:
    phase153_gate = {
        "status": phase153_status or "phase153_first_paper_contribution_refinement_ready_narrow_claims",
        "first_paper_draft_allowed_now": True,
        "phase153_model_training_allowed": False,
        "operator_training_allowed_now": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
    }
    return {
        "phase153_gate": _write_json(tmp_path / "p153/gate.json", phase153_gate),
        "phase153_contribution_table": _write_csv(
            tmp_path / "p153/contrib.csv",
            [{"contribution_id": "P153-CONTRIB-001", "contribution_title": "floor"}],
        ),
        "phase153_phrasing_guard_table": _write_csv(
            tmp_path / "p153/phrasing.csv",
            [{"guard_id": "P153-PHRASE-004", "unsafe_or_overbroad_phrase": "FNO success"}],
        ),
        "phase153_open_gap_table": _write_csv(
            tmp_path / "p153/gaps.csv",
            [{"gap_id": "P153-GAP-001", "blocks_submission": True}],
        ),
        "phase152_route_closure_table": _write_csv(
            tmp_path / "p152/routes.csv",
            [{"route_id": "P152-ROUTE-004", "route_status": "closed"}],
        ),
        "phase116_claim_status_table": _write_csv(
            tmp_path / "p116/claims.csv",
            [{"claim_id": "C61-MAIN-001", "status": "supported_for_v0"}],
        ),
    }


def test_phase154_answers_current_routes_verified_but_future_not_exhausted(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path),
    )

    gate = manifest["gate"]
    assert gate["status"] == (
        "phase154_route_coverage_audit_ready_current_routes_verified_future_not_exhausted"
    )
    assert gate["currently_executable_model_routes_verified"] is True
    assert gate["all_possible_future_schemes_exhausted"] is False
    assert gate["future_preconditioned_route_rows"] == 3
    assert gate["first_paper_draft_allowed_now"] is True
    assert gate["first_paper_submission_ready"] is False
    assert gate["new_model_claim_ready"] is False
    assert gate["phase154_model_training_allowed"] is False
    assert gate["operator_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False

    markdown = (tmp_path / "out/phase154_route_coverage_and_remaining_scheme_audit.md").read_text(
        encoding="utf-8"
    )
    assert "all currently opened and executable model/research routes have been verified" in markdown
    assert "future scheme space is not exhausted" in markdown
    assert "P154-DECISION-001" in markdown
    assert "|  |  |  |" not in markdown


def test_phase154_incomplete_if_phase153_gate_not_ready(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path, phase153_status="phase153_incomplete"),
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase154_route_coverage_audit_incomplete"
    assert gate["phase154_model_training_allowed"] is False
    assert gate["operator_training_allowed_now"] is False
    assert gate["a100_training_allowed_now"] is False
