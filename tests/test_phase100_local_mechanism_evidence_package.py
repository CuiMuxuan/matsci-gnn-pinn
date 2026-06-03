from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase100_local_mechanism_evidence_package.py")
    spec = importlib.util.spec_from_file_location("phase100_package", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase100_packages_local_mechanism_and_keeps_transfer_locked(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(Path(".").resolve(), tmp_path / "out")

    gate = manifest["gate"]
    assert manifest["phase"] == 100
    assert gate["status"] == "local_mechanism_package_ready_transfer_locked"
    assert gate["appendix_local_mechanism_ready"] is True
    assert gate["main_paper_transfer_claim_ready"] is False
    assert gate["phase101_registered_target_acquisition_allowed"] is True
    assert gate["am_bench_transfer_unlocked"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert gate["same_budget_boundary_disclosed"] is True
    assert manifest["counts"]["evidence_rows"] == 6
    assert manifest["counts"]["boundary_rows"] == 4
    assert manifest["counts"]["claim_rows"] == 4
    assert manifest["counts"]["allowed_claim_rows"] == 1
    assert manifest["counts"]["blocked_claim_rows"] == 3

    with (tmp_path / "out/phase100_local_mechanism_evidence_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        evidence_rows = list(csv.DictReader(handle))
    assert {row["row_id"] for row in evidence_rows} == {
        "P100-EVID-001",
        "P100-EVID-002",
        "P100-EVID-003",
        "P100-EVID-004",
        "P100-EVID-005",
        "P100-EVID-006",
    }
    same_budget = next(row for row in evidence_rows if row["row_id"] == "P100-EVID-006")
    assert same_budget["claim_scope"] == "boundary_disclosed"
    assert "gradient_q90_rmse" in same_budget["notes"]

    with (tmp_path / "out/phase100_local_mechanism_claim_use_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        claim_rows = list(csv.DictReader(handle))
    allowed = [row for row in claim_rows if row["allowed_now"] == "true"]
    blocked = [row for row in claim_rows if row["allowed_now"] == "false"]
    assert [row["claim_id"] for row in allowed] == ["P100-CLAIM-001"]
    assert {row["claim_id"] for row in blocked} == {
        "P100-CLAIM-002",
        "P100-CLAIM-003",
        "P100-CLAIM-004",
    }


def test_phase100_gate_blocks_without_phase99_package_permission():
    module = _load_module()

    gate = module.build_gate(
        phase96_gate={
            "status": "local_smoke_positive_transfer_design_only",
            "positive_mechanisms": ["fixed_green_function_features"],
        },
        phase98_gate={
            "status": "registered_surrogate_unlocked_no_a100",
            "am_bench_transfer_unlocked": False,
        },
        phase99_gate={
            "status": "closed_local_surrogate_negative",
            "phase100_local_mechanism_package_allowed": False,
            "a100_training_allowed_now": False,
            "a100_80gb_request_now": False,
        },
        evidence_rows=[],
        boundary_rows=[
            {
                "boundary_type": "same_budget_focused_boundary",
            }
        ],
        claim_rows=[
            {"allowed_now": True},
            {"allowed_now": False},
        ],
    )

    assert gate["status"] == "blocked_local_mechanism_package_incomplete"
    assert gate["appendix_local_mechanism_ready"] is False
    assert gate["phase101_registered_target_acquisition_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
