from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase152_paper_evidence_neural_operator_route_closure.py")
    spec = importlib.util.spec_from_file_location("phase152_route_closure", script)
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
        for row in rows:
            writer.writerow(row)
    return path


def _paths(tmp_path: Path, *, low_capacity_candidates: int = 0) -> dict[str, Path]:
    phase146_gate = _write_json(
        tmp_path / "p146/gate.json",
        {
            "status": "phase146_paper_evidence_refresh_ready_first_paper_narrow_claims",
            "first_paper_draft_allowed_now": True,
            "main_paper_floor": (
                "fixed-sampling broad12/broad21 spot_size under broad_process_v1, seeds 7/1/2"
            ),
            "phase146_model_training_allowed": False,
            "a100_training_allowed_now": False,
        },
    )
    phase149_gate = _write_json(
        tmp_path / "p149/gate.json",
        {
            "status": "phase149_neural_operator_readiness_closed_not_ready_for_operator_training",
            "blocker_rows": 5,
            "operator_training_allowed_now": False,
            "phase149_model_training_allowed": False,
            "a100_training_allowed_now": False,
        },
    )
    phase150_gate = _write_json(
        tmp_path / "p150/gate.json",
        {
            "status": "phase150_dense_tensorization_inventory_ready_phase151_fixed_grid_baseline_review",
            "present_source_rows": 6,
            "tensorizable_candidate_rows": 6,
            "operator_gap_ready_rows": 0,
            "operator_training_allowed_now": False,
            "phase150_model_training_allowed": False,
            "a100_training_allowed_now": False,
        },
    )
    phase151_gate = _write_json(
        tmp_path / "p151/gate.json",
        {
            "status": "phase151_fixed_grid_dense_baseline_closed_no_operator_gap",
            "split_contract_rows": 3,
            "leakage_safe_source_rows": 1,
            "phase152_low_capacity_dense_design_candidates": low_capacity_candidates,
            "operator_training_allowed_now": False,
            "phase151_model_training_allowed": False,
            "a100_training_allowed_now": False,
        },
    )
    return {
        "phase146_gate": phase146_gate,
        "phase149_gate": phase149_gate,
        "phase149_readiness_table": _write_csv(
            tmp_path / "p149/readiness.csv",
            [{"criterion_id": "r1", "blocks_operator_training": True}],
        ),
        "phase150_gate": phase150_gate,
        "phase150_inventory_table": _write_csv(
            tmp_path / "p150/inventory.csv",
            [
                {
                    "candidate_id": "dense_csv",
                    "present": True,
                    "tensorization_status": "candidate_indexed_dense_csv_needs_split_and_operator_baseline",
                }
            ],
        ),
        "phase151_gate": phase151_gate,
        "phase151_review_table": _write_csv(
            tmp_path / "p151/review.csv",
            [
                {
                    "candidate_id": "multiline",
                    "target": "target_frame_mean",
                    "strong_baseline_solved": True,
                    "status": "blocked_strong_baseline_solved",
                }
            ],
        ),
        "phase151_split_table": _write_csv(
            tmp_path / "p151/splits.csv",
            [
                {
                    "candidate_id": "single_line",
                    "split_contract_status": "diagnostic_frame_block_split_only",
                    "leakage_safe_split": False,
                },
                {
                    "candidate_id": "multiline",
                    "split_contract_status": "leakage_safe_line_group_split",
                    "leakage_safe_split": True,
                },
            ],
        ),
    }


def test_phase152_closes_neural_operator_route_and_keeps_claim_boundary(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path),
    )

    gate = manifest["gate"]
    assert gate["status"] == (
        "phase152_paper_evidence_refresh_ready_first_paper_narrow_claims_neural_operator_closed"
    )
    assert gate["first_paper_draft_allowed_now"] is True
    assert gate["neural_operator_route_closed_as_diagnostic"] is True
    assert gate["new_neural_operator_model_claim_ready"] is False
    assert gate["phase152_model_training_allowed"] is False
    assert gate["operator_training_allowed_now"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert manifest["counts"]["route_closure_rows"] == 4
    assert manifest["counts"]["operator_training_allowed_route_rows"] == 0

    markdown = (tmp_path / "out/phase152_paper_evidence_refresh.md").read_text(
        encoding="utf-8"
    )
    assert "Do not write FNO/neural-operator success" in markdown
    assert "P152-CLAIM-002" in markdown
    assert "complete GNN-PINN" in markdown
    assert "|  |  |  |" not in markdown


def test_phase152_incomplete_if_phase151_reopens_dense_design_candidates(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path, low_capacity_candidates=1),
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase152_paper_evidence_refresh_incomplete_neural_operator_closure"
    assert gate["neural_operator_route_closed_as_diagnostic"] is False
    assert gate["phase152_model_training_allowed"] is False
    assert gate["operator_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
