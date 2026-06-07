from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase147_literature_guided_model_roadmap.py")
    spec = importlib.util.spec_from_file_location("phase147_model_roadmap", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _phase_inputs(tmp_path: Path, *, unlock_phase114: bool = False) -> dict[str, Path]:
    return {
        "phase111_registered_target_closure": _write_json(
            tmp_path / "phase111_gate.json",
            {
                "status": "phase111_registered_target_closure_package_ready_sequence_branch_closed",
                "nist_ammt_sequence_branch_closed": True,
                "phase111_model_training_allowed": False,
                "a100_training_allowed_now": False,
                "a100_80gb_request_now": False,
            },
        ),
        "phase113_melt_pool_focused_review": _write_json(
            tmp_path / "phase113_gate.json",
            {
                "status": "phase113_melt_pool_focused_review_closed_validation_test_reversal",
                "phase113_model_training_allowed": False,
                "a100_training_allowed_now": False,
                "a100_80gb_request_now": False,
            },
        ),
        "phase114_gcode_strategy_source": _write_json(
            tmp_path / "phase114_gate.json",
            {
                "status": "phase114_gcode_strategy_source_gate_closed_no_guarded_baseline_gap",
                "phase114_model_training_allowed": bool(unlock_phase114),
                "a100_training_allowed_now": False,
                "a100_80gb_request_now": False,
            },
        ),
        "phase115_nist_ammt_diagnostic_closure": _write_json(
            tmp_path / "phase115_gate.json",
            {
                "status": "phase115_nist_ammt_diagnostic_closure_package_ready_all_new_branches_closed",
                "all_training_locks_verified": True,
                "phase115_model_training_allowed": False,
                "a100_training_allowed_now": False,
                "a100_80gb_request_now": False,
            },
        ),
        "phase146_paper_evidence_refresh": _write_json(
            tmp_path / "phase146_gate.json",
            {
                "status": "phase146_paper_evidence_refresh_ready_first_paper_narrow_claims",
                "first_paper_draft_allowed_now": True,
                "phase146_model_training_allowed": False,
                "a100_training_allowed_now": False,
                "a100_80gb_request_now": False,
            },
        ),
    }


def test_phase147_opens_only_no_training_phase148_design(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        phase_inputs=_phase_inputs(tmp_path),
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 147
    assert gate["status"] == "phase147_literature_guided_model_roadmap_ready_phase148_no_training_design"
    assert gate["phase148_no_training_design_allowed"] is True
    assert gate["recommended_phase148_route"] == "capl_path_contact_graph_audit"
    assert gate["new_main_paper_claim_ready"] is False
    assert gate["phase147_model_mechanism_allowed"] is False
    assert gate["phase147_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False

    with (tmp_path / "out/phase147_literature_route_audit_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        route_rows = list(csv.DictReader(handle))
    assert {row["route_id"] for row in route_rows} >= {
        "capl_path_history",
        "physics_hardcoded_gnn",
        "meltpoolgan",
        "neural_operator",
    }
    capl = next(row for row in route_rows if row["route_id"] == "capl_path_history")
    assert "phase114" in capl["prior_project_evidence"]
    assert capl["recommended_use"] == "only_if_finer_than_phase114"

    markdown = (tmp_path / "out/phase147_literature_guided_model_roadmap.md").read_text(
        encoding="utf-8"
    )
    assert "capl_path_contact_graph" in markdown
    assert "Model training allowed now: `false`" in markdown


def test_phase147_blocks_if_prior_training_lock_is_open(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        phase_inputs=_phase_inputs(tmp_path, unlock_phase114=True),
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase147_literature_guided_model_roadmap_incomplete"
    assert gate["phase148_no_training_design_allowed"] is False
    assert gate["phase147_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
