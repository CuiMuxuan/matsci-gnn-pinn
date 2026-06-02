from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase89_literature_venue_gap_resolution.py")
    spec = importlib.util.spec_from_file_location("phase89_lit_venue", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _gap_rows() -> list[dict[str, object]]:
    return [
        {
            "gap_id": "LIT_GAP-61-001",
            "location": "Introduction or Dataset paragraph",
            "claim_needing_support": "AM-Bench is a public additive-manufacturing benchmark suitable for thermal/process generalization studies.",
            "evidence_type_needed": "verified dataset citation or official NIST AM-Bench source",
            "suggested_search_or_material": "NIST AM-Bench official dataset page or peer-reviewed AM-Bench description",
            "blocks_current_phase61_draft": "no",
        },
        {
            "gap_id": "LIT_GAP-61-002",
            "location": "Related work",
            "claim_needing_support": "Physics-informed neural networks and process-conditioned neural models have known tradeoffs under sparse or heterogeneous thermal data.",
            "evidence_type_needed": "verified literature review and representative primary papers",
            "suggested_search_or_material": "PINN thermal modeling and process-conditioned neural operator papers",
            "blocks_current_phase61_draft": "no",
        },
        {
            "gap_id": "LIT_GAP-61-003",
            "location": "Target-venue adaptation",
            "claim_needing_support": "Final section order, citation density, and caption style match the target journal or conference.",
            "evidence_type_needed": "target venue author guide or 3-10 benchmark manuscripts",
            "suggested_search_or_material": "user-provided target venue or accepted examples",
            "blocks_current_phase61_draft": "no",
        },
    ]


def _paths(tmp_path: Path) -> dict[str, Path]:
    phase88_gate = {
        "status": "fallback_experimental_claim_complete",
        "experimental_claim_complete": True,
        "submission_ready": False,
        "open_submission_blockers": 2,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
    }
    phase88_manifest = {
        "phase": 88,
        "gate": phase88_gate,
        "counts": {
            "remaining_work_rows": 4,
            "literature_gap_rows": 3,
        },
    }
    return {
        "phase61_literature_gaps": _write_csv(tmp_path / "phase61_literature_gaps.csv", _gap_rows()),
        "phase88_remaining_work": _write_csv(
            tmp_path / "phase88_remaining_work.csv",
            [
                {
                    "work_id": "P88-WORK-LIT",
                    "category": "literature_verification",
                    "status": "open",
                    "required_input_or_gate": "3 gaps",
                    "blocks_submission": "true",
                    "blocks_experimental_claim": "false",
                    "next_action": "verify sources",
                },
                {
                    "work_id": "P88-WORK-VENUE",
                    "category": "target_venue_alignment",
                    "status": "open",
                    "required_input_or_gate": "target venue",
                    "blocks_submission": "true",
                    "blocks_experimental_claim": "false",
                    "next_action": "select venue",
                },
            ],
        ),
        "phase88_gate": _write_json(tmp_path / "phase88_gate.json", phase88_gate),
        "phase88_manifest": _write_json(tmp_path / "phase88_manifest.json", phase88_manifest),
        "mds2_2716_manifest": tmp_path / "unused_mds2_2716.yaml",
        "mds2_2718_manifest": tmp_path / "unused_mds2_2718.yaml",
    }


def test_phase89_resolves_core_literature_but_keeps_venue_unresolved(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(tmp_path, tmp_path / "out", paths=_paths(tmp_path))

    gate = manifest["gate"]
    assert manifest["phase"] == 89
    assert gate["status"] == "literature_core_resolved_venue_unresolved"
    assert gate["experimental_claim_complete"] is True
    assert gate["core_literature_ready"] is True
    assert gate["venue_alignment_ready"] is False
    assert gate["submission_ready"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert gate["verified_evidence_rows"] >= 8
    assert gate["unresolved_submission_blockers"] == 1

    with (tmp_path / manifest["outputs"]["gap_resolution_table"]).open(encoding="utf-8", newline="") as handle:
        gap_rows = list(csv.DictReader(handle))
    by_gap = {row["gap_id"]: row for row in gap_rows}
    assert by_gap["LIT_GAP-61-001"]["resolution_status"] == "resolved_writing_ready"
    assert by_gap["LIT_GAP-61-002"]["resolution_status"] == "resolved_writing_ready"
    assert by_gap["LIT_GAP-61-003"]["resolution_status"] == "unresolved_user_input_required"
    assert by_gap["LIT_GAP-61-003"]["blocks_submission_after_phase89"] == "true"


def test_phase89_outputs_evidence_handoff_and_manual_queue(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(tmp_path, tmp_path / "out", paths=_paths(tmp_path))

    with (tmp_path / manifest["outputs"]["evidence_register"]).open(encoding="utf-8", newline="") as handle:
        evidence_rows = list(csv.DictReader(handle))
    evidence_by_id = {row["evidence_id"]: row for row in evidence_rows}
    assert evidence_by_id["P89-EVD-PINN-001"]["doi"] == "10.1016/j.jcp.2018.10.045"
    assert evidence_by_id["P89-EVD-PINN-003"]["trust_state"] == "verified_doi_primary_url"
    assert evidence_by_id["P89-EVD-FILM-001"]["doi"] == "10.1609/aaai.v32i1.11671"
    assert evidence_by_id["P89-EVD-MDS2-2716"]["writing_ready"] == "true"

    with (tmp_path / manifest["outputs"]["writing_handoff"]).open(encoding="utf-8", newline="") as handle:
        handoff_rows = list(csv.DictReader(handle))
    by_handoff = {row["handoff_id"]: row for row in handoff_rows}
    assert "P89-EVD-MDS2-2716" in by_handoff["P89-HANDOFF-DATASET"]["evidence_ids"]
    assert by_handoff["P89-HANDOFF-VENUE"]["allowed_strength"] == "none_until_user_input"

    with (tmp_path / manifest["outputs"]["manual_verification_queue"]).open(encoding="utf-8", newline="") as handle:
        manual_rows = list(csv.DictReader(handle))
    assert len(manual_rows) == 1
    assert manual_rows[0]["category"] == "target_venue_alignment"
    assert manual_rows[0]["blocks_submission"] == "true"
