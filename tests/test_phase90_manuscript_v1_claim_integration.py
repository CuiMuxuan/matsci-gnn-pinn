from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase90_manuscript_v1_claim_integration.py")
    spec = importlib.util.spec_from_file_location("phase90_manuscript_v1", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


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


def _crosswalk_rows() -> list[dict[str, object]]:
    return [
        {
            "claim_anchor_id": "C61-MAIN-001",
            "claim_summary": "Fixed-sampling spot-size transfer is the main claim.",
            "manuscript_location": "Results",
            "support_type": "result",
            "support_locator": "phase60_main.csv",
            "evidence_register_key": "phase60_main",
            "allowed_claim_strength": "moderate",
            "verification_state": "writing_ready",
            "owner_skill": "paper-writing-workflow",
            "open_risk": "Do not claim density-invariant robustness.",
            "draft_sentence": "The fixed-sampling spot-size route is positive.",
        }
    ]


def _evidence_rows() -> list[dict[str, object]]:
    return [
        {
            "evidence_id": "P89-EVD-AMBench-001",
            "gap_id": "LIT_GAP-61-001",
            "claim_area": "AM-Bench context",
            "source_title": "Additive Manufacturing Benchmark Test Series (AM-Bench)",
            "authors_or_owner": "NIST",
            "venue_or_source": "NIST official page",
            "year": "2026 access",
            "doi": "",
            "stable_url": "https://www.nist.gov/ambench",
            "source_type": "official_project_page",
            "trust_state": "verified_official_url",
            "verification_trail": "verified",
            "supports_claim": "AM-Bench is a public benchmark context.",
            "allowed_claim_strength": "official_benchmark_context",
            "limitations": "Do not overclaim robustness.",
            "writing_ready": "true",
        },
        {
            "evidence_id": "P89-EVD-MDS2-2716",
            "gap_id": "LIT_GAP-61-001",
            "claim_area": "AM-Bench thermography",
            "source_title": "AM Bench 2022 Measurement Results Data",
            "authors_or_owner": "NIST",
            "venue_or_source": "NIST PDR",
            "year": "2024",
            "doi": "10.18434/mds2-2716",
            "stable_url": "https://data.nist.gov/od/id/mds2-2716",
            "source_type": "official_dataset_record",
            "trust_state": "verified_project_manifest_and_stable_url",
            "verification_trail": "manifest",
            "supports_claim": "Experiments use public thermography data.",
            "allowed_claim_strength": "dataset_specific",
            "limitations": "Registration remains blocked.",
            "writing_ready": "true",
        },
        {
            "evidence_id": "P89-EVD-PINN-001",
            "gap_id": "LIT_GAP-61-002",
            "claim_area": "PINN foundation",
            "source_title": "Physics-informed neural networks",
            "authors_or_owner": "Raissi, Perdikaris, and Karniadakis",
            "venue_or_source": "Journal of Computational Physics",
            "year": "2019",
            "doi": "10.1016/j.jcp.2018.10.045",
            "stable_url": "https://www.sciencedirect.com/science/article/pii/S0021999118307125",
            "source_type": "peer_reviewed_article",
            "trust_state": "verified_doi_primary_url",
            "verification_trail": "verified",
            "supports_claim": "PINNs support forward and inverse PDE settings.",
            "allowed_claim_strength": "foundational_method_context",
            "limitations": "Foundational support only.",
            "writing_ready": "true",
        },
        {
            "evidence_id": "P89-EVD-FILM-001",
            "gap_id": "LIT_GAP-61-002",
            "claim_area": "conditioning",
            "source_title": "FiLM: Visual Reasoning with a General Conditioning Layer",
            "authors_or_owner": "Perez et al.",
            "venue_or_source": "AAAI",
            "year": "2018",
            "doi": "10.1609/aaai.v32i1.11671",
            "stable_url": "https://ojs.aaai.org/index.php/AAAI/article/view/11671",
            "source_type": "peer_reviewed_conference_article",
            "trust_state": "verified_doi_primary_url",
            "verification_trail": "verified",
            "supports_claim": "Feature-wise affine modulation is recognized.",
            "allowed_claim_strength": "conditioning_mechanism_context",
            "limitations": "Mechanism support only.",
            "writing_ready": "true",
        },
    ]


def _handoff_rows() -> list[dict[str, object]]:
    return [
        {
            "handoff_id": "P89-HANDOFF-DATASET",
            "target_section": "Introduction/Dataset",
            "claim_anchor": "AM-Bench benchmark and data source framing",
            "allowed_claim": "AM-Bench can be described as a public NIST-led benchmark ecosystem.",
            "evidence_ids": "P89-EVD-AMBench-001;P89-EVD-MDS2-2716",
            "source_locator": "NIST sources",
            "allowed_strength": "dataset_context",
            "wording_guard": "Do not imply all modalities were used.",
            "unresolved_dependency": "",
        },
        {
            "handoff_id": "P89-HANDOFF-PINN",
            "target_section": "Related Work/Methods",
            "claim_anchor": "PINN framing",
            "allowed_claim": "PINNs are established for forward and inverse PDE problems.",
            "evidence_ids": "P89-EVD-PINN-001",
            "source_locator": "PINN source",
            "allowed_strength": "framing_and_limitation_context",
            "wording_guard": "Avoid exact failure-mode overclaiming.",
            "unresolved_dependency": "",
        },
        {
            "handoff_id": "P89-HANDOFF-CONDITIONING",
            "target_section": "Methods/Model",
            "claim_anchor": "process-conditioned route mechanism",
            "allowed_claim": "Feature-wise modulation is a recognized conditioning mechanism.",
            "evidence_ids": "P89-EVD-FILM-001",
            "source_locator": "FiLM source",
            "allowed_strength": "method_mechanism_context",
            "wording_guard": "Performance must cite Phase 55/60/74 artifacts.",
            "unresolved_dependency": "",
        },
        {
            "handoff_id": "P89-HANDOFF-VENUE",
            "target_section": "All final manuscript sections",
            "claim_anchor": "target venue style and citation density",
            "allowed_claim": "No final venue-specific style claim is allowed yet.",
            "evidence_ids": "",
            "source_locator": "",
            "allowed_strength": "none_until_user_input",
            "wording_guard": "Keep final style provisional.",
            "unresolved_dependency": "target venue, author guide, or accepted benchmark papers",
        },
    ]


def _paths(tmp_path: Path) -> dict[str, Path]:
    phase74_manifest = {
        "claim_boundary": {
            "main_claim": "fixed-sampling broad12/broad21 spot_size under broad_process_v1, seeds 7/1/2",
            "excluded_claims": ["density-invariant robustness"],
        }
    }
    phase88_gate = {
        "status": "fallback_experimental_claim_complete",
        "experimental_claim_complete": True,
        "submission_ready": False,
    }
    phase89_manifest = {
        "gate": {
            "status": "literature_core_resolved_venue_unresolved",
            "core_literature_ready": True,
            "venue_alignment_ready": False,
            "submission_ready": False,
        }
    }
    return {
        "phase61_results": _write_text(tmp_path / "phase61_results.md", "# Results\n\nResult text."),
        "phase61_methods": _write_text(tmp_path / "phase61_methods.md", "# Methods\n\nMethod text."),
        "phase61_captions": _write_text(tmp_path / "phase61_captions.md", "# Captions\n\nCaption text."),
        "phase61_crosswalk": _write_csv(tmp_path / "phase61_crosswalk.csv", _crosswalk_rows()),
        "phase74_manifest": _write_json(tmp_path / "phase74_manifest.json", phase74_manifest),
        "phase74_boundary": _write_csv(
            tmp_path / "phase74_boundary.csv",
            [
                {
                    "boundary_id": "C74-GATE-C",
                    "candidate_or_scope": "Candidate C",
                    "status": "blocked_by_registration_data",
                    "main_text_treatment": "future work only",
                    "appendix_treatment": "blocked gate",
                    "evidence_locator": "phase71_gate.json",
                    "reason": "no registration",
                }
            ],
        ),
        "phase88_gate": _write_json(tmp_path / "phase88_gate.json", phase88_gate),
        "phase89_evidence": _write_csv(tmp_path / "phase89_evidence.csv", _evidence_rows()),
        "phase89_gap_resolution": _write_csv(
            tmp_path / "phase89_gap_resolution.csv",
            [
                {
                    "gap_id": "LIT_GAP-61-003",
                    "location": "Target venue",
                    "original_claim_needing_support": "Venue style",
                    "resolution_status": "unresolved_user_input_required",
                    "resolved_by_evidence_ids": "",
                    "blocks_submission_after_phase89": "true",
                    "blocks_experimental_claim": "false",
                    "unresolved_reason": "No target venue",
                    "next_action": "Provide target venue.",
                }
            ],
        ),
        "phase89_handoff": _write_csv(tmp_path / "phase89_handoff.csv", _handoff_rows()),
        "phase89_manual_queue": _write_csv(
            tmp_path / "phase89_manual_queue.csv",
            [
                {
                    "queue_id": "P89-MANUAL-001",
                    "category": "target_venue_alignment",
                    "needed_input": "No target venue",
                    "reason": "Venue style",
                    "blocks_submission": "true",
                    "suggested_user_action": "Provide target venue.",
                }
            ],
        ),
        "phase89_manifest": _write_json(tmp_path / "phase89_manifest.json", phase89_manifest),
    }


def test_phase90_integrates_core_handoffs_but_keeps_venue_blocker(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(tmp_path, tmp_path / "out", paths=_paths(tmp_path))

    gate = manifest["gate"]
    assert manifest["phase"] == 90
    assert gate["status"] == "manuscript_v1_core_claims_integrated_venue_unresolved"
    assert gate["core_claims_integrated"] is True
    assert gate["core_literature_ready"] is True
    assert gate["venue_alignment_ready"] is False
    assert gate["submission_ready"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert gate["integrated_handoff_rows"] == 3
    assert gate["blocked_handoff_rows"] == 1
    assert gate["submission_blockers"] == 1

    manuscript = (tmp_path / manifest["outputs"]["manuscript_v1"]).read_text(encoding="utf-8")
    assert "Introduction And Dataset Context" in manuscript
    assert "Related Work And Method Context" in manuscript
    assert "Remaining Venue Dependency" in manuscript
    assert "P89-EVD-FILM-001" in manuscript
    assert "target venue, author guide, or accepted benchmark papers" in manuscript


def test_phase90_writes_audit_and_blocker_tables(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(tmp_path, tmp_path / "out", paths=_paths(tmp_path))

    with (tmp_path / manifest["outputs"]["literature_integration_table"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        integration_rows = list(csv.DictReader(handle))
    by_id = {row["integration_id"]: row for row in integration_rows}
    assert by_id["P89-HANDOFF-DATASET"]["integration_status"] == "integrated_writing_ready"
    assert by_id["P89-HANDOFF-VENUE"]["integration_status"] == "blocked_user_input_required"

    with (tmp_path / manifest["outputs"]["claim_evidence_audit"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        audit_rows = list(csv.DictReader(handle))
    assert any(row["claim_id"] == "C61-MAIN-001" for row in audit_rows)
    assert any(row["claim_id"] == "P89-EVD-FILM-001" for row in audit_rows)
    assert any(row["claim_id"] == "LIT_GAP-61-003" for row in audit_rows)

    with (tmp_path / manifest["outputs"]["venue_blocker_queue"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        blocker_rows = list(csv.DictReader(handle))
    assert len(blocker_rows) == 1
    assert blocker_rows[0]["blocks_submission"] == "true"
    assert blocker_rows[0]["blocks_phase90_core_claims"] == "false"
