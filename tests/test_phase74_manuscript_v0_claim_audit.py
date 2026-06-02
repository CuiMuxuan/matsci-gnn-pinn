from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase74_manuscript_v0_claim_audit.py")
    spec = importlib.util.spec_from_file_location("phase74_manuscript_v0", script)
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
        },
        {
            "claim_anchor_id": "C61-REL-001",
            "claim_summary": "A related-work claim still needs literature.",
            "manuscript_location": "Related Work",
            "support_type": "literature",
            "support_locator": "",
            "evidence_register_key": "lit_gap",
            "allowed_claim_strength": "none",
            "verification_state": "needs_verification",
            "owner_skill": "academic-research-verification",
            "open_risk": "Resolve literature before writing this claim.",
            "draft_sentence": "",
        },
    ]


def _gap_rows() -> list[dict[str, object]]:
    return [
        {
            "gap_id": "LIT_GAP-001",
            "location": "Introduction",
            "claim_needing_support": "AM-Bench benchmark context.",
            "evidence_type_needed": "verified citation",
            "suggested_search_or_material": "NIST AM-Bench source",
            "blocks_current_phase61_draft": "no",
        }
    ]


def _paths(tmp_path: Path, *, trainable_opened: bool = False) -> dict[str, Path]:
    phase60_manifest = {
        "claim_boundary": {
            "main_claim": "fixed-sampling broad12/broad21 spot_size under broad_process_v1, seeds 7/1/2",
            "excluded_claims": [
                "density-invariant robustness",
                "universal process-conditioning success",
            ],
        },
        "model_expansion_gate": {
            "decision": "block_density_failure_driven_model_expansion",
            "reason": "validation-visible correction does not beat the reference",
            "selected_variant": "blend:broad_process_v1->mean:alpha=1",
            "uses_test_for_selection": False,
        },
    }
    phase61_manifest = {
        "writing_stage_gate": {
            "mode": "section_draft",
            "active_gate": "draft_ready_for_internal_results_methods",
        }
    }
    phase68_manifest = {
        "current_decision": {
            "trainable_model_opened": trainable_opened,
            "reason": "current evidence opens no trainable branch directly",
        }
    }
    phase69_gate = {
        "candidate": "Candidate A",
        "status": "paused_no_training_signal",
        "reason": "no validation-visible spot-size signal",
    }
    phase70_gate = {
        "candidate": "Candidate B",
        "status": "blocked_no_validation_visible_route_policy_signal",
        "reason": "no route-policy signal",
    }
    phase71_gate = {
        "candidate": "Candidate C",
        "status": "blocked_by_registration_data",
        "reason": "no registered source path",
    }
    return {
        "phase57_ledger": _write_csv(tmp_path / "phase57_ledger.csv", [{"kind": "process_axis"}]),
        "phase60_main": _write_csv(tmp_path / "phase60_main.csv", [{"dataset": "broad12"}]),
        "phase60_route": _write_csv(tmp_path / "phase60_route.csv", [{"dataset": "broad12"}]),
        "phase60_stress": _write_csv(tmp_path / "phase60_stress.csv", [{"scenario": "stress"}]),
        "phase60_appendix": _write_csv(tmp_path / "phase60_appendix.csv", [{"phase": "33"}]),
        "phase60_next_gate": _write_csv(tmp_path / "phase60_next_gate.csv", [{"branch": "Candidate A"}]),
        "phase60_manifest": _write_json(tmp_path / "phase60_manifest.json", phase60_manifest),
        "phase61_results": _write_text(tmp_path / "phase61_results.md", "# Results\n\nSupported result."),
        "phase61_methods": _write_text(tmp_path / "phase61_methods.md", "# Methods\n\nSupported method."),
        "phase61_captions": _write_text(tmp_path / "phase61_captions.md", "# Captions\n"),
        "phase61_crosswalk": _write_csv(tmp_path / "phase61_crosswalk.csv", _crosswalk_rows()),
        "phase61_gaps": _write_csv(tmp_path / "phase61_gaps.csv", _gap_rows()),
        "phase61_manifest": _write_json(tmp_path / "phase61_manifest.json", phase61_manifest),
        "phase68_scorecard": _write_csv(
            tmp_path / "phase68_scorecard.csv",
            [{"candidate_id": "SUMMARY", "status": "summary"}],
        ),
        "phase68_manifest": _write_json(tmp_path / "phase68_manifest.json", phase68_manifest),
        "phase69_gate": _write_json(tmp_path / "phase69_gate.json", phase69_gate),
        "phase70_gate": _write_json(tmp_path / "phase70_gate.json", phase70_gate),
        "phase71_gate": _write_json(tmp_path / "phase71_gate.json", phase71_gate),
        "phase71_manifest": _write_json(
            tmp_path / "phase71_manifest.json",
            {"candidate_c_gate": phase71_gate},
        ),
    }


def test_phase74_locks_v0_claims_and_boundaries(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(tmp_path, tmp_path / "out", paths=_paths(tmp_path))

    gate = manifest["writing_stage_gate"]
    assert manifest["phase"] == 74
    assert gate["status"] == "ready_for_internal_manuscript_review"
    assert gate["main_claim_locked"] is True
    assert gate["trainable_model_opened_now"] is False
    assert gate["candidate_a_status"] == "paused_no_training_signal"
    assert gate["candidate_b_status"] == "blocked_no_validation_visible_route_policy_signal"
    assert gate["candidate_c_status"] == "blocked_by_registration_data"
    assert manifest["counts"]["claim_audit_rows"] == 4
    assert manifest["counts"]["unsupported_v0_claim_rows"] == 2
    assert manifest["counts"]["boundary_rows"] == 7

    with (tmp_path / manifest["outputs"]["claim_audit_table"]).open(encoding="utf-8", newline="") as handle:
        claim_rows = list(csv.DictReader(handle))
    assert any(row["claim_id"] == "C61-MAIN-001" and row["allowed_in_v0"] == "yes" for row in claim_rows)
    assert any(row["claim_id"] == "C61-REL-001" and row["allowed_in_v0"] == "no" for row in claim_rows)
    assert any(row["claim_id"] == "C74-LIT-LOCK" and row["allowed_in_v0"] == "no" for row in claim_rows)

    with (tmp_path / manifest["outputs"]["model_boundary_register"]).open(encoding="utf-8", newline="") as handle:
        boundary_rows = list(csv.DictReader(handle))
    assert any(row["boundary_id"] == "C74-GATE-A" for row in boundary_rows)
    assert any(row["boundary_id"] == "C74-GATE-C" for row in boundary_rows)


def test_phase74_records_trainable_model_opened_if_upstream_gate_changes(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        tmp_path,
        tmp_path / "out",
        paths=_paths(tmp_path, trainable_opened=True),
    )

    assert manifest["writing_stage_gate"]["trainable_model_opened_now"] is True
    with (tmp_path / manifest["outputs"]["model_boundary_register"]).open(encoding="utf-8", newline="") as handle:
        boundary_rows = list(csv.DictReader(handle))
    trainable_row = next(row for row in boundary_rows if row["boundary_id"] == "C74-GATE-TRAINABLE")
    assert trainable_row["status"] == "trainable_model_opened"
