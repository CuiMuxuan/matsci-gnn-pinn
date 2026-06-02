from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase68_validation_signal_scorecard.py")
    spec = importlib.util.spec_from_file_location("phase68_scorecard", script)
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


def _paths(tmp_path: Path) -> dict[str, Path]:
    phase60_manifest = _write_json(
        tmp_path / "phase60/manifest.json",
        {
            "phase": 60,
            "model_expansion_gate": {
                "decision": "block_density_failure_driven_model_expansion",
                "reason": "validation-visible correction does not beat the reference",
                "selected_variant": "blend:broad_process_v1->mean:alpha=1",
                "uses_test_for_selection": False,
            },
        },
    )
    next_gate = _write_csv(
        tmp_path / "phase60/next_gate.csv",
        [
            {
                "branch": "Candidate A: physically constrained spot-size conditioning",
                "status": "paused",
                "entry_condition": "Requires a new validation-visible spot-size signal.",
                "focused_validation": "broad12/broad21 spot_size",
                "seed_expansion_gate": "non-worse than floor",
                "manuscript_rule": "promote only if positive",
            },
            {
                "branch": "Candidate B: validation-auditable route policy",
                "status": "blocked by Phase 59 density gate",
                "entry_condition": "Route choice must be selected from train/validation evidence.",
                "focused_validation": "non-trainable policy first",
                "seed_expansion_gate": "non-worse than floor",
                "manuscript_rule": "future only",
            },
            {
                "branch": "Candidate C: heat-kernel or Green's-function source features",
                "status": "blocked by registration data",
                "entry_condition": "Requires aligned source-path data.",
                "focused_validation": "feature gate first",
                "seed_expansion_gate": "non-worse than floor",
                "manuscript_rule": "blocked",
            },
            {
                "branch": "External robustness / second dataset branch",
                "status": "deferred until manuscript package is draft-ready",
                "entry_condition": "Use after the claim package is internally complete.",
                "focused_validation": "local feasibility first",
                "seed_expansion_gate": "new frozen floor",
                "manuscript_rule": "external robustness",
            },
        ],
    )
    stress = _write_csv(
        tmp_path / "phase60/stress.csv",
        [
            {
                "scenario": "stronger_baseline_stress",
                "dataset": "broad12",
                "split": "spot_size",
                "metric": "Test RMSE",
                "status": "pass",
                "candidate": 136.0,
                "comparator": "mean: 151.0",
                "delta_vs_comparator": -15.0,
                "selected_variant": "",
                "manuscript_use": "supports fixed floor",
                "evidence": "phase58.json",
            },
            {
                "scenario": "alternate_density_stress",
                "dataset": "broad21",
                "split": "spot_size",
                "metric": "Test RMSE",
                "status": "boundary",
                "candidate": 153.0,
                "comparator": "mean: 139.0",
                "delta_vs_comparator": 14.0,
                "selected_variant": "",
                "manuscript_use": "boundary",
                "evidence": "phase58_density.json",
            },
            {
                "scenario": "residual_upper_bound_gate",
                "dataset": "broad21_density",
                "split": "test",
                "metric": "Test RMSE",
                "status": "blocks_model_expansion",
                "candidate": 153.0,
                "comparator": "mean: 139.0",
                "delta_vs_comparator": 14.0,
                "selected_variant": "blend:broad_process_v1->mean:alpha=1",
                "manuscript_use": "do not expand",
                "evidence": "phase59_upper.json",
            },
        ],
    )
    appendix = _write_csv(
        tmp_path / "phase60/appendix.csv",
        [
            {
                "phase": "49",
                "branch": "heat-kernel",
                "target": "Line_0_1",
                "result": "synthetic-positive AM-Bench-negative",
                "paper_use": "appendix diagnostic",
                "evidence": "docs/results/x.md",
            }
        ],
    )
    phase61_manifest = _write_json(
        tmp_path / "phase61/manifest.json",
        {
            "phase": 61,
            "writing_stage_gate": {
                "active_gate": "draft_ready_for_internal_results_methods; needs_verification_for_literature_context"
            },
        },
    )
    crosswalk = _write_csv(
        tmp_path / "phase61/crosswalk.csv",
        [
            {
                "claim_anchor_id": "C61-GATE-001",
                "claim_summary": "Candidate A is paused, Candidate B is blocked, and Candidate C is blocked by registration data.",
                "manuscript_location": "Discussion",
                "support_type": "result",
                "support_locator": "phase60 next gate",
                "evidence_register_key": "phase60_next_gate",
                "allowed_claim_strength": "moderate",
                "verification_state": "writing_ready",
                "owner_skill": "paper-writing-workflow",
                "open_risk": "future signal may reopen",
                "draft_sentence": "The next-branch table keeps Candidate A paused.",
            }
        ],
    )
    upper = _write_json(
        tmp_path / "phase59/upper.json",
        {
            "uses_test_for_selection": False,
            "selected_variant": {"name": "blend:broad_process_v1->mean:alpha=1"},
            "decision": {
                "selected_variant": "blend:broad_process_v1->mean:alpha=1",
                "selected_beats_reference_rmse": False,
            },
        },
    )
    return {
        "phase60_manifest": phase60_manifest,
        "phase60_next_gate": next_gate,
        "phase60_stress": stress,
        "phase60_appendix": appendix,
        "phase61_manifest": phase61_manifest,
        "phase61_crosswalk": crosswalk,
        "phase59_upper": upper,
    }


def test_phase68_builds_scorecard_and_action_queue(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(tmp_path, tmp_path / "out", paths=_paths(tmp_path))

    assert manifest["phase"] == 68
    assert manifest["counts"]["candidate_rows"] == 6
    assert manifest["counts"]["opened_trainable_candidates"] == 0
    assert manifest["current_decision"]["a100_80gb_request_now"] is False
    assert manifest["current_decision"]["trainable_model_opened"] is False
    outputs = manifest["outputs"]
    for path in outputs.values():
        assert (tmp_path / path).exists()
    with (tmp_path / outputs["candidate_signal_scorecard"]).open(encoding="utf-8", newline="") as handle:
        scorecard = list(csv.DictReader(handle))
    by_id = {row["candidate_id"]: row for row in scorecard}
    assert by_id["A"]["status"] == "paused_no_training_signal"
    assert by_id["B"]["status"] == "blocked_by_phase59_validation_gate"
    assert by_id["C"]["status"] == "blocked_by_registration_data"
    assert by_id["D"]["status"] == "deferred_requires_local_identifiability_gate"
    assert by_id["E"]["status"] == "open_for_data_planning_only"
    assert "A100-SXM4-80GB" in by_id["D"]["a100_80gb_trigger"]
    assert "no_trainable_model_opened" in by_id["SUMMARY"]["decision"]


def test_phase68_action_queue_keeps_80gb_as_conditional_resource_gate(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(tmp_path, tmp_path / "out", paths=_paths(tmp_path))

    with (tmp_path / manifest["outputs"]["next_action_queue"]).open(encoding="utf-8", newline="") as handle:
        actions = list(csv.DictReader(handle))

    by_id = {row["action_id"]: row for row in actions}
    assert by_id["P68-80GB-TRIGGER"]["status"] == "standing_gate"
    assert by_id["P68-80GB-TRIGGER"]["requires_a100_80gb"] == "conditional"
    assert "40GB" in by_id["P68-80GB-TRIGGER"]["exit_gate"]
    assert by_id["P68-SPOT-SIGNAL"]["requires_a100"] == "no"
