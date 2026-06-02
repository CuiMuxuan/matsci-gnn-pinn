from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase88_fallback_manuscript_finalization.py")
    spec = importlib.util.spec_from_file_location("phase88_finalization", script)
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
    phase60_manifest = {
        "claim_boundary": {
            "main_claim": "fixed-sampling broad12/broad21 spot_size under broad_process_v1, seeds 7/1/2",
            "excluded_claims": ["density-invariant robustness"],
        },
        "model_expansion_gate": {
            "decision": "block_density_failure_driven_model_expansion",
            "reason": "validation-visible correction does not beat the reference on the analysis split",
            "selected_variant": "blend:broad_process_v1->mean:alpha=1",
            "uses_test_for_selection": False,
        },
    }
    phase74_manifest = {
        "writing_stage_gate": {
            "status": "ready_for_internal_manuscript_review",
            "main_claim_locked": True,
            "literature_gap_rows": 3,
            "trainable_model_opened_now": False,
        }
    }
    phase75_manifest = {
        "gate_status": {
            "status": "blocked_by_local_ambench_gate",
            "phase76_seed7_allowed": False,
            "reason": "synthetic-positive but local AM-Bench-negative",
        }
    }
    phase79_manifest = {
        "gate": {
            "status": "local_surrogate_required_before_a100",
            "reason": "density debt blocks direct A100",
        }
    }
    phase80_manifest = {
        "gate": {
            "status": "blocked_by_local_surrogate_gate",
            "a100_seed7_allowed": False,
            "reason": "validation gain over identity is below the pre-declared minimum",
        }
    }
    phase81_manifest = {
        "gate": {
            "status": "blocked_no_registered_target",
            "a100_training_allowed_now": False,
            "a100_80gb_request_now": False,
            "reason": "No current route is coordinate-compatible and split-ready.",
            "next_action": "pursue pad camera-to-galvo registration or an external registered-target data card",
        }
    }
    return {
        "phase60_manifest": _write_json(tmp_path / "phase60_manifest.json", phase60_manifest),
        "phase60_main": tmp_path / "unused_main.csv",
        "phase60_route": tmp_path / "unused_route.csv",
        "phase60_stress": tmp_path / "unused_stress.csv",
        "phase60_appendix": _write_csv(
            tmp_path / "phase60_appendix.csv",
            [
                {
                    "phase": "33",
                    "branch": "Fourier spacetime features",
                    "target": "broad12",
                    "result": "negative",
                    "paper_use": "appendix diagnostic",
                    "evidence": "docs/results/phase33.md",
                }
            ],
        ),
        "phase61_manifest": tmp_path / "unused_phase61_manifest.json",
        "phase61_literature_gaps": _write_csv(
            tmp_path / "phase61_literature_gaps.csv",
            [
                {
                    "gap_id": "LIT_GAP-1",
                    "location": "Introduction",
                    "claim_needing_support": "AM-Bench context",
                    "evidence_type_needed": "verified citation",
                    "suggested_search_or_material": "NIST AM-Bench",
                    "blocks_current_phase61_draft": "no",
                }
            ],
        ),
        "phase74_manifest": _write_json(tmp_path / "phase74_manifest.json", phase74_manifest),
        "phase74_claim_audit": tmp_path / "unused_claim_audit.csv",
        "phase74_boundary": tmp_path / "unused_boundary.csv",
        "phase75_manifest": _write_json(tmp_path / "phase75_manifest.json", phase75_manifest),
        "phase79_manifest": _write_json(tmp_path / "phase79_manifest.json", phase79_manifest),
        "phase80_manifest": _write_json(tmp_path / "phase80_manifest.json", phase80_manifest),
        "phase81_manifest": _write_json(tmp_path / "phase81_manifest.json", phase81_manifest),
        "phase81_table": tmp_path / "unused_phase81_table.csv",
    }


def test_phase88_locks_fallback_claim_but_not_submission_ready(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(tmp_path, tmp_path / "out", paths=_paths(tmp_path))

    gate = manifest["gate"]
    assert manifest["phase"] == 88
    assert gate["status"] == "fallback_experimental_claim_complete"
    assert gate["experimental_claim_complete"] is True
    assert gate["submission_ready"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert gate["phase75_status"] == "blocked_by_local_ambench_gate"
    assert gate["phase80_status"] == "blocked_by_local_surrogate_gate"
    assert gate["phase81_status"] == "blocked_no_registered_target"
    assert gate["open_submission_blockers"] == 2

    claim_path = tmp_path / manifest["outputs"]["claim_lock_table"]
    with claim_path.open(encoding="utf-8", newline="") as handle:
        claim_rows = list(csv.DictReader(handle))
    assert any(row["lock_id"] == "P88-MAIN-LOCK" for row in claim_rows)
    assert any(row["lock_id"] == "P88-REGISTERED-TARGET" for row in claim_rows)


def test_phase88_appends_recent_negative_gates_and_remaining_work(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(tmp_path, tmp_path / "out", paths=_paths(tmp_path))

    appendix_path = tmp_path / manifest["outputs"]["appendix_diagnostic_table"]
    with appendix_path.open(encoding="utf-8", newline="") as handle:
        appendix_rows = list(csv.DictReader(handle))
    assert {row["appendix_id"] for row in appendix_rows} >= {
        "P88-APPX-075",
        "P88-APPX-079",
        "P88-APPX-080",
        "P88-APPX-081",
    }

    remaining_path = tmp_path / manifest["outputs"]["remaining_work_table"]
    with remaining_path.open(encoding="utf-8", newline="") as handle:
        remaining_rows = list(csv.DictReader(handle))
    by_id = {row["work_id"]: row for row in remaining_rows}
    assert by_id["P88-WORK-LIT"]["blocks_submission"] == "true"
    assert by_id["P88-WORK-A100"]["status"] == "blocked_no_training_gate"
    assert by_id["P88-WORK-A100"]["blocks_experimental_claim"] == "false"
