from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase92_benchmark_review_intake.py")
    spec = importlib.util.spec_from_file_location("phase92_intake", script)
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


def _phase89_evidence_rows() -> list[dict[str, object]]:
    return [
        {
            "evidence_id": "P89-EVD-AMBench-001",
            "gap_id": "LIT_GAP-61-001",
            "claim_area": "AM-Bench public benchmark context",
            "source_title": "AM-Bench",
            "authors_or_owner": "NIST",
            "venue_or_source": "NIST",
            "year": "2026 access",
            "doi": "",
            "stable_url": "https://www.nist.gov/ambench",
            "source_type": "official_project_page",
            "trust_state": "verified_official_url",
            "verification_trail": "verified",
            "supports_claim": "benchmark context",
            "allowed_claim_strength": "official_benchmark_context",
            "limitations": "context only",
            "writing_ready": "true",
        }
        for _ in range(8)
    ]


def _phase90_audit_rows() -> list[dict[str, object]]:
    return [
        {
            "claim_id": "C61-MAIN-001",
            "claim_type": "result",
            "source_anchor": "phase60_main_table",
            "support_locator": "phase60_main.csv",
            "verification_state": "writing_ready",
            "integrated_in_v1": "true",
            "allowed_strength": "moderate",
            "wording_guard": "limited to fixed-sampling spot_size",
            "blocker": "",
        }
    ]


def _phase91_main_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in ["broad12", "broad21"]:
        for metric in ["Test RMSE", "Hot q90 RMSE", "Gradient q90 RMSE"]:
            rows.append(
                {
                    "freeze_id": f"{dataset}-{metric}",
                    "dataset": dataset,
                    "split": "spot_size",
                    "route": "film/global_standard",
                    "metric": metric,
                    "broad_process_v1_mean": "1.0",
                    "broad_process_v1_std": "0.1",
                    "no_process_mean": "2.0",
                    "best_strong_baseline": "1.5",
                    "delta_vs_best_strong": "-0.5",
                    "n_seeds": "3",
                    "manuscript_role": "main_table_only_positive_performance_claim",
                    "claim_anchor": "C61-MAIN-001",
                    "freeze_status": "frozen_main_claim_row",
                }
            )
    return rows


def _phase91_route_rows() -> list[dict[str, object]]:
    return [
        {
            "freeze_id": f"P91-ROUTE-{index:03d}",
            "dataset": "broad12" if index <= 4 else "broad21",
            "split": ["laser_power", "line", "process", "scan_speed"][(index - 1) % 4],
            "classification": "route_guard_positive",
            "route": "none/none",
            "claim_use": "route-guard-only boundary evidence",
            "metrics_summary": "summary",
            "manuscript_role": "route_guard_boundary",
            "wording_guard": "guard",
            "freeze_status": "frozen_route_guard_row",
        }
        for index in range(1, 9)
    ]


def _phase91_stress_rows() -> list[dict[str, object]]:
    return [
        {
            "freeze_id": "P91-STRESS-001",
            "scenario": "stronger_baseline_stress",
            "dataset": "broad12",
            "split": "spot_size",
            "metric": "Test RMSE",
            "status": "pass",
            "candidate": "1.0",
            "comparator": "mean",
            "delta_vs_comparator": "-0.5",
            "manuscript_use": "support",
            "evidence": "phase58",
            "freeze_status": "frozen_support_row",
        },
        {
            "freeze_id": "P91-STRESS-002",
            "scenario": "alternate_density_stress",
            "dataset": "broad21",
            "split": "spot_size",
            "metric": "Test RMSE",
            "status": "boundary",
            "candidate": "2.0",
            "comparator": "mean",
            "delta_vs_comparator": "0.5",
            "manuscript_use": "boundary",
            "evidence": "phase58",
            "freeze_status": "frozen_boundary_row",
        },
    ]


def _phase91_figure_rows() -> list[dict[str, object]]:
    return [
        {
            "item_id": f"P91-ITEM-{index:03d}",
            "item_type": "figure" if index == 6 else "table",
            "manuscript_label": f"Item {index}",
            "source_artifact": "artifact",
            "caption_source": "caption",
            "claim_anchor": "C61-MAIN-001",
            "manuscript_role": "role",
            "freeze_status": "frozen",
            "venue_dependency": "venue dependent",
        }
        for index in range(1, 7)
    ]


def _benchmark_rows() -> list[dict[str, object]]:
    return [
        {
            "benchmark_id": f"BENCH-{index:03d}",
            "title_or_venue": f"Accepted benchmark paper {index}",
            "source_type": "benchmark_paper",
            "provided_reference": f"10.0000/example.{index}",
            "review_use": "target-near contribution framing",
        }
        for index in range(1, 4)
    ]


def _paths(tmp_path: Path, *, with_benchmarks: bool = False) -> dict[str, Path]:
    phase89_gate = {
        "status": "literature_core_resolved_venue_unresolved",
        "core_literature_ready": True,
        "venue_alignment_ready": False,
    }
    phase90_gate = {
        "status": "manuscript_v1_core_claims_integrated_venue_unresolved",
        "core_claims_integrated": True,
        "submission_blockers": 1,
    }
    phase91_gate = {
        "status": "table_figure_appendix_frozen_venue_unresolved",
        "table_figure_appendix_frozen": True,
        "figure_assets_exist": True,
    }
    phase91_manifest = {
        "phase60_claim_boundary": {
            "main_claim": "fixed-sampling broad12/broad21 spot_size",
            "excluded_claims": ["density-invariant robustness"],
        }
    }
    benchmark_path = tmp_path / "missing_benchmark_input.csv"
    if with_benchmarks:
        benchmark_path = _write_csv(tmp_path / "benchmark_input.csv", _benchmark_rows())
    return {
        "phase89_gate": _write_json(tmp_path / "phase89_gate.json", phase89_gate),
        "phase89_evidence_register": _write_csv(
            tmp_path / "phase89_evidence.csv", _phase89_evidence_rows()
        ),
        "phase89_manual_queue": _write_csv(
            tmp_path / "phase89_manual.csv",
            [
                {
                    "queue_id": "P89-MANUAL-001",
                    "category": "target_venue_alignment",
                    "needed_input": "No target venue, author guide, or accepted-paper benchmark set has been provided.",
                    "reason": "Final style depends on venue.",
                    "blocks_submission": "true",
                    "suggested_user_action": "provide target venue or benchmark papers",
                }
            ],
        ),
        "phase90_gate": _write_json(tmp_path / "phase90_gate.json", phase90_gate),
        "phase90_claim_audit": _write_csv(tmp_path / "phase90_audit.csv", _phase90_audit_rows()),
        "phase90_venue_blocker_queue": _write_csv(
            tmp_path / "phase90_blockers.csv",
            [
                {
                    "blocker_id": "P90-BLOCKER-001",
                    "category": "target_venue_alignment",
                    "status": "unresolved_user_input_required",
                    "required_input": "No target venue or benchmark papers.",
                    "blocks_submission": "true",
                    "blocks_phase90_core_claims": "false",
                    "next_action": "provide target venue or benchmark papers",
                }
            ],
        ),
        "phase91_gate": _write_json(tmp_path / "phase91_gate.json", phase91_gate),
        "phase91_manifest": _write_json(tmp_path / "phase91_manifest.json", phase91_manifest),
        "phase91_main_table": _write_csv(tmp_path / "phase91_main.csv", _phase91_main_rows()),
        "phase91_route_table": _write_csv(tmp_path / "phase91_route.csv", _phase91_route_rows()),
        "phase91_stress_table": _write_csv(tmp_path / "phase91_stress.csv", _phase91_stress_rows()),
        "phase91_figure_manifest": _write_csv(
            tmp_path / "phase91_figures.csv", _phase91_figure_rows()
        ),
        "benchmark_intake": benchmark_path,
    }


def test_phase92_blocks_when_target_benchmarks_are_missing(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(tmp_path, tmp_path / "out", paths=_paths(tmp_path))

    gate = manifest["gate"]
    assert manifest["phase"] == 92
    assert gate["status"] == "blocked_missing_target_benchmarks"
    assert gate["core_package_ready"] is True
    assert gate["benchmark_review_ready"] is False
    assert gate["target_benchmark_inputs"] == 0
    assert gate["usable_benchmark_inputs"] == 0
    assert gate["blocking_readiness_rows"] == 1
    assert gate["submission_ready"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False

    with (tmp_path / manifest["outputs"]["manual_benchmark_queue"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        manual_rows = list(csv.DictReader(handle))
    assert any(row["queue_id"] == "P92-MANUAL-001" for row in manual_rows)
    assert any(row["queue_id"] == "P92-MANUAL-002" for row in manual_rows)
    assert {row["blocks_model_training"] for row in manual_rows} >= {"false", "true"}


def test_phase92_can_become_review_ready_with_three_benchmark_inputs(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        tmp_path, tmp_path / "out", paths=_paths(tmp_path, with_benchmarks=True)
    )

    gate = manifest["gate"]
    assert gate["status"] == "benchmark_review_ready"
    assert gate["benchmark_review_ready"] is True
    assert gate["target_benchmark_inputs"] == 3
    assert gate["usable_benchmark_inputs"] == 3
    assert gate["blocking_readiness_rows"] == 0
    assert gate["submission_ready"] is False
    assert gate["a100_training_allowed_now"] is False

    with (tmp_path / manifest["outputs"]["benchmark_input_table"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        benchmark_rows = list(csv.DictReader(handle))
    assert {row["status"] for row in benchmark_rows} == {"usable_for_review"}

    with (tmp_path / manifest["outputs"]["claim_review_scope"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        scope_rows = list(csv.DictReader(handle))
    assert len(scope_rows) == 5
    assert any(row["manuscript_component"] == "route-guard boundary table" for row in scope_rows)
