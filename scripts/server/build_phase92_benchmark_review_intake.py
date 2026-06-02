#!/usr/bin/env python3
"""Build the Phase 92 benchmark-review intake/readiness package.

Phase 92 does not invent venue requirements or benchmark-paper comparisons.
It checks whether the Phase 90/91 manuscript package has enough external
target input for an internal benchmark review. If no target venue, author
guide, or accepted benchmark-paper set is available, the package must close as
an actionable blocker rather than a fake review.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


READINESS_FIELDS = (
    "readiness_id",
    "review_area",
    "current_state",
    "required_input",
    "evidence_anchor",
    "status",
    "blocks_submission",
    "next_action",
)

MANUAL_QUEUE_FIELDS = (
    "queue_id",
    "priority",
    "needed_input",
    "minimum_acceptance",
    "reason",
    "blocks_submission",
    "blocks_model_training",
    "suggested_user_action",
)

SCOPE_FIELDS = (
    "scope_id",
    "manuscript_component",
    "source_artifact",
    "review_question",
    "current_status",
    "venue_dependency",
)

BENCHMARK_FIELDS = (
    "benchmark_id",
    "title_or_venue",
    "source_type",
    "provided_reference",
    "review_use",
    "status",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Expected at least one row in {path}")
    return rows


def _read_optional_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    return _read_csv(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.9f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _display_path(path: Path, root: Path | None = None) -> str:
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def _default_paths(root: Path) -> dict[str, Path]:
    phase89 = root / "docs/results/phase89_literature_venue_gap_resolution"
    phase90 = root / "docs/results/phase90_manuscript_v1_claim_integration"
    phase91 = root / "docs/results/phase91_table_figure_appendix_freeze"
    return {
        "phase89_gate": phase89 / "phase89_literature_venue_gap_resolution_gate.json",
        "phase89_evidence_register": phase89 / "phase89_evidence_register.csv",
        "phase89_manual_queue": phase89 / "phase89_manual_verification_queue.csv",
        "phase90_gate": phase90 / "phase90_manuscript_v1_claim_integration_gate.json",
        "phase90_claim_audit": phase90 / "phase90_claim_evidence_audit.csv",
        "phase90_venue_blocker_queue": phase90 / "phase90_venue_blocker_queue.csv",
        "phase91_gate": phase91 / "phase91_table_figure_appendix_freeze_gate.json",
        "phase91_manifest": phase91 / "phase91_table_figure_appendix_freeze_manifest.json",
        "phase91_main_table": phase91 / "phase91_main_table_freeze.csv",
        "phase91_route_table": phase91 / "phase91_route_guard_table_freeze.csv",
        "phase91_stress_table": phase91 / "phase91_stress_boundary_table_freeze.csv",
        "phase91_figure_manifest": phase91 / "phase91_table_figure_caption_manifest.csv",
        "benchmark_intake": root / "docs/benchmark_review/phase92_target_benchmark_intake.csv",
    }


def _truthy(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return value.strip().lower() in {"true", "1", "yes", "y"}


def normalize_benchmark_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        title = row.get("title_or_venue") or row.get("title") or row.get("venue") or ""
        reference = row.get("provided_reference") or row.get("doi") or row.get("url") or ""
        source_type = row.get("source_type") or ("target_venue" if row.get("venue") else "benchmark_paper")
        status = "usable_for_review" if title and reference else "incomplete_reference"
        normalized.append(
            {
                "benchmark_id": row.get("benchmark_id") or f"P92-BENCH-{index:03d}",
                "title_or_venue": title,
                "source_type": source_type,
                "provided_reference": reference,
                "review_use": row.get("review_use") or "target-near contribution and formatting comparison",
                "status": status,
            }
        )
    return normalized


def build_scope_rows(
    *,
    main_rows: list[dict[str, str]],
    route_rows: list[dict[str, str]],
    stress_rows: list[dict[str, str]],
    figure_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    main_source = "docs/results/phase91_table_figure_appendix_freeze/phase91_main_table_freeze.csv"
    route_source = "docs/results/phase91_table_figure_appendix_freeze/phase91_route_guard_table_freeze.csv"
    stress_source = "docs/results/phase91_table_figure_appendix_freeze/phase91_stress_boundary_table_freeze.csv"
    figure_source = "docs/results/phase91_table_figure_appendix_freeze/phase91_table_figure_caption_manifest.csv"
    return [
        {
            "scope_id": "P92-SCOPE-001",
            "manuscript_component": "main performance table",
            "source_artifact": main_source,
            "review_question": "Do target-near papers expect the fixed-sampling spot_size claim to be framed as generalization, process conditioning, or benchmark evidence?",
            "current_status": f"ready_for_review_input; rows={len(main_rows)}",
            "venue_dependency": "requires target venue or benchmark papers to judge contribution strength",
        },
        {
            "scope_id": "P92-SCOPE-002",
            "manuscript_component": "route-guard boundary table",
            "source_artifact": route_source,
            "review_question": "Do target-near papers make similar boundary claims, and should route-guard-only axes move to appendix or remain in main text?",
            "current_status": f"ready_for_review_input; rows={len(route_rows)}",
            "venue_dependency": "requires benchmark-paper comparison for acceptable limitation prominence",
        },
        {
            "scope_id": "P92-SCOPE-003",
            "manuscript_component": "stress and boundary table",
            "source_artifact": stress_source,
            "review_question": "Is the stronger-baseline and density-boundary evidence sufficient for the target venue's robustness expectations?",
            "current_status": f"ready_for_review_input; rows={len(stress_rows)}",
            "venue_dependency": "requires target venue or accepted-paper robustness norms",
        },
        {
            "scope_id": "P92-SCOPE-004",
            "manuscript_component": "figure and caption package",
            "source_artifact": figure_source,
            "review_question": "Do captions, figure density, and supplement numbering match target-near paper conventions?",
            "current_status": f"ready_for_review_input; rows={len(figure_rows)}",
            "venue_dependency": "requires author guide or accepted-paper style examples",
        },
        {
            "scope_id": "P92-SCOPE-005",
            "manuscript_component": "appendix diagnostics and future-work gates",
            "source_artifact": "docs/results/phase91_table_figure_appendix_freeze/phase91_appendix_diagnostic_freeze.csv",
            "review_question": "Do negative diagnostics and future model gates provide enough transparency without weakening the main contribution framing?",
            "current_status": "ready_for_review_input",
            "venue_dependency": "requires target-near limitation and supplement norms",
        },
    ]


def build_readiness_rows(
    *,
    phase89_gate: dict[str, Any],
    phase90_gate: dict[str, Any],
    phase91_gate: dict[str, Any],
    benchmark_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    has_target_input = any(row["status"] == "usable_for_review" for row in benchmark_rows)
    enough_benchmark_papers = (
        sum(1 for row in benchmark_rows if row["status"] == "usable_for_review") >= 3
    )
    return [
        {
            "readiness_id": "P92-READY-001",
            "review_area": "experimental_evidence_package",
            "current_state": phase91_gate.get("status"),
            "required_input": "Phase 91 table/figure/appendix freeze",
            "evidence_anchor": "phase91_table_figure_appendix_freeze_gate.json",
            "status": "ready",
            "blocks_submission": False,
            "next_action": "use Phase 91 frozen artifacts as benchmark-review scope",
        },
        {
            "readiness_id": "P92-READY-002",
            "review_area": "core_literature_support",
            "current_state": phase89_gate.get("status"),
            "required_input": "verified AM-Bench, PINN, physics-informed ML, and FiLM evidence",
            "evidence_anchor": "phase89_evidence_register.csv",
            "status": "ready" if phase89_gate.get("core_literature_ready") else "blocked",
            "blocks_submission": not bool(phase89_gate.get("core_literature_ready")),
            "next_action": "use verified evidence register; do not add unverified citations",
        },
        {
            "readiness_id": "P92-READY-003",
            "review_area": "claim_evidence_traceability",
            "current_state": phase90_gate.get("status"),
            "required_input": "Phase 90 claim/evidence audit",
            "evidence_anchor": "phase90_claim_evidence_audit.csv",
            "status": "ready" if phase90_gate.get("core_claims_integrated") else "blocked",
            "blocks_submission": not bool(phase90_gate.get("core_claims_integrated")),
            "next_action": "audit every benchmark-review comment against claim anchors",
        },
        {
            "readiness_id": "P92-READY-004",
            "review_area": "target_venue_or_benchmark_papers",
            "current_state": f"usable_inputs={sum(1 for row in benchmark_rows if row['status'] == 'usable_for_review')}",
            "required_input": "target venue, author guide, or 3-10 accepted target-near benchmark papers",
            "evidence_anchor": "phase92_benchmark_input_table.csv",
            "status": "ready" if has_target_input and enough_benchmark_papers else "blocked_missing_target_benchmarks",
            "blocks_submission": not (has_target_input and enough_benchmark_papers),
            "next_action": "provide target venue/author guide or at least 3 benchmark papers before venue-specific review",
        },
        {
            "readiness_id": "P92-READY-005",
            "review_area": "model_training_governance",
            "current_state": "manuscript review only",
            "required_input": "local/no-training gate before any future model branch",
            "evidence_anchor": "task_plan.md Phase 92+ Long-Term Stepwise Execution and Effect-Validation Plan",
            "status": "ready_no_training",
            "blocks_submission": False,
            "next_action": "do not start speculative A100 training from Phase 92",
        },
    ]


def build_manual_queue(
    *,
    phase89_manual: list[dict[str, str]],
    phase90_blockers: list[dict[str, str]],
    benchmark_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    usable_count = sum(1 for row in benchmark_rows if row["status"] == "usable_for_review")
    queue: list[dict[str, Any]] = [
        {
            "queue_id": "P92-MANUAL-001",
            "priority": "P0",
            "needed_input": "target venue or author guide",
            "minimum_acceptance": "one named journal/conference with author instructions URL or local PDF",
            "reason": "final section order, figure/table formatting, citation density, and limitation placement are venue-dependent",
            "blocks_submission": True,
            "blocks_model_training": False,
            "suggested_user_action": "provide target venue and author guide before Phase 93 formatting",
        },
        {
            "queue_id": "P92-MANUAL-002",
            "priority": "P0",
            "needed_input": "accepted target-near benchmark papers",
            "minimum_acceptance": "at least 3 usable papers; 5 preferred; 10 maximum for this review package",
            "reason": "internal benchmark review needs target-near baselines, contribution framing, caption density, and limitation norms",
            "blocks_submission": usable_count < 3,
            "blocks_model_training": False,
            "suggested_user_action": "provide 3-10 accepted benchmark papers or their DOI/URL/title list",
        },
        {
            "queue_id": "P92-MANUAL-003",
            "priority": "P1",
            "needed_input": "retarget decision if benchmark review finds contribution too narrow",
            "minimum_acceptance": "choose retarget venue or open a separate gated Track B model branch",
            "reason": "Phase 92 should not reopen speculative model training without a design/local gate",
            "blocks_submission": False,
            "blocks_model_training": True,
            "suggested_user_action": "decide retarget-vs-new-model only after benchmark input exists",
        },
    ]
    if phase89_manual:
        queue.append(
            {
                "queue_id": "P92-MANUAL-004",
                "priority": "P0",
                "needed_input": phase89_manual[0].get("needed_input", "target venue alignment"),
                "minimum_acceptance": "resolve Phase 89 manual verification queue",
                "reason": phase89_manual[0].get("reason"),
                "blocks_submission": _truthy(phase89_manual[0].get("blocks_submission")),
                "blocks_model_training": False,
                "suggested_user_action": phase89_manual[0].get("suggested_user_action"),
            }
        )
    if phase90_blockers:
        queue.append(
            {
                "queue_id": "P92-MANUAL-005",
                "priority": "P0",
                "needed_input": phase90_blockers[0].get("required_input"),
                "minimum_acceptance": "resolve Phase 90 venue blocker queue",
                "reason": phase90_blockers[0].get("category"),
                "blocks_submission": _truthy(phase90_blockers[0].get("blocks_submission")),
                "blocks_model_training": False,
                "suggested_user_action": phase90_blockers[0].get("next_action"),
            }
        )
    return queue


def build_gate(
    *,
    readiness_rows: list[dict[str, Any]],
    manual_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
    scope_rows: list[dict[str, Any]],
    phase89_gate: dict[str, Any],
    phase90_gate: dict[str, Any],
    phase91_gate: dict[str, Any],
) -> dict[str, Any]:
    blocking_readiness = [
        row for row in readiness_rows if row["status"].startswith("blocked") or row["blocks_submission"]
    ]
    usable_benchmarks = [row for row in benchmark_rows if row["status"] == "usable_for_review"]
    core_ready = all(
        [
            phase89_gate.get("core_literature_ready"),
            phase90_gate.get("core_claims_integrated"),
            phase91_gate.get("table_figure_appendix_frozen"),
        ]
    )
    benchmark_review_ready = core_ready and len(usable_benchmarks) >= 3 and not blocking_readiness
    if benchmark_review_ready:
        status = "benchmark_review_ready"
        next_action = "perform benchmark-paper comparison and update manuscript review ledger"
    else:
        status = "blocked_missing_target_benchmarks"
        next_action = "request target venue, author guide, or 3-10 accepted benchmark papers before Phase 93"
    return {
        "status": status,
        "core_package_ready": bool(core_ready),
        "benchmark_review_ready": benchmark_review_ready,
        "target_benchmark_inputs": len(benchmark_rows),
        "usable_benchmark_inputs": len(usable_benchmarks),
        "minimum_required_benchmark_inputs": 3,
        "readiness_rows": len(readiness_rows),
        "blocking_readiness_rows": len(blocking_readiness),
        "manual_queue_rows": len(manual_rows),
        "review_scope_rows": len(scope_rows),
        "venue_alignment_ready": benchmark_review_ready,
        "submission_ready": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "phase89_status": phase89_gate.get("status"),
        "phase90_status": phase90_gate.get("status"),
        "phase91_status": phase91_gate.get("status"),
        "next_action": next_action,
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_csv_value(row.get(key)).replace("\n", " ") for key, _ in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def build_markdown(
    gate: dict[str, Any],
    readiness_rows: list[dict[str, Any]],
    manual_rows: list[dict[str, Any]],
    scope_rows: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "# Phase 92 Benchmark Review Intake",
            "",
            "## Gate Decision",
            "",
            f"Status: `{gate['status']}`.",
            f"Core package ready: `{str(gate['core_package_ready']).lower()}`.",
            f"Benchmark review ready: `{str(gate['benchmark_review_ready']).lower()}`.",
            f"Submission ready: `{str(gate['submission_ready']).lower()}`.",
            f"A100 training allowed now: `{str(gate['a100_training_allowed_now']).lower()}`.",
            "",
            "Phase 92 is an intake/readiness gate. It does not infer journal rules or accepted-paper norms without external target input.",
            "",
            "## Readiness Table",
            "",
            _markdown_table(
                readiness_rows,
                [
                    ("readiness_id", "Row"),
                    ("review_area", "Area"),
                    ("status", "Status"),
                    ("blocks_submission", "Blocks submission"),
                    ("next_action", "Next action"),
                ],
            ),
            "",
            "## Manual Queue",
            "",
            _markdown_table(
                manual_rows,
                [
                    ("queue_id", "Queue"),
                    ("priority", "Priority"),
                    ("needed_input", "Needed input"),
                    ("minimum_acceptance", "Minimum acceptance"),
                    ("blocks_submission", "Blocks submission"),
                ],
            ),
            "",
            "## Review Scope",
            "",
            _markdown_table(
                scope_rows,
                [
                    ("scope_id", "Scope"),
                    ("manuscript_component", "Component"),
                    ("current_status", "Status"),
                    ("venue_dependency", "Venue dependency"),
                ],
            ),
            "",
            "## Next Action",
            "",
            gate["next_action"],
            "",
        ]
    )


def build_package(
    root: Path,
    output_dir: Path,
    paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    resolved = _default_paths(root)
    if paths:
        resolved.update(paths)

    phase89_gate = _read_json(resolved["phase89_gate"])
    phase89_evidence = _read_csv(resolved["phase89_evidence_register"])
    phase89_manual = _read_csv(resolved["phase89_manual_queue"])
    phase90_gate = _read_json(resolved["phase90_gate"])
    phase90_audit = _read_csv(resolved["phase90_claim_audit"])
    phase90_blockers = _read_csv(resolved["phase90_venue_blocker_queue"])
    phase91_gate = _read_json(resolved["phase91_gate"])
    phase91_manifest = _read_json(resolved["phase91_manifest"])
    main_rows = _read_csv(resolved["phase91_main_table"])
    route_rows = _read_csv(resolved["phase91_route_table"])
    stress_rows = _read_csv(resolved["phase91_stress_table"])
    figure_rows = _read_csv(resolved["phase91_figure_manifest"])
    benchmark_rows = normalize_benchmark_rows(_read_optional_csv(resolved.get("benchmark_intake")))

    scope_rows = build_scope_rows(
        main_rows=main_rows,
        route_rows=route_rows,
        stress_rows=stress_rows,
        figure_rows=figure_rows,
    )
    readiness_rows = build_readiness_rows(
        phase89_gate=phase89_gate,
        phase90_gate=phase90_gate,
        phase91_gate=phase91_gate,
        benchmark_rows=benchmark_rows,
    )
    manual_rows = build_manual_queue(
        phase89_manual=phase89_manual,
        phase90_blockers=phase90_blockers,
        benchmark_rows=benchmark_rows,
    )
    gate = build_gate(
        readiness_rows=readiness_rows,
        manual_rows=manual_rows,
        benchmark_rows=benchmark_rows,
        scope_rows=scope_rows,
        phase89_gate=phase89_gate,
        phase90_gate=phase90_gate,
        phase91_gate=phase91_gate,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    readiness_path = output_dir / "phase92_benchmark_review_readiness_table.csv"
    manual_path = output_dir / "phase92_manual_benchmark_queue.csv"
    scope_path = output_dir / "phase92_claim_review_scope.csv"
    benchmark_path = output_dir / "phase92_benchmark_input_table.csv"
    gate_path = output_dir / "phase92_benchmark_review_intake_gate.json"
    markdown_path = output_dir / "phase92_benchmark_review_intake.md"
    manifest_path = output_dir / "phase92_benchmark_review_intake_manifest.json"

    _write_csv(readiness_path, readiness_rows, READINESS_FIELDS)
    _write_csv(manual_path, manual_rows, MANUAL_QUEUE_FIELDS)
    _write_csv(scope_path, scope_rows, SCOPE_FIELDS)
    _write_csv(benchmark_path, benchmark_rows, BENCHMARK_FIELDS)
    _write_json(gate_path, gate)
    markdown_path.write_text(
        build_markdown(gate, readiness_rows, manual_rows, scope_rows),
        encoding="utf-8",
    )

    manifest = {
        "phase": 92,
        "objective": "benchmark_review_intake_readiness",
        "inputs": {key: _display_path(path, root) for key, path in sorted(resolved.items())},
        "outputs": {
            "readiness_table": _display_path(readiness_path, root),
            "manual_benchmark_queue": _display_path(manual_path, root),
            "claim_review_scope": _display_path(scope_path, root),
            "benchmark_input_table": _display_path(benchmark_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "verified_evidence_rows": len(phase89_evidence),
            "phase90_audit_rows": len(phase90_audit),
            "phase91_main_rows": len(main_rows),
            "phase91_route_rows": len(route_rows),
            "phase91_stress_rows": len(stress_rows),
            "figure_manifest_rows": len(figure_rows),
            "benchmark_input_rows": len(benchmark_rows),
            "readiness_rows": len(readiness_rows),
            "manual_queue_rows": len(manual_rows),
            "scope_rows": len(scope_rows),
        },
        "gate": gate,
        "phase89_gate": {
            "status": phase89_gate.get("status"),
            "core_literature_ready": phase89_gate.get("core_literature_ready"),
            "venue_alignment_ready": phase89_gate.get("venue_alignment_ready"),
        },
        "phase90_gate": {
            "status": phase90_gate.get("status"),
            "core_claims_integrated": phase90_gate.get("core_claims_integrated"),
            "submission_blockers": phase90_gate.get("submission_blockers"),
        },
        "phase91_gate": {
            "status": phase91_gate.get("status"),
            "table_figure_appendix_frozen": phase91_gate.get("table_figure_appendix_frozen"),
            "figure_assets_exist": phase91_gate.get("figure_assets_exist"),
        },
        "phase91_claim_boundary": phase91_manifest.get("phase60_claim_boundary"),
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase92_benchmark_review_intake"),
    )
    parser.add_argument(
        "--benchmark-intake",
        type=Path,
        default=None,
        help="Optional CSV with target venue or benchmark-paper references.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    paths = None
    if args.benchmark_intake is not None:
        benchmark_intake = args.benchmark_intake
        if not benchmark_intake.is_absolute():
            benchmark_intake = root / benchmark_intake
        paths = {"benchmark_intake": benchmark_intake}
    manifest = build_package(root=root, output_dir=output_dir, paths=paths)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
