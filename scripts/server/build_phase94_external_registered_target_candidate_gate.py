#!/usr/bin/env python3
"""Build the Phase 94 external registered-target candidate gate.

Phase 92 blocks submission-facing benchmark review until target venue or
benchmark papers are supplied. Phase 94 keeps model progress moving without
violating that blocker: it triages possible new data/model-entry routes and
decides whether any may enter a local/no-training design gate. It does not
open A100 training.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CANDIDATE_FIELDS = (
    "candidate_id",
    "candidate_family",
    "source_name",
    "source_url",
    "source_type",
    "public_reproducibility",
    "registration_status",
    "target_relevance",
    "model_innovation_fit",
    "allowed_next_gate",
    "a100_training_allowed",
    "a100_80gb_request_now",
    "status",
    "priority",
    "stop_condition",
    "next_action",
    "evidence",
)

DESIGN_QUEUE_FIELDS = (
    "queue_id",
    "priority",
    "candidate_id",
    "design_task",
    "minimum_artifact",
    "pass_condition",
    "stop_condition",
    "allowed_compute",
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
    phase81 = root / "docs/results/phase81_registered_target_intake_gate"
    phase92 = root / "docs/results/phase92_benchmark_review_intake"
    return {
        "phase81_gate": phase81 / "phase81_registered_target_gate.json",
        "phase81_table": phase81 / "phase81_registered_target_intake_table.csv",
        "phase92_gate": phase92 / "phase92_benchmark_review_intake_gate.json",
        "phase92_manifest": phase92 / "phase92_benchmark_review_intake_manifest.json",
    }


def _phase81_has_open_registered_target(phase81_gate: dict[str, Any]) -> bool:
    return int(phase81_gate.get("open_registered_target_count") or 0) > 0


def build_candidate_rows(
    *,
    phase81_gate: dict[str, Any],
    phase81_rows: list[dict[str, str]],
    phase92_gate: dict[str, Any],
) -> list[dict[str, Any]]:
    has_registered_target = _phase81_has_open_registered_target(phase81_gate)
    phase92_blocked = phase92_gate.get("status") == "blocked_missing_target_benchmarks"
    preferred_phase81 = phase81_gate.get("preferred_next_route") or "ambench_mds2_2716_pad_thermography_xypt"
    pad_row = next(
        (row for row in phase81_rows if row.get("route_id") == preferred_phase81),
        phase81_rows[0],
    )
    return [
        {
            "candidate_id": "P94-CAND-AMBNCH-PAD-REG",
            "candidate_family": "current_ambench_registration_followup",
            "source_name": "NIST AM-Bench / mds2-2716 pad thermography plus XYPT",
            "source_url": "https://data.nist.gov/od/id/mds2-2716",
            "source_type": "official_dataset_record",
            "public_reproducibility": "strong_existing_manifest",
            "registration_status": pad_row.get("coordinate_registration_status"),
            "target_relevance": "highest for current manuscript and heat-kernel/source-path features",
            "model_innovation_fit": "registered heat-kernel, Green's-function, source-path features",
            "allowed_next_gate": "data_registration_evidence_update_only",
            "a100_training_allowed": False,
            "a100_80gb_request_now": False,
            "status": "blocked_until_pad_registration_evidence",
            "priority": 1,
            "stop_condition": "no documented pad camera-to-galvo mapping or equivalent registration",
            "next_action": "search or obtain pad camera-to-galvo registration metadata before feature/model work",
            "evidence": "Phase 81 preferred route remains blocked by missing paper-facing registration.",
        },
        {
            "candidate_id": "P94-CAND-PFHUB-PINN",
            "candidate_family": "external_physics_benchmark",
            "source_name": "PFHub phase-field benchmark problems",
            "source_url": "https://pages.nist.gov/pfhub/benchmarks/",
            "source_type": "official_benchmark_collection",
            "public_reproducibility": "public_benchmark_specs",
            "registration_status": "synthetic_or_benchmark_registered_by_problem_definition",
            "target_relevance": "supports mechanism testing, not a direct AM-Bench process-transfer replacement",
            "model_innovation_fit": "Bayesian/adaptive PINN, Green's-function features, meta-learning quick adaptation precheck",
            "allowed_next_gate": "phase95_local_synthetic_benchmark_design",
            "a100_training_allowed": False,
            "a100_80gb_request_now": False,
            "status": "open_for_local_design_gate",
            "priority": 2 if not has_registered_target else 4,
            "stop_condition": "cannot preserve global/hot/gradient metrics on AM-Bench after synthetic success",
            "next_action": "build a local/no-training Phase 95 design gate for one PFHub-style physics benchmark",
            "evidence": "PFHub gives public physics benchmark definitions that can test model mechanisms before A100 validation.",
        },
        {
            "candidate_id": "P94-CAND-EXACA-SIM",
            "candidate_family": "simulation_augmented_target",
            "source_name": "ExaCA cellular-automata solidification code",
            "source_url": "https://github.com/LLNL/ExaCA",
            "source_type": "official_open_source_code",
            "public_reproducibility": "open_code_not_yet_project_data_card",
            "registration_status": "requires_generated_dataset_and_alignment_card",
            "target_relevance": "possible microstructure/process augmentation after a source manifest is created",
            "model_innovation_fit": "GCN or graph-conditioned process/microstructure branch after data-card gate",
            "allowed_next_gate": "simulation_data_card_only",
            "a100_training_allowed": False,
            "a100_80gb_request_now": False,
            "status": "blocked_until_simulation_data_card",
            "priority": 3,
            "stop_condition": "generated fields cannot be linked to AM-Bench process variables or strong baselines",
            "next_action": "create an ExaCA data-card proposal before any generated-data model training",
            "evidence": "ExaCA is useful as a reproducible simulator, but no generated target table exists in this repo.",
        },
        {
            "candidate_id": "P94-CAND-EXT-THERMAL",
            "candidate_family": "external_registered_dataset",
            "source_name": "external public registered thermal/process dataset",
            "source_url": "",
            "source_type": "to_be_supplied",
            "public_reproducibility": "missing_source_manifest",
            "registration_status": "missing_data_card",
            "target_relevance": "could become a second-paper or model-contribution route",
            "model_innovation_fit": "heat-kernel, GCN/attention, Bayesian PINN, or meta-learning only after intake",
            "allowed_next_gate": "source_manifest_and_data_card_required",
            "a100_training_allowed": False,
            "a100_80gb_request_now": False,
            "status": "blocked_no_external_data_card",
            "priority": 5,
            "stop_condition": "private, non-reproducible, unregistered, or lacks process/physics targets",
            "next_action": "provide or identify a public dataset with source manifest and registration story",
            "evidence": "Phase 81 external route remains blocked without a source manifest.",
        },
        {
            "candidate_id": "P94-CAND-MANUSCRIPT",
            "candidate_family": "submission_track_dependency",
            "source_name": "target venue or accepted benchmark papers",
            "source_url": "",
            "source_type": "user_supplied_review_input",
            "public_reproducibility": "not_applicable_to_model_training",
            "registration_status": "not_a_data_target",
            "target_relevance": "required for Phase 93 manuscript formatting and benchmark review",
            "model_innovation_fit": "none; prevents fake venue-specific claims",
            "allowed_next_gate": "benchmark_review_only",
            "a100_training_allowed": False,
            "a100_80gb_request_now": False,
            "status": "blocked_missing_target_benchmarks" if phase92_blocked else "review_input_available",
            "priority": 0,
            "stop_condition": "no target venue, author guide, or 3-10 benchmark papers",
            "next_action": "collect target venue or benchmark papers for Phase 93, separate from model exploration",
            "evidence": f"Phase 92 status is {phase92_gate.get('status')}.",
        },
    ]


def build_design_queue(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    open_rows = [row for row in candidate_rows if row["status"] == "open_for_local_design_gate"]
    queue: list[dict[str, Any]] = []
    for index, row in enumerate(sorted(open_rows, key=lambda item: int(item["priority"])), start=1):
        queue.append(
            {
                "queue_id": f"P94-DESIGN-{index:03d}",
                "priority": f"P{index}",
                "candidate_id": row["candidate_id"],
                "design_task": "write candidate_design.json and local/no-training benchmark protocol",
                "minimum_artifact": "docs/results/phase95_* candidate design package with pass/fail metrics",
                "pass_condition": "mechanism has a measurable target and does not depend on test-set selection",
                "stop_condition": row["stop_condition"],
                "allowed_compute": "local CPU/GPU smoke only; no A100 training",
            }
        )
    if not queue:
        queue.append(
            {
                "queue_id": "P94-DESIGN-000",
                "priority": "P0",
                "candidate_id": "none",
                "design_task": "no local design gate is currently open",
                "minimum_artifact": "resolve registration, external data card, or benchmark input blocker",
                "pass_condition": "one candidate status becomes open_for_local_design_gate",
                "stop_condition": "all candidates remain missing registration/data-card evidence",
                "allowed_compute": "none",
            }
        )
    return queue


def build_gate(candidate_rows: list[dict[str, Any]], design_rows: list[dict[str, Any]]) -> dict[str, Any]:
    open_design = [row for row in candidate_rows if row["status"] == "open_for_local_design_gate"]
    a100_open = [row for row in candidate_rows if row["a100_training_allowed"]]
    blocked = [row for row in candidate_rows if row["status"].startswith("blocked")]
    if open_design:
        status = "opened_local_design_gate_no_a100"
        next_action = "enter Phase 95 local/no-training design gate for the highest-priority open candidate"
        preferred = sorted(open_design, key=lambda row: int(row["priority"]))[0]["candidate_id"]
    else:
        status = "blocked_no_open_candidate"
        next_action = "resolve registration, external data-card, or benchmark-input blockers"
        preferred = "none"
    return {
        "status": status,
        "preferred_next_candidate": preferred,
        "open_local_design_candidates": len(open_design),
        "blocked_candidates": len(blocked),
        "candidate_rows": len(candidate_rows),
        "design_queue_rows": len(design_rows),
        "phase95_local_gate_allowed": bool(open_design),
        "a100_training_allowed_now": bool(a100_open),
        "a100_80gb_request_now": False,
        "submission_ready": False,
        "next_action": next_action,
        "required_before_a100_training": [
            "candidate_design.json",
            "synthetic/no-training or local benchmark gate",
            "baseline comparison",
            "non-worse global/focused metrics",
            "no-test-leakage selection",
        ],
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
    candidate_rows: list[dict[str, Any]],
    design_rows: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "# Phase 94 External Registered-Target Candidate Gate",
            "",
            "## Gate Decision",
            "",
            f"Status: `{gate['status']}`.",
            f"Preferred next candidate: `{gate['preferred_next_candidate']}`.",
            f"Phase 95 local gate allowed: `{str(gate['phase95_local_gate_allowed']).lower()}`.",
            f"A100 training allowed now: `{str(gate['a100_training_allowed_now']).lower()}`.",
            f"A100-SXM4-80GB request now: `{str(gate['a100_80gb_request_now']).lower()}`.",
            "",
            "Phase 94 only opens local/no-training design work. It does not open A100 training or submission-ready claims.",
            "",
            "## Candidate Triage",
            "",
            _markdown_table(
                candidate_rows,
                [
                    ("candidate_id", "Candidate"),
                    ("source_name", "Source"),
                    ("status", "Status"),
                    ("allowed_next_gate", "Allowed next gate"),
                    ("priority", "Priority"),
                ],
            ),
            "",
            "## Design Queue",
            "",
            _markdown_table(
                design_rows,
                [
                    ("queue_id", "Queue"),
                    ("candidate_id", "Candidate"),
                    ("design_task", "Task"),
                    ("allowed_compute", "Compute"),
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

    phase81_gate = _read_json(resolved["phase81_gate"])
    phase81_rows = _read_csv(resolved["phase81_table"])
    phase92_gate = _read_json(resolved["phase92_gate"])
    phase92_manifest = _read_json(resolved["phase92_manifest"])

    candidate_rows = build_candidate_rows(
        phase81_gate=phase81_gate,
        phase81_rows=phase81_rows,
        phase92_gate=phase92_gate,
    )
    design_rows = build_design_queue(candidate_rows)
    gate = build_gate(candidate_rows, design_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "phase94_external_candidate_triage.csv"
    design_path = output_dir / "phase94_local_design_queue.csv"
    gate_path = output_dir / "phase94_external_registered_target_candidate_gate.json"
    markdown_path = output_dir / "phase94_external_registered_target_candidate_gate.md"
    manifest_path = output_dir / "phase94_external_registered_target_candidate_gate_manifest.json"

    _write_csv(candidate_path, candidate_rows, CANDIDATE_FIELDS)
    _write_csv(design_path, design_rows, DESIGN_QUEUE_FIELDS)
    _write_json(gate_path, gate)
    markdown_path.write_text(build_markdown(gate, candidate_rows, design_rows), encoding="utf-8")

    manifest = {
        "phase": 94,
        "objective": "external_registered_target_candidate_gate",
        "inputs": {key: _display_path(path, root) for key, path in sorted(resolved.items())},
        "outputs": {
            "candidate_triage": _display_path(candidate_path, root),
            "local_design_queue": _display_path(design_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "candidate_rows": len(candidate_rows),
            "design_queue_rows": len(design_rows),
            "open_local_design_candidates": gate["open_local_design_candidates"],
            "blocked_candidates": gate["blocked_candidates"],
        },
        "gate": gate,
        "phase81_gate": {
            "status": phase81_gate.get("status"),
            "open_registered_target_count": phase81_gate.get("open_registered_target_count"),
            "preferred_next_route": phase81_gate.get("preferred_next_route"),
        },
        "phase92_gate": {
            "status": phase92_gate.get("status"),
            "benchmark_review_ready": phase92_gate.get("benchmark_review_ready"),
            "usable_benchmark_inputs": phase92_gate.get("usable_benchmark_inputs"),
        },
        "phase92_manifest_gate": phase92_manifest.get("gate"),
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase94_external_registered_target_candidate_gate"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    manifest = build_package(root=root, output_dir=output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
