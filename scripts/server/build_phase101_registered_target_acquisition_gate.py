#!/usr/bin/env python3
"""Build the Phase 101 registered-target acquisition gate.

Phase 101 decides whether the Phase 100 local mechanism evidence can move to a
real registered AM-Bench or external target. Generated analytic surrogates are
kept as local mechanism evidence and cannot unlock transfer training.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


TARGET_FIELDS = (
    "target_id",
    "source_family",
    "dataset_id",
    "target_type",
    "public_reproducibility",
    "source_manifest_status",
    "registration_status",
    "split_plan_status",
    "baseline_plan_status",
    "physical_source_path_meaning",
    "phase102_baseline_smoke_allowed",
    "a100_training_allowed",
    "status",
    "priority",
    "next_action",
    "evidence",
)

MANUAL_QUEUE_FIELDS = (
    "queue_id",
    "priority",
    "target_id",
    "missing_component",
    "minimum_evidence",
    "unlock_condition",
    "stop_condition",
)

PROTOCOL_FIELDS = (
    "protocol_id",
    "component",
    "requirement",
    "current_status",
    "pass_condition",
    "stop_condition",
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


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _display_path(path: Path, root: Path | None = None) -> str:
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def _default_paths(root: Path) -> dict[str, Path]:
    phase81 = root / "docs/results/phase81_registered_target_intake_gate"
    phase94 = root / "docs/results/phase94_external_registered_target_candidate_gate"
    phase98 = root / "docs/results/phase98_registered_target_unlock_gate"
    phase100 = root / "docs/results/phase100_local_mechanism_evidence_package"
    return {
        "phase81_gate": phase81 / "phase81_registered_target_gate.json",
        "phase81_intake_table": phase81 / "phase81_registered_target_intake_table.csv",
        "phase94_gate": phase94 / "phase94_external_registered_target_candidate_gate.json",
        "phase94_candidate_table": phase94 / "phase94_external_candidate_triage.csv",
        "phase98_unlock_table": phase98 / "phase98_unlock_candidate_table.csv",
        "phase100_gate": phase100 / "phase100_local_mechanism_evidence_package.json",
        "phase100_boundary_table": phase100 / "phase100_local_mechanism_boundary_table.csv",
    }


def _row_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str] | None:
    for row in rows:
        if row.get(key) == value:
            return row
    return None


def _is_ready(row: dict[str, Any]) -> bool:
    return bool(row.get("phase102_baseline_smoke_allowed"))


def _physical_source_path_meaning(registration_status: str, route_status: str) -> str:
    registration = registration_status.strip().lower()
    status = route_status.strip().lower()
    if "must_be_verified" in registration:
        return "unverified"
    blocker_terms = (
        "missing",
        "no ",
        "not_registered",
        "independent-rescale",
        "independent rescale",
        "diagnostic",
        "only",
    )
    if status.startswith("blocked") or any(term in registration for term in blocker_terms):
        return "blocked"
    pass_terms = (
        "camera_to_galvo",
        "camera-to-galvo",
        "scan_path_alignment",
        "scan-path alignment",
        "registration_available",
        "registered_by_analytic_definition",
        "documented",
        "exact",
        "aligned",
    )
    if any(term in registration for term in pass_terms):
        return "yes"
    return "unverified"


def _target_from_phase81(row: dict[str, str]) -> dict[str, Any]:
    route_id = row["route_id"]
    registration = row.get("coordinate_registration_status", "")
    route_status = row.get("status", "")
    source_manifest = row.get("source_manifest", "")
    split_status = row.get("split_status", "")
    baseline_status = row.get("baseline_entry_status", "")
    physical_meaning = _physical_source_path_meaning(registration, route_status)
    ready = (
        route_status == "open_registered_target"
        and source_manifest != ""
        and "available" in split_status
        and "baseline" in baseline_status
        and physical_meaning == "yes"
    )
    if ready:
        status = "phase102_registered_baseline_smoke_ready_no_a100"
        next_action = "enter Phase 102 baseline-first smoke"
    elif route_status == "blocked_missing_registration":
        status = "blocked_registration_evidence_required"
        next_action = row.get("next_action") or "provide coordinate registration evidence"
    elif route_status == "blocked_no_data_card":
        status = "blocked_source_manifest_data_card_required"
        next_action = row.get("next_action") or "provide public source manifest and data card"
    elif route_status.startswith("diagnostic"):
        status = "diagnostic_only_not_registered_transfer"
        next_action = row.get("next_action") or "keep as diagnostic evidence"
    else:
        status = route_status or "blocked"
        next_action = row.get("next_action") or "resolve registered target requirements"
    return {
        "target_id": f"phase101_{route_id}",
        "source_family": row.get("route_family", ""),
        "dataset_id": row.get("dataset_id", ""),
        "target_type": row.get("target_family", ""),
        "public_reproducibility": "source_manifest_present" if source_manifest else "missing_source_manifest",
        "source_manifest_status": "present" if source_manifest else "missing",
        "registration_status": registration,
        "split_plan_status": split_status,
        "baseline_plan_status": baseline_status,
        "physical_source_path_meaning": physical_meaning,
        "phase102_baseline_smoke_allowed": ready,
        "a100_training_allowed": False,
        "status": status,
        "priority": int(row.get("priority") or 99),
        "next_action": next_action,
        "evidence": row.get("evidence", ""),
    }


def _external_target_from_phase94(row: dict[str, str]) -> dict[str, Any]:
    candidate_id = row["candidate_id"]
    is_external = candidate_id in {"P94-CAND-EXT-THERMAL", "P94-CAND-EXACA-SIM"}
    if not is_external:
        raise ValueError(f"Unsupported external target row: {candidate_id}")
    if candidate_id == "P94-CAND-EXACA-SIM":
        status = "blocked_until_simulation_data_card"
        missing = "generated_dataset_and_alignment_card"
        source_family = "simulation_augmented_target"
    else:
        status = "blocked_no_external_data_card"
        missing = "source_manifest_and_registration_story"
        source_family = "external_registered_dataset"
    return {
        "target_id": f"phase101_{candidate_id.lower().replace('-', '_')}",
        "source_family": source_family,
        "dataset_id": row.get("source_name", ""),
        "target_type": row.get("source_type", ""),
        "public_reproducibility": row.get("public_reproducibility", ""),
        "source_manifest_status": "missing",
        "registration_status": row.get("registration_status", ""),
        "split_plan_status": "missing_train_validation_test_split",
        "baseline_plan_status": "baseline_table_required",
        "physical_source_path_meaning": "unverified",
        "phase102_baseline_smoke_allowed": False,
        "a100_training_allowed": False,
        "status": status,
        "priority": int(row.get("priority") or 99),
        "next_action": row.get("next_action") or f"provide {missing}",
        "evidence": row.get("evidence", ""),
    }


def _surrogate_target_from_phase100(phase100_gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_id": "phase101_generated_registered_surrogate_closed",
        "source_family": "generated_registered_surrogate",
        "dataset_id": "phase98_generated_pfhub_registered_surrogate_v1",
        "target_type": "analytic_heat_source_surrogate",
        "public_reproducibility": "repo_generated_reproducible_artifact",
        "source_manifest_status": "present",
        "registration_status": "registered_by_analytic_definition",
        "split_plan_status": "fixed_train_validation_test_grids",
        "baseline_plan_status": "phase99_baseline_first_smoke_complete",
        "physical_source_path_meaning": "local_surrogate_only",
        "phase102_baseline_smoke_allowed": False,
        "a100_training_allowed": False,
        "status": "closed_local_mechanism_not_transfer_target",
        "priority": 90,
        "next_action": "do not treat analytic surrogate as AM-Bench/external transfer evidence",
        "evidence": (
            f"Phase 100 status is {phase100_gate.get('status')}; "
            f"appendix_ready={str(phase100_gate.get('appendix_local_mechanism_ready')).lower()}"
        ),
    }


def build_target_rows(
    *,
    phase81_rows: list[dict[str, str]],
    phase94_rows: list[dict[str, str]],
    phase100_gate: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [_target_from_phase81(row) for row in phase81_rows]
    for candidate_id in ("P94-CAND-EXACA-SIM", "P94-CAND-EXT-THERMAL"):
        row = _row_by(phase94_rows, "candidate_id", candidate_id)
        if row is not None:
            rows.append(_external_target_from_phase94(row))
    rows.append(_surrogate_target_from_phase100(phase100_gate))
    return sorted(rows, key=lambda row: int(row["priority"]))


def build_manual_queue(target_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in target_rows:
        if row["phase102_baseline_smoke_allowed"] or row["status"].startswith("closed_local"):
            continue
        if "registration" in row["status"]:
            missing_component = "coordinate_registration"
            minimum_evidence = "camera-to-galvo mapping, scan-path alignment, or equivalent coordinate registration"
        elif "data_card" in row["status"] or row["source_manifest_status"] == "missing":
            missing_component = "source_manifest_data_card"
            minimum_evidence = "public source manifest, license/reproducibility note, split plan, and baseline plan"
        else:
            missing_component = "registered_transfer_evidence"
            minimum_evidence = "proof that source/path features map physically to the target observations"
        rows.append(
            {
                "queue_id": f"P101-QUEUE-{len(rows) + 1:03d}",
                "priority": row["priority"],
                "target_id": row["target_id"],
                "missing_component": missing_component,
                "minimum_evidence": minimum_evidence,
                "unlock_condition": "target row becomes public, registered, split-ready, and baseline-ready",
                "stop_condition": row["next_action"],
            }
        )
    return rows


def build_protocol_rows() -> list[dict[str, Any]]:
    return [
        {
            "protocol_id": "P101-PROT-001",
            "component": "real_transfer_target",
            "requirement": "A Phase 102 target must be a real AM-Bench or external registered target, not the generated analytic surrogate.",
            "current_status": "enforced_by_phase101",
            "pass_condition": "target source_family is not generated_registered_surrogate and registration is physical",
            "stop_condition": "only analytic or synthetic-local evidence is available",
        },
        {
            "protocol_id": "P101-PROT-002",
            "component": "source_manifest",
            "requirement": "Target needs a public source manifest or generated-data card with reproducibility and licensing notes.",
            "current_status": "missing_for_external_and_simulation_targets",
            "pass_condition": "source manifest, checksums or generator, and license note are present",
            "stop_condition": "private or non-reproducible data",
        },
        {
            "protocol_id": "P101-PROT-003",
            "component": "registration",
            "requirement": "Source/path coordinates must map to target observations without test-label alignment.",
            "current_status": "missing_for_current_ambench_transfer_routes",
            "pass_condition": "camera-to-galvo mapping, scan-path alignment, or equivalent registration exists",
            "stop_condition": "independent rescale or unregistered pixel/process features only",
        },
        {
            "protocol_id": "P101-PROT-004",
            "component": "compute_governance",
            "requirement": "Phase 101 may only open Phase 102 local baseline-first smoke, not A100 training.",
            "current_status": "a100_locked",
            "pass_condition": "A100 flags remain false",
            "stop_condition": "any target requests A100 before baseline-first smoke",
        },
    ]


def build_gate(
    *,
    phase81_gate: dict[str, Any],
    phase100_gate: dict[str, Any],
    target_rows: list[dict[str, Any]],
    manual_queue: list[dict[str, Any]],
) -> dict[str, Any]:
    phase100_allows = bool(phase100_gate.get("phase101_registered_target_acquisition_allowed"))
    ready_rows = [row for row in target_rows if _is_ready(row)]
    real_ready_rows = [
        row for row in ready_rows if row.get("source_family") != "generated_registered_surrogate"
    ]
    if not phase100_allows:
        status = "blocked_by_phase100_package_gate"
        next_action = "repair Phase 100 local mechanism package before target acquisition"
        phase102_allowed = False
    elif real_ready_rows:
        status = "registered_target_acquired_phase102_allowed"
        next_action = "enter Phase 102 baseline-first smoke on the highest-priority real registered target"
        phase102_allowed = True
    else:
        status = "blocked_no_real_registered_target"
        next_action = "provide pad registration evidence or an external registered-target data card"
        phase102_allowed = False
    return {
        "status": status,
        "source_phase81_status": phase81_gate.get("status"),
        "source_phase100_status": phase100_gate.get("status"),
        "phase102_baseline_smoke_allowed": phase102_allowed,
        "ready_real_registered_target_rows": len(real_ready_rows),
        "target_rows": len(target_rows),
        "manual_queue_rows": len(manual_queue),
        "open_registered_target_count_phase81": phase81_gate.get("open_registered_target_count"),
        "preferred_next_route_phase81": phase81_gate.get("preferred_next_route"),
        "analytic_surrogate_closed_for_transfer": True,
        "am_bench_transfer_unlocked": phase102_allowed,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "submission_ready": False,
        "next_action": next_action,
        "required_before_a100_training": [
            "Phase 102 baseline-first smoke on a real registered target",
            "non-worse global/hot/gradient metrics",
            "strong baseline comparison",
            "validation-only selection",
            "server validation from a pushed commit",
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
    *,
    gate: dict[str, Any],
    target_rows: list[dict[str, Any]],
    manual_queue: list[dict[str, Any]],
    protocol_rows: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "# Phase 101 Registered Target Acquisition Gate",
            "",
            "## Gate Decision",
            "",
            f"Status: `{gate['status']}`.",
            f"Phase 102 baseline smoke allowed: `{str(gate['phase102_baseline_smoke_allowed']).lower()}`.",
            f"AM-Bench transfer unlocked: `{str(gate['am_bench_transfer_unlocked']).lower()}`.",
            f"A100 training allowed now: `{str(gate['a100_training_allowed_now']).lower()}`.",
            f"A100-SXM4-80GB request now: `{str(gate['a100_80gb_request_now']).lower()}`.",
            "",
            "Phase 101 requires a real registered AM-Bench or external target. The generated analytic surrogate is closed for transfer use.",
            "",
            "## Target Rows",
            "",
            _markdown_table(
                target_rows,
                [
                    ("target_id", "Target"),
                    ("dataset_id", "Dataset"),
                    ("registration_status", "Registration"),
                    ("phase102_baseline_smoke_allowed", "Phase 102"),
                    ("status", "Status"),
                    ("next_action", "Next action"),
                ],
            ),
            "",
            "## Manual Queue",
            "",
            _markdown_table(
                manual_queue,
                [
                    ("queue_id", "Queue"),
                    ("target_id", "Target"),
                    ("missing_component", "Missing"),
                    ("minimum_evidence", "Minimum evidence"),
                ],
            ),
            "",
            "## Protocol",
            "",
            _markdown_table(
                protocol_rows,
                [
                    ("protocol_id", "Protocol"),
                    ("component", "Component"),
                    ("requirement", "Requirement"),
                    ("current_status", "Current status"),
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
    phase81_rows = _read_csv(resolved["phase81_intake_table"])
    phase94_gate = _read_json(resolved["phase94_gate"])
    phase94_rows = _read_csv(resolved["phase94_candidate_table"])
    phase98_rows = _read_csv(resolved["phase98_unlock_table"])
    phase100_gate = _read_json(resolved["phase100_gate"])

    target_rows = build_target_rows(
        phase81_rows=phase81_rows,
        phase94_rows=phase94_rows,
        phase100_gate=phase100_gate,
    )
    manual_queue = build_manual_queue(target_rows)
    protocol_rows = build_protocol_rows()
    gate = build_gate(
        phase81_gate=phase81_gate,
        phase100_gate=phase100_gate,
        target_rows=target_rows,
        manual_queue=manual_queue,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "phase101_registered_target_acquisition_table.csv"
    queue_path = output_dir / "phase101_registered_target_manual_queue.csv"
    protocol_path = output_dir / "phase101_registered_target_protocol.csv"
    gate_path = output_dir / "phase101_registered_target_acquisition_gate.json"
    markdown_path = output_dir / "phase101_registered_target_acquisition_gate.md"
    manifest_path = output_dir / "phase101_registered_target_acquisition_gate_manifest.json"

    _write_csv(target_path, target_rows, TARGET_FIELDS)
    _write_csv(queue_path, manual_queue, MANUAL_QUEUE_FIELDS)
    _write_csv(protocol_path, protocol_rows, PROTOCOL_FIELDS)
    _write_json(gate_path, gate)
    markdown_path.write_text(
        build_markdown(
            gate=gate,
            target_rows=target_rows,
            manual_queue=manual_queue,
            protocol_rows=protocol_rows,
        ),
        encoding="utf-8",
    )

    manifest = {
        "phase": 101,
        "objective": "registered_target_acquisition_gate",
        "inputs": {key: _display_path(path, root) for key, path in sorted(resolved.items())},
        "outputs": {
            "target_table": _display_path(target_path, root),
            "manual_queue": _display_path(queue_path, root),
            "protocol": _display_path(protocol_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "target_rows": len(target_rows),
            "manual_queue_rows": len(manual_queue),
            "protocol_rows": len(protocol_rows),
            "ready_real_registered_target_rows": gate["ready_real_registered_target_rows"],
            "phase98_unlock_rows_read": len(phase98_rows),
        },
        "gate": gate,
        "source_gates": {
            "phase81_status": phase81_gate.get("status"),
            "phase94_status": phase94_gate.get("status"),
            "phase100_status": phase100_gate.get("status"),
        },
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase101_registered_target_acquisition_gate"),
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
