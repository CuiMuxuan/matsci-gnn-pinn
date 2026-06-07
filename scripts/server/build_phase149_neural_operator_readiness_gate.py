#!/usr/bin/env python3
"""Build Phase 149 Neural Operator/FNO readiness gate.

This phase audits whether the current project evidence supports starting a
neural-operator branch. It consumes only existing small artifacts and does not
read raw data, tensorize fields, train models, or request larger GPUs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("docs/results/phase149_neural_operator_readiness_gate")

READINESS_FIELDS = (
    "criterion_id",
    "criterion",
    "required_for_operator_training",
    "observed_project_state",
    "evidence_source",
    "status",
    "blocks_operator_training",
    "next_action",
)
DECISION_FIELDS = (
    "decision_id",
    "route",
    "decision",
    "rationale",
    "phase150_allowed",
    "operator_training_allowed",
    "a100_training_allowed_now",
    "a100_80gb_request_now",
    "evidence_anchor",
)
BOUNDARY_FIELDS = (
    "boundary_id",
    "claim_or_route",
    "status",
    "allowed_wording",
    "blocked_wording",
    "evidence_anchor",
)

PHASE_INPUTS = {
    "phase116_gate": Path(
        "docs/results/phase116_paper_evidence_consolidation/"
        "phase116_paper_evidence_consolidation_gate.json"
    ),
    "phase116_positive_floor": Path(
        "docs/results/phase116_paper_evidence_consolidation/phase116_positive_floor_table.csv"
    ),
    "phase148_gate": Path(
        "docs/results/phase148_nist_ammt_path_contact_graph_audit/"
        "phase148_nist_ammt_path_contact_graph_audit_gate.json"
    ),
    "phase33_fourier_diagnostic": Path("docs/results/ambench_multiline_process_fourier_spacetime_v1.md"),
    "phase55_spot_size_seed_validation": Path(
        "docs/results/ambench_multiline_process_spot_size_seed_validation_v1.md"
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _display_path(path: Path, root: Path | None = None) -> str:
    try:
        if root is not None:
            return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        pass
    return path.as_posix()


def _is_false(value: Any) -> bool:
    if isinstance(value, bool):
        return value is False
    if isinstance(value, str):
        return value.strip().lower() in {"false", "0", "no", ""}
    return not bool(value)


def _has_text(text: str, *needles: str) -> bool:
    lower = text.lower()
    return all(needle.lower() in lower for needle in needles)


def build_readiness_rows(
    *,
    phase116_gate: dict[str, Any],
    positive_floor_rows: list[dict[str, str]],
    phase148_gate: dict[str, Any],
    phase33_text: str,
    phase55_text: str,
    phase_inputs: dict[str, Path],
    root: Path,
) -> list[dict[str, Any]]:
    positive_floor_ready = bool(phase116_gate.get("positive_floor_ready")) and bool(positive_floor_rows)
    locks_false = (
        _is_false(phase116_gate.get("phase116_model_training_allowed"))
        and _is_false(phase116_gate.get("a100_training_allowed_now"))
        and _is_false(phase116_gate.get("a100_80gb_request_now"))
        and _is_false(phase148_gate.get("phase148_model_training_allowed"))
        and _is_false(phase148_gate.get("a100_training_allowed_now"))
        and _is_false(phase148_gate.get("a100_80gb_request_now"))
    )
    fourier_negative = _has_text(phase33_text, "Fourier", "negative") or _has_text(
        phase33_text, "worsened", "broad_process_v1"
    )
    spot_size_route_floor = positive_floor_ready and any(
        row.get("split") == "spot_size" and row.get("manuscript_use") == "current_main_text_floor"
        for row in positive_floor_rows
    )
    seed_validated = _has_text(phase55_text, "seed-validated", "spot_size")
    path_contact_closed = phase148_gate.get("status") == "phase148_path_contact_graph_audit_closed_no_guarded_graph_gap"

    return [
        {
            "criterion_id": "P149-READY-001",
            "criterion": "dense_operator_tensor_dataset",
            "required_for_operator_training": "committed or server-local tensor grid manifest with fixed spatial shape and split provenance",
            "observed_project_state": "no dense tensor/grid manifest is part of the paper-facing floor; current floor is tabular route-guarded spot_size evidence",
            "evidence_source": _display_path(phase_inputs["phase116_positive_floor"], root),
            "status": "blocked_missing_dense_tensor_manifest",
            "blocks_operator_training": True,
            "next_action": "open a no-training dense tensorization inventory before any FNO training",
        },
        {
            "criterion_id": "P149-READY-002",
            "criterion": "operator_target_stability",
            "required_for_operator_training": "stable dense field target that survives strong baselines and split/shortcut guards",
            "observed_project_state": "only the narrow spot_size route floor is stable; NIST AMMT path-contact and melt-pool/sequence branches remain diagnostics",
            "evidence_source": _display_path(phase_inputs["phase148_gate"], root),
            "status": "blocked_no_operator_target_gap",
            "blocks_operator_training": True,
            "next_action": "do not convert closed NIST AMMT diagnostics into operator targets",
        },
        {
            "criterion_id": "P149-READY-003",
            "criterion": "spectral_representation_prior",
            "required_for_operator_training": "Fourier/spectral representation should not already be a negative diagnostic on the closest broad-process route",
            "observed_project_state": "Phase 33 Fourier spacetime representation was a negative broad-process diagnostic"
            if fourier_negative
            else "Phase 33 Fourier diagnostic evidence missing or inconclusive",
            "evidence_source": _display_path(phase_inputs["phase33_fourier_diagnostic"], root),
            "status": "blocked_fourier_proxy_negative" if fourier_negative else "blocked_fourier_evidence_incomplete",
            "blocks_operator_training": True,
            "next_action": "require a fresh dense readiness gate rather than scaling FNO from the Phase 33 result",
        },
        {
            "criterion_id": "P149-READY-004",
            "criterion": "existing_positive_floor_shape",
            "required_for_operator_training": "positive floor should be a field/operator prediction problem, not only a scalar route-selection result",
            "observed_project_state": "spot_size floor is seed-validated and useful for paper one, but it is not an operator-learning target"
            if spot_size_route_floor and seed_validated
            else "positive floor evidence is incomplete",
            "evidence_source": _display_path(phase_inputs["phase55_spot_size_seed_validation"], root),
            "status": "blocked_floor_not_operator_target" if spot_size_route_floor else "blocked_floor_incomplete",
            "blocks_operator_training": True,
            "next_action": "keep paper-one floor unchanged; do not relabel it as FNO/operator success",
        },
        {
            "criterion_id": "P149-READY-005",
            "criterion": "training_and_compute_locks",
            "required_for_operator_training": "previous diagnostic locks must remain false and a 40GB bottleneck must be measured before 80GB is requested",
            "observed_project_state": "all relevant training/A100 locks remain false; no 40GB bottleneck exists"
            if locks_false and path_contact_closed
            else "training locks or closure evidence are incomplete",
            "evidence_source": _display_path(phase_inputs["phase148_gate"], root),
            "status": "locked_no_operator_training",
            "blocks_operator_training": True,
            "next_action": "continue on A800 with no-training readiness gates only",
        },
    ]


def build_decision_rows(
    *, readiness_rows: list[dict[str, Any]], phase_inputs: dict[str, Path], root: Path
) -> list[dict[str, Any]]:
    blockers = [row for row in readiness_rows if row["blocks_operator_training"]]
    operator_training_allowed = not blockers
    phase150_allowed = bool(blockers)
    return [
        {
            "decision_id": "P149-DECISION-001",
            "route": "neural_operator_fno",
            "decision": "closed_not_ready_for_training" if blockers else "ready_for_focused_operator_training",
            "rationale": (
                "FNO/operator learning is externally credible but current project evidence lacks a dense tensor target, "
                "operator-specific baseline gap, and positive spectral proxy."
            ),
            "phase150_allowed": phase150_allowed,
            "operator_training_allowed": operator_training_allowed,
            "a100_training_allowed_now": False,
            "a100_80gb_request_now": False,
            "evidence_anchor": _display_path(phase_inputs["phase116_gate"], root),
        },
        {
            "decision_id": "P149-DECISION-002",
            "route": "dense_tensorization_inventory",
            "decision": "allow_no_training_inventory",
            "rationale": (
                "The only safe continuation of the neural-operator route is an inventory of whether server-local "
                "thermal fields can be tensorized with stable splits and strong baseline guards."
            ),
            "phase150_allowed": True,
            "operator_training_allowed": False,
            "a100_training_allowed_now": False,
            "a100_80gb_request_now": False,
            "evidence_anchor": _display_path(phase_inputs["phase33_fourier_diagnostic"], root),
        },
    ]


def build_boundary_rows(*, phase_inputs: dict[str, Path], root: Path) -> list[dict[str, Any]]:
    return [
        {
            "boundary_id": "P149-BOUNDARY-001",
            "claim_or_route": "neural_operator_fno",
            "status": "not_ready",
            "allowed_wording": "FNO/neural operators remain future work pending dense tensor readiness",
            "blocked_wording": "FNO success, operator-learning contribution, dense field operator model",
            "evidence_anchor": _display_path(phase_inputs["phase33_fourier_diagnostic"], root),
        },
        {
            "boundary_id": "P149-BOUNDARY-002",
            "claim_or_route": "first_paper_floor",
            "status": "unchanged",
            "allowed_wording": "route-guarded fixed-sampling broad12/broad21 spot_size under broad_process_v1",
            "blocked_wording": "recast the spot_size floor as a neural-operator result",
            "evidence_anchor": _display_path(phase_inputs["phase116_positive_floor"], root),
        },
        {
            "boundary_id": "P149-BOUNDARY-003",
            "claim_or_route": "compute",
            "status": "a800_sufficient",
            "allowed_wording": "no-training readiness and inventory gates on A800",
            "blocked_wording": "A100-SXM4-80GB request without measured 40GB operator-training bottleneck",
            "evidence_anchor": _display_path(phase_inputs["phase148_gate"], root),
        },
    ]


def build_gate(
    *,
    readiness_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    phase116_gate: dict[str, Any],
    phase148_gate: dict[str, Any],
) -> dict[str, Any]:
    blocker_rows = [row for row in readiness_rows if row["blocks_operator_training"]]
    phase150_allowed = any(row["phase150_allowed"] for row in decision_rows)
    if blocker_rows:
        status = "phase149_neural_operator_readiness_closed_not_ready_for_operator_training"
        next_action = (
            "do not train FNO/neural operators; if continuing this route, implement Phase 150 "
            "as a no-training dense tensorization inventory and baseline-gap audit"
        )
    else:
        status = "phase149_neural_operator_readiness_ready_focused_operator_training"
        next_action = "run focused low-capacity operator training on A800 before any 80GB request"
    return {
        "status": status,
        "readiness_rows": len(readiness_rows),
        "blocker_rows": len(blocker_rows),
        "phase150_dense_tensorization_inventory_allowed": phase150_allowed,
        "phase116_positive_floor_ready": bool(phase116_gate.get("positive_floor_ready")),
        "phase148_gate_status": phase148_gate.get("status"),
        "operator_training_allowed_now": False,
        "phase149_model_mechanism_allowed": False,
        "phase149_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": next_action,
    }


def _markdown_table(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[str]:
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    body = [
        "| " + " | ".join(_csv_value(row.get(field)) for field in fields) + " |"
        for row in rows
    ]
    return [header, sep, *body]


def _write_markdown(
    path: Path,
    *,
    gate: dict[str, Any],
    readiness_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Phase 149 Neural Operator Readiness Gate",
        "",
        f"- Status: `{gate['status']}`",
        f"- Blocker rows: `{gate['blocker_rows']}`",
        f"- Phase 150 dense tensorization inventory allowed: `{gate['phase150_dense_tensorization_inventory_allowed']}`",
        "- Operator training allowed now: `false`",
        "- A100 training allowed now: `false`",
        "",
        "## Readiness Audit",
        "",
        *_markdown_table(readiness_rows, READINESS_FIELDS),
        "",
        "## Decisions",
        "",
        *_markdown_table(decision_rows, DECISION_FIELDS),
        "",
        "## Boundaries",
        "",
        *_markdown_table(boundary_rows, BOUNDARY_FIELDS),
        "",
        f"Next action: {gate['next_action']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_package(
    *,
    root: Path,
    output_dir: Path,
    phase_inputs: dict[str, Path] | None = None,
) -> dict[str, Any]:
    phase_inputs = dict(phase_inputs or PHASE_INPUTS)
    resolved = {
        key: path if path.is_absolute() else root / path
        for key, path in phase_inputs.items()
    }
    phase116_gate = _read_json(resolved["phase116_gate"])
    positive_floor_rows = _read_csv(resolved["phase116_positive_floor"])
    phase148_gate = _read_json(resolved["phase148_gate"])
    phase33_text = _read_text(resolved["phase33_fourier_diagnostic"])
    phase55_text = _read_text(resolved["phase55_spot_size_seed_validation"])

    output_dir.mkdir(parents=True, exist_ok=True)
    readiness_rows = build_readiness_rows(
        phase116_gate=phase116_gate,
        positive_floor_rows=positive_floor_rows,
        phase148_gate=phase148_gate,
        phase33_text=phase33_text,
        phase55_text=phase55_text,
        phase_inputs=resolved,
        root=root,
    )
    decision_rows = build_decision_rows(readiness_rows=readiness_rows, phase_inputs=resolved, root=root)
    boundary_rows = build_boundary_rows(phase_inputs=resolved, root=root)
    gate = build_gate(
        readiness_rows=readiness_rows,
        decision_rows=decision_rows,
        phase116_gate=phase116_gate,
        phase148_gate=phase148_gate,
    )

    readiness_path = output_dir / "phase149_neural_operator_readiness_table.csv"
    decision_path = output_dir / "phase149_neural_operator_decision_table.csv"
    boundary_path = output_dir / "phase149_neural_operator_boundary_table.csv"
    gate_path = output_dir / "phase149_neural_operator_readiness_gate.json"
    markdown_path = output_dir / "phase149_neural_operator_readiness_gate.md"
    manifest_path = output_dir / "phase149_neural_operator_readiness_manifest.json"

    _write_csv(readiness_path, readiness_rows, READINESS_FIELDS)
    _write_csv(decision_path, decision_rows, DECISION_FIELDS)
    _write_csv(boundary_path, boundary_rows, BOUNDARY_FIELDS)
    _write_json(gate_path, gate)
    _write_markdown(
        markdown_path,
        gate=gate,
        readiness_rows=readiness_rows,
        decision_rows=decision_rows,
        boundary_rows=boundary_rows,
    )
    manifest = {
        "phase": 149,
        "objective": "neural_operator_fno_readiness_gate_no_training",
        "inputs": {key: _display_path(path, root) for key, path in resolved.items()},
        "outputs": {
            "readiness_table": _display_path(readiness_path, root),
            "decision_table": _display_path(decision_path, root),
            "boundary_table": _display_path(boundary_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown_summary": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "readiness_rows": len(readiness_rows),
            "blocker_rows": len([row for row in readiness_rows if row["blocks_operator_training"]]),
            "decision_rows": len(decision_rows),
            "boundary_rows": len(boundary_rows),
        },
        "gate": gate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = build_package(root=root, output_dir=output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
