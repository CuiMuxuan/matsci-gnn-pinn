#!/usr/bin/env python3
"""Build Phase 147 literature-guided model roadmap.

This phase converts the latest literature/project search into a guarded project
roadmap. It consumes only existing small closure/evidence artifacts and does
not read raw data, run baselines, or open model training.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("docs/results/phase147_literature_guided_model_roadmap")

ROUTE_FIELDS = (
    "route_id",
    "external_pattern",
    "source_or_project",
    "stable_url",
    "external_takeaway",
    "project_overlap",
    "prior_project_evidence",
    "risk_if_repeated",
    "recommended_use",
    "next_phase_action",
    "trust_state",
)
DECISION_FIELDS = (
    "decision_id",
    "candidate_route",
    "decision",
    "rationale",
    "phase148_allowed",
    "model_training_allowed",
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
    "phase111_registered_target_closure": Path(
        "docs/results/phase111_nist_ammt_registered_target_closure_package/"
        "phase111_nist_ammt_registered_target_closure_gate.json"
    ),
    "phase113_melt_pool_focused_review": Path(
        "docs/results/phase113_nist_ammt_melt_pool_focused_review/"
        "phase113_nist_ammt_melt_pool_focused_review_gate.json"
    ),
    "phase114_gcode_strategy_source": Path(
        "docs/results/phase114_nist_ammt_gcode_strategy_source_gate/"
        "phase114_nist_ammt_gcode_strategy_source_gate.json"
    ),
    "phase115_nist_ammt_diagnostic_closure": Path(
        "docs/results/phase115_nist_ammt_diagnostic_closure_package/"
        "phase115_nist_ammt_diagnostic_closure_gate.json"
    ),
    "phase146_paper_evidence_refresh": Path(
        "docs/results/phase146_paper_evidence_refresh/phase146_paper_evidence_refresh_gate.json"
    ),
}

LITERATURE_ROUTES = (
    {
        "route_id": "thermal_pinn",
        "external_pattern": "AM thermal PINN for temperature prediction and parameter identification",
        "source_or_project": "Physics_informed_AM",
        "stable_url": "https://github.com/ShuhengLiao/Physics_informed_AM",
        "external_takeaway": "PINNs are credible for AM thermal fields, but PDE soft constraints alone are not a new branch here.",
        "project_overlap": "macro_pinn_and_pde_residual",
        "prior_project_evidence_key": "phase146_paper_evidence_refresh",
        "risk_if_repeated": "would repackage existing Macro PINN/PDE residual work without a new baseline-visible gap",
        "recommended_use": "background_and_method_positioning",
        "next_phase_action": "do not open standalone thermal-PINN training from this route",
        "trust_state": "candidate_verified_url",
    },
    {
        "route_id": "geometry_agnostic_gnn",
        "external_pattern": "geometry/topology-aware GNN thermal surrogate for AM",
        "source_or_project": "Geometry-agnostic data-driven thermal modeling of AM via graph neural networks",
        "stable_url": "https://www.scholars.northwestern.edu/en/publications/geometry-agnostic-data-driven-thermal-modeling-of-additive-manufa",
        "external_takeaway": "Graph topology should encode heat-transfer neighborhoods rather than only scalar process descriptors.",
        "project_overlap": "gnn_conditioning_and_process_route_guards",
        "prior_project_evidence_key": "phase111_registered_target_closure",
        "risk_if_repeated": "generic GNN conditioning has failed without a guarded target/source gap",
        "recommended_use": "architecture_inspiration_only",
        "next_phase_action": "allow only no-training topology-feature gate before any GNN mechanism",
        "trust_state": "candidate_verified_url",
    },
    {
        "route_id": "physics_hardcoded_gnn",
        "external_pattern": "physics-hardcoded graph dynamics plus learned nonlinear residual",
        "source_or_project": "MAM-PhyGNN-style physics-hardcoded graph model",
        "stable_url": "https://www.sciencedirect.com/science/article/pii/S221486042600134X",
        "external_takeaway": "A fixed diffusion/Laplacian branch plus small learned residual is a stronger inductive bias than blind routing.",
        "project_overlap": "future_low_capacity_graph_residual",
        "prior_project_evidence_key": "phase115_nist_ammt_diagnostic_closure",
        "risk_if_repeated": "premature mechanism design would violate closed NIST AMMT gates",
        "recommended_use": "phase148_design_target_if_no_training_gate_passes",
        "next_phase_action": "defer implementation until a no-training topology gate beats controls",
        "trust_state": "candidate_verified_url",
    },
    {
        "route_id": "capl_path_history",
        "external_pattern": "contact-aware path-level thermal-history features",
        "source_or_project": "CAPL/path-level thermal history and MeltpoolGAN-related route",
        "stable_url": "https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=936133",
        "external_takeaway": "Ordered scan/contact/reheat relations are more relevant than scalar source/path summaries.",
        "project_overlap": "nist_ammt_gcode_and_source_path_gates",
        "prior_project_evidence_key": "phase114_gcode_strategy_source",
        "risk_if_repeated": "simple G-code strategy/source summaries already closed with no guarded baseline gap",
        "recommended_use": "only_if_finer_than_phase114",
        "next_phase_action": "open no-training path-contact graph audit only with shuffled/shortcut controls",
        "trust_state": "candidate_verified_url",
    },
    {
        "route_id": "meltpoolgan",
        "external_pattern": "path-history-conditioned melt-pool image/shape prediction",
        "source_or_project": "MeltpoolGAN",
        "stable_url": "https://www.sciencedirect.com/science/article/pii/S2214860424001416",
        "external_takeaway": "Melt-pool imagery can be modeled from path-level thermal history, but target split stability is critical.",
        "project_overlap": "phase112_113_melt_pool_targets",
        "prior_project_evidence_key": "phase113_melt_pool_focused_review",
        "risk_if_repeated": "Phase 112 melt-pool candidates reversed on test versus mean guard",
        "recommended_use": "negative_control_and_related_work",
        "next_phase_action": "do not train on current Phase 112 melt-pool targets",
        "trust_state": "candidate_verified_url",
    },
    {
        "route_id": "neural_operator",
        "external_pattern": "FNO/neural-operator thermal digital twin",
        "source_or_project": "Fourier neural operator AM thermal-field work",
        "stable_url": "https://arxiv.org/abs/2307.01804",
        "external_takeaway": "Operator learning may scale field prediction, but it needs a cleaner dense target regime than current gates provide.",
        "project_overlap": "phase33_fourier_diagnostic_and_broad_process_floor",
        "prior_project_evidence_key": "phase146_paper_evidence_refresh",
        "risk_if_repeated": "Fourier features already degraded broad-process spot-size routes; FNO would be premature",
        "recommended_use": "future_large_data_candidate",
        "next_phase_action": "do not request 80GB or train neural operators without a seed-positive dense gate",
        "trust_state": "candidate_verified_url",
    },
    {
        "route_id": "microstructure_gnn",
        "external_pattern": "microstructure/grain graph neural networks",
        "source_or_project": "PEGN / GrainGNN-style microstructure graph learning",
        "stable_url": "https://www.nature.com/articles/s41524-022-00890-9",
        "external_takeaway": "Microstructure graph learning is relevant, but it requires stronger physical registration than current data provide.",
        "project_overlap": "phase17_22_real_micro_diagnostics",
        "prior_project_evidence_key": "phase146_paper_evidence_refresh",
        "risk_if_repeated": "current microstructure branches are weak-positive diagnostics, not stable claims",
        "recommended_use": "limitations_and_future_work",
        "next_phase_action": "do not claim microstructure GNN success in the first paper",
        "trust_state": "candidate_verified_url",
    },
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


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


def _lock_fields_false(gate: dict[str, Any]) -> bool:
    return all(
        _is_false(gate.get(key))
        for key in (
            "phase111_model_training_allowed",
            "phase113_model_training_allowed",
            "phase114_model_training_allowed",
            "phase115_model_training_allowed",
            "phase146_model_training_allowed",
            "a100_training_allowed_now",
            "a100_80gb_request_now",
        )
        if key in gate
    )


def build_route_rows(*, phase_inputs: dict[str, Path], root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for route in LITERATURE_ROUTES:
        prior_key = route["prior_project_evidence_key"]
        rows.append(
            {
                **{key: route[key] for key in ROUTE_FIELDS if key in route},
                "prior_project_evidence": _display_path(phase_inputs[prior_key], root),
            }
        )
    return rows


def build_decision_rows(
    *,
    phase111_gate: dict[str, Any],
    phase113_gate: dict[str, Any],
    phase114_gate: dict[str, Any],
    phase115_gate: dict[str, Any],
    phase146_gate: dict[str, Any],
    phase_inputs: dict[str, Path],
    root: Path,
) -> list[dict[str, Any]]:
    locks_ok = all(
        _lock_fields_false(gate)
        for gate in (phase111_gate, phase113_gate, phase114_gate, phase115_gate, phase146_gate)
    )
    nist_closed = (
        bool(phase111_gate.get("nist_ammt_sequence_branch_closed", True))
        and bool(phase115_gate.get("all_training_locks_verified", True))
        and "closed" in str(phase113_gate.get("status", ""))
        and "closed" in str(phase114_gate.get("status", ""))
    )
    floor_ready = bool(phase146_gate.get("first_paper_draft_allowed_now"))
    phase148_allowed = bool(locks_ok and nist_closed and floor_ready)
    return [
        {
            "decision_id": "P147-DECISION-001",
            "candidate_route": "capl_path_contact_graph",
            "decision": "allow_no_training_design" if phase148_allowed else "blocked",
            "rationale": (
                "External CAPL/MeltpoolGAN routes motivate ordered path-contact/reheat graphs, "
                "but Phase 114 already closed simple G-code strategy summaries; the next branch "
                "must be finer-grained and baseline-first."
            ),
            "phase148_allowed": phase148_allowed,
            "model_training_allowed": False,
            "a100_training_allowed_now": False,
            "a100_80gb_request_now": False,
            "evidence_anchor": _display_path(phase_inputs["phase114_gcode_strategy_source"], root),
        },
        {
            "decision_id": "P147-DECISION-002",
            "candidate_route": "physics_hardcoded_graph_residual",
            "decision": "defer_until_no_training_gap",
            "rationale": (
                "MAM-PhyGNN-style inductive bias is promising, but mechanism design must wait "
                "until a topology/source gate beats strong baselines and shuffled controls."
            ),
            "phase148_allowed": False,
            "model_training_allowed": False,
            "a100_training_allowed_now": False,
            "a100_80gb_request_now": False,
            "evidence_anchor": _display_path(phase_inputs["phase115_nist_ammt_diagnostic_closure"], root),
        },
        {
            "decision_id": "P147-DECISION-003",
            "candidate_route": "neural_operator_or_microstructure_gnn",
            "decision": "keep_as_future_work",
            "rationale": (
                "Neural operators and microstructure GNNs are externally credible, but current "
                "project evidence lacks a stable dense/operator target or physically registered "
                "microstructure gap."
            ),
            "phase148_allowed": False,
            "model_training_allowed": False,
            "a100_training_allowed_now": False,
            "a100_80gb_request_now": False,
            "evidence_anchor": _display_path(phase_inputs["phase146_paper_evidence_refresh"], root),
        },
    ]


def build_boundary_rows(*, phase_inputs: dict[str, Path], root: Path) -> list[dict[str, Any]]:
    return [
        {
            "boundary_id": "P147-BOUNDARY-001",
            "claim_or_route": "first_paper_main_claim",
            "status": "unchanged",
            "allowed_wording": "route-guarded fixed-sampling broad12/broad21 spot_size under broad_process_v1",
            "blocked_wording": "complete GNN-PINN or general process-condition modeling",
            "evidence_anchor": _display_path(phase_inputs["phase146_paper_evidence_refresh"], root),
        },
        {
            "boundary_id": "P147-BOUNDARY-002",
            "claim_or_route": "capl_path_contact_graph",
            "status": "no_training_design_only",
            "allowed_wording": "literature-motivated no-training path-contact graph audit",
            "blocked_wording": "CAPL/G-code success, source-path/Green success, or graph model success",
            "evidence_anchor": _display_path(phase_inputs["phase114_gcode_strategy_source"], root),
        },
        {
            "boundary_id": "P147-BOUNDARY-003",
            "claim_or_route": "compute",
            "status": "a800_sufficient",
            "allowed_wording": "continue with small artifact gates and A800 reproduction",
            "blocked_wording": "request A100-SXM4-80GB before a seed-positive 40GB blocker",
            "evidence_anchor": _display_path(phase_inputs["phase115_nist_ammt_diagnostic_closure"], root),
        },
    ]


def build_gate(
    *,
    phase111_gate: dict[str, Any],
    phase113_gate: dict[str, Any],
    phase114_gate: dict[str, Any],
    phase115_gate: dict[str, Any],
    phase146_gate: dict[str, Any],
    decision_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    phase148_allowed = any(row["phase148_allowed"] for row in decision_rows)
    locks_verified = all(
        _lock_fields_false(gate)
        for gate in (phase111_gate, phase113_gate, phase114_gate, phase115_gate, phase146_gate)
    )
    if phase148_allowed and locks_verified:
        status = "phase147_literature_guided_model_roadmap_ready_phase148_no_training_design"
        next_action = (
            "implement Phase 148 no-training path-contact graph audit only; compare against "
            "Phase 114 G-code strategy, scalar source, shuffled sequence, layer/time, and camera controls"
        )
    else:
        status = "phase147_literature_guided_model_roadmap_incomplete"
        next_action = "repair missing closure evidence or training-lock violations before opening a new branch"
    return {
        "status": status,
        "phase148_no_training_design_allowed": bool(phase148_allowed and locks_verified),
        "recommended_phase148_route": "capl_path_contact_graph_audit" if phase148_allowed else None,
        "route_rows": len(LITERATURE_ROUTES),
        "decision_rows": len(decision_rows),
        "previous_nist_diagnostics_locked": locks_verified,
        "phase146_first_paper_draft_allowed": bool(phase146_gate.get("first_paper_draft_allowed_now")),
        "new_main_paper_claim_ready": False,
        "phase147_model_mechanism_allowed": False,
        "phase147_model_training_allowed": False,
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
    route_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Phase 147 Literature-Guided Model Roadmap",
        "",
        f"- Status: `{gate['status']}`",
        f"- Recommended Phase 148 route: `{gate['recommended_phase148_route']}`",
        f"- Phase 148 no-training design allowed: `{gate['phase148_no_training_design_allowed']}`",
        "- Model mechanism allowed now: `false`",
        "- Model training allowed now: `false`",
        "- A100 training allowed now: `false`",
        "",
        "## Route Audit",
        "",
        *_markdown_table(route_rows, ("route_id", "recommended_use", "next_phase_action", "prior_project_evidence")),
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
    phase111_gate = _read_json(resolved["phase111_registered_target_closure"])
    phase113_gate = _read_json(resolved["phase113_melt_pool_focused_review"])
    phase114_gate = _read_json(resolved["phase114_gcode_strategy_source"])
    phase115_gate = _read_json(resolved["phase115_nist_ammt_diagnostic_closure"])
    phase146_gate = _read_json(resolved["phase146_paper_evidence_refresh"])

    output_dir.mkdir(parents=True, exist_ok=True)
    route_rows = build_route_rows(phase_inputs=resolved, root=root)
    decision_rows = build_decision_rows(
        phase111_gate=phase111_gate,
        phase113_gate=phase113_gate,
        phase114_gate=phase114_gate,
        phase115_gate=phase115_gate,
        phase146_gate=phase146_gate,
        phase_inputs=resolved,
        root=root,
    )
    boundary_rows = build_boundary_rows(phase_inputs=resolved, root=root)
    gate = build_gate(
        phase111_gate=phase111_gate,
        phase113_gate=phase113_gate,
        phase114_gate=phase114_gate,
        phase115_gate=phase115_gate,
        phase146_gate=phase146_gate,
        decision_rows=decision_rows,
    )

    route_path = output_dir / "phase147_literature_route_audit_table.csv"
    decision_path = output_dir / "phase147_model_roadmap_decision_table.csv"
    boundary_path = output_dir / "phase147_claim_boundary_table.csv"
    gate_path = output_dir / "phase147_literature_guided_model_roadmap_gate.json"
    markdown_path = output_dir / "phase147_literature_guided_model_roadmap.md"
    manifest_path = output_dir / "phase147_literature_guided_model_roadmap_manifest.json"

    _write_csv(route_path, route_rows, ROUTE_FIELDS)
    _write_csv(decision_path, decision_rows, DECISION_FIELDS)
    _write_csv(boundary_path, boundary_rows, BOUNDARY_FIELDS)
    _write_json(gate_path, gate)
    _write_markdown(
        markdown_path,
        gate=gate,
        route_rows=route_rows,
        decision_rows=decision_rows,
        boundary_rows=boundary_rows,
    )

    manifest = {
        "phase": 147,
        "objective": "literature_guided_model_roadmap_no_training",
        "inputs": {key: _display_path(path, root) for key, path in resolved.items()},
        "outputs": {
            "route_audit_table": _display_path(route_path, root),
            "decision_table": _display_path(decision_path, root),
            "boundary_table": _display_path(boundary_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown_summary": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "route_rows": len(route_rows),
            "decision_rows": len(decision_rows),
            "boundary_rows": len(boundary_rows),
            "phase148_allowed_decisions": sum(1 for row in decision_rows if row["phase148_allowed"]),
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
