#!/usr/bin/env python3
"""Build Phase 168 hidden-source/closure redesign gate.

Phase 168 is a no-training redesign package. It consumes the Phase 167 closure
of the failure-informed sampler PINN smoke and selects a different next route:
hidden-source / closure identifiability before any new PINN training.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("docs/results/phase168_hidden_source_closure_redesign_gate")

PHASE_INPUTS = {
    "phase167_gate": Path(
        "docs/results/phase167_low_budget_pinn_smoke/"
        "phase167_low_budget_pinn_smoke_gate.json"
    ),
    "phase167_summary_table": Path(
        "docs/results/phase167_low_budget_pinn_smoke/"
        "phase167_variant_summary_table.csv"
    ),
    "phase166_reference_table": Path(
        "docs/results/phase166_low_budget_pinn_smoke_design_gate/"
        "phase166_route_reference_table.csv"
    ),
}

EVIDENCE_FIELDS = (
    "evidence_id",
    "source_phase",
    "finding",
    "metric_anchor",
    "route_consequence",
)

ROUTE_FIELDS = (
    "route_id",
    "route_name",
    "mechanism",
    "decision",
    "why_after_phase167",
    "opens_training_now",
    "opens_a100_now",
)

DESIGN_FIELDS = (
    "design_id",
    "component",
    "phase169_requirement",
    "control_or_guard",
    "claim_boundary",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _stable(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 10)
    if isinstance(value, dict):
        return {key: _stable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stable(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(_stable(payload), indent=2, sort_keys=True) + "\n")


def _csv_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{round(value, 10):.10g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_stable(value), sort_keys=True)
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def _display_path(path: Path, root: Path | None = None) -> str:
    if root is None:
        return str(path).replace("\\", "/")
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _is_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def _is_false(value: Any) -> bool:
    if value is False or value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "0", "false", "none", "no"}
    return False


def _summary_row(
    summary_rows: list[dict[str, str]],
    variant_id: str,
    split: str,
) -> dict[str, str]:
    for row in summary_rows:
        if row.get("variant_id") == variant_id and row.get("split") == split:
            return row
    return {}


def build_evidence_rows(
    *,
    phase167_gate: dict[str, Any],
    phase167_summary_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    uniform_val = _summary_row(phase167_summary_rows, "uniform_grid_pinn", "val")
    adaptive_val = _summary_row(
        phase167_summary_rows,
        "failure_informed_hot_gradient_pinn",
        "val",
    )
    wrong_prior_val = _summary_row(
        phase167_summary_rows,
        "wrong_prior_failure_sampler_control",
        "val",
    )
    data_only_val = _summary_row(
        phase167_summary_rows,
        "data_only_tiny_mlp_no_residual",
        "val",
    )
    return [
        {
            "evidence_id": "P168-EVID-001",
            "source_phase": "phase167",
            "finding": "validation_selected_uniform_control",
            "metric_anchor": (
                f"selected={phase167_gate.get('selected_variant')}; "
                f"best_control={phase167_gate.get('best_control_variant')}"
            ),
            "route_consequence": "do_not_continue_same_failure_sampler_pinn",
        },
        {
            "evidence_id": "P168-EVID-002",
            "source_phase": "phase167",
            "finding": "adaptive_sampler_model_score_worse_than_uniform",
            "metric_anchor": (
                "adaptive_val_score="
                f"{adaptive_val.get('selection_score_mean', '')}; "
                "uniform_val_score="
                f"{uniform_val.get('selection_score_mean', '')}"
            ),
            "route_consequence": "sampler_coverage_alone_is_not_model_mechanism",
        },
        {
            "evidence_id": "P168-EVID-003",
            "source_phase": "phase167",
            "finding": "wrong_source_prior_control_failed_strongly",
            "metric_anchor": (
                "wrong_prior_val_score="
                f"{wrong_prior_val.get('selection_score_mean', '')}; "
                "data_only_val_score="
                f"{data_only_val.get('selection_score_mean', '')}"
            ),
            "route_consequence": "hidden_source_or_closure_identifiability_is_the_next_physical_pain_point",
        },
        {
            "evidence_id": "P168-EVID-004",
            "source_phase": "phase166_literature",
            "finding": "bayesian_inverse_and_parametric_heat_pinn_references_remain_relevant",
            "metric_anchor": "B-PINN, EKI-BPINN, AM thermal PINN, LPBF parametric PINN",
            "route_consequence": "use_for_identifiability_design_not_immediate_training",
        },
    ]


def build_route_rows() -> list[dict[str, Any]]:
    return [
        {
            "route_id": "P168-ROUTE-001",
            "route_name": "hidden_source_closure_identifiability_gate",
            "mechanism": "infer low-dimensional moving-source and residual-closure parameters before training a PINN",
            "decision": "selected_for_phase169_no_training_gate",
            "why_after_phase167": (
                "Phase 167 shows sampler choice alone loses to uniform control, while wrong source prior "
                "is strongly harmful; the physical pain point is source/closure identifiability."
            ),
            "opens_training_now": False,
            "opens_a100_now": False,
        },
        {
            "route_id": "P168-ROUTE-002",
            "route_name": "adaptive_loss_balancing_gate",
            "mechanism": "balance data/residual/source losses after a source/closure candidate is identifiable",
            "decision": "defer_until_phase169_identifiability_passes",
            "why_after_phase167": "loss balancing without a better source/closure target would retune the same failed smoke",
            "opens_training_now": False,
            "opens_a100_now": False,
        },
        {
            "route_id": "P168-ROUTE-003",
            "route_name": "lightweight_bayesian_neural_uq",
            "mechanism": "ensemble or EKI-style Bayesian neural posterior for uncertainty",
            "decision": "defer_until_deterministic_closure_gap_exists",
            "why_after_phase167": "full Bayesian neural inference is not justified before a deterministic mechanism gap",
            "opens_training_now": False,
            "opens_a100_now": False,
        },
        {
            "route_id": "P168-ROUTE-004",
            "route_name": "gcn_or_path_graph_pinn",
            "mechanism": "graph-structured non-grid PDE residual route",
            "decision": "blocked_by_prior_path_graph_guard",
            "why_after_phase167": "Phase 148 already closed current path-contact graph evidence",
            "opens_training_now": False,
            "opens_a100_now": False,
        },
        {
            "route_id": "P168-ROUTE-005",
            "route_name": "cnn_or_neural_operator_dense_route",
            "mechanism": "fixed-grid dense field residual completion",
            "decision": "blocked_by_prior_dense_baseline_guard",
            "why_after_phase167": "Phase 151 found no leakage-safe dense baseline gap",
            "opens_training_now": False,
            "opens_a100_now": False,
        },
    ]


def build_design_rows() -> list[dict[str, Any]]:
    return [
        {
            "design_id": "P168-DESIGN-001",
            "component": "target_physics",
            "phase169_requirement": "recover source center, width, amplitude, diffusivity, and a bounded residual-closure coefficient",
            "control_or_guard": "grid least-squares, wrong-source prior, and flexible non-physical regression controls",
            "claim_boundary": "identifiability only; no trained Bayesian PINN claim",
        },
        {
            "design_id": "P168-DESIGN-002",
            "component": "data_contract",
            "phase169_requirement": "synthetic sparse sensors with train/val/test parameter shifts and fixed random seeds",
            "control_or_guard": "validation-only route selection and shifted test reporting",
            "claim_boundary": "no AM-Bench or NIST AMMT data in Phase 169",
        },
        {
            "design_id": "P168-DESIGN-003",
            "component": "model_contract",
            "phase169_requirement": "no neural training; compare posterior/grid/linearized/strong-baseline estimators",
            "control_or_guard": "candidate must beat best non-neural control on validation and avoid test reversal",
            "claim_boundary": "opens at most a later low-budget mechanism smoke design",
        },
        {
            "design_id": "P168-DESIGN-004",
            "component": "compute_contract",
            "phase169_requirement": "CPU/local or A800 reproduction only for no-training numerical grids",
            "control_or_guard": "a100_training_allowed_now=false and a100_80gb_request_now=false",
            "claim_boundary": "80GB request requires a later seed-positive branch with measured 40GB blockage",
        },
        {
            "design_id": "P168-DESIGN-005",
            "component": "failure_exit",
            "phase169_requirement": "close if source/closure parameters are solved by controls or not identifiable",
            "control_or_guard": "blocking audits must remain explicit",
            "claim_boundary": "do not proceed to adaptive loss, Bayesian neural, GCN, or CNN/operator training after a failed identifiability gate",
        },
    ]


def build_gate(
    *,
    phase167_gate: dict[str, Any],
    phase166_reference_rows: list[dict[str, str]],
    evidence_rows: list[dict[str, Any]],
    route_rows: list[dict[str, Any]],
    design_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    phase167_closed = (
        phase167_gate.get("status")
        == "phase167_low_budget_pinn_smoke_closed_no_stable_model_gain"
        and phase167_gate.get("selected_variant") == "uniform_grid_pinn"
        and not _is_true(phase167_gate.get("phase168_focused_review_allowed"))
    )
    has_sampler_failure_evidence = any(
        row.get("finding") == "adaptive_sampler_model_score_worse_than_uniform"
        for row in evidence_rows
    )
    has_source_prior_evidence = any(
        row.get("finding") == "wrong_source_prior_control_failed_strongly"
        for row in evidence_rows
    )
    selected_routes = [
        row
        for row in route_rows
        if row.get("decision") == "selected_for_phase169_no_training_gate"
    ]
    no_training_now = all(_is_false(row.get("opens_training_now")) for row in route_rows)
    no_a100_now = all(_is_false(row.get("opens_a100_now")) for row in route_rows)
    reference_ok = len(phase166_reference_rows) >= 8
    design_ok = len(design_rows) >= 5
    complete = (
        phase167_closed
        and has_sampler_failure_evidence
        and has_source_prior_evidence
        and len(selected_routes) == 1
        and selected_routes[0].get("route_name") == "hidden_source_closure_identifiability_gate"
        and no_training_now
        and no_a100_now
        and reference_ok
        and design_ok
    )
    blockers: list[str] = []
    if not phase167_closed:
        blockers.append("phase167_not_closed_by_uniform_control")
    if not has_sampler_failure_evidence:
        blockers.append("missing_sampler_failure_evidence")
    if not has_source_prior_evidence:
        blockers.append("missing_source_prior_evidence")
    if len(selected_routes) != 1:
        blockers.append("selected_route_count_guard")
    if selected_routes and selected_routes[0].get("route_name") != "hidden_source_closure_identifiability_gate":
        blockers.append("wrong_selected_route")
    if not no_training_now:
        blockers.append("route_attempted_training_now")
    if not no_a100_now:
        blockers.append("route_attempted_a100_now")
    if not reference_ok:
        blockers.append("insufficient_reference_context")
    if not design_ok:
        blockers.append("incomplete_phase169_design_contract")
    return {
        "status": (
            "phase168_hidden_source_closure_redesign_ready_phase169_identifiability_gate"
            if complete
            else "phase168_hidden_source_closure_redesign_incomplete"
        ),
        "selected_next_route": (
            selected_routes[0]["route_name"] if len(selected_routes) == 1 else None
        ),
        "phase169_hidden_source_closure_identifiability_gate_allowed": bool(complete),
        "phase168_retrain_same_sampler_route_allowed": False,
        "phase168_model_mechanism_allowed": False,
        "phase168_model_training_allowed": False,
        "phase169_training_allowed_now": False,
        "bayesian_pinn_training_allowed_now": False,
        "adaptive_sampling_training_allowed_now": False,
        "gcn_pinn_training_allowed_now": False,
        "cnn_operator_training_allowed_now": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "evidence_rows": len(evidence_rows),
        "route_rows": len(route_rows),
        "design_rows": len(design_rows),
        "reference_rows": len(phase166_reference_rows),
        "blocking_audits": blockers,
        "next_action": (
            "enter Phase 169 no-training hidden-source/closure identifiability gate"
            if complete
            else "repair the redesign package before any new route"
        ),
    }


def _markdown_table(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[str]:
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    body = [
        "| " + " | ".join(_csv_value(row.get(field, "")) for field in fields)
        + " |"
        for row in rows
    ]
    return [header, sep, *body]


def build_markdown(
    *,
    gate: dict[str, Any],
    evidence_rows: list[dict[str, Any]],
    route_rows: list[dict[str, Any]],
    design_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Phase 168 Hidden-Source/Closure Redesign Gate",
        "",
        "## Gate",
        f"- Status: `{gate['status']}`",
        f"- Selected next route: `{gate['selected_next_route']}`",
        f"- Phase 169 identifiability gate allowed: `{_csv_value(gate['phase169_hidden_source_closure_identifiability_gate_allowed'])}`",
        f"- Retrain same sampler route allowed: `{_csv_value(gate['phase168_retrain_same_sampler_route_allowed'])}`",
        f"- Phase 168 model training allowed: `{_csv_value(gate['phase168_model_training_allowed'])}`",
        f"- Phase 169 training allowed now: `{_csv_value(gate['phase169_training_allowed_now'])}`",
        f"- A100 training allowed now: `{_csv_value(gate['a100_training_allowed_now'])}`",
        f"- A100-SXM4-80GB request now: `{_csv_value(gate['a100_80gb_request_now'])}`",
        "",
        "## Interpretation",
        (
            "Phase 168 redirects the PINN route away from sampler retuning. The next "
            "candidate must first prove hidden-source/closure identifiability against "
            "non-neural controls before any neural training is reconsidered."
        ),
        "",
        "## Evidence",
        *_markdown_table(evidence_rows, EVIDENCE_FIELDS),
        "",
        "## Routes",
        *_markdown_table(route_rows, ROUTE_FIELDS),
        "",
        "## Phase 169 Design Contract",
        *_markdown_table(design_rows, DESIGN_FIELDS),
        "",
    ]
    return "\n".join(lines)


def build_package(*, root: Path, output_dir: Path, phase_inputs: dict[str, Path]) -> dict[str, Any]:
    root = root.resolve()
    output_dir = output_dir if output_dir.is_absolute() else root / output_dir
    resolved = {
        name: path if path.is_absolute() else root / path
        for name, path in phase_inputs.items()
    }
    phase167_gate = _read_json(resolved["phase167_gate"])
    phase167_summary_rows = _read_csv(resolved["phase167_summary_table"])
    phase166_reference_rows = _read_csv(resolved["phase166_reference_table"])
    evidence_rows = build_evidence_rows(
        phase167_gate=phase167_gate,
        phase167_summary_rows=phase167_summary_rows,
    )
    route_rows = build_route_rows()
    design_rows = build_design_rows()
    gate = build_gate(
        phase167_gate=phase167_gate,
        phase166_reference_rows=phase166_reference_rows,
        evidence_rows=evidence_rows,
        route_rows=route_rows,
        design_rows=design_rows,
    )

    evidence_path = output_dir / "phase168_evidence_table.csv"
    route_path = output_dir / "phase168_route_redesign_table.csv"
    design_path = output_dir / "phase168_phase169_design_contract_table.csv"
    gate_path = output_dir / "phase168_hidden_source_closure_redesign_gate.json"
    markdown_path = output_dir / "phase168_hidden_source_closure_redesign_gate.md"
    manifest_path = output_dir / "phase168_hidden_source_closure_redesign_manifest.json"

    _write_csv(evidence_path, evidence_rows, EVIDENCE_FIELDS)
    _write_csv(route_path, route_rows, ROUTE_FIELDS)
    _write_csv(design_path, design_rows, DESIGN_FIELDS)
    _write_json(gate_path, gate)
    with markdown_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            build_markdown(
                gate=gate,
                evidence_rows=evidence_rows,
                route_rows=route_rows,
                design_rows=design_rows,
            )
        )

    manifest = {
        "phase": 168,
        "description": "no-training hidden-source/closure mechanism redesign gate",
        "inputs": {name: _display_path(path, root) for name, path in resolved.items()},
        "outputs": {
            "evidence_table": _display_path(evidence_path, root),
            "route_table": _display_path(route_path, root),
            "design_contract_table": _display_path(design_path, root),
            "gate": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "evidence_rows": len(evidence_rows),
            "route_rows": len(route_rows),
            "design_rows": len(design_rows),
            "phase166_reference_rows": len(phase166_reference_rows),
            "phase167_summary_rows": len(phase167_summary_rows),
        },
        "gate": gate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    for name, default in PHASE_INPUTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=default)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase_inputs = {name: getattr(args, name) for name in PHASE_INPUTS}
    manifest = build_package(root=args.root, output_dir=args.output_dir, phase_inputs=phase_inputs)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
