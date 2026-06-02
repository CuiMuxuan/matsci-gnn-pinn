#!/usr/bin/env python3
"""Build the Phase 95 PFHub-style local/no-training physics design gate.

Phase 95 turns the Phase 94 PFHub candidate into a concrete design contract.
It selects a small public-physics benchmark route, defines candidate mechanisms,
baselines, metrics, leakage controls, and stop conditions, and decides whether a
future local smoke gate may be implemented. It does not run A100 training.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROTOCOL_FIELDS = (
    "protocol_id",
    "component",
    "requirement",
    "artifact",
    "status",
    "pass_condition",
    "stop_condition",
)

MECHANISM_FIELDS = (
    "mechanism_id",
    "mechanism",
    "physics_role",
    "expected_signal",
    "baseline_comparator",
    "risk",
    "status",
)

METRIC_FIELDS = (
    "metric_id",
    "metric",
    "scope",
    "pass_rule",
    "regression_guard",
    "paper_use",
)

BASELINE_FIELDS = (
    "baseline_id",
    "baseline",
    "purpose",
    "required_for_gate",
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
    phase94 = root / "docs/results/phase94_external_registered_target_candidate_gate"
    return {
        "phase94_gate": phase94 / "phase94_external_registered_target_candidate_gate.json",
        "phase94_candidate_triage": phase94 / "phase94_external_candidate_triage.csv",
        "phase94_design_queue": phase94 / "phase94_local_design_queue.csv",
    }


def _phase94_candidate(rows: list[dict[str, str]]) -> dict[str, str]:
    for row in rows:
        if row.get("candidate_id") == "P94-CAND-PFHUB-PINN":
            return row
    raise ValueError("Missing P94-CAND-PFHUB-PINN candidate row")


def build_candidate_design(phase94_candidate: dict[str, str]) -> dict[str, Any]:
    return {
        "candidate_id": "phase95_pfhub_local_physics_v1",
        "source_candidate": phase94_candidate.get("candidate_id"),
        "source_url": phase94_candidate.get("source_url"),
        "candidate_family": "PFHub-style local physics benchmark precheck",
        "selected_benchmark_style": "PFHub benchmark-derived Allen-Cahn / heat-diffusion surrogate",
        "paper_facing_hypothesis": (
            "A lightweight physics benchmark can test whether Green's-function features, "
            "Bayesian/adaptive sampling, or meta-learning adaptation have a measurable "
            "mechanistic signal before any AM-Bench or A100 validation."
        ),
        "not_a_claim": [
            "not AM-Bench performance evidence",
            "not submission-ready venue alignment",
            "not a replacement for Phase 55/60/74 spot_size floor",
            "not permission for A100 broad12/broad21 training",
        ],
        "candidate_mechanisms": [
            "fixed Green's-function / heat-kernel feature augmentation",
            "Bayesian active collocation selection with uncertainty and residual proxies",
            "small meta-initialization/adaptation probe over related PDE coefficients",
        ],
        "leakage_controls": [
            "fixed analytic train/validation/test grids before selecting mechanism",
            "validation-only mechanism choice",
            "no test-set selection of acquisition strategy or hyperparameters",
            "all random seeds and generated fields written to manifest",
        ],
        "a100_policy": "A100 training is not allowed from Phase 95; at most Phase 96 local smoke may be opened.",
        "a100_80gb_policy": "A100-SXM4-80GB is not justified by design-gate work.",
    }


def build_protocol_rows() -> list[dict[str, Any]]:
    return [
        {
            "protocol_id": "P95-PROT-001",
            "component": "benchmark_target",
            "requirement": "Define a small analytic PDE target inspired by PFHub-style benchmark practice.",
            "artifact": "candidate_design.json selected_benchmark_style",
            "status": "defined",
            "pass_condition": "target has known PDE residual and fixed train/validation/test grids",
            "stop_condition": "target has no measurable physics signal or no held-out split",
        },
        {
            "protocol_id": "P95-PROT-002",
            "component": "candidate_mechanisms",
            "requirement": "Predeclare Green's-function features, Bayesian/adaptive sampling, and meta-adaptation probes.",
            "artifact": "phase95_mechanism_matrix.csv",
            "status": "defined",
            "pass_condition": "each mechanism maps to one expected metric signal",
            "stop_condition": "mechanism is architecture novelty without a measurable failure mode",
        },
        {
            "protocol_id": "P95-PROT-003",
            "component": "baselines",
            "requirement": "Compare against analytic prior, mean/RBF interpolation, vanilla PINN, and random collocation.",
            "artifact": "phase95_baseline_contract.csv",
            "status": "defined",
            "pass_condition": "every mechanism has a deterministic comparator",
            "stop_condition": "no stronger or simpler baseline is available",
        },
        {
            "protocol_id": "P95-PROT-004",
            "component": "metrics",
            "requirement": "Use global RMSE, residual RMSE, hot/top-gradient region RMSE, calibration if Bayesian, and adaptation steps if meta-learning.",
            "artifact": "phase95_metric_contract.csv",
            "status": "defined",
            "pass_condition": "global metric is non-worse and one predeclared focused metric improves",
            "stop_condition": "focused gain is bought by global collapse",
        },
        {
            "protocol_id": "P95-PROT-005",
            "component": "compute_governance",
            "requirement": "Allow only local/no-training design and possible future local smoke.",
            "artifact": "phase95_pfhub_physics_design_gate.json",
            "status": "defined",
            "pass_condition": "A100 flags remain false",
            "stop_condition": "any path tries to open A100 without a local gate",
        },
    ]


def build_mechanism_rows() -> list[dict[str, Any]]:
    return [
        {
            "mechanism_id": "P95-MECH-001",
            "mechanism": "fixed_green_function_features",
            "physics_role": "inject analytic diffusion/source-distance basis into a small surrogate",
            "expected_signal": "lower residual RMSE and top-gradient RMSE without global RMSE regression",
            "baseline_comparator": "vanilla PINN or RBF interpolation on identical collocation points",
            "risk": "synthetic success may not transfer to AM-Bench without registration",
            "status": "eligible_for_phase96_local_smoke",
        },
        {
            "mechanism_id": "P95-MECH-002",
            "mechanism": "bayesian_adaptive_collocation",
            "physics_role": "select informative collocation points by uncertainty/residual proxies",
            "expected_signal": "same or better RMSE with fewer collocation points and calibrated coverage",
            "baseline_comparator": "random collocation with same budget",
            "risk": "uncertainty can over-focus regions and harm global error, as Phase 75 warned",
            "status": "eligible_for_phase96_local_smoke",
        },
        {
            "mechanism_id": "P95-MECH-003",
            "mechanism": "small_meta_adaptation_probe",
            "physics_role": "test rapid adaptation across related PDE coefficients",
            "expected_signal": "fewer adaptation steps to reach validation threshold",
            "baseline_comparator": "same architecture trained from scratch",
            "risk": "meta-learning may add complexity without AM-Bench relevance",
            "status": "design_only_until_simpler_mechanisms_fail",
        },
    ]


def build_metric_rows() -> list[dict[str, Any]]:
    return [
        {
            "metric_id": "P95-MET-001",
            "metric": "global_rmse",
            "scope": "held-out validation/test grid",
            "pass_rule": "non-worse than the strongest baseline",
            "regression_guard": "any global collapse closes the candidate",
            "paper_use": "gate only, not AM-Bench manuscript evidence",
        },
        {
            "metric_id": "P95-MET-002",
            "metric": "pde_residual_rmse",
            "scope": "collocation grid",
            "pass_rule": "improves over vanilla PINN/RBF comparator",
            "regression_guard": "must not hide worse prediction RMSE",
            "paper_use": "mechanism evidence",
        },
        {
            "metric_id": "P95-MET-003",
            "metric": "hot_or_top_gradient_region_rmse",
            "scope": "predeclared high-value/high-gradient region",
            "pass_rule": "improves while global RMSE is non-worse",
            "regression_guard": "same Phase 75 region-gain/global-collapse guard",
            "paper_use": "focused-mechanism evidence",
        },
        {
            "metric_id": "P95-MET-004",
            "metric": "coverage_or_adaptation_efficiency",
            "scope": "Bayesian or meta-learning subgate only",
            "pass_rule": "coverage within tolerance or fewer adaptation steps",
            "regression_guard": "does not override RMSE guards",
            "paper_use": "secondary mechanism evidence",
        },
    ]


def build_baseline_rows() -> list[dict[str, Any]]:
    return [
        {
            "baseline_id": "P95-BASE-001",
            "baseline": "analytic_or_manufactured_solution_prior",
            "purpose": "sanity-check generated benchmark fields and residuals",
            "required_for_gate": True,
            "status": "required",
        },
        {
            "baseline_id": "P95-BASE-002",
            "baseline": "mean_or_low_order_interpolation",
            "purpose": "simple non-neural comparator",
            "required_for_gate": True,
            "status": "required",
        },
        {
            "baseline_id": "P95-BASE-003",
            "baseline": "vanilla_pinn_same_budget",
            "purpose": "physics-informed neural comparator without candidate mechanism",
            "required_for_gate": True,
            "status": "required",
        },
        {
            "baseline_id": "P95-BASE-004",
            "baseline": "random_collocation_same_budget",
            "purpose": "adaptive-sampling comparator",
            "required_for_gate": True,
            "status": "required_for_bayesian_sampling",
        },
    ]


def build_gate(
    *,
    phase94_gate: dict[str, Any],
    phase94_candidate: dict[str, str],
    protocol_rows: list[dict[str, Any]],
    mechanism_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    phase94_allows = bool(phase94_gate.get("phase95_local_gate_allowed"))
    candidate_open = phase94_candidate.get("status") == "open_for_local_design_gate"
    complete = all(row["status"] == "defined" for row in protocol_rows)
    has_smoke_mechanism = any(
        row["status"] == "eligible_for_phase96_local_smoke" for row in mechanism_rows
    )
    required_baselines = [row for row in baseline_rows if row["required_for_gate"]]
    if phase94_allows and candidate_open and complete and has_smoke_mechanism and required_baselines:
        status = "local_design_ready_no_a100"
        next_action = "implement Phase 96 local smoke only; do not start A100 training"
        phase96_allowed = True
    else:
        status = "blocked_design_incomplete"
        next_action = "repair missing design, mechanism, metric, or baseline contract"
        phase96_allowed = False
    return {
        "status": status,
        "source_phase94_status": phase94_gate.get("status"),
        "source_candidate": phase94_candidate.get("candidate_id"),
        "phase96_local_smoke_allowed": phase96_allowed,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "submission_ready": False,
        "protocol_rows": len(protocol_rows),
        "mechanism_rows": len(mechanism_rows),
        "metric_rows": len(metric_rows),
        "baseline_rows": len(baseline_rows),
        "eligible_smoke_mechanisms": sum(
            1 for row in mechanism_rows if row["status"] == "eligible_for_phase96_local_smoke"
        ),
        "next_action": next_action,
        "required_before_a100_training": [
            "Phase 96 local smoke results",
            "baseline deltas",
            "non-worse global metric",
            "focused metric improvement",
            "explicit AM-Bench transfer gate before broad12/broad21 validation",
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
    mechanism_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "# Phase 95 PFHub-Style Physics Design Gate",
            "",
            "## Gate Decision",
            "",
            f"Status: `{gate['status']}`.",
            f"Phase 96 local smoke allowed: `{str(gate['phase96_local_smoke_allowed']).lower()}`.",
            f"A100 training allowed now: `{str(gate['a100_training_allowed_now']).lower()}`.",
            f"A100-SXM4-80GB request now: `{str(gate['a100_80gb_request_now']).lower()}`.",
            "",
            "Phase 95 is a design gate only. It prepares a local smoke protocol and cannot support paper claims by itself.",
            "",
            "## Mechanism Matrix",
            "",
            _markdown_table(
                mechanism_rows,
                [
                    ("mechanism_id", "Mechanism"),
                    ("mechanism", "Name"),
                    ("expected_signal", "Expected signal"),
                    ("status", "Status"),
                ],
            ),
            "",
            "## Metric Contract",
            "",
            _markdown_table(
                metric_rows,
                [
                    ("metric_id", "Metric"),
                    ("metric", "Name"),
                    ("pass_rule", "Pass rule"),
                    ("regression_guard", "Guard"),
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

    phase94_gate = _read_json(resolved["phase94_gate"])
    phase94_candidates = _read_csv(resolved["phase94_candidate_triage"])
    phase94_queue = _read_csv(resolved["phase94_design_queue"])
    phase94_candidate = _phase94_candidate(phase94_candidates)

    candidate_design = build_candidate_design(phase94_candidate)
    protocol_rows = build_protocol_rows()
    mechanism_rows = build_mechanism_rows()
    metric_rows = build_metric_rows()
    baseline_rows = build_baseline_rows()
    gate = build_gate(
        phase94_gate=phase94_gate,
        phase94_candidate=phase94_candidate,
        protocol_rows=protocol_rows,
        mechanism_rows=mechanism_rows,
        metric_rows=metric_rows,
        baseline_rows=baseline_rows,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    design_path = output_dir / "phase95_candidate_design.json"
    protocol_path = output_dir / "phase95_design_protocol.csv"
    mechanism_path = output_dir / "phase95_mechanism_matrix.csv"
    metric_path = output_dir / "phase95_metric_contract.csv"
    baseline_path = output_dir / "phase95_baseline_contract.csv"
    gate_path = output_dir / "phase95_pfhub_physics_design_gate.json"
    markdown_path = output_dir / "phase95_pfhub_physics_design_gate.md"
    manifest_path = output_dir / "phase95_pfhub_physics_design_gate_manifest.json"

    _write_json(design_path, candidate_design)
    _write_csv(protocol_path, protocol_rows, PROTOCOL_FIELDS)
    _write_csv(mechanism_path, mechanism_rows, MECHANISM_FIELDS)
    _write_csv(metric_path, metric_rows, METRIC_FIELDS)
    _write_csv(baseline_path, baseline_rows, BASELINE_FIELDS)
    _write_json(gate_path, gate)
    markdown_path.write_text(build_markdown(gate, mechanism_rows, metric_rows), encoding="utf-8")

    manifest = {
        "phase": 95,
        "objective": "pfhub_style_local_physics_design_gate",
        "inputs": {key: _display_path(path, root) for key, path in sorted(resolved.items())},
        "outputs": {
            "candidate_design": _display_path(design_path, root),
            "design_protocol": _display_path(protocol_path, root),
            "mechanism_matrix": _display_path(mechanism_path, root),
            "metric_contract": _display_path(metric_path, root),
            "baseline_contract": _display_path(baseline_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "phase94_queue_rows": len(phase94_queue),
            "protocol_rows": len(protocol_rows),
            "mechanism_rows": len(mechanism_rows),
            "metric_rows": len(metric_rows),
            "baseline_rows": len(baseline_rows),
        },
        "gate": gate,
        "phase94_gate": {
            "status": phase94_gate.get("status"),
            "preferred_next_candidate": phase94_gate.get("preferred_next_candidate"),
            "phase95_local_gate_allowed": phase94_gate.get("phase95_local_gate_allowed"),
            "a100_training_allowed_now": phase94_gate.get("a100_training_allowed_now"),
        },
        "source_candidate": phase94_candidate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase95_pfhub_physics_design_gate"),
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
