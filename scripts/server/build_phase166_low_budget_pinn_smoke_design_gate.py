#!/usr/bin/env python3
"""Build Phase 166 low-budget PINN smoke design gate.

Phase 166 is a design package only. It converts the Phase 164 calibrated
Bayesian inverse-heat positive and the Phase 165 failure-informed sampler
positive into a bounded local smoke protocol. No PINN training is executed here.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("docs/results/phase166_low_budget_pinn_smoke_design_gate")

PHASE_INPUTS = {
    "phase164_gate": Path(
        "docs/results/phase164_synthetic_bayesian_inverse_heat_identifiability_gate/"
        "phase164_synthetic_bayesian_inverse_heat_identifiability_gate.json"
    ),
    "phase165_gate": Path(
        "docs/results/phase165_adaptive_residual_sampler_gate/"
        "phase165_adaptive_residual_sampler_gate.json"
    ),
    "phase165_metric_table": Path(
        "docs/results/phase165_adaptive_residual_sampler_gate/"
        "phase165_sampler_metric_table.csv"
    ),
}

DESIGN_FIELDS = (
    "design_id",
    "component",
    "decision",
    "bound",
    "rationale",
    "opens_training_now",
)

CONTROL_FIELDS = (
    "control_id",
    "control_name",
    "role",
    "required_metric",
    "promotion_requirement",
)

COMPUTE_FIELDS = (
    "resource_id",
    "resource",
    "limit",
    "allowed_now",
    "escalation_rule",
)

REFERENCE_FIELDS = (
    "reference_id",
    "title",
    "year",
    "doi",
    "source_url",
    "verification_source",
    "route_implication",
    "phase166_decision",
)

RISK_FIELDS = (
    "risk_id",
    "risk",
    "guard",
    "closure_action",
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


def build_design_rows(*, phase165_gate: dict[str, Any]) -> list[dict[str, Any]]:
    selected_sampler = phase165_gate.get("selected_sampler", "unknown")
    return [
        {
            "design_id": "P166-DESIGN-001",
            "component": "task_scope",
            "decision": "synthetic_1d_moving_heat_source_only",
            "bound": "no AM-Bench, no NIST AMMT, no external raw data",
            "rationale": "The Phase 164/165 positives are synthetic and must not be promoted to AM data.",
            "opens_training_now": False,
        },
        {
            "design_id": "P166-DESIGN-002",
            "component": "model_budget",
            "decision": "tiny_mlp_pinn_2_hidden_layers_width_32",
            "bound": "max 3 seeds, max 1500 optimizer steps, max 512 collocation points per step",
            "rationale": "Smoke should test mechanism plausibility, not scale.",
            "opens_training_now": False,
        },
        {
            "design_id": "P166-DESIGN-003",
            "component": "sampler_variants",
            "decision": f"compare_uniform_grid_control_vs_{selected_sampler}",
            "bound": "equal collocation point budget and identical data sensors",
            "rationale": "Isolates the Phase 165 sampler contribution.",
            "opens_training_now": False,
        },
        {
            "design_id": "P166-DESIGN-004",
            "component": "bayesian_inverse_variant",
            "decision": "use_calibrated_grid_posterior_as_parameter_prior_diagnostic",
            "bound": "posterior informs reporting/calibration only; no full Bayesian neural net",
            "rationale": "Matches the Phase 164 calibrated hidden-parameter result without expensive BNN training.",
            "opens_training_now": False,
        },
        {
            "design_id": "P166-DESIGN-005",
            "component": "metrics",
            "decision": "validation_only_selection_rmse_residual_hot_gradient_parameter_error",
            "bound": "test metrics reported once; no hyperparameter tuning on test",
            "rationale": "Preserves the gate discipline used in earlier phases.",
            "opens_training_now": False,
        },
        {
            "design_id": "P166-DESIGN-006",
            "component": "promotion_rule",
            "decision": "promote_only_if_failure_sampler_beats_uniform_and_data_only_controls",
            "bound": ">=0.03 validation relative RMSE gain, no test reversal >1.05, parameter error non-worse",
            "rationale": "Prevents sampler-only coverage from being misread as model performance.",
            "opens_training_now": False,
        },
    ]


def build_control_rows() -> list[dict[str, Any]]:
    return [
        {
            "control_id": "P166-CTRL-001",
            "control_name": "train_mean_or_sensor_interpolation",
            "role": "non-neural target baseline",
            "required_metric": "temperature RMSE and hot/gradient q90 RMSE",
            "promotion_requirement": "tiny PINN must beat this on validation and avoid test reversal",
        },
        {
            "control_id": "P166-CTRL-002",
            "control_name": "data_only_tiny_mlp_no_residual",
            "role": "tests whether physics residual adds value",
            "required_metric": "temperature RMSE, residual RMSE, parameter error",
            "promotion_requirement": "PINN residual variants must beat or match data-only MLP",
        },
        {
            "control_id": "P166-CTRL-003",
            "control_name": "uniform_grid_pinn",
            "role": "equal-budget sampler control",
            "required_metric": "same optimizer steps and collocation budget",
            "promotion_requirement": "failure-informed sampler must beat uniform on validation",
        },
        {
            "control_id": "P166-CTRL-004",
            "control_name": "wrong_parameter_prior_control",
            "role": "tests hidden-parameter interpretability",
            "required_metric": "diffusivity/source-width error and prediction RMSE",
            "promotion_requirement": "calibrated prior route must not be worse than wrong-prior control",
        },
        {
            "control_id": "P166-CTRL-005",
            "control_name": "seed_stability_control",
            "role": "prevents single-seed promotion",
            "required_metric": "three seeds when the smoke is cheap enough",
            "promotion_requirement": "mean gain positive and worst seed not worse than uniform by >5%",
        },
    ]


def build_compute_rows() -> list[dict[str, Any]]:
    return [
        {
            "resource_id": "P166-COMPUTE-001",
            "resource": "local_cpu_or_existing_local_gpu",
            "limit": "preferred for Phase 167 smoke if torch is available",
            "allowed_now": True,
            "escalation_rule": "use A800 only if local torch/runtime blocks the tiny synthetic smoke",
        },
        {
            "resource_id": "P166-COMPUTE-002",
            "resource": "A800_40GB",
            "limit": "allowed only for reproduction or if local environment cannot import torch",
            "allowed_now": False,
            "escalation_rule": "still no AM-Bench training and no large sweeps",
        },
        {
            "resource_id": "P166-COMPUTE-003",
            "resource": "A100_SXM4_80GB",
            "limit": "not justified",
            "allowed_now": False,
            "escalation_rule": "request only after a seed-positive branch hits measured 40GB memory/runtime blockage",
        },
    ]


def build_reference_rows() -> list[dict[str, Any]]:
    return [
        {
            "reference_id": "P166-REF-001",
            "title": "B-PINNs: Bayesian physics-informed neural networks for forward and inverse PDE problems with noisy data",
            "year": 2020,
            "doi": "10.1016/j.jcp.2020.109913",
            "source_url": "https://www.osti.gov/pages/biblio/2282008",
            "verification_source": "OSTI record and DOI metadata",
            "route_implication": "Bayesian treatment is appropriate for noisy inverse PDE settings, but full BNN/HMC is heavier than a smoke gate.",
            "phase166_decision": "Use calibrated grid-posterior diagnostics first; do not claim Bayesian PINN training.",
        },
        {
            "reference_id": "P166-REF-002",
            "title": "Efficient Bayesian Physics Informed Neural Networks for inverse problems via Ensemble Kalman Inversion",
            "year": 2024,
            "doi": "10.1016/j.jcp.2024.113006",
            "source_url": "https://dl.acm.org/doi/10.1016/j.jcp.2024.113006",
            "verification_source": "Publisher/Crossref-indexed DOI metadata",
            "route_implication": "Efficient Bayesian inverse-PINN variants exist and can become a later lightweight UQ branch.",
            "phase166_decision": "Defer EKI-style Bayesian neural inference until a tiny deterministic PINN smoke passes.",
        },
        {
            "reference_id": "P166-REF-003",
            "title": "A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks",
            "year": 2023,
            "doi": "10.1016/j.cma.2022.115671",
            "source_url": "https://github.com/lu-group/pinn-sampling",
            "verification_source": "arXiv record and official code repository citation",
            "route_implication": "RAD/RAR-D-style sampling is a valid comparator for residual-point efficiency.",
            "phase166_decision": "Keep uniform/jittered controls and failure-informed sampler as equal-budget variants.",
        },
        {
            "reference_id": "P166-REF-004",
            "title": "Failure-Informed Adaptive Sampling for PINNs",
            "year": 2023,
            "doi": "10.1137/22M1527763",
            "source_url": "https://epubs.siam.org/doi/abs/10.1137/22M1527763",
            "verification_source": "SIAM DOI landing page",
            "route_implication": "Failure probability/error-indicator sampling supports Phase 165's sampler choice.",
            "phase166_decision": "Require model-error improvement, not sampler coverage alone, before promotion.",
        },
        {
            "reference_id": "P166-REF-005",
            "title": "Machine learning for metal additive manufacturing: predicting temperature and melt pool fluid dynamics using physics-informed neural networks",
            "year": 2021,
            "doi": "10.1007/s00466-020-01952-9",
            "source_url": "https://link.springer.com/article/10.1007/s00466-020-01952-9",
            "verification_source": "University publication record with Springer DOI",
            "route_implication": "AM thermal PINNs are plausible, but AM-Bench claims require separate guarded data evidence.",
            "phase166_decision": "Run synthetic inverse-heat smoke before returning to AM-Bench or NIST AMMT.",
        },
        {
            "reference_id": "P166-REF-006",
            "title": "Single-track thermal analysis of laser powder bed fusion process: Parametric solution through physics-informed neural networks",
            "year": 2023,
            "doi": "10.1016/j.cma.2023.116019",
            "source_url": "https://www.research-collection.ethz.ch/handle/20.500.11850/607001",
            "verification_source": "ETH repository and DOI metadata",
            "route_implication": "Parametric heat PINNs align with the user's hidden-physics/parameter-inference goal.",
            "phase166_decision": "Track diffusivity/source-width parameter error as a smoke metric.",
        },
        {
            "reference_id": "P166-REF-007",
            "title": "Physics-informed graph neural Galerkin networks: A unified framework for solving PDE-governed forward and inverse problems",
            "year": 2022,
            "doi": "10.1016/j.cma.2021.114502",
            "source_url": "https://par.nsf.gov/biblio/10338460-physics-informed-graph-neural-galerkin-networks-unified-framework-solving-pde-governed-forward-inverse-problems",
            "verification_source": "NSF public-access manuscript and DOI metadata",
            "route_implication": "GCN/graph discretizations are relevant for non-grid PDE domains.",
            "phase166_decision": "Defer graph-PINN training because current project path-graph routes were already closed by earlier guards.",
        },
        {
            "reference_id": "P166-REF-008",
            "title": "Physics-informed machine learning-based real-time long-horizon temperature fields prediction in metallic additive manufacturing",
            "year": 2025,
            "doi": "10.1038/s44172-025-00501-7",
            "source_url": "https://www.nature.com/articles/s44172-025-00501-7",
            "verification_source": "Nature Communications Engineering article page",
            "route_implication": "Hybrid recurrent/CNN physics-informed AM models support future dense-field routes.",
            "phase166_decision": "Do not open neural-operator/CNN dense training until leakage-safe dense targets are reopened.",
        },
    ]


def build_risk_rows() -> list[dict[str, Any]]:
    return [
        {
            "risk_id": "P166-RISK-001",
            "risk": "synthetic overfitting",
            "guard": "test_shifted scenario and wrong-prior control",
            "closure_action": "close as synthetic diagnostic if shifted test reverses",
        },
        {
            "risk_id": "P166-RISK-002",
            "risk": "sampler advantage without model advantage",
            "guard": "uniform-grid PINN and data-only MLP controls",
            "closure_action": "do not promote if coverage gain does not reduce validation error",
        },
        {
            "risk_id": "P166-RISK-003",
            "risk": "Bayesian language overclaim",
            "guard": "calibrated grid posterior only; no full BNN claim",
            "closure_action": "write as posterior diagnostic, not Bayesian PINN success",
        },
        {
            "risk_id": "P166-RISK-004",
            "risk": "compute creep",
            "guard": "max steps/width/seeds and no AM-Bench raw data",
            "closure_action": "stop and redesign instead of scaling",
        },
    ]


def build_gate(
    *,
    phase164_gate: dict[str, Any],
    phase165_gate: dict[str, Any],
    metric_rows: list[dict[str, str]],
    design_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    compute_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    risk_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    phase164_ready = (
        phase164_gate.get("status")
        == "phase164_synthetic_bayesian_inverse_heat_identifiability_ready_phase165_sampler_gate"
    )
    phase165_ready = (
        phase165_gate.get("status")
        == "phase165_adaptive_residual_sampler_ready_low_budget_pinn_smoke_design"
        and _is_true(phase165_gate.get("phase166_low_budget_pinn_smoke_design_allowed"))
    )
    sampler_positive = float(phase165_gate.get("validation_score_gain_vs_best_control", 0.0)) >= 0.08
    shifted_positive = float(phase165_gate.get("test_score_gain_vs_best_control", 0.0)) >= 0.05
    boundary_ok = (
        float(phase165_gate.get("selected_validation_boundary_fraction", 0.0)) >= 0.08
        and float(phase165_gate.get("selected_test_boundary_fraction", 0.0)) >= 0.08
    )
    enough_controls = len(control_rows) >= 5
    enough_references = len(reference_rows) >= 8
    compute_bounded = any(row["resource"] == "local_cpu_or_existing_local_gpu" for row in compute_rows)
    no_a100_80gb = all(not _is_true(row["allowed_now"]) for row in compute_rows if row["resource"] == "A100_SXM4_80GB")
    no_training_now = all(_is_false(row["opens_training_now"]) for row in design_rows)
    complete = (
        phase164_ready
        and phase165_ready
        and sampler_positive
        and shifted_positive
        and boundary_ok
        and enough_controls
        and enough_references
        and compute_bounded
        and no_a100_80gb
        and no_training_now
        and len(metric_rows) >= 10
        and len(risk_rows) >= 4
    )
    blockers: list[str] = []
    if not phase164_ready:
        blockers.append("phase164_gate_not_ready")
    if not phase165_ready:
        blockers.append("phase165_gate_not_ready")
    if not sampler_positive:
        blockers.append("sampler_validation_gain_guard")
    if not shifted_positive:
        blockers.append("sampler_shifted_test_gain_guard")
    if not boundary_ok:
        blockers.append("sampler_boundary_coverage_guard")
    if not enough_controls:
        blockers.append("missing_required_controls")
    if not enough_references:
        blockers.append("missing_verified_route_references")
    if not compute_bounded:
        blockers.append("missing_bounded_compute_plan")
    if not no_a100_80gb:
        blockers.append("a100_80gb_request_not_allowed")
    if not no_training_now:
        blockers.append("design_attempted_training_now")
    return {
        "status": (
            "phase166_low_budget_pinn_smoke_design_ready_phase167_local_smoke"
            if complete
            else "phase166_low_budget_pinn_smoke_design_incomplete"
        ),
        "phase167_local_low_budget_pinn_smoke_allowed": bool(complete),
        "selected_sampler": phase165_gate.get("selected_sampler"),
        "required_control_rows": len(control_rows),
        "reference_rows": len(reference_rows),
        "design_rows": len(design_rows),
        "risk_rows": len(risk_rows),
        "blocking_audits": blockers,
        "phase166_model_mechanism_allowed": False,
        "phase166_model_training_allowed": False,
        "phase167_training_allowed_now": False,
        "bayesian_pinn_training_allowed_now": False,
        "adaptive_sampling_training_allowed_now": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": (
            "enter Phase 167 local low-budget synthetic PINN smoke only if local torch/runtime is suitable"
            if complete
            else "repair the smoke design before any training"
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
    design_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    compute_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    risk_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Phase 166 Low-Budget PINN Smoke Design Gate",
        "",
        "## Gate",
        f"- Status: `{gate['status']}`",
        f"- Phase 167 local low-budget PINN smoke allowed: `{_csv_value(gate['phase167_local_low_budget_pinn_smoke_allowed'])}`",
        f"- Phase 166 model training allowed: `{_csv_value(gate['phase166_model_training_allowed'])}`",
        f"- Phase 167 training allowed now: `{_csv_value(gate['phase167_training_allowed_now'])}`",
        f"- A100 training allowed now: `{_csv_value(gate['a100_training_allowed_now'])}`",
        f"- A100-SXM4-80GB request now: `{_csv_value(gate['a100_80gb_request_now'])}`",
        "",
        "## Interpretation",
        (
            "This is a design gate only. It permits a later local synthetic smoke protocol "
            "if the design is complete, but it does not execute training or support AM-Bench claims."
        ),
        "",
        "## Design",
        *_markdown_table(design_rows, DESIGN_FIELDS),
        "",
        "## Controls",
        *_markdown_table(control_rows, CONTROL_FIELDS),
        "",
        "## Compute",
        *_markdown_table(compute_rows, COMPUTE_FIELDS),
        "",
        "## Route References",
        *_markdown_table(reference_rows, REFERENCE_FIELDS),
        "",
        "## Risks",
        *_markdown_table(risk_rows, RISK_FIELDS),
        "",
    ]
    return "\n".join(lines)


def build_package(*, root: Path, output_dir: Path, phase_inputs: dict[str, Path]) -> dict[str, Any]:
    root = root.resolve()
    output_dir = output_dir if output_dir.is_absolute() else root / output_dir
    resolved = {
        name: path if path.is_absolute() else root / path for name, path in phase_inputs.items()
    }
    phase164_gate = _read_json(resolved["phase164_gate"])
    phase165_gate = _read_json(resolved["phase165_gate"])
    metric_rows = _read_csv(resolved["phase165_metric_table"])
    design_rows = build_design_rows(phase165_gate=phase165_gate)
    control_rows = build_control_rows()
    compute_rows = build_compute_rows()
    reference_rows = build_reference_rows()
    risk_rows = build_risk_rows()
    gate = build_gate(
        phase164_gate=phase164_gate,
        phase165_gate=phase165_gate,
        metric_rows=metric_rows,
        design_rows=design_rows,
        control_rows=control_rows,
        compute_rows=compute_rows,
        reference_rows=reference_rows,
        risk_rows=risk_rows,
    )

    design_path = output_dir / "phase166_smoke_design_table.csv"
    control_path = output_dir / "phase166_control_table.csv"
    compute_path = output_dir / "phase166_compute_envelope_table.csv"
    reference_path = output_dir / "phase166_route_reference_table.csv"
    risk_path = output_dir / "phase166_risk_table.csv"
    gate_path = output_dir / "phase166_low_budget_pinn_smoke_design_gate.json"
    markdown_path = output_dir / "phase166_low_budget_pinn_smoke_design_gate.md"
    manifest_path = output_dir / "phase166_low_budget_pinn_smoke_design_manifest.json"

    _write_csv(design_path, design_rows, DESIGN_FIELDS)
    _write_csv(control_path, control_rows, CONTROL_FIELDS)
    _write_csv(compute_path, compute_rows, COMPUTE_FIELDS)
    _write_csv(reference_path, reference_rows, REFERENCE_FIELDS)
    _write_csv(risk_path, risk_rows, RISK_FIELDS)
    _write_json(gate_path, gate)
    with markdown_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            build_markdown(
                gate=gate,
                design_rows=design_rows,
                control_rows=control_rows,
                compute_rows=compute_rows,
                reference_rows=reference_rows,
                risk_rows=risk_rows,
            )
        )

    manifest = {
        "phase": 166,
        "description": "low-budget synthetic PINN smoke design gate",
        "inputs": {name: _display_path(path, root) for name, path in resolved.items()},
        "outputs": {
            "design_table": _display_path(design_path, root),
            "control_table": _display_path(control_path, root),
            "compute_envelope_table": _display_path(compute_path, root),
            "route_reference_table": _display_path(reference_path, root),
            "risk_table": _display_path(risk_path, root),
            "gate": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "design_rows": len(design_rows),
            "control_rows": len(control_rows),
            "compute_rows": len(compute_rows),
            "reference_rows": len(reference_rows),
            "risk_rows": len(risk_rows),
            "phase165_metric_rows": len(metric_rows),
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
