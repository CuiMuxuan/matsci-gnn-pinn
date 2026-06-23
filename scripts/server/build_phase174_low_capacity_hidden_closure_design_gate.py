#!/usr/bin/env python3
"""Build Phase 174 low-capacity hidden-closure mechanism design gate.

Phase 174 is design-only. It converts the Phase 173 synthetic trainable-latent
positive into a bounded low-capacity smoke protocol. It does not train a PINN,
does not read AM-Bench/NIST raw data, and does not request A100-SXM4-80GB.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("docs/results/phase174_low_capacity_hidden_closure_design_gate")

PHASE_INPUTS = {
    "phase173_gate": Path(
        "docs/results/phase173_trainable_hidden_closure_low_budget_smoke/"
        "phase173_trainable_hidden_closure_low_budget_smoke_gate.json"
    ),
    "phase173_variant_summary_table": Path(
        "docs/results/phase173_trainable_hidden_closure_low_budget_smoke/"
        "phase173_variant_summary_table.csv"
    ),
    "phase173_training_audit_table": Path(
        "docs/results/phase173_trainable_hidden_closure_low_budget_smoke/"
        "phase173_training_audit_table.csv"
    ),
}

DESIGN_FIELDS = (
    "design_id",
    "component",
    "decision",
    "bound",
    "phase173_evidence",
    "opens_training_now",
)

CONTROL_FIELDS = (
    "control_id",
    "control_name",
    "role",
    "required_metric",
    "promotion_requirement",
)

METRIC_FIELDS = (
    "metric_id",
    "metric_or_guard",
    "threshold",
    "selection_use",
    "rationale",
)

COMPUTE_FIELDS = (
    "resource_id",
    "resource",
    "allowed_now",
    "limit",
    "escalation_rule",
)

PROMOTION_FIELDS = (
    "rule_id",
    "rule",
    "threshold",
    "failure_action",
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


def _summary_lookup(rows: list[dict[str, str]], variant_id: str, split: str) -> dict[str, str]:
    for row in rows:
        if row.get("variant_id") == variant_id and row.get("split") == split:
            return row
    raise KeyError((variant_id, split))


def build_design_rows(*, phase173_gate: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "design_id": "P174-DESIGN-001",
            "component": "task_scope",
            "decision": "synthetic_sparse_sensor_inverse_heat_only",
            "bound": "no AM-Bench, no NIST AMMT, no external raw data",
            "phase173_evidence": "Phase 173 positive is synthetic and NumPy-only.",
            "opens_training_now": False,
        },
        {
            "design_id": "P174-DESIGN-002",
            "component": "candidate_mechanism",
            "decision": "low_capacity_explicit_latent_closure_head",
            "bound": "two explicit source latents plus one bounded closure head; no free residual field",
            "phase173_evidence": phase173_gate.get("candidate_variant", "unknown"),
            "opens_training_now": False,
        },
        {
            "design_id": "P174-DESIGN-003",
            "component": "initialization",
            "decision": "posterior_warm_start_plus_phase173_latent_solution",
            "bound": "initialization and prior audit only; not Bayesian neural posterior training",
            "phase173_evidence": "Phase 173 beat posterior-only and Phase 171 controls.",
            "opens_training_now": False,
        },
        {
            "design_id": "P174-DESIGN-004",
            "component": "model_capacity",
            "decision": "tiny_residual_free_parameter_head_or_numpy_differentiable_loop",
            "bound": "max 3 seeds, max 800 tiny steps in later smoke; Phase 174 does not train",
            "phase173_evidence": "Phase 173 max rounds per start and function evaluations stayed bounded.",
            "opens_training_now": False,
        },
        {
            "design_id": "P174-DESIGN-005",
            "component": "loss_policy",
            "decision": "data_fit_plus_closure_interpretability_plus_source_prior_guard",
            "bound": "no adaptive sampler retuning, no residual MLP, no test tuning",
            "phase173_evidence": "Phase 173 selected explicit latent route, not sampler route.",
            "opens_training_now": False,
        },
        {
            "design_id": "P174-DESIGN-006",
            "component": "selection_protocol",
            "decision": "validation_only_selection_shifted_test_once",
            "bound": "must beat Phase 173, Phase 171, posterior, grid, and uniform-start controls",
            "phase173_evidence": "Phase 173 seed stability pass rate was 1.0.",
            "opens_training_now": False,
        },
        {
            "design_id": "P174-DESIGN-007",
            "component": "claim_boundary",
            "decision": "may_open_phase175_smoke_but_not_training_now",
            "bound": "all Phase 174 training and A100 locks remain false",
            "phase173_evidence": "Phase 173 opened design only.",
            "opens_training_now": False,
        },
    ]


def build_control_rows() -> list[dict[str, Any]]:
    return [
        {
            "control_id": "P174-CTRL-001",
            "control_name": "phase173_tiny_explicit_latent_hidden_closure_smoke",
            "role": "must beat or explain the current trainable-latent positive",
            "required_metric": "selection score, field RMSE, closure error",
            "promotion_requirement": "future low-capacity smoke must improve validation or close as unnecessary",
        },
        {
            "control_id": "P174-CTRL-002",
            "control_name": "phase171_numpy_closure_head_control",
            "role": "non-trainable mechanism floor",
            "required_metric": "closure_abs_error and selection_score",
            "promotion_requirement": "candidate must preserve Phase 171 closure interpretability gains",
        },
        {
            "control_id": "P174-CTRL-003",
            "control_name": "posterior_only_calibrated_bayesian_no_neural",
            "role": "strong no-neural posterior control",
            "required_metric": "field RMSE, closure error, coverage",
            "promotion_requirement": "candidate must improve validation without degrading coverage",
        },
        {
            "control_id": "P174-CTRL-004",
            "control_name": "grid_least_squares_source_closure_control",
            "role": "non-Bayesian inverse control",
            "required_metric": "field RMSE and closure error",
            "promotion_requirement": "candidate must beat validation score and avoid test reversal",
        },
        {
            "control_id": "P174-CTRL-005",
            "control_name": "no_closure_source_control",
            "role": "closure necessity control",
            "required_metric": "field RMSE and closure error",
            "promotion_requirement": "candidate must show closure term remains necessary",
        },
        {
            "control_id": "P174-CTRL-006",
            "control_name": "wrong_source_prior_control",
            "role": "hidden-source interpretability control",
            "required_metric": "field RMSE and closure error",
            "promotion_requirement": "candidate must not be solved by wrong source prior",
        },
        {
            "control_id": "P174-CTRL-007",
            "control_name": "data_only_tiny_control",
            "role": "tests whether physics/closure adds value",
            "required_metric": "field RMSE, hot q90, gradient q90",
            "promotion_requirement": "candidate must beat data-only validation score",
        },
        {
            "control_id": "P174-CTRL-008",
            "control_name": "uniform_grid_latent_trainable_control",
            "role": "same trainable search budget without posterior/grid starts",
            "required_metric": "selection_score and coverage penalty",
            "promotion_requirement": "candidate must beat or justify this control",
        },
        {
            "control_id": "P174-CTRL-009",
            "control_name": "failure_sampler_retrain_block",
            "role": "prevents repeating Phase 167",
            "required_metric": "must remain non-selected",
            "promotion_requirement": "cannot be the selected route",
        },
        {
            "control_id": "P174-CTRL-010",
            "control_name": "seed_stability_control",
            "role": "prevents single-seed promotion",
            "required_metric": "three seeds when cheap",
            "promotion_requirement": "all seeds pass or gate closes",
        },
    ]


def build_metric_rows() -> list[dict[str, Any]]:
    return [
        {
            "metric_id": "P174-METRIC-001",
            "metric_or_guard": "validation_selection_score",
            "threshold": "future candidate must improve vs Phase 173 and best control",
            "selection_use": "validation only",
            "rationale": "prevents test tuning and controls solve-it artifacts",
        },
        {
            "metric_id": "P174-METRIC-002",
            "metric_or_guard": "field_rmse",
            "threshold": "must not trade closure gain for large field degradation",
            "selection_use": "validation/test audit",
            "rationale": "keeps the route predictive, not only interpretable",
        },
        {
            "metric_id": "P174-METRIC-003",
            "metric_or_guard": "closure_abs_error",
            "threshold": "must improve or match Phase 173 within tolerance",
            "selection_use": "validation/test audit",
            "rationale": "preserves hidden-closure interpretability",
        },
        {
            "metric_id": "P174-METRIC-004",
            "metric_or_guard": "coverage90_mean",
            "threshold": "must remain in [0.65, 1.0] if intervals are reported",
            "selection_use": "validation/test audit",
            "rationale": "keeps calibrated inference language bounded",
        },
        {
            "metric_id": "P174-METRIC-005",
            "metric_or_guard": "test_reversal_ratio",
            "threshold": "future shifted-test ratio <= 1.02 vs best control",
            "selection_use": "test once after validation selection",
            "rationale": "closes unstable split behavior",
        },
        {
            "metric_id": "P174-METRIC-006",
            "metric_or_guard": "hot_q90_gradient_q90_rmse",
            "threshold": "must not degrade both region metrics",
            "selection_use": "validation/test audit",
            "rationale": "retains thermal hot/gradient relevance",
        },
        {
            "metric_id": "P174-METRIC-007",
            "metric_or_guard": "budget_and_seed_stability",
            "threshold": "max 3 seeds and bounded tiny steps; all seeds pass",
            "selection_use": "gate audit",
            "rationale": "prevents compute creep and single-seed promotion",
        },
    ]


def build_compute_rows() -> list[dict[str, Any]]:
    return [
        {
            "resource_id": "P174-COMPUTE-001",
            "resource": "local_cpu_numpy_or_tiny_torch",
            "allowed_now": False,
            "limit": "design only in Phase 174; Phase 175 may run only if explicitly opened",
            "escalation_rule": "prefer NumPy/local smoke before any GPU use",
        },
        {
            "resource_id": "P174-COMPUTE-002",
            "resource": "A800_40GB",
            "allowed_now": False,
            "limit": "reproduce design only now; future smoke must stay tiny",
            "escalation_rule": "allowed only by a later Phase 175 runner",
        },
        {
            "resource_id": "P174-COMPUTE-003",
            "resource": "A100_SXM4_80GB",
            "allowed_now": False,
            "limit": "not justified",
            "escalation_rule": "request only after a seed-positive branch hits measured 40GB blockage",
        },
    ]


def build_promotion_rows() -> list[dict[str, Any]]:
    return [
        {
            "rule_id": "P174-PROMOTE-001",
            "rule": "phase175_entry",
            "threshold": "Phase 174 design gate passes with all training locks false",
            "failure_action": "repair design before training",
        },
        {
            "rule_id": "P174-PROMOTE-002",
            "rule": "low_capacity_validation_gain",
            "threshold": "future candidate improves validation score vs Phase 173 and controls",
            "failure_action": "close as unnecessary low-capacity mechanism",
        },
        {
            "rule_id": "P174-PROMOTE-003",
            "rule": "closure_interpretability",
            "threshold": "future closure error improves or matches Phase 173 without field degradation",
            "failure_action": "write as predictive diagnostic only",
        },
        {
            "rule_id": "P174-PROMOTE-004",
            "rule": "test_reversal",
            "threshold": "future shifted-test reversal ratio <=1.02",
            "failure_action": "close and do not tune test",
        },
        {
            "rule_id": "P174-PROMOTE-005",
            "rule": "claim_boundary",
            "threshold": "future result remains synthetic until separate AM data gate",
            "failure_action": "do not claim AM-Bench, Bayesian PINN, GNN/CNN/operator, or 80GB success",
        },
    ]


def build_risk_rows() -> list[dict[str, Any]]:
    return [
        {
            "risk_id": "P174-RISK-001",
            "risk": "low-capacity model adds no value over Phase 173 explicit latents",
            "guard": "Phase 173 candidate is a required control",
            "closure_action": "close if validation improvement is absent",
        },
        {
            "risk_id": "P174-RISK-002",
            "risk": "overclaiming Bayesian PINN",
            "guard": "posterior warm-start is initialization only",
            "closure_action": "write as latent mechanism design only",
        },
        {
            "risk_id": "P174-RISK-003",
            "risk": "sampler retuning repeats Phase 167 failure",
            "guard": "failure_sampler_retrain_block remains required",
            "closure_action": "do not select sampler route",
        },
        {
            "risk_id": "P174-RISK-004",
            "risk": "capacity or compute creep",
            "guard": "Phase 174 is design-only and future budget is bounded",
            "closure_action": "stop instead of scaling",
        },
        {
            "risk_id": "P174-RISK-005",
            "risk": "synthetic-to-AM overreach",
            "guard": "no raw data and no AM claim",
            "closure_action": "require later baseline-first AM data gate",
        },
    ]


def build_gate(
    *,
    phase173_gate: dict[str, Any],
    variant_summary_rows: list[dict[str, str]],
    training_audit_rows: list[dict[str, str]],
    design_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    compute_rows: list[dict[str, Any]],
    promotion_rows: list[dict[str, Any]],
    risk_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    phase173_ready = (
        phase173_gate.get("status")
        == "phase173_trainable_hidden_closure_low_budget_smoke_ready_phase174_low_capacity_hidden_closure_design"
        and _is_true(phase173_gate.get("phase174_low_capacity_hidden_closure_design_allowed"))
        and not _is_true(phase173_gate.get("phase173_model_training_allowed"))
    )
    candidate_id = "tiny_explicit_latent_hidden_closure_smoke"
    phase173_candidate_positive = (
        phase173_gate.get("selected_variant") == candidate_id
        and float(phase173_gate.get("validation_score_gain_vs_best_control", 0.0)) >= 0.005
        and float(phase173_gate.get("phase171_validation_score_gain", 0.0)) >= 0.010
        and float(phase173_gate.get("phase171_test_closure_gain", 0.0)) >= 0.002
        and float(phase173_gate.get("seed_stability_pass_rate", 0.0)) >= 1.0
        and not phase173_gate.get("blocking_audits")
    )
    candidate_val = _summary_lookup(variant_summary_rows, candidate_id, "val")
    candidate_test = _summary_lookup(variant_summary_rows, candidate_id, "test")
    uniform_val = _summary_lookup(
        variant_summary_rows,
        "uniform_grid_latent_trainable_control",
        "val",
    )
    summary_consistent = (
        abs(
            float(candidate_val["selection_score_mean"])
            - float(phase173_gate.get("candidate_validation_selection_score", 0.0))
        )
        <= 1e-9
        and abs(
            float(candidate_test["selection_score_mean"])
            - float(phase173_gate.get("candidate_test_selection_score", 0.0))
        )
        <= 1e-9
        and float(candidate_val["selection_score_mean"])
        < float(uniform_val["selection_score_mean"])
    )
    trainable_rows = [
        row
        for row in training_audit_rows
        if row.get("variant_id") == candidate_id
    ]
    budget_ready = (
        len(trainable_rows) >= 200
        and max(int(row["start_count"]) for row in trainable_rows) <= 3
        and max(int(row["max_rounds_per_start"]) for row in trainable_rows) <= 48
        and max(int(row["executed_rounds_total"]) for row in trainable_rows) <= 144
        and max(int(row["function_evaluations_total"]) for row in trainable_rows) <= 1200
    )
    no_training_now = (
        all(not _is_true(row["opens_training_now"]) for row in design_rows)
        and all(not _is_true(row["allowed_now"]) for row in compute_rows)
    )
    complete = (
        phase173_ready
        and phase173_candidate_positive
        and summary_consistent
        and budget_ready
        and len(design_rows) >= 7
        and len(control_rows) >= 10
        and len(metric_rows) >= 7
        and len(compute_rows) >= 3
        and len(promotion_rows) >= 5
        and len(risk_rows) >= 5
        and no_training_now
    )
    blockers: list[str] = []
    if not phase173_ready:
        blockers.append("phase173_gate_not_ready")
    if not phase173_candidate_positive:
        blockers.append("phase173_candidate_positive_guard")
    if not summary_consistent:
        blockers.append("phase173_summary_consistency_guard")
    if not budget_ready:
        blockers.append("phase173_budget_guard")
    if len(design_rows) < 7:
        blockers.append("missing_design_rows")
    if len(control_rows) < 10:
        blockers.append("missing_control_rows")
    if len(metric_rows) < 7:
        blockers.append("missing_metric_rows")
    if len(compute_rows) < 3:
        blockers.append("missing_compute_rows")
    if len(promotion_rows) < 5:
        blockers.append("missing_promotion_rows")
    if len(risk_rows) < 5:
        blockers.append("missing_risk_rows")
    if not no_training_now:
        blockers.append("phase174_attempted_training_now")
    return {
        "status": (
            "phase174_low_capacity_hidden_closure_design_ready_phase175_low_capacity_smoke"
            if complete
            else "phase174_low_capacity_hidden_closure_design_incomplete"
        ),
        "candidate_low_capacity_route": "low_capacity_explicit_latent_hidden_closure_head",
        "phase173_candidate_variant": phase173_gate.get("candidate_variant"),
        "phase173_selected_variant": phase173_gate.get("selected_variant"),
        "phase173_best_control_variant": phase173_gate.get("best_control_variant"),
        "phase173_validation_gain": phase173_gate.get("validation_score_gain_vs_best_control"),
        "phase173_phase171_validation_score_gain": phase173_gate.get("phase171_validation_score_gain"),
        "phase173_phase171_test_closure_gain": phase173_gate.get("phase171_test_closure_gain"),
        "phase173_seed_stability_pass_rate": phase173_gate.get("seed_stability_pass_rate"),
        "phase173_trainable_audit_rows": len(trainable_rows),
        "design_rows": len(design_rows),
        "control_rows": len(control_rows),
        "metric_rows": len(metric_rows),
        "compute_rows": len(compute_rows),
        "promotion_rows": len(promotion_rows),
        "risk_rows": len(risk_rows),
        "blocking_audits": blockers,
        "phase175_low_capacity_smoke_allowed": bool(complete),
        "phase174_model_mechanism_allowed": False,
        "phase174_model_training_allowed": False,
        "phase175_training_allowed_now": False,
        "bayesian_pinn_training_allowed_now": False,
        "adaptive_sampling_training_allowed_now": False,
        "gcn_pinn_training_allowed_now": False,
        "cnn_operator_training_allowed_now": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": (
            "enter Phase 175 bounded low-capacity hidden-closure smoke"
            if complete
            else "repair Phase 174 design before training"
        ),
    }


def _markdown_table(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[str]:
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    body = [
        "| " + " | ".join(_csv_value(row.get(field, "")) for field in fields) + " |"
        for row in rows
    ]
    return [header, sep, *body]


def build_markdown(
    *,
    gate: dict[str, Any],
    design_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    compute_rows: list[dict[str, Any]],
    promotion_rows: list[dict[str, Any]],
    risk_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Phase 174 Low-Capacity Hidden-Closure Design Gate",
        "",
        "## Gate",
        f"- Status: `{gate['status']}`",
        f"- Candidate low-capacity route: `{gate['candidate_low_capacity_route']}`",
        f"- Phase 175 low-capacity smoke allowed: `{_csv_value(gate['phase175_low_capacity_smoke_allowed'])}`",
        f"- Phase 174 model training allowed: `{_csv_value(gate['phase174_model_training_allowed'])}`",
        f"- Phase 175 training allowed now: `{_csv_value(gate['phase175_training_allowed_now'])}`",
        f"- A100 training allowed now: `{_csv_value(gate['a100_training_allowed_now'])}`",
        f"- A100-SXM4-80GB request now: `{_csv_value(gate['a100_80gb_request_now'])}`",
        "",
        "## Interpretation",
        (
            "This is a design gate only. It translates the Phase 173 synthetic "
            "trainable-latent positive into a future bounded low-capacity smoke "
            "protocol, but it does not execute training and does not support "
            "AM-Bench, Bayesian PINN, GCN, CNN, operator, or A100-80GB claims."
        ),
        "",
        "## Design",
        *_markdown_table(design_rows, DESIGN_FIELDS),
        "",
        "## Controls",
        *_markdown_table(control_rows, CONTROL_FIELDS),
        "",
        "## Metrics",
        *_markdown_table(metric_rows, METRIC_FIELDS),
        "",
        "## Compute",
        *_markdown_table(compute_rows, COMPUTE_FIELDS),
        "",
        "## Promotion Rules",
        *_markdown_table(promotion_rows, PROMOTION_FIELDS),
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
        name: path if path.is_absolute() else root / path
        for name, path in phase_inputs.items()
    }
    phase173_gate = _read_json(resolved["phase173_gate"])
    variant_summary_rows = _read_csv(resolved["phase173_variant_summary_table"])
    training_audit_rows = _read_csv(resolved["phase173_training_audit_table"])
    design_rows = build_design_rows(phase173_gate=phase173_gate)
    control_rows = build_control_rows()
    metric_rows = build_metric_rows()
    compute_rows = build_compute_rows()
    promotion_rows = build_promotion_rows()
    risk_rows = build_risk_rows()
    gate = build_gate(
        phase173_gate=phase173_gate,
        variant_summary_rows=variant_summary_rows,
        training_audit_rows=training_audit_rows,
        design_rows=design_rows,
        control_rows=control_rows,
        metric_rows=metric_rows,
        compute_rows=compute_rows,
        promotion_rows=promotion_rows,
        risk_rows=risk_rows,
    )

    design_path = output_dir / "phase174_low_capacity_design_table.csv"
    control_path = output_dir / "phase174_control_table.csv"
    metric_path = output_dir / "phase174_metric_guard_table.csv"
    compute_path = output_dir / "phase174_compute_envelope_table.csv"
    promotion_path = output_dir / "phase174_promotion_rule_table.csv"
    risk_path = output_dir / "phase174_risk_table.csv"
    gate_path = output_dir / "phase174_low_capacity_hidden_closure_design_gate.json"
    markdown_path = output_dir / "phase174_low_capacity_hidden_closure_design_gate.md"
    manifest_path = output_dir / "phase174_low_capacity_hidden_closure_design_manifest.json"

    _write_csv(design_path, design_rows, DESIGN_FIELDS)
    _write_csv(control_path, control_rows, CONTROL_FIELDS)
    _write_csv(metric_path, metric_rows, METRIC_FIELDS)
    _write_csv(compute_path, compute_rows, COMPUTE_FIELDS)
    _write_csv(promotion_path, promotion_rows, PROMOTION_FIELDS)
    _write_csv(risk_path, risk_rows, RISK_FIELDS)
    _write_json(gate_path, gate)
    with markdown_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            build_markdown(
                gate=gate,
                design_rows=design_rows,
                control_rows=control_rows,
                metric_rows=metric_rows,
                compute_rows=compute_rows,
                promotion_rows=promotion_rows,
                risk_rows=risk_rows,
            )
        )

    manifest = {
        "phase": 174,
        "description": "low-capacity hidden-closure mechanism design gate",
        "inputs": {name: _display_path(path, root) for name, path in resolved.items()},
        "outputs": {
            "design_table": _display_path(design_path, root),
            "control_table": _display_path(control_path, root),
            "metric_guard_table": _display_path(metric_path, root),
            "compute_envelope_table": _display_path(compute_path, root),
            "promotion_rule_table": _display_path(promotion_path, root),
            "risk_table": _display_path(risk_path, root),
            "gate": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "design_rows": len(design_rows),
            "control_rows": len(control_rows),
            "metric_rows": len(metric_rows),
            "compute_rows": len(compute_rows),
            "promotion_rows": len(promotion_rows),
            "risk_rows": len(risk_rows),
            "phase173_summary_rows": len(variant_summary_rows),
            "phase173_training_audit_rows": len(training_audit_rows),
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
