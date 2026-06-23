#!/usr/bin/env python3
"""Build Phase 170 hidden-closure mechanism smoke design gate.

Phase 170 is a design package only. It converts the Phase 169 calibrated
hidden-source/closure identifiability positive into a bounded low-budget smoke
protocol. No PINN, Bayesian neural, adaptive sampler, AM-Bench, or A100 training
is executed here.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path(
    "docs/results/phase170_hidden_closure_mechanism_smoke_design_gate"
)

PHASE_INPUTS = {
    "phase167_gate": Path(
        "docs/results/phase167_low_budget_pinn_smoke/"
        "phase167_low_budget_pinn_smoke_gate.json"
    ),
    "phase169_gate": Path(
        "docs/results/phase169_hidden_source_closure_identifiability_gate/"
        "phase169_hidden_source_closure_identifiability_gate.json"
    ),
    "phase169_metric_table": Path(
        "docs/results/phase169_hidden_source_closure_identifiability_gate/"
        "phase169_hidden_source_metric_table.csv"
    ),
}

EVIDENCE_FIELDS = (
    "evidence_id",
    "source",
    "finding",
    "design_implication",
    "claim_boundary",
)

MECHANISM_FIELDS = (
    "mechanism_id",
    "component",
    "design_choice",
    "bound",
    "phase169_evidence",
    "opens_training_now",
)

CONTROL_FIELDS = (
    "control_id",
    "control_name",
    "role",
    "required_metric",
    "promotion_requirement",
)

LOSS_METRIC_FIELDS = (
    "metric_id",
    "metric_or_loss",
    "split_use",
    "guard",
    "rationale",
)

COMPUTE_FIELDS = (
    "resource_id",
    "resource",
    "phase170_use",
    "limit",
    "training_allowed_now",
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


def _metric_contract_ready(metric_rows: list[dict[str, str]]) -> bool:
    required = {
        ("calibrated_bayesian_hidden_source_closure_posterior", "val"),
        ("calibrated_bayesian_hidden_source_closure_posterior", "test"),
        ("grid_least_squares_source_closure_control", "val"),
        ("grid_least_squares_source_closure_control", "test"),
        ("no_closure_source_control", "val"),
        ("extra_trees_sensor_control", "test"),
    }
    present = {(row.get("method", ""), row.get("split", "")) for row in metric_rows}
    return required.issubset(present) and len(metric_rows) >= 18


def build_evidence_rows(
    *, phase167_gate: dict[str, Any], phase169_gate: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": "P170-EVID-001",
            "source": "Phase 167 low-budget synthetic PINN smoke",
            "finding": (
                f"validation selected {phase167_gate.get('selected_variant', 'unknown')} "
                "and blocked the failure-informed sampler route"
            ),
            "design_implication": "Do not retune the same sampler-PINN route.",
            "claim_boundary": "Sampler coverage is not model-error evidence.",
        },
        {
            "evidence_id": "P170-EVID-002",
            "source": "Phase 169 hidden-source/closure identifiability gate",
            "finding": (
                f"selected {phase169_gate.get('selected_method', 'unknown')} "
                f"with validation gain {phase169_gate.get('validation_score_gain_vs_best_control', 'unknown')}"
            ),
            "design_implication": "Use hidden source/closure parameters as the next mechanism target.",
            "claim_boundary": "Still synthetic and no-training.",
        },
        {
            "evidence_id": "P170-EVID-003",
            "source": "Phase 169 best control",
            "finding": (
                f"best control was {phase169_gate.get('best_control_method', 'unknown')}; "
                "point estimates are competitive but calibration is absent"
            ),
            "design_implication": "Keep grid least-squares as a required non-neural control.",
            "claim_boundary": "Bayesian language must be tied to calibrated intervals, not point RMSE only.",
        },
        {
            "evidence_id": "P170-EVID-004",
            "source": "B-PINN and efficient Bayesian PINN literature",
            "finding": "Bayesian inverse-PDE uncertainty is plausible but full neural posterior training is heavier.",
            "design_implication": "Use calibrated posterior warm-start/regularization only in the next smoke.",
            "claim_boundary": "Do not claim Bayesian PINN training until a later trained gate passes.",
        },
        {
            "evidence_id": "P170-EVID-005",
            "source": "AM thermal PINN literature",
            "finding": "Thermal PINNs are relevant to AM, but AM data claims require separate source guards.",
            "design_implication": "Keep Phase 171 synthetic before any AM-Bench or NIST AMMT route.",
            "claim_boundary": "No AM-Bench model claim from Phase 170.",
        },
        {
            "evidence_id": "P170-EVID-006",
            "source": "Phase 148 and Phase 151 route closures",
            "finding": "Current path-graph/GCN and dense CNN/operator routes remain blocked.",
            "design_implication": "Do not reopen graph or operator mechanisms inside this smoke design.",
            "claim_boundary": "No GCN/CNN/operator success claim.",
        },
    ]


def build_mechanism_rows(*, phase169_gate: dict[str, Any]) -> list[dict[str, Any]]:
    selected = phase169_gate.get("selected_method", "unknown")
    return [
        {
            "mechanism_id": "P170-MECH-001",
            "component": "task_scope",
            "design_choice": "synthetic_sparse_sensor_inverse_heat_only",
            "bound": "no AM-Bench, no NIST AMMT, no raw data",
            "phase169_evidence": "Phase 169 is synthetic and no-training.",
            "opens_training_now": False,
        },
        {
            "mechanism_id": "P170-MECH-002",
            "component": "hidden_parameter_head",
            "design_choice": "learn_center_shift_source_width_closure_coeff_as_explicit_latents",
            "bound": "three scalar latents with bounded physical ranges",
            "phase169_evidence": "center/width/closure were identifiable under sparse sensors.",
            "opens_training_now": False,
        },
        {
            "mechanism_id": "P170-MECH-003",
            "component": "posterior_warm_start",
            "design_choice": f"use_{selected}_as_initialization_or_teacher_diagnostic",
            "bound": "warm-start/reporting only; no full BNN, HMC, VI, or EKI training now",
            "phase169_evidence": "calibrated posterior beat the non-Bayesian guard on validation score.",
            "opens_training_now": False,
        },
        {
            "mechanism_id": "P170-MECH-004",
            "component": "physics_residual",
            "design_choice": "add_hidden_closure_term_to_heat_residual_with_bounded_weight",
            "bound": "closure term is scalar and interpretable; no free residual MLP",
            "phase169_evidence": "no-closure source control was much worse than calibrated closure inference.",
            "opens_training_now": False,
        },
        {
            "mechanism_id": "P170-MECH-005",
            "component": "sampler_policy",
            "design_choice": "uniform_grid_primary_with_optional_fixed_quota_hot_gradient_ablation",
            "bound": "no retuning of Phase 167 failure-informed sampler as the main candidate",
            "phase169_evidence": "Phase 167 selected the uniform-grid PINN control.",
            "opens_training_now": False,
        },
        {
            "mechanism_id": "P170-MECH-006",
            "component": "loss_balancing",
            "design_choice": "bounded_adaptive_weights_between_data_residual_and_latent_prior_terms",
            "bound": "weights clipped to registered ranges and reported in artifacts",
            "phase169_evidence": "parameter calibration matters; loss weights must not hide parameter error.",
            "opens_training_now": False,
        },
        {
            "mechanism_id": "P170-MECH-007",
            "component": "selection_protocol",
            "design_choice": "validation_only_selection_with_shifted_test_once",
            "bound": "no test tuning and no promotion from one seed",
            "phase169_evidence": "Phase 169 used validation-only selection and shifted-test reversal guard.",
            "opens_training_now": False,
        },
    ]


def build_control_rows() -> list[dict[str, Any]]:
    return [
        {
            "control_id": "P170-CTRL-001",
            "control_name": "posterior_only_calibrated_bayesian_no_neural",
            "role": "tests whether training adds value beyond Phase 169 inference",
            "required_metric": "field RMSE, closure RMSE, coverage, selection score",
            "promotion_requirement": "trained mechanism must beat or complement posterior-only on validation",
        },
        {
            "control_id": "P170-CTRL-002",
            "control_name": "grid_least_squares_source_closure_control",
            "role": "required non-Bayesian inverse control",
            "required_metric": "joint normalized parameter RMSE and field RMSE",
            "promotion_requirement": "candidate must beat validation score and avoid test reversal >1.05",
        },
        {
            "control_id": "P170-CTRL-003",
            "control_name": "no_closure_source_control",
            "role": "tests whether closure term is necessary",
            "required_metric": "closure coefficient RMSE and field RMSE",
            "promotion_requirement": "candidate must reduce validation error and closure error",
        },
        {
            "control_id": "P170-CTRL-004",
            "control_name": "uniform_grid_pinn_control",
            "role": "equal-budget trained PINN control from Phase 167 lesson",
            "required_metric": "same optimizer steps, architecture width, and collocation budget",
            "promotion_requirement": "candidate must beat uniform validation score",
        },
        {
            "control_id": "P170-CTRL-005",
            "control_name": "data_only_tiny_mlp_no_residual",
            "role": "tests whether physics residual helps",
            "required_metric": "field RMSE, hot q90, gradient q90",
            "promotion_requirement": "physics candidate must beat data-only on validation",
        },
        {
            "control_id": "P170-CTRL-006",
            "control_name": "wrong_source_prior_control",
            "role": "tests hidden-source interpretability",
            "required_metric": "parameter RMSE and field RMSE",
            "promotion_requirement": "candidate must not be solved by a wrong prior",
        },
        {
            "control_id": "P170-CTRL-007",
            "control_name": "failure_sampler_retrain_block",
            "role": "prevents repeating Phase 167 failed route",
            "required_metric": "registered as blocked unless used only as fixed ablation",
            "promotion_requirement": "cannot be the main selected candidate in Phase 171",
        },
        {
            "control_id": "P170-CTRL-008",
            "control_name": "seed_stability_control",
            "role": "prevents single-seed promotion",
            "required_metric": "at least three seeds when the smoke remains cheap",
            "promotion_requirement": "mean gain positive and worst seed not worse than best control by >5%",
        },
    ]


def build_loss_metric_rows() -> list[dict[str, Any]]:
    return [
        {
            "metric_id": "P170-METRIC-001",
            "metric_or_loss": "validation_selection_score",
            "split_use": "validation only",
            "guard": "primary selection; lower is better",
            "rationale": "Keeps model selection off the shifted test split.",
        },
        {
            "metric_id": "P170-METRIC-002",
            "metric_or_loss": "field_rmse",
            "split_use": "train/val/test",
            "guard": "candidate validation gain >=0.03 vs best trained control",
            "rationale": "Tests actual model error, unlike Phase 169 parameter-only inference.",
        },
        {
            "metric_id": "P170-METRIC-003",
            "metric_or_loss": "closure_coeff_rmse",
            "split_use": "validation and test",
            "guard": "validation <=0.020 and test <=0.025 unless field error gain is clearly larger",
            "rationale": "Preserves the interpretable closure signal from Phase 169.",
        },
        {
            "metric_id": "P170-METRIC-004",
            "metric_or_loss": "posterior_or_interval_coverage",
            "split_use": "validation and test",
            "guard": "coverage in [0.65, 1.0] for any Bayesian-calibrated report",
            "rationale": "Prevents overclaiming Bayesian point estimates.",
        },
        {
            "metric_id": "P170-METRIC-005",
            "metric_or_loss": "hot_q90_and_gradient_q90_rmse",
            "split_use": "validation and test",
            "guard": "must not degrade both region metrics while improving only global RMSE",
            "rationale": "Matches AM thermal-region concerns without using AM data.",
        },
        {
            "metric_id": "P170-METRIC-006",
            "metric_or_loss": "bounded_adaptive_loss_weights",
            "split_use": "artifact audit",
            "guard": "weights must remain within registered ranges",
            "rationale": "Loss balancing is a mechanism only if weights remain interpretable.",
        },
        {
            "metric_id": "P170-METRIC-007",
            "metric_or_loss": "runtime_and_memory",
            "split_use": "artifact audit",
            "guard": "small smoke must finish without 80GB hardware",
            "rationale": "A100-SXM4-80GB remains unjustified until measured 40GB blockage.",
        },
    ]


def build_compute_rows() -> list[dict[str, Any]]:
    return [
        {
            "resource_id": "P170-COMPUTE-001",
            "resource": "local_python",
            "phase170_use": "design artifact generation and tests",
            "limit": "no torch training and no raw data",
            "training_allowed_now": False,
            "escalation_rule": "none",
        },
        {
            "resource_id": "P170-COMPUTE-002",
            "resource": "A800_40GB",
            "phase170_use": "reproduce design artifacts and tests only",
            "limit": "runner must leave training locks false",
            "training_allowed_now": False,
            "escalation_rule": "use for Phase 171 smoke only if that later phase explicitly runs it",
        },
        {
            "resource_id": "P170-COMPUTE-003",
            "resource": "future_phase171_small_smoke",
            "phase170_use": "planned only",
            "limit": "tiny synthetic, three seeds, no AM raw data",
            "training_allowed_now": False,
            "escalation_rule": "must be implemented in Phase 171, not Phase 170",
        },
        {
            "resource_id": "P170-COMPUTE-004",
            "resource": "A100_SXM4_80GB",
            "phase170_use": "not allowed",
            "limit": "not justified",
            "training_allowed_now": False,
            "escalation_rule": "request only after a later seed-positive branch has measured 40GB blockage",
        },
    ]


def build_promotion_rows() -> list[dict[str, Any]]:
    return [
        {
            "rule_id": "P170-PROMOTE-001",
            "rule": "phase171_entry",
            "threshold": "Phase 170 gate passes with all training/A100 locks false",
            "failure_action": "repair design before any training",
        },
        {
            "rule_id": "P170-PROMOTE-002",
            "rule": "validation_gain",
            "threshold": "future Phase 171 candidate validation score gain >=0.03 vs best trained control",
            "failure_action": "close as diagnostic",
        },
        {
            "rule_id": "P170-PROMOTE-003",
            "rule": "test_reversal",
            "threshold": "future shifted-test reversal ratio <=1.05",
            "failure_action": "close and do not tune test",
        },
        {
            "rule_id": "P170-PROMOTE-004",
            "rule": "closure_interpretability",
            "threshold": "future closure coefficient RMSE remains within Phase 169-scale guard",
            "failure_action": "write as predictive diagnostic only, not interpretable mechanism",
        },
        {
            "rule_id": "P170-PROMOTE-005",
            "rule": "claim_boundary",
            "threshold": "future result remains synthetic until separate AM data guard",
            "failure_action": "do not write AM-Bench or general GNN-PINN claim",
        },
    ]


def build_risk_rows() -> list[dict[str, Any]]:
    return [
        {
            "risk_id": "P170-RISK-001",
            "risk": "repeating Phase 167 sampler route",
            "guard": "failure_sampler_retrain_block is a required control",
            "closure_action": "close if the main candidate is sampler retuning",
        },
        {
            "risk_id": "P170-RISK-002",
            "risk": "Bayesian overclaim",
            "guard": "posterior calibration must be reported separately from neural training",
            "closure_action": "write as calibrated inference diagnostic only",
        },
        {
            "risk_id": "P170-RISK-003",
            "risk": "grid-search control solves the problem",
            "guard": "grid least-squares remains a required non-neural baseline",
            "closure_action": "do not train if controls leave no modeling gap",
        },
        {
            "risk_id": "P170-RISK-004",
            "risk": "synthetic-to-AM overreach",
            "guard": "no raw AM data and no AM claim in this route",
            "closure_action": "require a later AM data gate before AM-Bench claims",
        },
        {
            "risk_id": "P170-RISK-005",
            "risk": "compute creep",
            "guard": "all Phase 170 training and A100/80GB locks remain false",
            "closure_action": "stop and redesign instead of scaling",
        },
    ]


def build_gate(
    *,
    phase167_gate: dict[str, Any],
    phase169_gate: dict[str, Any],
    metric_rows: list[dict[str, str]],
    evidence_rows: list[dict[str, Any]],
    mechanism_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    loss_metric_rows: list[dict[str, Any]],
    compute_rows: list[dict[str, Any]],
    promotion_rows: list[dict[str, Any]],
    risk_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    phase167_closed = (
        phase167_gate.get("status") == "phase167_low_budget_pinn_smoke_closed_no_stable_model_gain"
        and phase167_gate.get("selected_variant") == "uniform_grid_pinn"
        and not _is_true(phase167_gate.get("phase167_model_claim_allowed"))
    )
    phase169_ready = (
        phase169_gate.get("status")
        == "phase169_hidden_source_closure_identifiability_ready_phase170_low_budget_mechanism_design"
        and _is_true(phase169_gate.get("phase170_low_budget_mechanism_design_allowed"))
        and not _is_true(phase169_gate.get("phase169_model_training_allowed"))
    )
    phase169_identifiable = (
        float(phase169_gate.get("validation_score_gain_vs_best_control", 0.0)) >= 0.02
        and float(phase169_gate.get("test_reversal_ratio_vs_best_control", 99.0)) <= 1.05
        and float(phase169_gate.get("candidate_validation_closure_coeff_rmse", 99.0)) <= 0.020
        and float(phase169_gate.get("candidate_test_closure_coeff_rmse", 99.0)) <= 0.025
        and 0.65 <= float(phase169_gate.get("candidate_validation_coverage90_mean", 0.0)) <= 1.0
        and 0.65 <= float(phase169_gate.get("candidate_test_coverage90_mean", 0.0)) <= 1.0
        and not phase169_gate.get("blocking_audits")
    )
    no_training_now = (
        all(not _is_true(row["opens_training_now"]) for row in mechanism_rows)
        and all(not _is_true(row["training_allowed_now"]) for row in compute_rows)
    )
    complete = (
        phase167_closed
        and phase169_ready
        and phase169_identifiable
        and _metric_contract_ready(metric_rows)
        and len(evidence_rows) >= 6
        and len(mechanism_rows) >= 7
        and len(control_rows) >= 8
        and len(loss_metric_rows) >= 7
        and len(compute_rows) >= 4
        and len(promotion_rows) >= 5
        and len(risk_rows) >= 5
        and no_training_now
    )
    blockers: list[str] = []
    if not phase167_closed:
        blockers.append("phase167_sampler_smoke_not_closed_as_control")
    if not phase169_ready:
        blockers.append("phase169_gate_not_ready")
    if not phase169_identifiable:
        blockers.append("phase169_identifiability_guard")
    if not _metric_contract_ready(metric_rows):
        blockers.append("phase169_metric_contract_missing")
    if len(evidence_rows) < 6:
        blockers.append("missing_evidence_rows")
    if len(mechanism_rows) < 7:
        blockers.append("missing_mechanism_rows")
    if len(control_rows) < 8:
        blockers.append("missing_control_rows")
    if len(loss_metric_rows) < 7:
        blockers.append("missing_loss_metric_rows")
    if len(compute_rows) < 4:
        blockers.append("missing_compute_rows")
    if len(promotion_rows) < 5:
        blockers.append("missing_promotion_rows")
    if len(risk_rows) < 5:
        blockers.append("missing_risk_rows")
    if not no_training_now:
        blockers.append("phase170_attempted_training_now")
    return {
        "status": (
            "phase170_hidden_closure_mechanism_smoke_design_ready_phase171_low_budget_smoke"
            if complete
            else "phase170_hidden_closure_mechanism_smoke_design_incomplete"
        ),
        "candidate_mechanism": "calibrated_hidden_source_closure_parameter_head_design",
        "selected_phase169_method": phase169_gate.get("selected_method"),
        "phase169_best_control_method": phase169_gate.get("best_control_method"),
        "phase169_validation_score_gain": phase169_gate.get("validation_score_gain_vs_best_control"),
        "phase169_test_reversal_ratio": phase169_gate.get("test_reversal_ratio_vs_best_control"),
        "phase171_low_budget_hidden_closure_smoke_allowed": bool(complete),
        "evidence_rows": len(evidence_rows),
        "mechanism_rows": len(mechanism_rows),
        "control_rows": len(control_rows),
        "loss_metric_rows": len(loss_metric_rows),
        "compute_rows": len(compute_rows),
        "promotion_rows": len(promotion_rows),
        "risk_rows": len(risk_rows),
        "phase169_metric_rows": len(metric_rows),
        "blocking_audits": blockers,
        "phase170_model_mechanism_allowed": False,
        "phase170_model_training_allowed": False,
        "phase171_training_allowed_now": False,
        "bayesian_pinn_training_allowed_now": False,
        "adaptive_sampling_training_allowed_now": False,
        "gcn_pinn_training_allowed_now": False,
        "cnn_operator_training_allowed_now": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": (
            "enter Phase 171 bounded low-budget hidden-closure smoke implementation"
            if complete
            else "repair the Phase 170 design before any training"
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
    evidence_rows: list[dict[str, Any]],
    mechanism_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    loss_metric_rows: list[dict[str, Any]],
    compute_rows: list[dict[str, Any]],
    promotion_rows: list[dict[str, Any]],
    risk_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Phase 170 Hidden-Closure Mechanism Smoke Design Gate",
        "",
        "## Gate",
        f"- Status: `{gate['status']}`",
        f"- Candidate mechanism: `{gate['candidate_mechanism']}`",
        f"- Phase 171 low-budget hidden-closure smoke allowed: `{_csv_value(gate['phase171_low_budget_hidden_closure_smoke_allowed'])}`",
        f"- Phase 170 model training allowed: `{_csv_value(gate['phase170_model_training_allowed'])}`",
        f"- Phase 171 training allowed now: `{_csv_value(gate['phase171_training_allowed_now'])}`",
        f"- A100 training allowed now: `{_csv_value(gate['a100_training_allowed_now'])}`",
        f"- A100-SXM4-80GB request now: `{_csv_value(gate['a100_80gb_request_now'])}`",
        "",
        "## Interpretation",
        (
            "This is a design gate only. It does not execute training, does not reopen "
            "the Phase 167 sampler-retuning route, and does not support AM-Bench, "
            "Bayesian PINN, GCN, CNN, or neural-operator claims."
        ),
        "",
        "## Evidence",
        *_markdown_table(evidence_rows, EVIDENCE_FIELDS),
        "",
        "## Mechanism",
        *_markdown_table(mechanism_rows, MECHANISM_FIELDS),
        "",
        "## Controls",
        *_markdown_table(control_rows, CONTROL_FIELDS),
        "",
        "## Losses And Metrics",
        *_markdown_table(loss_metric_rows, LOSS_METRIC_FIELDS),
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
    phase167_gate = _read_json(resolved["phase167_gate"])
    phase169_gate = _read_json(resolved["phase169_gate"])
    metric_rows = _read_csv(resolved["phase169_metric_table"])
    evidence_rows = build_evidence_rows(
        phase167_gate=phase167_gate,
        phase169_gate=phase169_gate,
    )
    mechanism_rows = build_mechanism_rows(phase169_gate=phase169_gate)
    control_rows = build_control_rows()
    loss_metric_rows = build_loss_metric_rows()
    compute_rows = build_compute_rows()
    promotion_rows = build_promotion_rows()
    risk_rows = build_risk_rows()
    gate = build_gate(
        phase167_gate=phase167_gate,
        phase169_gate=phase169_gate,
        metric_rows=metric_rows,
        evidence_rows=evidence_rows,
        mechanism_rows=mechanism_rows,
        control_rows=control_rows,
        loss_metric_rows=loss_metric_rows,
        compute_rows=compute_rows,
        promotion_rows=promotion_rows,
        risk_rows=risk_rows,
    )

    evidence_path = output_dir / "phase170_evidence_table.csv"
    mechanism_path = output_dir / "phase170_mechanism_design_table.csv"
    control_path = output_dir / "phase170_control_table.csv"
    loss_metric_path = output_dir / "phase170_loss_metric_table.csv"
    compute_path = output_dir / "phase170_compute_envelope_table.csv"
    promotion_path = output_dir / "phase170_promotion_rule_table.csv"
    risk_path = output_dir / "phase170_risk_table.csv"
    gate_path = output_dir / "phase170_hidden_closure_mechanism_smoke_design_gate.json"
    markdown_path = output_dir / "phase170_hidden_closure_mechanism_smoke_design_gate.md"
    manifest_path = output_dir / "phase170_hidden_closure_mechanism_smoke_design_manifest.json"

    _write_csv(evidence_path, evidence_rows, EVIDENCE_FIELDS)
    _write_csv(mechanism_path, mechanism_rows, MECHANISM_FIELDS)
    _write_csv(control_path, control_rows, CONTROL_FIELDS)
    _write_csv(loss_metric_path, loss_metric_rows, LOSS_METRIC_FIELDS)
    _write_csv(compute_path, compute_rows, COMPUTE_FIELDS)
    _write_csv(promotion_path, promotion_rows, PROMOTION_FIELDS)
    _write_csv(risk_path, risk_rows, RISK_FIELDS)
    _write_json(gate_path, gate)
    with markdown_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            build_markdown(
                gate=gate,
                evidence_rows=evidence_rows,
                mechanism_rows=mechanism_rows,
                control_rows=control_rows,
                loss_metric_rows=loss_metric_rows,
                compute_rows=compute_rows,
                promotion_rows=promotion_rows,
                risk_rows=risk_rows,
            )
        )

    manifest = {
        "phase": 170,
        "description": "hidden-source/closure low-budget mechanism smoke design gate",
        "inputs": {name: _display_path(path, root) for name, path in resolved.items()},
        "outputs": {
            "evidence_table": _display_path(evidence_path, root),
            "mechanism_design_table": _display_path(mechanism_path, root),
            "control_table": _display_path(control_path, root),
            "loss_metric_table": _display_path(loss_metric_path, root),
            "compute_envelope_table": _display_path(compute_path, root),
            "promotion_rule_table": _display_path(promotion_path, root),
            "risk_table": _display_path(risk_path, root),
            "gate": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "evidence_rows": len(evidence_rows),
            "mechanism_rows": len(mechanism_rows),
            "control_rows": len(control_rows),
            "loss_metric_rows": len(loss_metric_rows),
            "compute_rows": len(compute_rows),
            "promotion_rows": len(promotion_rows),
            "risk_rows": len(risk_rows),
            "phase169_metric_rows": len(metric_rows),
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
