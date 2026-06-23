#!/usr/bin/env python3
"""Build Phase 177 uncertainty-guided latent acquisition design gate.

Phase 177 is a design-only gate after the Phase 176 evidence refresh. It defines
a materially different next mechanism: posterior/ensemble uncertainty-guided
selection of sparse synthetic observations for hidden source/closure recovery.
It does not train, read raw data, reopen the Phase 175 low-capacity head, or
request A100-SXM4-80GB.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path(
    "docs/results/phase177_uncertainty_guided_latent_acquisition_design_gate"
)

PHASE_INPUTS = {
    "phase176_gate": Path(
        "docs/results/phase176_hidden_closure_evidence_refresh/"
        "phase176_hidden_closure_evidence_refresh_gate.json"
    ),
    "phase176_route_table": Path(
        "docs/results/phase176_hidden_closure_evidence_refresh/"
        "phase176_hidden_closure_route_evidence_table.csv"
    ),
    "phase176_claim_table": Path(
        "docs/results/phase176_hidden_closure_evidence_refresh/"
        "phase176_claim_boundary_refresh_table.csv"
    ),
    "phase169_gate": Path(
        "docs/results/phase169_hidden_source_closure_identifiability_gate/"
        "phase169_hidden_source_closure_identifiability_gate.json"
    ),
    "phase173_gate": Path(
        "docs/results/phase173_trainable_hidden_closure_low_budget_smoke/"
        "phase173_trainable_hidden_closure_low_budget_smoke_gate.json"
    ),
    "phase175_gate": Path(
        "docs/results/phase175_low_capacity_hidden_closure_smoke/"
        "phase175_low_capacity_hidden_closure_smoke_gate.json"
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
    "materially_different_from_phase175",
    "opens_training_now",
)

ACQUISITION_FIELDS = (
    "acquisition_id",
    "policy",
    "input_signal",
    "selection_rule",
    "leakage_guard",
    "candidate_for_phase178",
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
    "split_use",
    "threshold",
    "rationale",
)

COMPUTE_FIELDS = (
    "resource_id",
    "resource",
    "phase177_use",
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
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


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
    path.write_text(json.dumps(_stable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
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
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _display_path(path: Path, root: Path | None = None) -> str:
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


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


def build_evidence_rows(
    *,
    phase176_gate: dict[str, Any],
    phase169_gate: dict[str, Any],
    phase173_gate: dict[str, Any],
    phase175_gate: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": "P177-EVID-001",
            "source": "Phase 176 evidence refresh",
            "finding": (
                "synthetic hidden-closure claims are allowed only narrowly; "
                f"Phase 177 design allowed={phase176_gate.get('phase177_materially_different_mechanism_design_allowed')}"
            ),
            "design_implication": "The next route must be materially different and design-only.",
            "claim_boundary": "No second-paper core claim or training permission yet.",
        },
        {
            "evidence_id": "P177-EVID-002",
            "source": "Phase 169 calibrated posterior",
            "finding": (
                f"validation gain {phase169_gate.get('validation_score_gain_vs_best_control')} "
                f"with coverage {phase169_gate.get('candidate_validation_coverage90_mean')}"
            ),
            "design_implication": "Use posterior uncertainty as a candidate acquisition signal.",
            "claim_boundary": "Still no Bayesian neural training claim.",
        },
        {
            "evidence_id": "P177-EVID-003",
            "source": "Phase 173 explicit latent smoke",
            "finding": (
                f"explicit latent route beat {phase173_gate.get('best_control_variant')} "
                f"with validation gain {phase173_gate.get('validation_score_gain_vs_best_control')}"
            ),
            "design_implication": "Use latent disagreement, not a larger head, to choose new observations.",
            "claim_boundary": "Tiny synthetic result only.",
        },
        {
            "evidence_id": "P177-EVID-004",
            "source": "Phase 175 closure",
            "finding": (
                f"low-capacity candidate {phase175_gate.get('candidate_variant')} closed with "
                f"gain {phase175_gate.get('validation_score_gain_vs_best_control')}"
            ),
            "design_implication": "Block low-capacity head retuning and avoid capacity expansion.",
            "claim_boundary": "The low-capacity head is a negative route.",
        },
        {
            "evidence_id": "P177-EVID-005",
            "source": "Phase 167 sampler-to-PINN closure",
            "finding": "sampler coverage alone did not become a trained PINN gain.",
            "design_implication": "Acquisition must target posterior contraction and latent error, not only hot-gradient coverage.",
            "claim_boundary": "No adaptive PINN training claim.",
        },
        {
            "evidence_id": "P177-EVID-006",
            "source": "Phase 148/151 route closures",
            "finding": "current graph/path and dense operator routes remain blocked.",
            "design_implication": "Do not add GCN/CNN/operator machinery in this design gate.",
            "claim_boundary": "No graph/operator success claim.",
        },
    ]


def build_mechanism_rows() -> list[dict[str, Any]]:
    return [
        {
            "mechanism_id": "P177-MECH-001",
            "component": "task_scope",
            "design_choice": "controlled_synthetic_inverse_heat_sparse_observation_acquisition",
            "bound": "no AM-Bench, no NIST AMMT, no registered camera target, no raw data",
            "materially_different_from_phase175": True,
            "opens_training_now": False,
        },
        {
            "mechanism_id": "P177-MECH-002",
            "component": "candidate_route",
            "design_choice": "posterior_ensemble_uncertainty_guided_latent_acquisition",
            "bound": "choose new observation locations before any model update; no low-capacity head",
            "materially_different_from_phase175": True,
            "opens_training_now": False,
        },
        {
            "mechanism_id": "P177-MECH-003",
            "component": "uncertainty_source",
            "design_choice": "combine_phase169_posterior_variance_and_phase173_latent_disagreement",
            "bound": "uncertainty is computed from train/validation-safe synthetic sensors only",
            "materially_different_from_phase175": True,
            "opens_training_now": False,
        },
        {
            "mechanism_id": "P177-MECH-004",
            "component": "objective",
            "design_choice": "maximize_expected_closure_posterior_contraction_under_sparse_budget",
            "bound": "registered acquisition budget; no test-target or shifted-test peeking",
            "materially_different_from_phase175": True,
            "opens_training_now": False,
        },
        {
            "mechanism_id": "P177-MECH-005",
            "component": "physics_guard",
            "design_choice": "retain_explicit_center_width_closure_latents_with_bounded_ranges",
            "bound": "no free residual field, no residual MLP, no density proxy retuning",
            "materially_different_from_phase175": True,
            "opens_training_now": False,
        },
        {
            "mechanism_id": "P177-MECH-006",
            "component": "selection_protocol",
            "design_choice": "validation_only_acquisition_policy_selection_shifted_test_once",
            "bound": "Phase 178 may run only a no-training acquisition utility smoke if opened",
            "materially_different_from_phase175": True,
            "opens_training_now": False,
        },
        {
            "mechanism_id": "P177-MECH-007",
            "component": "claim_boundary",
            "design_choice": "mechanism_design_not_model_training",
            "bound": "all training/A100/80GB locks remain false",
            "materially_different_from_phase175": True,
            "opens_training_now": False,
        },
    ]


def build_acquisition_rows() -> list[dict[str, Any]]:
    return [
        {
            "acquisition_id": "P177-ACQ-001",
            "policy": "posterior_entropy_reduction_candidate",
            "input_signal": "Phase 169 calibrated posterior variance over center, width, closure coefficient",
            "selection_rule": "rank candidate sensor locations by expected entropy reduction",
            "leakage_guard": "candidate pool generated from train/validation synthetic coordinates only",
            "candidate_for_phase178": True,
        },
        {
            "acquisition_id": "P177-ACQ-002",
            "policy": "latent_ensemble_disagreement_candidate",
            "input_signal": "Phase 173 explicit latent ensemble prediction disagreement",
            "selection_rule": "rank by disagreement in field and closure-sensitive regions",
            "leakage_guard": "no shifted-test error or target residual is used",
            "candidate_for_phase178": True,
        },
        {
            "acquisition_id": "P177-ACQ-003",
            "policy": "hybrid_uncertainty_hot_gradient_candidate",
            "input_signal": "posterior contraction score plus fixed hot/gradient quota",
            "selection_rule": "allocate a bounded quota to high-gradient regions after uncertainty ranking",
            "leakage_guard": "hot/gradient field computed analytically in synthetic train split only",
            "candidate_for_phase178": True,
        },
        {
            "acquisition_id": "P177-ACQ-004",
            "policy": "uniform_budget_control",
            "input_signal": "none",
            "selection_rule": "same number of new observation points on a uniform grid",
            "leakage_guard": "same candidate pool and budget",
            "candidate_for_phase178": False,
        },
        {
            "acquisition_id": "P177-ACQ-005",
            "policy": "random_budget_control",
            "input_signal": "seeded random coordinate order",
            "selection_rule": "same number of new observation points by deterministic random seed",
            "leakage_guard": "same candidate pool and budget",
            "candidate_for_phase178": False,
        },
        {
            "acquisition_id": "P177-ACQ-006",
            "policy": "oracle_test_target_block",
            "input_signal": "test error, shifted-test labels, or target residuals",
            "selection_rule": "prohibited",
            "leakage_guard": "any use closes the gate",
            "candidate_for_phase178": False,
        },
    ]


def build_control_rows() -> list[dict[str, Any]]:
    return [
        {
            "control_id": "P177-CTRL-001",
            "control_name": "phase173_tiny_explicit_latent_hidden_closure_smoke",
            "role": "current strongest synthetic mechanism floor",
            "required_metric": "selection score, field RMSE, closure error",
            "promotion_requirement": "future acquisition smoke must improve latent/closure metrics over this floor",
        },
        {
            "control_id": "P177-CTRL-002",
            "control_name": "phase169_posterior_only_calibrated_bayesian_no_neural",
            "role": "uncertainty baseline",
            "required_metric": "posterior contraction, interval coverage, closure RMSE",
            "promotion_requirement": "candidate must improve uncertainty or closure recovery beyond posterior-only",
        },
        {
            "control_id": "P177-CTRL-003",
            "control_name": "grid_least_squares_source_closure_control",
            "role": "strong deterministic inverse control",
            "required_metric": "joint normalized RMSE and closure coefficient error",
            "promotion_requirement": "candidate must beat validation score or close as solved",
        },
        {
            "control_id": "P177-CTRL-004",
            "control_name": "uniform_budget_acquisition_control",
            "role": "same budget acquisition baseline",
            "required_metric": "posterior contraction and validation selection score",
            "promotion_requirement": "candidate must beat uniform acquisition",
        },
        {
            "control_id": "P177-CTRL-005",
            "control_name": "random_budget_acquisition_control",
            "role": "seeded random observation baseline",
            "required_metric": "mean and worst-seed contraction",
            "promotion_requirement": "candidate must beat random mean and avoid worst-seed collapse",
        },
        {
            "control_id": "P177-CTRL-006",
            "control_name": "no_new_observation_control",
            "role": "tests whether acquisition itself adds value",
            "required_metric": "posterior contraction and closure error delta",
            "promotion_requirement": "candidate must improve beyond no-acquisition",
        },
        {
            "control_id": "P177-CTRL-007",
            "control_name": "phase175_low_capacity_head_retrain_block",
            "role": "blocks the closed Phase 175 route",
            "required_metric": "must remain non-selected",
            "promotion_requirement": "any reuse of the low-capacity head closes the design",
        },
        {
            "control_id": "P177-CTRL-008",
            "control_name": "failure_sampler_retrain_block",
            "role": "blocks Phase 167 sampler retuning",
            "required_metric": "must remain diagnostic-only",
            "promotion_requirement": "candidate cannot be only hot-gradient sampler retuning",
        },
        {
            "control_id": "P177-CTRL-009",
            "control_name": "oracle_target_leakage_block",
            "role": "prevents test/target leakage",
            "required_metric": "must be absent from acquisition features",
            "promotion_requirement": "any oracle target use closes the branch",
        },
        {
            "control_id": "P177-CTRL-010",
            "control_name": "seed_stability_control",
            "role": "prevents single-seed acquisition promotion",
            "required_metric": "three deterministic seeds when Phase 178 runs",
            "promotion_requirement": "mean gain positive and worst seed within registered tolerance",
        },
    ]


def build_metric_rows() -> list[dict[str, Any]]:
    return [
        {
            "metric_id": "P177-METRIC-001",
            "metric_or_guard": "validation_selection_score",
            "split_use": "validation only",
            "threshold": "future acquisition candidate must beat best control",
            "rationale": "prevents promotion from design text alone",
        },
        {
            "metric_id": "P177-METRIC-002",
            "metric_or_guard": "posterior_entropy_or_variance_contraction",
            "split_use": "validation audit",
            "threshold": "candidate contraction gain > uniform and random controls",
            "rationale": "tests the acquisition mechanism directly",
        },
        {
            "metric_id": "P177-METRIC-003",
            "metric_or_guard": "closure_abs_error_delta",
            "split_use": "validation/test audit",
            "threshold": "must improve or match Phase 173 closure error",
            "rationale": "keeps hidden-closure recovery as the target",
        },
        {
            "metric_id": "P177-METRIC-004",
            "metric_or_guard": "field_rmse_delta",
            "split_use": "validation/test audit",
            "threshold": "must not degrade field RMSE while improving uncertainty",
            "rationale": "prevents pure uncertainty-only gains",
        },
        {
            "metric_id": "P177-METRIC-005",
            "metric_or_guard": "coverage90_mean",
            "split_use": "validation/test audit",
            "threshold": "remain in [0.65, 1.0] if intervals are reported",
            "rationale": "bounds Bayesian interpretation",
        },
        {
            "metric_id": "P177-METRIC-006",
            "metric_or_guard": "acquisition_diversity",
            "split_use": "artifact audit",
            "threshold": "no duplicate points; bounded boundary fraction",
            "rationale": "prevents collapse into one local hot spot",
        },
        {
            "metric_id": "P177-METRIC-007",
            "metric_or_guard": "test_reversal_ratio",
            "split_use": "shifted test once",
            "threshold": "<=1.02 vs best acquisition control",
            "rationale": "prevents validation-only overfit",
        },
        {
            "metric_id": "P177-METRIC-008",
            "metric_or_guard": "no_training_no_raw_data_lock",
            "split_use": "gate audit",
            "threshold": "all training/raw-data/A100-80GB locks false",
            "rationale": "keeps Phase 177 as design-only",
        },
    ]


def build_compute_rows() -> list[dict[str, Any]]:
    return [
        {
            "resource_id": "P177-COMPUTE-001",
            "resource": "local_python",
            "phase177_use": "design artifact generation and tests",
            "limit": "no torch training, no raw data",
            "training_allowed_now": False,
            "escalation_rule": "none",
        },
        {
            "resource_id": "P177-COMPUTE-002",
            "resource": "future_phase178_no_training_smoke",
            "phase177_use": "planned only",
            "limit": "analytic/synthetic acquisition utility only if Phase 177 passes",
            "training_allowed_now": False,
            "escalation_rule": "must be implemented in Phase 178, not Phase 177",
        },
        {
            "resource_id": "P177-COMPUTE-003",
            "resource": "A800_40GB",
            "phase177_use": "reproduce design artifacts and tests only",
            "limit": "no training; server optional for reproduction",
            "training_allowed_now": False,
            "escalation_rule": "do not use as evidence for larger hardware",
        },
        {
            "resource_id": "P177-COMPUTE-004",
            "resource": "A100_SXM4_80GB",
            "phase177_use": "not allowed",
            "limit": "not justified",
            "training_allowed_now": False,
            "escalation_rule": "request only after later seed-positive branch hits measured 40GB blockage",
        },
    ]


def build_promotion_rows() -> list[dict[str, Any]]:
    return [
        {
            "rule_id": "P177-PROMOTE-001",
            "rule": "phase178_entry",
            "threshold": "Phase 177 design passes with all training/A100 locks false",
            "failure_action": "repair design before any smoke",
        },
        {
            "rule_id": "P177-PROMOTE-002",
            "rule": "materially_different_route",
            "threshold": "candidate route is posterior/ensemble acquisition, not Phase 175 low-capacity head",
            "failure_action": "close as retune attempt",
        },
        {
            "rule_id": "P177-PROMOTE-003",
            "rule": "acquisition_utility_gain",
            "threshold": "future no-training smoke must improve posterior contraction vs uniform/random/no-acquisition",
            "failure_action": "close as solved by controls",
        },
        {
            "rule_id": "P177-PROMOTE-004",
            "rule": "closure_and_field_guard",
            "threshold": "future smoke must improve closure without field/test reversal",
            "failure_action": "write as diagnostic only",
        },
        {
            "rule_id": "P177-PROMOTE-005",
            "rule": "claim_boundary",
            "threshold": "future smoke remains synthetic/no-training until a separate gate permits more",
            "failure_action": "do not claim Bayesian PINN, AM-Bench, graph/operator, or 80GB success",
        },
    ]


def build_risk_rows() -> list[dict[str, Any]]:
    return [
        {
            "risk_id": "P177-RISK-001",
            "risk": "retuning the closed Phase 175 low-capacity head",
            "guard": "explicit low-capacity head retrain block",
            "closure_action": "close if selected route uses the Phase 175 head",
        },
        {
            "risk_id": "P177-RISK-002",
            "risk": "sampler coverage repeats Phase 167 without model utility",
            "guard": "posterior contraction and closure error are primary metrics",
            "closure_action": "close if only hot-gradient coverage improves",
        },
        {
            "risk_id": "P177-RISK-003",
            "risk": "oracle target leakage in acquisition",
            "guard": "candidate pool excludes shifted-test labels and target residuals",
            "closure_action": "close immediately on leakage",
        },
        {
            "risk_id": "P177-RISK-004",
            "risk": "Bayesian overclaim",
            "guard": "coverage and calibration are audit metrics only",
            "closure_action": "do not write full Bayesian PINN success",
        },
        {
            "risk_id": "P177-RISK-005",
            "risk": "synthetic-to-AM overreach",
            "guard": "no raw AM data and no AM claim",
            "closure_action": "require separate baseline-first AM data gate",
        },
        {
            "risk_id": "P177-RISK-006",
            "risk": "compute creep",
            "guard": "all Phase 177 training/A100/80GB locks are false",
            "closure_action": "stop instead of scaling",
        },
    ]


def build_gate(
    *,
    phase176_gate: dict[str, Any],
    phase176_route_rows: list[dict[str, str]],
    phase176_claim_rows: list[dict[str, str]],
    phase169_gate: dict[str, Any],
    phase173_gate: dict[str, Any],
    phase175_gate: dict[str, Any],
    evidence_rows: list[dict[str, Any]],
    mechanism_rows: list[dict[str, Any]],
    acquisition_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    compute_rows: list[dict[str, Any]],
    promotion_rows: list[dict[str, Any]],
    risk_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    phase176_ready = (
        phase176_gate.get("status")
        == "phase176_hidden_closure_evidence_refresh_ready_synthetic_claims_low_capacity_closed"
        and _is_true(phase176_gate.get("phase177_materially_different_mechanism_design_allowed"))
        and _is_false(phase176_gate.get("phase176_model_training_allowed"))
        and _is_false(phase176_gate.get("phase177_training_allowed_now"))
        and _is_false(phase176_gate.get("a100_80gb_request_now"))
        and len(phase176_route_rows) >= 7
        and len(phase176_claim_rows) >= 7
    )
    phase169_positive = (
        phase169_gate.get("status")
        == "phase169_hidden_source_closure_identifiability_ready_phase170_low_budget_mechanism_design"
        and float(phase169_gate.get("validation_score_gain_vs_best_control", 0.0)) > 0.02
    )
    phase173_positive = (
        phase173_gate.get("status")
        == "phase173_trainable_hidden_closure_low_budget_smoke_ready_phase174_low_capacity_hidden_closure_design"
        and phase173_gate.get("selected_variant") == "tiny_explicit_latent_hidden_closure_smoke"
        and float(phase173_gate.get("validation_score_gain_vs_best_control", 0.0)) > 0.005
    )
    phase175_closed = (
        phase175_gate.get("status")
        == "phase175_low_capacity_hidden_closure_smoke_closed_no_incremental_gain"
        and phase175_gate.get("candidate_variant") == "low_capacity_explicit_latent_hidden_closure_head"
        and _is_false(phase175_gate.get("phase176_focused_review_allowed"))
    )
    candidate_route = "posterior_ensemble_uncertainty_guided_latent_acquisition"
    materially_different = (
        candidate_route != phase175_gate.get("candidate_variant")
        and all(_is_true(row["materially_different_from_phase175"]) for row in mechanism_rows)
        and not any("low_capacity_explicit_latent_hidden_closure_head" in row["design_choice"] for row in mechanism_rows)
    )
    no_training_now = (
        all(not _is_true(row["opens_training_now"]) for row in mechanism_rows)
        and all(not _is_true(row["training_allowed_now"]) for row in compute_rows)
    )
    complete = (
        phase176_ready
        and phase169_positive
        and phase173_positive
        and phase175_closed
        and materially_different
        and no_training_now
        and len(evidence_rows) >= 6
        and len(mechanism_rows) >= 7
        and len(acquisition_rows) >= 6
        and len(control_rows) >= 10
        and len(metric_rows) >= 8
        and len(compute_rows) >= 4
        and len(promotion_rows) >= 5
        and len(risk_rows) >= 6
    )
    blockers: list[str] = []
    if not phase176_ready:
        blockers.append("phase176_gate_not_ready")
    if not phase169_positive:
        blockers.append("phase169_positive_evidence_missing")
    if not phase173_positive:
        blockers.append("phase173_positive_evidence_missing")
    if not phase175_closed:
        blockers.append("phase175_low_capacity_closure_missing")
    if not materially_different:
        blockers.append("not_materially_different_from_phase175")
    if not no_training_now:
        blockers.append("phase177_attempted_training_now")
    if len(evidence_rows) < 6:
        blockers.append("missing_evidence_rows")
    if len(mechanism_rows) < 7:
        blockers.append("missing_mechanism_rows")
    if len(acquisition_rows) < 6:
        blockers.append("missing_acquisition_rows")
    if len(control_rows) < 10:
        blockers.append("missing_control_rows")
    if len(metric_rows) < 8:
        blockers.append("missing_metric_rows")
    if len(compute_rows) < 4:
        blockers.append("missing_compute_rows")
    if len(promotion_rows) < 5:
        blockers.append("missing_promotion_rows")
    if len(risk_rows) < 6:
        blockers.append("missing_risk_rows")
    return {
        "status": (
            "phase177_uncertainty_guided_latent_acquisition_design_ready_phase178_no_training_smoke"
            if complete
            else "phase177_uncertainty_guided_latent_acquisition_design_incomplete"
        ),
        "candidate_mechanism": candidate_route,
        "materially_different_from_phase175": bool(materially_different),
        "phase176_hidden_closure_branch_refreshed": bool(phase176_ready),
        "phase169_identifiability_positive_preserved": bool(phase169_positive),
        "phase173_tiny_trainable_positive_preserved": bool(phase173_positive),
        "phase175_low_capacity_route_closed": bool(phase175_closed),
        "phase178_no_training_acquisition_smoke_allowed": bool(complete),
        "phase177_model_mechanism_allowed": False,
        "phase177_model_training_allowed": False,
        "phase178_training_allowed_now": False,
        "bayesian_pinn_training_allowed_now": False,
        "adaptive_sampling_training_allowed_now": False,
        "gcn_pinn_training_allowed_now": False,
        "cnn_operator_training_allowed_now": False,
        "am_bench_training_allowed_now": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "evidence_rows": len(evidence_rows),
        "mechanism_rows": len(mechanism_rows),
        "acquisition_rows": len(acquisition_rows),
        "control_rows": len(control_rows),
        "metric_rows": len(metric_rows),
        "compute_rows": len(compute_rows),
        "promotion_rows": len(promotion_rows),
        "risk_rows": len(risk_rows),
        "blocking_audits": blockers,
        "next_action": (
            "enter Phase 178 no-training uncertainty-guided acquisition smoke"
            if complete
            else "repair Phase 177 design before any smoke"
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
    acquisition_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    compute_rows: list[dict[str, Any]],
    promotion_rows: list[dict[str, Any]],
    risk_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Phase 177 Uncertainty-Guided Latent Acquisition Design Gate",
        "",
        "## Gate",
        f"- Status: `{gate['status']}`",
        f"- Candidate mechanism: `{gate['candidate_mechanism']}`",
        f"- Materially different from Phase 175: `{_csv_value(gate['materially_different_from_phase175'])}`",
        f"- Phase 178 no-training acquisition smoke allowed: `{_csv_value(gate['phase178_no_training_acquisition_smoke_allowed'])}`",
        f"- Phase 177 model training allowed: `{_csv_value(gate['phase177_model_training_allowed'])}`",
        f"- Phase 178 training allowed now: `{_csv_value(gate['phase178_training_allowed_now'])}`",
        f"- A100 training allowed now: `{_csv_value(gate['a100_training_allowed_now'])}`",
        f"- A100-SXM4-80GB request now: `{_csv_value(gate['a100_80gb_request_now'])}`",
        "",
        "## Interpretation",
        (
            "This is a design-only gate. It pivots away from the closed Phase 175 "
            "low-capacity head toward uncertainty-guided sparse observation "
            "acquisition for explicit source/closure latents. It does not train "
            "a PINN, does not read AM data, and does not open graph, CNN, operator, "
            "or 80GB claims."
        ),
        "",
        "## Evidence",
        *_markdown_table(evidence_rows, EVIDENCE_FIELDS),
        "",
        "## Mechanism",
        *_markdown_table(mechanism_rows, MECHANISM_FIELDS),
        "",
        "## Acquisition Policies",
        *_markdown_table(acquisition_rows, ACQUISITION_FIELDS),
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
        name: path if path.is_absolute() else root / path for name, path in phase_inputs.items()
    }
    phase176_gate = _read_json(resolved["phase176_gate"])
    phase176_route_rows = _read_csv(resolved["phase176_route_table"])
    phase176_claim_rows = _read_csv(resolved["phase176_claim_table"])
    phase169_gate = _read_json(resolved["phase169_gate"])
    phase173_gate = _read_json(resolved["phase173_gate"])
    phase175_gate = _read_json(resolved["phase175_gate"])
    evidence_rows = build_evidence_rows(
        phase176_gate=phase176_gate,
        phase169_gate=phase169_gate,
        phase173_gate=phase173_gate,
        phase175_gate=phase175_gate,
    )
    mechanism_rows = build_mechanism_rows()
    acquisition_rows = build_acquisition_rows()
    control_rows = build_control_rows()
    metric_rows = build_metric_rows()
    compute_rows = build_compute_rows()
    promotion_rows = build_promotion_rows()
    risk_rows = build_risk_rows()
    gate = build_gate(
        phase176_gate=phase176_gate,
        phase176_route_rows=phase176_route_rows,
        phase176_claim_rows=phase176_claim_rows,
        phase169_gate=phase169_gate,
        phase173_gate=phase173_gate,
        phase175_gate=phase175_gate,
        evidence_rows=evidence_rows,
        mechanism_rows=mechanism_rows,
        acquisition_rows=acquisition_rows,
        control_rows=control_rows,
        metric_rows=metric_rows,
        compute_rows=compute_rows,
        promotion_rows=promotion_rows,
        risk_rows=risk_rows,
    )

    evidence_path = output_dir / "phase177_evidence_table.csv"
    mechanism_path = output_dir / "phase177_mechanism_design_table.csv"
    acquisition_path = output_dir / "phase177_acquisition_policy_table.csv"
    control_path = output_dir / "phase177_control_table.csv"
    metric_path = output_dir / "phase177_metric_guard_table.csv"
    compute_path = output_dir / "phase177_compute_envelope_table.csv"
    promotion_path = output_dir / "phase177_promotion_rule_table.csv"
    risk_path = output_dir / "phase177_risk_table.csv"
    gate_path = output_dir / "phase177_uncertainty_guided_latent_acquisition_design_gate.json"
    markdown_path = output_dir / "phase177_uncertainty_guided_latent_acquisition_design_gate.md"
    manifest_path = output_dir / "phase177_uncertainty_guided_latent_acquisition_design_manifest.json"

    _write_csv(evidence_path, evidence_rows, EVIDENCE_FIELDS)
    _write_csv(mechanism_path, mechanism_rows, MECHANISM_FIELDS)
    _write_csv(acquisition_path, acquisition_rows, ACQUISITION_FIELDS)
    _write_csv(control_path, control_rows, CONTROL_FIELDS)
    _write_csv(metric_path, metric_rows, METRIC_FIELDS)
    _write_csv(compute_path, compute_rows, COMPUTE_FIELDS)
    _write_csv(promotion_path, promotion_rows, PROMOTION_FIELDS)
    _write_csv(risk_path, risk_rows, RISK_FIELDS)
    _write_json(gate_path, gate)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        build_markdown(
            gate=gate,
            evidence_rows=evidence_rows,
            mechanism_rows=mechanism_rows,
            acquisition_rows=acquisition_rows,
            control_rows=control_rows,
            metric_rows=metric_rows,
            compute_rows=compute_rows,
            promotion_rows=promotion_rows,
            risk_rows=risk_rows,
        ),
        encoding="utf-8",
    )

    manifest = {
        "phase": 177,
        "description": "uncertainty-guided latent acquisition design gate",
        "inputs": {name: _display_path(path, root) for name, path in sorted(resolved.items())},
        "outputs": {
            "evidence_table": _display_path(evidence_path, root),
            "mechanism_design_table": _display_path(mechanism_path, root),
            "acquisition_policy_table": _display_path(acquisition_path, root),
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
            "evidence_rows": len(evidence_rows),
            "mechanism_rows": len(mechanism_rows),
            "acquisition_rows": len(acquisition_rows),
            "control_rows": len(control_rows),
            "metric_rows": len(metric_rows),
            "compute_rows": len(compute_rows),
            "promotion_rows": len(promotion_rows),
            "risk_rows": len(risk_rows),
            "phase176_route_rows": len(phase176_route_rows),
            "phase176_claim_rows": len(phase176_claim_rows),
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
    manifest = build_package(
        root=args.root,
        output_dir=args.output_dir,
        phase_inputs=phase_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
