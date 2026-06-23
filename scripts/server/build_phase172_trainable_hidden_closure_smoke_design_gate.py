#!/usr/bin/env python3
"""Build Phase 172 trainable hidden-closure smoke design gate.

Phase 172 is design-only. It converts the Phase 171 NumPy-only closure-head
positive into a tightly bounded trainable synthetic smoke protocol. It does not
start training and does not request A100 or A100-SXM4-80GB.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path(
    "docs/results/phase172_trainable_hidden_closure_smoke_design_gate"
)

PHASE_INPUTS = {
    "phase171_gate": Path(
        "docs/results/phase171_hidden_closure_low_budget_smoke/"
        "phase171_hidden_closure_low_budget_smoke_gate.json"
    ),
    "phase171_variant_summary_table": Path(
        "docs/results/phase171_hidden_closure_low_budget_smoke/"
        "phase171_variant_summary_table.csv"
    ),
    "phase171_seed_summary_table": Path(
        "docs/results/phase171_hidden_closure_low_budget_smoke/"
        "phase171_seed_summary_table.csv"
    ),
}

DESIGN_FIELDS = (
    "design_id",
    "component",
    "decision",
    "bound",
    "phase171_evidence",
    "opens_training_now",
)

CONTROL_FIELDS = (
    "control_id",
    "control_name",
    "role",
    "required_metric",
    "promotion_requirement",
)

LOSS_FIELDS = (
    "loss_id",
    "loss_or_metric",
    "weight_or_guard",
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


def build_design_rows(*, phase171_gate: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "design_id": "P172-DESIGN-001",
            "component": "task_scope",
            "decision": "synthetic_sparse_sensor_inverse_heat_only",
            "bound": "no AM-Bench, no NIST AMMT, no external raw data",
            "phase171_evidence": "Phase 171 positive is synthetic and NumPy-only.",
            "opens_training_now": False,
        },
        {
            "design_id": "P172-DESIGN-002",
            "component": "trainable_mechanism",
            "decision": "tiny_explicit_latent_head_for_center_width_closure",
            "bound": "three scalar latents plus bounded correction head; no residual MLP",
            "phase171_evidence": phase171_gate.get("candidate_variant", "unknown"),
            "opens_training_now": False,
        },
        {
            "design_id": "P172-DESIGN-003",
            "component": "initialization",
            "decision": "initialize_from_phase171_calibrated_closure_head_and_phase169_posterior",
            "bound": "initialization only; not a Bayesian neural posterior",
            "phase171_evidence": "posterior closure gain was positive across seeds.",
            "opens_training_now": False,
        },
        {
            "design_id": "P172-DESIGN-004",
            "component": "model_budget",
            "decision": "tiny_width_16_or_numpy_differentiable_optimizer_smoke",
            "bound": "max 3 seeds, max 600 steps, no AM data, no GPU requirement",
            "phase171_evidence": "Phase 171 runner is small and deterministic.",
            "opens_training_now": False,
        },
        {
            "design_id": "P172-DESIGN-005",
            "component": "sampler_policy",
            "decision": "uniform_sensor_and_collocation_primary",
            "bound": "failure-informed sampler remains blocked except as non-selected audit row",
            "phase171_evidence": "Phase 167 sampler retuning failed; Phase 171 blocked it.",
            "opens_training_now": False,
        },
        {
            "design_id": "P172-DESIGN-006",
            "component": "selection_protocol",
            "decision": "validation_only_selection_shifted_test_once",
            "bound": "no test tuning, no single-seed promotion",
            "phase171_evidence": "Phase 171 seed stability pass rate was positive.",
            "opens_training_now": False,
        },
        {
            "design_id": "P172-DESIGN-007",
            "component": "claim_boundary",
            "decision": "design_may_open_phase173_smoke_but_not_training_now",
            "bound": "all Phase 172 training and A100 locks remain false",
            "phase171_evidence": "Phase 171 opened design only.",
            "opens_training_now": False,
        },
    ]


def build_control_rows() -> list[dict[str, Any]]:
    return [
        {
            "control_id": "P172-CTRL-001",
            "control_name": "phase171_numpy_closure_head",
            "role": "must beat or complement the non-trainable mechanism",
            "required_metric": "selection score, closure error, field RMSE",
            "promotion_requirement": "trainable smoke must improve validation score or close as unnecessary",
        },
        {
            "control_id": "P172-CTRL-002",
            "control_name": "posterior_only_calibrated_bayesian_no_neural",
            "role": "strong no-neural posterior control",
            "required_metric": "field RMSE, closure error, coverage",
            "promotion_requirement": "candidate must preserve field RMSE and improve interpretable closure",
        },
        {
            "control_id": "P172-CTRL-003",
            "control_name": "grid_least_squares_source_closure_control",
            "role": "non-Bayesian inverse control",
            "required_metric": "field RMSE, closure error",
            "promotion_requirement": "candidate must beat validation score and avoid test reversal",
        },
        {
            "control_id": "P172-CTRL-004",
            "control_name": "no_closure_source_control",
            "role": "closure necessity control",
            "required_metric": "field RMSE and closure error",
            "promotion_requirement": "candidate must show closure term remains necessary",
        },
        {
            "control_id": "P172-CTRL-005",
            "control_name": "data_only_tiny_control",
            "role": "tests whether physics/closure adds value",
            "required_metric": "field RMSE, hot q90, gradient q90",
            "promotion_requirement": "candidate must beat data-only validation score",
        },
        {
            "control_id": "P172-CTRL-006",
            "control_name": "wrong_source_prior_control",
            "role": "hidden-source interpretability control",
            "required_metric": "field RMSE and closure error",
            "promotion_requirement": "candidate must not be solved by wrong source prior",
        },
        {
            "control_id": "P172-CTRL-007",
            "control_name": "uniform_grid_pinn_control",
            "role": "future tiny trained baseline if Phase 173 executes",
            "required_metric": "same trainable budget",
            "promotion_requirement": "candidate must beat or match it on validation",
        },
        {
            "control_id": "P172-CTRL-008",
            "control_name": "failure_sampler_retrain_block",
            "role": "prevents repeating Phase 167",
            "required_metric": "must remain non-selected",
            "promotion_requirement": "cannot be the selected route",
        },
        {
            "control_id": "P172-CTRL-009",
            "control_name": "seed_stability_control",
            "role": "prevents single-seed promotion",
            "required_metric": "three seeds when cheap",
            "promotion_requirement": "all seeds pass or gate closes",
        },
    ]


def build_loss_rows() -> list[dict[str, Any]]:
    return [
        {
            "loss_id": "P172-LOSS-001",
            "loss_or_metric": "sensor_data_loss",
            "weight_or_guard": "primary data fit",
            "selection_use": "train only",
            "rationale": "keeps the trainable smoke tied to observed sparse sensors",
        },
        {
            "loss_id": "P172-LOSS-002",
            "loss_or_metric": "heat_residual_with_explicit_closure",
            "weight_or_guard": "bounded registered range [0.01, 0.10]",
            "selection_use": "train only",
            "rationale": "tests a physical closure mechanism without a free residual MLP",
        },
        {
            "loss_id": "P172-LOSS-003",
            "loss_or_metric": "latent_prior_penalty",
            "weight_or_guard": "posterior warm-start prior, not full Bayesian training",
            "selection_use": "train audit",
            "rationale": "uses Phase 171 inference as a bounded prior",
        },
        {
            "loss_id": "P172-LOSS-004",
            "loss_or_metric": "validation_selection_score",
            "weight_or_guard": "candidate must beat controls",
            "selection_use": "validation only",
            "rationale": "prevents test tuning",
        },
        {
            "loss_id": "P172-LOSS-005",
            "loss_or_metric": "closure_abs_error",
            "weight_or_guard": "must improve vs posterior-only",
            "selection_use": "validation and test audit",
            "rationale": "preserves interpretability",
        },
        {
            "loss_id": "P172-LOSS-006",
            "loss_or_metric": "coverage90_mean",
            "weight_or_guard": "must remain in [0.65, 1.0] if intervals are reported",
            "selection_use": "validation/test audit",
            "rationale": "keeps Bayesian language calibrated",
        },
        {
            "loss_id": "P172-LOSS-007",
            "loss_or_metric": "hot_q90_gradient_q90_rmse",
            "weight_or_guard": "must not degrade both region metrics",
            "selection_use": "validation/test audit",
            "rationale": "retains thermal hot/gradient relevance",
        },
    ]


def build_compute_rows() -> list[dict[str, Any]]:
    return [
        {
            "resource_id": "P172-COMPUTE-001",
            "resource": "local_cpu_numpy_or_tiny_torch",
            "allowed_now": False,
            "limit": "design only in Phase 172; Phase 173 may run if explicitly opened",
            "escalation_rule": "prefer local if torch import works; otherwise A800 tiny smoke",
        },
        {
            "resource_id": "P172-COMPUTE-002",
            "resource": "A800_40GB",
            "allowed_now": False,
            "limit": "reproduce design only now; future smoke must stay tiny",
            "escalation_rule": "allowed only by a later Phase 173 runner",
        },
        {
            "resource_id": "P172-COMPUTE-003",
            "resource": "A100_SXM4_80GB",
            "allowed_now": False,
            "limit": "not justified",
            "escalation_rule": "request only after a seed-positive branch hits measured 40GB blockage",
        },
    ]


def build_promotion_rows() -> list[dict[str, Any]]:
    return [
        {
            "rule_id": "P172-PROMOTE-001",
            "rule": "phase173_entry",
            "threshold": "Phase 172 design gate passes with all training locks false",
            "failure_action": "repair design before training",
        },
        {
            "rule_id": "P172-PROMOTE-002",
            "rule": "trainable_validation_gain",
            "threshold": "future trainable candidate validation score improves vs Phase 171 and posterior-only controls",
            "failure_action": "close as unnecessary trainable mechanism",
        },
        {
            "rule_id": "P172-PROMOTE-003",
            "rule": "test_reversal",
            "threshold": "future shifted-test reversal ratio <=1.05",
            "failure_action": "close and do not tune test",
        },
        {
            "rule_id": "P172-PROMOTE-004",
            "rule": "closure_interpretability",
            "threshold": "future closure error improves or matches Phase 171 without field degradation",
            "failure_action": "write as predictive diagnostic only",
        },
        {
            "rule_id": "P172-PROMOTE-005",
            "rule": "claim_boundary",
            "threshold": "future result remains synthetic until separate AM data gate",
            "failure_action": "do not claim AM-Bench, Bayesian PINN, or full GNN-PINN success",
        },
    ]


def build_risk_rows() -> list[dict[str, Any]]:
    return [
        {
            "risk_id": "P172-RISK-001",
            "risk": "trainable smoke adds no value over Phase 171",
            "guard": "Phase 171 closure head is a required control",
            "closure_action": "close if trainable variant does not improve validation",
        },
        {
            "risk_id": "P172-RISK-002",
            "risk": "overclaiming Bayesian PINN",
            "guard": "posterior warm-start is not neural posterior inference",
            "closure_action": "write as initialized latent model only",
        },
        {
            "risk_id": "P172-RISK-003",
            "risk": "sampler retuning repeats Phase 167 failure",
            "guard": "failure_sampler_retrain_block remains required",
            "closure_action": "do not select sampler route",
        },
        {
            "risk_id": "P172-RISK-004",
            "risk": "compute creep",
            "guard": "no training in Phase 172 and future max budget registered",
            "closure_action": "stop instead of scaling",
        },
        {
            "risk_id": "P172-RISK-005",
            "risk": "synthetic-to-AM overreach",
            "guard": "no raw data and no AM claim",
            "closure_action": "require later AM data gate",
        },
    ]


def _summary_lookup(rows: list[dict[str, str]], variant_id: str, split: str) -> dict[str, str]:
    for row in rows:
        if row.get("variant_id") == variant_id and row.get("split") == split:
            return row
    raise KeyError((variant_id, split))


def build_gate(
    *,
    phase171_gate: dict[str, Any],
    variant_summary_rows: list[dict[str, str]],
    seed_summary_rows: list[dict[str, str]],
    design_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    loss_rows: list[dict[str, Any]],
    compute_rows: list[dict[str, Any]],
    promotion_rows: list[dict[str, Any]],
    risk_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    phase171_ready = (
        phase171_gate.get("status")
        == "phase171_hidden_closure_low_budget_smoke_ready_phase172_trainable_design"
        and _is_true(phase171_gate.get("phase172_trainable_hidden_closure_design_allowed"))
        and not _is_true(phase171_gate.get("phase171_model_training_allowed"))
    )
    candidate_val = _summary_lookup(
        variant_summary_rows,
        "calibrated_hidden_source_closure_parameter_head",
        "val",
    )
    posterior_val = _summary_lookup(
        variant_summary_rows,
        "posterior_only_calibrated_bayesian_no_neural",
        "val",
    )
    candidate_positive = (
        float(phase171_gate.get("validation_score_gain_vs_best_control", 0.0)) > 0.0
        and float(phase171_gate.get("posterior_validation_closure_gain", 0.0)) > 0.0
        and float(phase171_gate.get("posterior_test_closure_gain", 0.0)) > 0.0
        and float(phase171_gate.get("seed_stability_pass_rate", 0.0)) >= 1.0
        and not phase171_gate.get("blocking_audits")
    )
    phase171_control_gap_small = (
        abs(
            float(candidate_val["field_rmse_mean"])
            - float(posterior_val["field_rmse_mean"])
        )
        <= 1e-10
    )
    enough_seed_rows = len(seed_summary_rows) >= 42
    no_training_now = (
        all(not _is_true(row["opens_training_now"]) for row in design_rows)
        and all(not _is_true(row["allowed_now"]) for row in compute_rows)
    )
    complete = (
        phase171_ready
        and candidate_positive
        and phase171_control_gap_small
        and enough_seed_rows
        and len(design_rows) >= 7
        and len(control_rows) >= 9
        and len(loss_rows) >= 7
        and len(compute_rows) >= 3
        and len(promotion_rows) >= 5
        and len(risk_rows) >= 5
        and no_training_now
    )
    blockers: list[str] = []
    if not phase171_ready:
        blockers.append("phase171_gate_not_ready")
    if not candidate_positive:
        blockers.append("phase171_candidate_positive_guard")
    if not phase171_control_gap_small:
        blockers.append("phase171_field_reversal_guard")
    if not enough_seed_rows:
        blockers.append("phase171_seed_summary_missing")
    if len(design_rows) < 7:
        blockers.append("missing_design_rows")
    if len(control_rows) < 9:
        blockers.append("missing_control_rows")
    if len(loss_rows) < 7:
        blockers.append("missing_loss_rows")
    if len(compute_rows) < 3:
        blockers.append("missing_compute_rows")
    if len(promotion_rows) < 5:
        blockers.append("missing_promotion_rows")
    if len(risk_rows) < 5:
        blockers.append("missing_risk_rows")
    if not no_training_now:
        blockers.append("phase172_attempted_training_now")
    return {
        "status": (
            "phase172_trainable_hidden_closure_smoke_design_ready_phase173_low_budget_trainable_smoke"
            if complete
            else "phase172_trainable_hidden_closure_smoke_design_incomplete"
        ),
        "candidate_trainable_route": "tiny_explicit_latent_hidden_closure_smoke",
        "phase171_candidate_variant": phase171_gate.get("candidate_variant"),
        "phase171_best_control_variant": phase171_gate.get("best_control_variant"),
        "phase171_validation_gain": phase171_gate.get("validation_score_gain_vs_best_control"),
        "phase171_posterior_test_closure_gain": phase171_gate.get("posterior_test_closure_gain"),
        "phase173_low_budget_trainable_smoke_allowed": bool(complete),
        "design_rows": len(design_rows),
        "control_rows": len(control_rows),
        "loss_rows": len(loss_rows),
        "compute_rows": len(compute_rows),
        "promotion_rows": len(promotion_rows),
        "risk_rows": len(risk_rows),
        "phase171_seed_summary_rows": len(seed_summary_rows),
        "blocking_audits": blockers,
        "phase172_model_mechanism_allowed": False,
        "phase172_model_training_allowed": False,
        "phase173_training_allowed_now": False,
        "bayesian_pinn_training_allowed_now": False,
        "adaptive_sampling_training_allowed_now": False,
        "gcn_pinn_training_allowed_now": False,
        "cnn_operator_training_allowed_now": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": (
            "enter Phase 173 bounded low-budget trainable hidden-closure smoke"
            if complete
            else "repair Phase 172 design before training"
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
    loss_rows: list[dict[str, Any]],
    compute_rows: list[dict[str, Any]],
    promotion_rows: list[dict[str, Any]],
    risk_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Phase 172 Trainable Hidden-Closure Smoke Design Gate",
        "",
        "## Gate",
        f"- Status: `{gate['status']}`",
        f"- Candidate trainable route: `{gate['candidate_trainable_route']}`",
        f"- Phase 173 low-budget trainable smoke allowed: `{_csv_value(gate['phase173_low_budget_trainable_smoke_allowed'])}`",
        f"- Phase 172 model training allowed: `{_csv_value(gate['phase172_model_training_allowed'])}`",
        f"- Phase 173 training allowed now: `{_csv_value(gate['phase173_training_allowed_now'])}`",
        f"- A100 training allowed now: `{_csv_value(gate['a100_training_allowed_now'])}`",
        f"- A100-SXM4-80GB request now: `{_csv_value(gate['a100_80gb_request_now'])}`",
        "",
        "## Interpretation",
        (
            "This is a design gate only. It may allow a later tiny trainable synthetic "
            "smoke, but it does not execute training and does not support AM-Bench, "
            "Bayesian PINN, GCN, CNN, operator, or A100-80GB claims."
        ),
        "",
        "## Design",
        *_markdown_table(design_rows, DESIGN_FIELDS),
        "",
        "## Controls",
        *_markdown_table(control_rows, CONTROL_FIELDS),
        "",
        "## Losses",
        *_markdown_table(loss_rows, LOSS_FIELDS),
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
    phase171_gate = _read_json(resolved["phase171_gate"])
    variant_summary_rows = _read_csv(resolved["phase171_variant_summary_table"])
    seed_summary_rows = _read_csv(resolved["phase171_seed_summary_table"])
    design_rows = build_design_rows(phase171_gate=phase171_gate)
    control_rows = build_control_rows()
    loss_rows = build_loss_rows()
    compute_rows = build_compute_rows()
    promotion_rows = build_promotion_rows()
    risk_rows = build_risk_rows()
    gate = build_gate(
        phase171_gate=phase171_gate,
        variant_summary_rows=variant_summary_rows,
        seed_summary_rows=seed_summary_rows,
        design_rows=design_rows,
        control_rows=control_rows,
        loss_rows=loss_rows,
        compute_rows=compute_rows,
        promotion_rows=promotion_rows,
        risk_rows=risk_rows,
    )

    design_path = output_dir / "phase172_trainable_design_table.csv"
    control_path = output_dir / "phase172_control_table.csv"
    loss_path = output_dir / "phase172_loss_table.csv"
    compute_path = output_dir / "phase172_compute_envelope_table.csv"
    promotion_path = output_dir / "phase172_promotion_rule_table.csv"
    risk_path = output_dir / "phase172_risk_table.csv"
    gate_path = output_dir / "phase172_trainable_hidden_closure_smoke_design_gate.json"
    markdown_path = output_dir / "phase172_trainable_hidden_closure_smoke_design_gate.md"
    manifest_path = output_dir / "phase172_trainable_hidden_closure_smoke_design_manifest.json"

    _write_csv(design_path, design_rows, DESIGN_FIELDS)
    _write_csv(control_path, control_rows, CONTROL_FIELDS)
    _write_csv(loss_path, loss_rows, LOSS_FIELDS)
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
                loss_rows=loss_rows,
                compute_rows=compute_rows,
                promotion_rows=promotion_rows,
                risk_rows=risk_rows,
            )
        )

    manifest = {
        "phase": 172,
        "description": "trainable hidden-closure low-budget smoke design gate",
        "inputs": {name: _display_path(path, root) for name, path in resolved.items()},
        "outputs": {
            "design_table": _display_path(design_path, root),
            "control_table": _display_path(control_path, root),
            "loss_table": _display_path(loss_path, root),
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
            "loss_rows": len(loss_rows),
            "compute_rows": len(compute_rows),
            "promotion_rows": len(promotion_rows),
            "risk_rows": len(risk_rows),
            "phase171_summary_rows": len(variant_summary_rows),
            "phase171_seed_summary_rows": len(seed_summary_rows),
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
