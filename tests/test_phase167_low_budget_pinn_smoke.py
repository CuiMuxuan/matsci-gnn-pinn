from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase167_low_budget_pinn_smoke.py")
    spec = importlib.util.spec_from_file_location("phase167_low_budget_pinn_smoke", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _phase166_gate(status: str | None = None) -> dict[str, object]:
    return {
        "status": status or "phase166_low_budget_pinn_smoke_design_ready_phase167_local_smoke",
        "phase167_local_low_budget_pinn_smoke_allowed": True,
        "phase166_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
    }


def _variant_rows() -> list[dict[str, object]]:
    return [
        {
            "variant_id": "data_only_tiny_mlp_no_residual",
            "family": "control",
            "sampler_id": "none",
        },
        {
            "variant_id": "uniform_grid_pinn",
            "family": "control",
            "sampler_id": "uniform_grid_control",
        },
        {
            "variant_id": "wrong_prior_failure_sampler_control",
            "family": "control",
            "sampler_id": "failure_informed_hot_gradient",
        },
        {
            "variant_id": "failure_informed_hot_gradient_pinn",
            "family": "adaptive_candidate",
            "sampler_id": "failure_informed_hot_gradient",
        },
    ]


def _summary_rows(*, adaptive_best: bool = True) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base_scores = {
        "data_only_tiny_mlp_no_residual": 0.44,
        "uniform_grid_pinn": 0.40,
        "wrong_prior_failure_sampler_control": 0.43,
        "failure_informed_hot_gradient_pinn": 0.36 if adaptive_best else 0.42,
    }
    for variant_id, score in base_scores.items():
        family = "adaptive_candidate" if variant_id == "failure_informed_hot_gradient_pinn" else "control"
        for split, multiplier in (("val", 1.0), ("test", 1.02)):
            rows.append(
                {
                    "variant_id": variant_id,
                    "family": family,
                    "sampler_id": "failure_informed_hot_gradient",
                    "split": split,
                    "seed_count": 3,
                    "temperature_rmse_mean": score * multiplier,
                    "temperature_rmse_std": 0.01,
                    "hot_q90_rmse_mean": score * multiplier,
                    "gradient_q90_rmse_mean": score * multiplier,
                    "residual_rmse_mean": 0.2,
                    "selection_score_mean": score * multiplier,
                    "selection_score_std": 0.02,
                }
            )
    return rows


def test_phase167_gate_promotes_only_adaptive_smoke_candidate():
    module = _load_module()
    gate = module.build_gate(
        phase166_gate=_phase166_gate(),
        variant_rows=_variant_rows(),
        summary_rows=_summary_rows(adaptive_best=True),
        local_torch_blocked=True,
    )

    assert gate["status"] == "phase167_low_budget_pinn_smoke_ready_phase168_focused_review"
    assert gate["selected_variant"] == "failure_informed_hot_gradient_pinn"
    assert gate["phase168_focused_review_allowed"] is True
    assert gate["phase167_model_mechanism_allowed"] is False
    assert gate["phase167_model_claim_allowed"] is False
    assert gate["bayesian_pinn_training_allowed_now"] is False
    assert gate["adaptive_sampling_training_allowed_now"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False


def test_phase167_closes_when_control_selected():
    module = _load_module()
    gate = module.build_gate(
        phase166_gate=_phase166_gate(),
        variant_rows=_variant_rows(),
        summary_rows=_summary_rows(adaptive_best=False),
        local_torch_blocked=True,
    )

    assert gate["status"] == "phase167_low_budget_pinn_smoke_closed_no_stable_model_gain"
    assert "validation_selected_control_variant" in gate["blocking_audits"]
    assert gate["phase168_focused_review_allowed"] is False
    assert gate["a100_80gb_request_now"] is False


def test_phase167_closes_if_phase166_not_ready():
    module = _load_module()
    gate = module.build_gate(
        phase166_gate=_phase166_gate("phase166_incomplete"),
        variant_rows=_variant_rows(),
        summary_rows=_summary_rows(adaptive_best=True),
        local_torch_blocked=True,
    )

    assert gate["status"] == "phase167_low_budget_pinn_smoke_closed_no_stable_model_gain"
    assert "phase166_gate_not_ready" in gate["blocking_audits"]
    assert gate["phase168_focused_review_allowed"] is False


def test_phase167_static_tables_have_expected_variants():
    module = _load_module()
    rows = module.build_variant_rows()
    payload = json.dumps(rows)
    assert "data_only_tiny_mlp_no_residual" in payload
    assert "uniform_grid_pinn" in payload
    assert "wrong_prior_failure_sampler_control" in payload
    assert "failure_informed_hot_gradient_pinn" in payload
