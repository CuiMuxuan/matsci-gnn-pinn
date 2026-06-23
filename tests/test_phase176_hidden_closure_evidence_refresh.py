from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase176_hidden_closure_evidence_refresh.py")
    spec = importlib.util.spec_from_file_location("phase176_evidence_refresh", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _paths(tmp_path: Path, *, phase175_status: str | None = None) -> dict[str, Path]:
    return {
        "phase169_gate": _write_json(
            tmp_path / "p169/gate.json",
            {
                "status": "phase169_hidden_source_closure_identifiability_ready_phase170_low_budget_mechanism_design",
                "best_control_method": "grid_least_squares_source_closure_control",
                "validation_score_gain_vs_best_control": 0.0707838724,
                "phase169_model_training_allowed": False,
                "a100_training_allowed_now": False,
                "a100_80gb_request_now": False,
            },
        ),
        "phase170_gate": _write_json(
            tmp_path / "p170/gate.json",
            {
                "status": "phase170_hidden_closure_mechanism_smoke_design_ready_phase171_low_budget_smoke",
                "control_rows": 8,
                "promotion_rows": 5,
                "phase170_model_training_allowed": False,
                "a100_training_allowed_now": False,
                "a100_80gb_request_now": False,
            },
        ),
        "phase171_gate": _write_json(
            tmp_path / "p171/gate.json",
            {
                "status": "phase171_hidden_closure_low_budget_smoke_ready_phase172_trainable_design",
                "selected_variant": "calibrated_hidden_source_closure_parameter_head",
                "best_control_variant": "posterior_only_calibrated_bayesian_no_neural",
                "validation_score_gain_vs_best_control": 0.0011359227,
                "seed_stability_pass_rate": 1.0,
                "phase171_model_training_allowed": False,
                "a100_training_allowed_now": False,
                "a100_80gb_request_now": False,
            },
        ),
        "phase172_gate": _write_json(
            tmp_path / "p172/gate.json",
            {
                "status": "phase172_trainable_hidden_closure_smoke_design_ready_phase173_low_budget_trainable_smoke",
                "candidate_trainable_route": "tiny_explicit_latent_hidden_closure_smoke",
                "phase172_model_training_allowed": False,
                "a100_training_allowed_now": False,
                "a100_80gb_request_now": False,
            },
        ),
        "phase173_gate": _write_json(
            tmp_path / "p173/gate.json",
            {
                "status": "phase173_trainable_hidden_closure_low_budget_smoke_ready_phase174_low_capacity_hidden_closure_design",
                "selected_variant": "tiny_explicit_latent_hidden_closure_smoke",
                "best_control_variant": "uniform_grid_latent_trainable_control",
                "validation_score_gain_vs_best_control": 0.0166680534,
                "test_reversal_ratio_vs_best_control": 0.5605813176,
                "seed_stability_pass_rate": 1.0,
                "phase173_model_training_allowed": False,
                "a100_training_allowed_now": False,
                "a100_80gb_request_now": False,
            },
        ),
        "phase174_gate": _write_json(
            tmp_path / "p174/gate.json",
            {
                "status": "phase174_low_capacity_hidden_closure_design_ready_phase175_low_capacity_smoke",
                "candidate_low_capacity_route": "low_capacity_explicit_latent_hidden_closure_head",
                "phase174_model_training_allowed": False,
                "a100_training_allowed_now": False,
                "a100_80gb_request_now": False,
            },
        ),
        "phase175_gate": _write_json(
            tmp_path / "p175/gate.json",
            {
                "status": phase175_status
                or "phase175_low_capacity_hidden_closure_smoke_closed_no_incremental_gain",
                "selected_variant": "phase173_tiny_explicit_latent_hidden_closure_smoke",
                "candidate_variant": "low_capacity_explicit_latent_hidden_closure_head",
                "validation_score_gain_vs_best_control": -0.0004223396,
                "test_reversal_ratio_vs_best_control": 1.0307333811,
                "seed_stability_pass_rate": 0.3333333333,
                "blocking_audits": [
                    "validation_selected_control_variant",
                    "validation_gain_vs_best_control",
                ],
                "phase176_focused_review_allowed": False,
                "phase175_model_training_allowed": False,
                "a100_training_allowed_now": False,
                "a100_80gb_request_now": False,
            },
        ),
    }


def test_phase176_refreshes_branch_and_preserves_training_locks(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path),
    )

    gate = manifest["gate"]
    assert gate["status"] == (
        "phase176_hidden_closure_evidence_refresh_ready_synthetic_claims_low_capacity_closed"
    )
    assert gate["synthetic_hidden_closure_claim_allowed_now"] is True
    assert gate["second_paper_core_claim_ready"] is False
    assert gate["low_capacity_head_claim_ready"] is False
    assert gate["phase177_materially_different_mechanism_design_allowed"] is True
    assert gate["phase176_model_training_allowed"] is False
    assert gate["phase177_training_allowed_now"] is False
    assert gate["bayesian_pinn_training_allowed_now"] is False
    assert gate["gcn_pinn_training_allowed_now"] is False
    assert gate["cnn_operator_training_allowed_now"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert manifest["counts"]["route_evidence_rows"] == 7
    assert manifest["counts"]["model_training_allowed_route_rows"] == 0

    markdown = (tmp_path / "out/phase176_hidden_closure_evidence_refresh.md").read_text(
        encoding="utf-8"
    )
    assert "controlled synthetic inverse-heat tasks" in markdown
    assert "Do not claim the Phase 174/175 low-capacity head" in markdown
    assert "A100-SXM4-80GB request now: `false`" in markdown
    assert "|  |  |" not in markdown


def test_phase176_incomplete_if_phase175_not_closed(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path, phase175_status="phase175_unexpected_open"),
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase176_hidden_closure_evidence_refresh_incomplete"
    assert gate["phase175_low_capacity_route_closed"] is False
    assert gate["phase177_materially_different_mechanism_design_allowed"] is False
    assert gate["phase176_model_training_allowed"] is False
    assert gate["a100_80gb_request_now"] is False
