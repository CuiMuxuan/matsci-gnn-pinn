from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path(
        "scripts/server/build_phase177_uncertainty_guided_latent_acquisition_design_gate.py"
    )
    spec = importlib.util.spec_from_file_location("phase177_acquisition_design", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _phase176_route_rows() -> list[dict[str, object]]:
    return [
        {
            "route_id": f"P176-ROUTE-{idx:03d}",
            "phase": f"phase{168 + idx}",
            "model_training_allowed": False,
            "a100_training_allowed_now": False,
            "a100_80gb_request_now": False,
        }
        for idx in range(1, 8)
    ]


def _phase176_claim_rows() -> list[dict[str, object]]:
    return [
        {
            "claim_id": f"P176-CLAIM-{idx:03d}",
            "claim_area": "synthetic_hidden_closure",
            "claim_status": "allowed_narrow_positive" if idx < 3 else "blocked_success_claim",
        }
        for idx in range(1, 8)
    ]


def _paths(
    tmp_path: Path,
    *,
    phase176_ready: bool = True,
    phase175_closed: bool = True,
) -> dict[str, Path]:
    return {
        "phase176_gate": _write_json(
            tmp_path / "p176/gate.json",
            {
                "status": "phase176_hidden_closure_evidence_refresh_ready_synthetic_claims_low_capacity_closed"
                if phase176_ready
                else "phase176_hidden_closure_evidence_refresh_incomplete",
                "phase177_materially_different_mechanism_design_allowed": phase176_ready,
                "phase176_model_training_allowed": False,
                "phase177_training_allowed_now": False,
                "a100_80gb_request_now": False,
            },
        ),
        "phase176_route_table": _write_csv(
            tmp_path / "p176/routes.csv",
            _phase176_route_rows(),
        ),
        "phase176_claim_table": _write_csv(
            tmp_path / "p176/claims.csv",
            _phase176_claim_rows(),
        ),
        "phase169_gate": _write_json(
            tmp_path / "p169/gate.json",
            {
                "status": "phase169_hidden_source_closure_identifiability_ready_phase170_low_budget_mechanism_design",
                "validation_score_gain_vs_best_control": 0.0707838724,
                "candidate_validation_coverage90_mean": 0.9375,
            },
        ),
        "phase173_gate": _write_json(
            tmp_path / "p173/gate.json",
            {
                "status": "phase173_trainable_hidden_closure_low_budget_smoke_ready_phase174_low_capacity_hidden_closure_design",
                "selected_variant": "tiny_explicit_latent_hidden_closure_smoke",
                "best_control_variant": "uniform_grid_latent_trainable_control",
                "validation_score_gain_vs_best_control": 0.0166680534,
            },
        ),
        "phase175_gate": _write_json(
            tmp_path / "p175/gate.json",
            {
                "status": "phase175_low_capacity_hidden_closure_smoke_closed_no_incremental_gain"
                if phase175_closed
                else "phase175_low_capacity_hidden_closure_smoke_ready_focused_review",
                "candidate_variant": "low_capacity_explicit_latent_hidden_closure_head",
                "validation_score_gain_vs_best_control": -0.0004223396,
                "phase176_focused_review_allowed": False,
            },
        ),
    }


def test_phase177_builds_materially_different_design_gate(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path),
    )

    gate = manifest["gate"]
    assert gate["status"] == (
        "phase177_uncertainty_guided_latent_acquisition_design_ready_phase178_no_training_smoke"
    )
    assert gate["candidate_mechanism"] == "posterior_ensemble_uncertainty_guided_latent_acquisition"
    assert gate["materially_different_from_phase175"] is True
    assert gate["phase178_no_training_acquisition_smoke_allowed"] is True
    assert gate["phase177_model_mechanism_allowed"] is False
    assert gate["phase177_model_training_allowed"] is False
    assert gate["phase178_training_allowed_now"] is False
    assert gate["bayesian_pinn_training_allowed_now"] is False
    assert gate["adaptive_sampling_training_allowed_now"] is False
    assert gate["gcn_pinn_training_allowed_now"] is False
    assert gate["cnn_operator_training_allowed_now"] is False
    assert gate["am_bench_training_allowed_now"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert manifest["counts"]["control_rows"] >= 10
    assert manifest["counts"]["acquisition_rows"] >= 6

    markdown = (
        tmp_path / "out/phase177_uncertainty_guided_latent_acquisition_design_gate.md"
    ).read_text(encoding="utf-8")
    assert "design-only gate" in markdown
    assert "posterior_ensemble_uncertainty_guided_latent_acquisition" in markdown
    assert "low-capacity head" in markdown
    assert "|  |  |" not in markdown


def test_phase177_incomplete_without_phase176_permission(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path, phase176_ready=False),
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase177_uncertainty_guided_latent_acquisition_design_incomplete"
    assert "phase176_gate_not_ready" in gate["blocking_audits"]
    assert gate["phase178_no_training_acquisition_smoke_allowed"] is False
    assert gate["phase177_model_training_allowed"] is False
    assert gate["a100_80gb_request_now"] is False


def test_phase177_incomplete_if_low_capacity_closure_missing(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path, phase175_closed=False),
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase177_uncertainty_guided_latent_acquisition_design_incomplete"
    assert "phase175_low_capacity_closure_missing" in gate["blocking_audits"]
    assert gate["phase178_training_allowed_now"] is False
    assert gate["a100_training_allowed_now"] is False
