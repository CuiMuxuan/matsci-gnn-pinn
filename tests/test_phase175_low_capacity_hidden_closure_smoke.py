from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase175_low_capacity_hidden_closure_smoke.py")
    spec = importlib.util.spec_from_file_location("phase175_low_capacity_smoke", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
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


def _phase174_controls() -> list[dict[str, object]]:
    names = [
        "phase173_tiny_explicit_latent_hidden_closure_smoke",
        "phase171_numpy_closure_head_control",
        "posterior_only_calibrated_bayesian_no_neural",
        "grid_least_squares_source_closure_control",
        "no_closure_source_control",
        "wrong_source_prior_control",
        "data_only_tiny_control",
        "uniform_grid_latent_trainable_control",
        "failure_sampler_retrain_block",
        "seed_stability_control",
    ]
    return [
        {
            "control_id": f"P174-CTRL-{idx:03d}",
            "control_name": name,
            "role": "required",
            "required_metric": "selection_score",
            "promotion_requirement": "candidate must pass",
        }
        for idx, name in enumerate(names, start=1)
    ]


def _paths(
    tmp_path: Path,
    *,
    phase174_status: str | None = None,
    phase175_allowed: bool = True,
    phase173_status: str | None = None,
    phase174_design_allowed: bool = True,
) -> dict[str, Path]:
    phase174_gate = {
        "status": phase174_status
        or "phase174_low_capacity_hidden_closure_design_ready_phase175_low_capacity_smoke",
        "phase175_low_capacity_smoke_allowed": phase175_allowed,
        "phase174_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
    }
    phase173_gate = {
        "status": phase173_status
        or "phase173_trainable_hidden_closure_low_budget_smoke_ready_phase174_low_capacity_hidden_closure_design",
        "phase174_low_capacity_hidden_closure_design_allowed": phase174_design_allowed,
        "phase173_model_training_allowed": False,
    }
    return {
        "phase174_gate": _write_json(tmp_path / "p174/gate.json", phase174_gate),
        "phase174_control_table": _write_csv(
            tmp_path / "p174/control.csv",
            _phase174_controls(),
        ),
        "phase173_gate": _write_json(tmp_path / "p173/gate.json", phase173_gate),
    }


def _closed_smoke_payload(module):
    variant_rows = module.build_variant_rows()
    summary_rows = [
        {
            "variant_id": "low_capacity_explicit_latent_hidden_closure_head",
            "family": "mechanism_candidate",
            "split": "val",
            "seed_count": 3,
            "case_count": 48,
            "field_rmse_mean": 0.0066,
            "hot_q90_rmse_mean": 0.0088,
            "gradient_q90_rmse_mean": 0.0091,
            "closure_abs_error_mean": 0.0159,
            "coverage90_mean": 0.88,
            "selection_score_mean": 0.0243,
            "selection_score_std": 0.01,
        },
        {
            "variant_id": "low_capacity_explicit_latent_hidden_closure_head",
            "family": "mechanism_candidate",
            "split": "test",
            "seed_count": 3,
            "case_count": 48,
            "field_rmse_mean": 0.0063,
            "hot_q90_rmse_mean": 0.0078,
            "gradient_q90_rmse_mean": 0.0090,
            "closure_abs_error_mean": 0.0142,
            "coverage90_mean": 0.94,
            "selection_score_mean": 0.0220,
            "selection_score_std": 0.01,
        },
        {
            "variant_id": "phase173_tiny_explicit_latent_hidden_closure_smoke",
            "family": "control",
            "split": "val",
            "seed_count": 3,
            "case_count": 48,
            "field_rmse_mean": 0.0063,
            "hot_q90_rmse_mean": 0.0086,
            "gradient_q90_rmse_mean": 0.0090,
            "closure_abs_error_mean": 0.0160,
            "coverage90_mean": 0.88,
            "selection_score_mean": 0.0238,
            "selection_score_std": 0.01,
        },
        {
            "variant_id": "phase173_tiny_explicit_latent_hidden_closure_smoke",
            "family": "control",
            "split": "test",
            "seed_count": 3,
            "case_count": 48,
            "field_rmse_mean": 0.0060,
            "hot_q90_rmse_mean": 0.0075,
            "gradient_q90_rmse_mean": 0.0088,
            "closure_abs_error_mean": 0.0140,
            "coverage90_mean": 0.94,
            "selection_score_mean": 0.0214,
            "selection_score_std": 0.01,
        },
    ]
    seed_summary_rows = [
        {
            "seed": seed,
            "variant_id": variant,
            "split": split,
            "case_count": 16,
            "field_rmse_mean": 0.006,
            "closure_abs_error_mean": 0.014,
            "selection_score_mean": score,
        }
        for seed in (171, 172, 173)
        for split in ("val", "test")
        for variant, score in (
            ("low_capacity_explicit_latent_hidden_closure_head", 0.0243),
            ("phase173_tiny_explicit_latent_hidden_closure_smoke", 0.0238),
        )
    ]
    training_audit_rows = [
        {
            "seed": 171 + idx % 3,
            "case_id": f"P169-CASE-{idx:03d}",
            "split": "val",
            "variant_id": "low_capacity_explicit_latent_hidden_closure_head",
            "start_count": 3,
            "max_rounds_per_start": 48,
            "executed_rounds_total": 77,
            "function_evaluations_total": 619,
            "ridge_alpha": 1e-6,
            "learned_center_shift": 0.0,
            "learned_source_width": 0.064,
            "raw_closure_coeff": 0.01,
            "head_center_shift": 0.0,
            "head_source_width": 0.064,
            "head_closure_coeff": 0.01,
        }
        for idx in range(48)
    ]
    return {
        "variant_rows": variant_rows,
        "calibration_rows": [],
        "training_audit_rows": training_audit_rows,
        "case_metric_rows": [],
        "summary_rows": summary_rows,
        "seed_summary_rows": seed_summary_rows,
    }


def test_phase175_closes_when_low_capacity_fails_phase173_guard(tmp_path: Path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "run_smoke", lambda: _closed_smoke_payload(module))

    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path),
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase175_low_capacity_hidden_closure_smoke_closed_no_incremental_gain"
    assert gate["selected_variant"] == "phase173_tiny_explicit_latent_hidden_closure_smoke"
    assert "validation_selected_control_variant" in gate["blocking_audits"]
    assert "phase173_validation_score_gain_guard" in gate["blocking_audits"]
    assert gate["phase176_focused_review_allowed"] is False
    assert gate["phase175_model_mechanism_allowed"] is False
    assert gate["phase175_model_training_allowed"] is False
    assert gate["phase176_training_allowed_now"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False

    markdown = (
        tmp_path / "out/phase175_low_capacity_hidden_closure_smoke.md"
    ).read_text(encoding="utf-8")
    assert "did not beat the simpler Phase 173" in markdown
    assert "|  |  |" not in markdown


def test_phase175_closes_when_phase174_not_ready(tmp_path: Path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "run_smoke", lambda: _closed_smoke_payload(module))

    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(
            tmp_path,
            phase174_status="phase174_low_capacity_hidden_closure_design_incomplete",
            phase175_allowed=False,
        ),
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase175_low_capacity_hidden_closure_smoke_closed_no_incremental_gain"
    assert "phase174_gate_not_ready" in gate["blocking_audits"]
    assert gate["phase176_focused_review_allowed"] is False


def test_phase175_blocks_if_required_controls_missing(tmp_path: Path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "run_smoke", lambda: _closed_smoke_payload(module))
    paths = _paths(tmp_path)
    _write_csv(tmp_path / "p174/control.csv", _phase174_controls()[:3])

    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=paths,
    )

    gate = manifest["gate"]
    assert "phase174_control_contract_missing" in gate["blocking_audits"]
    assert gate["phase176_focused_review_allowed"] is False
