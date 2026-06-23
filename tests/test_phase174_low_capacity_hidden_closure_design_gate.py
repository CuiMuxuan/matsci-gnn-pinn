from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase174_low_capacity_hidden_closure_design_gate.py")
    spec = importlib.util.spec_from_file_location("phase174_low_capacity_design", script)
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


def _summary_rows() -> list[dict[str, object]]:
    return [
        {
            "variant_id": "tiny_explicit_latent_hidden_closure_smoke",
            "family": "mechanism_candidate",
            "split": "val",
            "seed_count": 3,
            "case_count": 48,
            "field_rmse_mean": 0.0062717940,
            "hot_q90_rmse_mean": 0.0085512203,
            "gradient_q90_rmse_mean": 0.0089651896,
            "closure_abs_error_mean": 0.0160014312,
            "coverage90_mean": 0.8819444444,
            "selection_score_mean": 0.0238240749,
            "selection_score_std": 0.0111767491,
        },
        {
            "variant_id": "tiny_explicit_latent_hidden_closure_smoke",
            "family": "mechanism_candidate",
            "split": "test",
            "seed_count": 3,
            "case_count": 48,
            "field_rmse_mean": 0.0059616267,
            "hot_q90_rmse_mean": 0.0074503416,
            "gradient_q90_rmse_mean": 0.0087659511,
            "closure_abs_error_mean": 0.0140415939,
            "coverage90_mean": 0.9375,
            "selection_score_mean": 0.0213594856,
            "selection_score_std": 0.0091051469,
        },
        {
            "variant_id": "uniform_grid_latent_trainable_control",
            "family": "control",
            "split": "val",
            "seed_count": 3,
            "case_count": 48,
            "field_rmse_mean": 0.0062720170,
            "hot_q90_rmse_mean": 0.0085552192,
            "gradient_q90_rmse_mean": 0.0089679272,
            "closure_abs_error_mean": 0.0154830329,
            "coverage90_mean": 0.0,
            "selection_score_mean": 0.0404921283,
            "selection_score_std": 0.0106266041,
        },
    ]


def _audit_rows(count: int = 240) -> list[dict[str, object]]:
    return [
        {
            "seed": 171 + (idx % 3),
            "case_id": f"P169-CASE-{idx:03d}",
            "split": "train" if idx % 5 == 0 else "val",
            "variant_id": "tiny_explicit_latent_hidden_closure_smoke",
            "start_count": 3,
            "max_rounds_per_start": 48,
            "executed_rounds_total": 77,
            "function_evaluations_total": 619,
            "train_objective": 0.001,
            "learned_center_shift": 0.0,
            "learned_source_width": 0.064,
            "raw_closure_coeff": 0.02,
            "calibrated_closure_coeff": 0.02,
        }
        for idx in range(count)
    ]


def _paths(
    tmp_path: Path,
    *,
    phase173_status: str | None = None,
    phase174_allowed: bool = True,
    audit_count: int = 240,
) -> dict[str, Path]:
    phase173_gate = {
        "status": phase173_status
        or "phase173_trainable_hidden_closure_low_budget_smoke_ready_phase174_low_capacity_hidden_closure_design",
        "selected_variant": "tiny_explicit_latent_hidden_closure_smoke",
        "candidate_variant": "tiny_explicit_latent_hidden_closure_smoke",
        "best_control_variant": "uniform_grid_latent_trainable_control",
        "validation_score_gain_vs_best_control": 0.0166680534,
        "candidate_validation_selection_score": 0.0238240749,
        "candidate_test_selection_score": 0.0213594856,
        "phase171_validation_score_gain": 0.0372880265,
        "phase171_test_closure_gain": 0.0032139974,
        "seed_stability_pass_rate": 1.0,
        "blocking_audits": [],
        "phase174_low_capacity_hidden_closure_design_allowed": phase174_allowed,
        "phase173_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
    }
    return {
        "phase173_gate": _write_json(tmp_path / "p173/gate.json", phase173_gate),
        "phase173_variant_summary_table": _write_csv(
            tmp_path / "p173/summary.csv",
            _summary_rows(),
        ),
        "phase173_training_audit_table": _write_csv(
            tmp_path / "p173/training_audit.csv",
            _audit_rows(audit_count),
        ),
    }


def test_phase174_builds_design_gate_without_training(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path),
    )

    gate = manifest["gate"]
    assert gate["status"] == (
        "phase174_low_capacity_hidden_closure_design_ready_phase175_low_capacity_smoke"
    )
    assert gate["phase175_low_capacity_smoke_allowed"] is True
    assert gate["phase174_model_mechanism_allowed"] is False
    assert gate["phase174_model_training_allowed"] is False
    assert gate["phase175_training_allowed_now"] is False
    assert gate["bayesian_pinn_training_allowed_now"] is False
    assert gate["adaptive_sampling_training_allowed_now"] is False
    assert gate["gcn_pinn_training_allowed_now"] is False
    assert gate["cnn_operator_training_allowed_now"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert manifest["counts"]["design_rows"] >= 7
    assert manifest["counts"]["control_rows"] >= 10
    assert manifest["counts"]["metric_rows"] >= 7

    markdown = (
        tmp_path / "out/phase174_low_capacity_hidden_closure_design_gate.md"
    ).read_text(encoding="utf-8")
    assert "does not execute training" in markdown
    assert "low_capacity_explicit_latent_hidden_closure_head" in markdown
    assert "|  |  |" not in markdown


def test_phase174_closes_when_phase173_not_ready(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(
            tmp_path,
            phase173_status="phase173_trainable_hidden_closure_low_budget_smoke_closed_no_stable_trainable_gain",
            phase174_allowed=False,
        ),
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase174_low_capacity_hidden_closure_design_incomplete"
    assert "phase173_gate_not_ready" in gate["blocking_audits"]
    assert gate["phase175_low_capacity_smoke_allowed"] is False
    assert gate["phase174_model_training_allowed"] is False
    assert gate["a100_80gb_request_now"] is False


def test_phase174_blocks_when_training_audit_missing(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path, audit_count=4),
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase174_low_capacity_hidden_closure_design_incomplete"
    assert "phase173_budget_guard" in gate["blocking_audits"]
    assert gate["phase175_low_capacity_smoke_allowed"] is False
