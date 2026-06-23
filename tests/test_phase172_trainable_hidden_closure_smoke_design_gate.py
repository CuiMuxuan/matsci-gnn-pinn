from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase172_trainable_hidden_closure_smoke_design_gate.py")
    spec = importlib.util.spec_from_file_location("phase172_trainable_design_gate", script)
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


def _variant_summary_rows() -> list[dict[str, object]]:
    return [
        {
            "variant_id": "calibrated_hidden_source_closure_parameter_head",
            "family": "mechanism_candidate",
            "split": "val",
            "seed_count": 3,
            "case_count": 48,
            "field_rmse_mean": 0.024,
            "closure_abs_error_mean": 0.026,
            "selection_score_mean": 0.061,
        },
        {
            "variant_id": "posterior_only_calibrated_bayesian_no_neural",
            "family": "control",
            "split": "val",
            "seed_count": 3,
            "case_count": 48,
            "field_rmse_mean": 0.024,
            "closure_abs_error_mean": 0.028,
            "selection_score_mean": 0.062,
        },
    ]


def _seed_summary_rows(count: int = 42) -> list[dict[str, object]]:
    return [
        {
            "seed": 171 + (idx % 3),
            "variant_id": f"variant_{idx % 7}",
            "split": "val" if idx % 2 == 0 else "test",
            "case_count": 16,
            "field_rmse_mean": 0.02,
            "closure_abs_error_mean": 0.02,
            "selection_score_mean": 0.06,
        }
        for idx in range(count)
    ]


def _paths(
    tmp_path: Path,
    *,
    phase171_status: str | None = None,
    phase172_allowed: bool = True,
    seed_row_count: int = 42,
) -> dict[str, Path]:
    phase171_gate = {
        "status": phase171_status
        or "phase171_hidden_closure_low_budget_smoke_ready_phase172_trainable_design",
        "candidate_variant": "calibrated_hidden_source_closure_parameter_head",
        "best_control_variant": "posterior_only_calibrated_bayesian_no_neural",
        "validation_score_gain_vs_best_control": 0.0011,
        "posterior_validation_closure_gain": 0.0015,
        "posterior_test_closure_gain": 0.0031,
        "seed_stability_pass_rate": 1.0,
        "blocking_audits": [],
        "phase172_trainable_hidden_closure_design_allowed": phase172_allowed,
        "phase171_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
    }
    return {
        "phase171_gate": _write_json(tmp_path / "p171/gate.json", phase171_gate),
        "phase171_variant_summary_table": _write_csv(
            tmp_path / "p171/summary.csv",
            _variant_summary_rows(),
        ),
        "phase171_seed_summary_table": _write_csv(
            tmp_path / "p171/seed_summary.csv",
            _seed_summary_rows(seed_row_count),
        ),
    }


def test_phase172_builds_design_gate_without_training(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path),
    )

    gate = manifest["gate"]
    assert gate["status"] == (
        "phase172_trainable_hidden_closure_smoke_design_ready_phase173_low_budget_trainable_smoke"
    )
    assert gate["phase173_low_budget_trainable_smoke_allowed"] is True
    assert gate["phase172_model_mechanism_allowed"] is False
    assert gate["phase172_model_training_allowed"] is False
    assert gate["phase173_training_allowed_now"] is False
    assert gate["bayesian_pinn_training_allowed_now"] is False
    assert gate["adaptive_sampling_training_allowed_now"] is False
    assert gate["gcn_pinn_training_allowed_now"] is False
    assert gate["cnn_operator_training_allowed_now"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert manifest["counts"]["design_rows"] >= 7
    assert manifest["counts"]["control_rows"] >= 9
    assert manifest["counts"]["loss_rows"] >= 7

    markdown = (
        tmp_path / "out/phase172_trainable_hidden_closure_smoke_design_gate.md"
    ).read_text(encoding="utf-8")
    assert "does not execute training" in markdown
    assert "tiny_explicit_latent_hidden_closure_smoke" in markdown
    assert "|  |  |" not in markdown


def test_phase172_closes_when_phase171_not_ready(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(
            tmp_path,
            phase171_status="phase171_hidden_closure_low_budget_smoke_closed_no_stable_mechanism_gain",
            phase172_allowed=False,
        ),
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase172_trainable_hidden_closure_smoke_design_incomplete"
    assert "phase171_gate_not_ready" in gate["blocking_audits"]
    assert gate["phase173_low_budget_trainable_smoke_allowed"] is False
    assert gate["phase172_model_training_allowed"] is False
    assert gate["a100_80gb_request_now"] is False


def test_phase172_blocks_when_seed_summary_missing(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path, seed_row_count=4),
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase172_trainable_hidden_closure_smoke_design_incomplete"
    assert "phase171_seed_summary_missing" in gate["blocking_audits"]
    assert gate["phase173_low_budget_trainable_smoke_allowed"] is False
