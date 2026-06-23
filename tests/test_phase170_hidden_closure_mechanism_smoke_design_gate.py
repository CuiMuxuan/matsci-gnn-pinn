from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase170_hidden_closure_mechanism_smoke_design_gate.py")
    spec = importlib.util.spec_from_file_location("phase170_mechanism_design_gate", script)
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


def _metric_rows() -> list[dict[str, object]]:
    methods = [
        ("calibrated_bayesian_hidden_source_closure_posterior", "bayesian_candidate"),
        ("bayesian_hidden_source_closure_posterior", "bayesian_candidate"),
        ("grid_least_squares_source_closure_control", "control"),
        ("no_closure_source_control", "control"),
        ("moment_linearized_closure_control", "control"),
        ("extra_trees_sensor_control", "control"),
    ]
    rows = []
    for method, family in methods:
        for split in ("train", "val", "test"):
            rows.append(
                {
                    "method": method,
                    "method_family": family,
                    "split": split,
                    "case_count": 16,
                    "center_shift_rmse": 0.001,
                    "source_width_rmse": 0.001,
                    "closure_coeff_rmse": 0.015,
                    "joint_normalized_rmse": 0.05,
                    "coverage90_mean": 0.9 if family == "bayesian_candidate" else 0.0,
                    "calibration_gap": 0.0 if family == "bayesian_candidate" else 0.9,
                    "selection_score": 0.05 if family == "bayesian_candidate" else 0.12,
                }
            )
    return rows


def _paths(
    tmp_path: Path,
    *,
    phase169_status: str | None = None,
    phase171_allowed: bool = True,
    metric_rows: list[dict[str, object]] | None = None,
) -> dict[str, Path]:
    phase167_gate = {
        "status": "phase167_low_budget_pinn_smoke_closed_no_stable_model_gain",
        "selected_variant": "uniform_grid_pinn",
        "best_control_variant": "uniform_grid_pinn",
        "phase167_model_claim_allowed": False,
        "phase167_model_mechanism_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
    }
    phase169_gate = {
        "status": phase169_status
        or "phase169_hidden_source_closure_identifiability_ready_phase170_low_budget_mechanism_design",
        "selected_method": "calibrated_bayesian_hidden_source_closure_posterior",
        "candidate_method": "calibrated_bayesian_hidden_source_closure_posterior",
        "best_control_method": "grid_least_squares_source_closure_control",
        "validation_score_gain_vs_best_control": 0.071,
        "test_reversal_ratio_vs_best_control": 1.01,
        "candidate_validation_closure_coeff_rmse": 0.015,
        "candidate_test_closure_coeff_rmse": 0.023,
        "candidate_validation_coverage90_mean": 0.94,
        "candidate_test_coverage90_mean": 0.89,
        "blocking_audits": [],
        "phase170_low_budget_mechanism_design_allowed": phase171_allowed,
        "phase169_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
    }
    return {
        "phase167_gate": _write_json(tmp_path / "p167/gate.json", phase167_gate),
        "phase169_gate": _write_json(tmp_path / "p169/gate.json", phase169_gate),
        "phase169_metric_table": _write_csv(
            tmp_path / "p169/metric.csv",
            metric_rows if metric_rows is not None else _metric_rows(),
        ),
    }


def test_phase170_builds_design_gate_without_training(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path),
    )

    gate = manifest["gate"]
    assert gate["status"] == (
        "phase170_hidden_closure_mechanism_smoke_design_ready_phase171_low_budget_smoke"
    )
    assert gate["phase171_low_budget_hidden_closure_smoke_allowed"] is True
    assert gate["phase170_model_mechanism_allowed"] is False
    assert gate["phase170_model_training_allowed"] is False
    assert gate["phase171_training_allowed_now"] is False
    assert gate["bayesian_pinn_training_allowed_now"] is False
    assert gate["adaptive_sampling_training_allowed_now"] is False
    assert gate["gcn_pinn_training_allowed_now"] is False
    assert gate["cnn_operator_training_allowed_now"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert manifest["counts"]["evidence_rows"] >= 6
    assert manifest["counts"]["mechanism_rows"] >= 7
    assert manifest["counts"]["control_rows"] >= 8
    assert manifest["counts"]["loss_metric_rows"] >= 7
    assert manifest["counts"]["risk_rows"] >= 5

    markdown = (
        tmp_path / "out/phase170_hidden_closure_mechanism_smoke_design_gate.md"
    ).read_text(encoding="utf-8")
    assert "does not execute training" in markdown
    assert "grid_least_squares_source_closure_control" in markdown
    assert "failure_sampler_retrain_block" in markdown
    assert "|  |  |" not in markdown

    controls = (tmp_path / "out/phase170_control_table.csv").read_text(encoding="utf-8")
    assert "posterior_only_calibrated_bayesian_no_neural" in controls
    assert "uniform_grid_pinn_control" in controls


def test_phase170_closes_when_phase169_not_ready(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(
            tmp_path,
            phase169_status="phase169_hidden_source_closure_identifiability_closed_no_guarded_identifiability",
            phase171_allowed=False,
        ),
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase170_hidden_closure_mechanism_smoke_design_incomplete"
    assert "phase169_gate_not_ready" in gate["blocking_audits"]
    assert gate["phase171_low_budget_hidden_closure_smoke_allowed"] is False
    assert gate["phase170_model_training_allowed"] is False
    assert gate["a100_80gb_request_now"] is False


def test_phase170_blocks_when_metric_contract_missing(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path, metric_rows=_metric_rows()[:4]),
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase170_hidden_closure_mechanism_smoke_design_incomplete"
    assert "phase169_metric_contract_missing" in gate["blocking_audits"]
    assert gate["phase171_low_budget_hidden_closure_smoke_allowed"] is False
