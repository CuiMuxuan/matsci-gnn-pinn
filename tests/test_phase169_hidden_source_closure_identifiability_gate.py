from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    script = Path(
        "scripts/server/build_phase169_hidden_source_closure_identifiability_gate.py"
    )
    spec = importlib.util.spec_from_file_location("phase169_identifiability_gate", script)
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


def _paths(tmp_path: Path, *, phase168_status: str | None = None) -> dict[str, Path]:
    phase168_gate = {
        "status": phase168_status
        or "phase168_hidden_source_closure_redesign_ready_phase169_identifiability_gate",
        "phase169_hidden_source_closure_identifiability_gate_allowed": True,
        "phase168_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
    }
    design_rows = [
        {
            "design_id": f"P168-DESIGN-{idx:03d}",
            "component": "contract",
        }
        for idx in range(5)
    ]
    return {
        "phase168_gate": _write_json(tmp_path / "p168/gate.json", phase168_gate),
        "phase168_design_table": _write_csv(tmp_path / "p168/design.csv", design_rows),
    }


def _fast_cases(generate_cases):
    return generate_cases(seed=169)[:20]


def _fixture_prediction(case, *, center_error: float, width_error: float, closure_error: float, interval: bool):
    pred_center = case.center_shift + center_error
    pred_width = case.source_width + width_error
    pred_closure = case.closure_coeff + closure_error
    margin = 0.025 if interval else 0.0
    return {
        "pred_center_shift": pred_center,
        "center_shift_ci90_low": case.center_shift - margin if interval else "",
        "center_shift_ci90_high": case.center_shift + margin if interval else "",
        "pred_source_width": pred_width,
        "source_width_ci90_low": case.source_width - margin if interval else "",
        "source_width_ci90_high": case.source_width + margin if interval else "",
        "pred_closure_coeff": pred_closure,
        "closure_coeff_ci90_low": case.closure_coeff - margin if interval else "",
        "closure_coeff_ci90_high": case.closure_coeff + margin if interval else "",
    }


def _fast_predictions(module, cases):
    rows = []
    for case in cases:
        method_predictions = {
            "bayesian_hidden_source_closure_posterior": _fixture_prediction(
                case,
                center_error=0.001,
                width_error=0.001,
                closure_error=0.010,
                interval=False,
            ),
            "calibrated_bayesian_hidden_source_closure_posterior": _fixture_prediction(
                case,
                center_error=0.001,
                width_error=0.001,
                closure_error=0.010,
                interval=True,
            ),
            "grid_least_squares_source_closure_control": _fixture_prediction(
                case,
                center_error=0.006,
                width_error=0.006,
                closure_error=0.055,
                interval=False,
            ),
            "no_closure_source_control": _fixture_prediction(
                case,
                center_error=0.004,
                width_error=0.004,
                closure_error=-case.closure_coeff,
                interval=False,
            ),
            "moment_linearized_closure_control": _fixture_prediction(
                case,
                center_error=0.020,
                width_error=0.020,
                closure_error=0.080,
                interval=False,
            ),
            "extra_trees_sensor_control": _fixture_prediction(
                case,
                center_error=0.015,
                width_error=0.015,
                closure_error=0.065,
                interval=False,
            ),
        }
        for method, pred in method_predictions.items():
            rows.append(
                {
                    "method": method,
                    "case_id": case.case_id,
                    "split": case.split,
                    "true_center_shift": case.center_shift,
                    "pred_center_shift": pred["pred_center_shift"],
                    "center_shift_ci90_low": pred["center_shift_ci90_low"],
                    "center_shift_ci90_high": pred["center_shift_ci90_high"],
                    "true_source_width": case.source_width,
                    "pred_source_width": pred["pred_source_width"],
                    "source_width_ci90_low": pred["source_width_ci90_low"],
                    "source_width_ci90_high": pred["source_width_ci90_high"],
                    "true_closure_coeff": case.closure_coeff,
                    "pred_closure_coeff": pred["pred_closure_coeff"],
                    "closure_coeff_ci90_low": pred["closure_coeff_ci90_low"],
                    "closure_coeff_ci90_high": pred["closure_coeff_ci90_high"],
                }
            )
    return rows


def test_phase169_builds_no_training_identifiability_gate(tmp_path: Path, monkeypatch):
    module = _load_module()
    original_generate_cases = module.generate_cases
    monkeypatch.setattr(module, "generate_cases", lambda: _fast_cases(original_generate_cases))
    monkeypatch.setattr(module, "build_predictions", lambda cases: _fast_predictions(module, cases))
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path),
    )

    gate = manifest["gate"]
    assert gate["status"] == (
        "phase169_hidden_source_closure_identifiability_ready_phase170_low_budget_mechanism_design"
    )
    assert gate["candidate_method"] == "calibrated_bayesian_hidden_source_closure_posterior"
    assert gate["validation_score_gain_vs_best_control"] >= 0.008
    assert gate["test_reversal_ratio_vs_best_control"] <= 1.05
    assert gate["candidate_validation_closure_coeff_rmse"] <= 0.020
    assert gate["candidate_test_closure_coeff_rmse"] <= 0.025
    assert gate["phase170_low_budget_mechanism_design_allowed"] is True
    assert gate["phase169_model_mechanism_allowed"] is False
    assert gate["phase169_model_training_allowed"] is False
    assert gate["phase170_training_allowed_now"] is False
    assert gate["bayesian_pinn_training_allowed_now"] is False
    assert gate["adaptive_sampling_training_allowed_now"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert manifest["counts"]["case_rows"] == 20
    assert manifest["counts"]["prediction_rows"] == 120

    metric_table = (tmp_path / "out/phase169_hidden_source_metric_table.csv").read_text(
        encoding="utf-8"
    )
    assert "calibrated_bayesian_hidden_source_closure_posterior" in metric_table
    assert "extra_trees_sensor_control" in metric_table
    assert "no_closure_source_control" in metric_table

    markdown = (
        tmp_path / "out/phase169_hidden_source_closure_identifiability_gate.md"
    ).read_text(encoding="utf-8")
    assert "does not train a PINN" in markdown
    assert "|  |  |" not in markdown


def test_phase169_closes_if_phase168_not_ready(tmp_path: Path, monkeypatch):
    module = _load_module()
    original_generate_cases = module.generate_cases
    monkeypatch.setattr(module, "generate_cases", lambda: _fast_cases(original_generate_cases))
    monkeypatch.setattr(module, "build_predictions", lambda cases: _fast_predictions(module, cases))
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path, phase168_status="phase168_incomplete"),
    )

    gate = manifest["gate"]
    assert gate["status"] == (
        "phase169_hidden_source_closure_identifiability_closed_no_guarded_identifiability"
    )
    assert "phase168_gate_not_ready" in gate["blocking_audits"]
    assert gate["phase170_low_budget_mechanism_design_allowed"] is False
    assert gate["phase169_model_training_allowed"] is False
    assert gate["a100_80gb_request_now"] is False
