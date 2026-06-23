from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    script = Path(
        "scripts/server/build_phase164_synthetic_bayesian_inverse_heat_identifiability_gate.py"
    )
    spec = importlib.util.spec_from_file_location("phase164_inverse_heat_gate", script)
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


def _paths(tmp_path: Path, *, phase163_status: str | None = None) -> dict[str, Path]:
    phase163_gate = {
        "status": phase163_status
        or "phase163_pinn_bayesian_hybrid_roadmap_ready_phase164_synthetic_inverse_gate",
        "phase164_no_training_design_allowed": True,
        "phase163_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
    }
    route_rows = [
        {
            "route_id": "P163-ROUTE-001",
            "route_name": "bayesian_inverse_heat_parameter_synthetic_gate",
            "training_allowed_now": False,
            "a100_80gb_request_now": False,
        },
        {
            "route_id": "P163-ROUTE-002",
            "route_name": "adaptive_residual_sampler_heat_gate",
            "training_allowed_now": False,
            "a100_80gb_request_now": False,
        },
    ]
    return {
        "phase163_gate": _write_json(tmp_path / "p163/gate.json", phase163_gate),
        "phase163_route_table": _write_csv(tmp_path / "p163/routes.csv", route_rows),
    }


def test_phase164_builds_no_training_identifiability_gate(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path),
    )

    gate = manifest["gate"]
    assert gate["status"] == (
        "phase164_synthetic_bayesian_inverse_heat_identifiability_ready_phase165_sampler_gate"
    )
    assert gate["selected_method"] == "calibrated_bayesian_grid_posterior"
    assert gate["validation_score_gain_vs_best_control"] >= 0.005
    assert gate["test_reversal_ratio_vs_best_control"] <= 1.05
    assert gate["phase165_adaptive_sampler_gate_allowed"] is True
    assert gate["phase164_low_capacity_training_allowed"] is False
    assert gate["phase164_model_mechanism_allowed"] is False
    assert gate["phase164_model_training_allowed"] is False
    assert gate["bayesian_pinn_training_allowed_now"] is False
    assert gate["adaptive_sampling_training_allowed_now"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert manifest["counts"]["case_rows"] == 24
    assert manifest["counts"]["prediction_rows"] == 120

    metric_table = (tmp_path / "out/phase164_inverse_metric_table.csv").read_text(
        encoding="utf-8"
    )
    assert "bayesian_grid_posterior" in metric_table
    assert "extra_trees_sensor_control" in metric_table

    markdown = (
        tmp_path / "out/phase164_synthetic_bayesian_inverse_heat_identifiability_gate.md"
    ).read_text(encoding="utf-8")
    assert "does not permit AM-Bench Bayesian PINN training" in markdown
    assert "|  |  |" not in markdown


def test_phase164_incomplete_if_phase163_not_ready(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path, phase163_status="phase163_incomplete"),
    )

    gate = manifest["gate"]
    assert gate["status"] == (
        "phase164_synthetic_bayesian_inverse_heat_identifiability_closed_no_guarded_gain"
    )
    assert "phase163_gate_not_ready" in gate["blocking_audits"]
    assert gate["phase165_adaptive_sampler_gate_allowed"] is False
    assert gate["phase164_model_training_allowed"] is False
    assert gate["a100_80gb_request_now"] is False
