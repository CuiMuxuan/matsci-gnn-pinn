from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase165_adaptive_residual_sampler_gate.py")
    spec = importlib.util.spec_from_file_location("phase165_sampler_gate", script)
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


def _paths(tmp_path: Path, *, phase164_status: str | None = None) -> dict[str, Path]:
    phase164_gate = {
        "status": phase164_status
        or "phase164_synthetic_bayesian_inverse_heat_identifiability_ready_phase165_sampler_gate",
        "phase165_adaptive_sampler_gate_allowed": True,
        "phase164_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
    }
    literature_rows = [
        {"source_id": "P163-LIT-003", "trust_state": "verified"},
        {"source_id": "P163-LIT-004", "trust_state": "verified"},
        {"source_id": "P163-LIT-010", "trust_state": "verified"},
    ]
    return {
        "phase164_gate": _write_json(tmp_path / "p164/gate.json", phase164_gate),
        "phase163_literature_table": _write_csv(
            tmp_path / "p163/literature.csv",
            literature_rows,
        ),
    }


def test_phase165_builds_no_training_sampler_gate(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path),
    )

    gate = manifest["gate"]
    assert gate["status"] == (
        "phase165_adaptive_residual_sampler_ready_low_budget_pinn_smoke_design"
    )
    assert gate["selected_sampler"] == "failure_informed_hot_gradient"
    assert gate["validation_score_gain_vs_best_control"] >= 0.08
    assert gate["test_score_gain_vs_best_control"] >= 0.05
    assert gate["selected_validation_boundary_fraction"] >= 0.08
    assert gate["selected_test_boundary_fraction"] >= 0.08
    assert gate["phase166_low_budget_pinn_smoke_design_allowed"] is True
    assert gate["phase165_low_capacity_training_allowed"] is False
    assert gate["phase165_model_mechanism_allowed"] is False
    assert gate["phase165_model_training_allowed"] is False
    assert gate["bayesian_pinn_training_allowed_now"] is False
    assert gate["adaptive_sampling_training_allowed_now"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert manifest["counts"]["sampler_rows"] == 5
    assert manifest["counts"]["seed_metric_rows"] == 30
    assert manifest["counts"]["metric_rows"] == 10

    metric_table = (tmp_path / "out/phase165_sampler_metric_table.csv").read_text(
        encoding="utf-8"
    )
    assert "failure_informed_hot_gradient" in metric_table
    assert "uniform_grid_control" in metric_table

    markdown = (tmp_path / "out/phase165_adaptive_residual_sampler_gate.md").read_text(
        encoding="utf-8"
    )
    assert "does not train a PINN" in markdown
    assert "|  |  |" not in markdown


def test_phase165_closes_if_phase164_not_ready(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path, phase164_status="phase164_closed"),
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase165_adaptive_residual_sampler_closed_no_stable_sampler_gain"
    assert "phase164_gate_not_ready" in gate["blocking_audits"]
    assert gate["phase166_low_budget_pinn_smoke_design_allowed"] is False
    assert gate["phase165_model_training_allowed"] is False
    assert gate["a100_80gb_request_now"] is False
