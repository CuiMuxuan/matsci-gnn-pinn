from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase166_low_budget_pinn_smoke_design_gate.py")
    spec = importlib.util.spec_from_file_location("phase166_smoke_design_gate", script)
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


def _paths(
    tmp_path: Path,
    *,
    phase165_status: str | None = None,
    phase166_allowed: bool = True,
    phase165_gain: float = 0.25,
) -> dict[str, Path]:
    phase164_gate = {
        "status": "phase164_synthetic_bayesian_inverse_heat_identifiability_ready_phase165_sampler_gate",
        "phase165_adaptive_sampler_gate_allowed": True,
        "phase164_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
    }
    phase165_gate = {
        "status": phase165_status
        or "phase165_adaptive_residual_sampler_ready_low_budget_pinn_smoke_design",
        "phase166_low_budget_pinn_smoke_design_allowed": phase166_allowed,
        "selected_sampler": "failure_informed_hot_gradient",
        "validation_score_gain_vs_best_control": phase165_gain,
        "test_score_gain_vs_best_control": 0.24,
        "selected_validation_boundary_fraction": 0.21,
        "selected_test_boundary_fraction": 0.20,
        "phase165_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
    }
    metric_rows = [
        {"scenario": "validation_nominal", "sampler": f"sampler_{idx}", "score": idx}
        for idx in range(10)
    ]
    return {
        "phase164_gate": _write_json(tmp_path / "p164/gate.json", phase164_gate),
        "phase165_gate": _write_json(tmp_path / "p165/gate.json", phase165_gate),
        "phase165_metric_table": _write_csv(
            tmp_path / "p165/metric_table.csv",
            metric_rows,
        ),
    }


def test_phase166_builds_design_gate_without_training(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path),
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase166_low_budget_pinn_smoke_design_ready_phase167_local_smoke"
    assert gate["phase167_local_low_budget_pinn_smoke_allowed"] is True
    assert gate["phase166_model_mechanism_allowed"] is False
    assert gate["phase166_model_training_allowed"] is False
    assert gate["phase167_training_allowed_now"] is False
    assert gate["bayesian_pinn_training_allowed_now"] is False
    assert gate["adaptive_sampling_training_allowed_now"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert manifest["counts"]["design_rows"] >= 6
    assert manifest["counts"]["control_rows"] >= 5
    assert manifest["counts"]["reference_rows"] >= 8
    assert manifest["counts"]["risk_rows"] >= 4

    markdown = (tmp_path / "out/phase166_low_budget_pinn_smoke_design_gate.md").read_text(
        encoding="utf-8"
    )
    assert "does not execute training" in markdown
    assert "Route References" in markdown
    assert "10.1016/j.jcp.2020.109913" in markdown
    assert "|  |  |" not in markdown

    references = (tmp_path / "out/phase166_route_reference_table.csv").read_text(
        encoding="utf-8"
    )
    assert "Failure-Informed Adaptive Sampling for PINNs" in references
    assert "graph neural Galerkin" in references


def test_phase166_closes_when_phase165_gate_not_ready(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(
            tmp_path,
            phase165_status="phase165_adaptive_residual_sampler_closed_no_stable_sampler_gain",
            phase166_allowed=False,
        ),
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase166_low_budget_pinn_smoke_design_incomplete"
    assert "phase165_gate_not_ready" in gate["blocking_audits"]
    assert gate["phase167_local_low_budget_pinn_smoke_allowed"] is False
    assert gate["phase166_model_training_allowed"] is False
    assert gate["a100_80gb_request_now"] is False


def test_phase166_blocks_when_sampler_gain_removed(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path, phase165_gain=0.01),
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase166_low_budget_pinn_smoke_design_incomplete"
    assert "sampler_validation_gain_guard" in gate["blocking_audits"]
    assert gate["phase167_local_low_budget_pinn_smoke_allowed"] is False
