from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase163_pinn_bayesian_hybrid_roadmap.py")
    spec = importlib.util.spec_from_file_location("phase163_pinn_roadmap", script)
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


def _paths(tmp_path: Path, *, phase154_status: str | None = None) -> dict[str, Path]:
    phase154_gate = {
        "status": phase154_status
        or "phase154_route_coverage_audit_ready_current_routes_verified_future_not_exhausted",
        "currently_executable_model_routes_verified": True,
        "all_possible_future_schemes_exhausted": False,
        "phase154_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
    }
    phase162_gate = {
        "status": "phase162_uci_steel_industry_energy_closed_no_stable_guarded_gap",
        "phase162_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
    }
    route_rows = [
        {
            "route_id": "P154-ROUTE-003",
            "route_family": "registered_source_path_green_capl",
            "current_status": "closed_diagnostic_under_current_registration",
        },
        {
            "route_id": "P154-ROUTE-002",
            "route_family": "neural_operator_fno_dense_field",
            "current_status": "closed_diagnostic_no_operator_gap",
        },
    ]
    return {
        "phase154_gate": _write_json(tmp_path / "p154/gate.json", phase154_gate),
        "phase154_route_table": _write_csv(tmp_path / "p154/routes.csv", route_rows),
        "phase162_gate": _write_json(tmp_path / "p162/gate.json", phase162_gate),
    }


def test_phase163_builds_no_training_pinn_roadmap(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path),
    )

    gate = manifest["gate"]
    assert (
        gate["status"]
        == "phase163_pinn_bayesian_hybrid_roadmap_ready_phase164_synthetic_inverse_gate"
    )
    assert gate["verified_literature_rows"] >= 10
    assert gate["recommended_next_phase"] == (
        "phase164_synthetic_bayesian_inverse_heat_identifiability_gate"
    )
    assert gate["phase164_no_training_design_allowed"] is True
    assert gate["phase164_low_capacity_training_allowed"] is False
    assert gate["phase163_model_mechanism_allowed"] is False
    assert gate["phase163_model_training_allowed"] is False
    assert gate["bayesian_pinn_training_allowed_now"] is False
    assert gate["adaptive_sampling_training_allowed_now"] is False
    assert gate["gcn_pinn_training_allowed_now"] is False
    assert gate["cnn_pinn_training_allowed_now"] is False
    assert gate["operator_training_allowed_now"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert manifest["counts"]["route_candidate_rows"] == 6
    assert manifest["counts"]["execution_queue_rows"] == 3
    assert manifest["counts"]["training_allowed_route_rows"] == 0

    route_table = (tmp_path / "out/phase163_route_candidate_table.csv").read_text(
        encoding="utf-8"
    )
    assert "bayesian_inverse_heat_parameter_synthetic_gate" in route_table
    assert "gcn_or_path_graph_pinn_residual" in route_table
    assert "Phase 148" in route_table

    markdown = (tmp_path / "out/phase163_pinn_bayesian_hybrid_roadmap.md").read_text(
        encoding="utf-8"
    )
    assert "Bayesian/adaptive/hybrid PINN" in markdown
    assert "local no-training synthetic inverse-heat identifiability gate" in markdown
    assert "|  |  |" not in markdown


def test_phase163_incomplete_if_phase154_not_ready(tmp_path: Path):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path, phase154_status="phase154_incomplete"),
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase163_pinn_bayesian_hybrid_roadmap_incomplete"
    assert gate["phase164_no_training_design_allowed"] is False
    assert gate["phase163_model_training_allowed"] is False
    assert gate["a100_80gb_request_now"] is False
