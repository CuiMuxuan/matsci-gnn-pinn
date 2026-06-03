from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase96_pfhub_local_smoke_gate.py")
    spec = importlib.util.spec_from_file_location("phase96_gate", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _paths(tmp_path: Path, *, phase96_allowed: bool = True) -> dict[str, Path]:
    phase95_gate = {
        "status": "local_design_ready_no_a100" if phase96_allowed else "blocked_design_incomplete",
        "source_candidate": "P94-CAND-PFHUB-PINN",
        "phase96_local_smoke_allowed": phase96_allowed,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
    }
    phase95_design = {
        "candidate_id": "phase95_pfhub_local_physics_v1",
        "source_candidate": "P94-CAND-PFHUB-PINN",
        "selected_benchmark_style": "PFHub benchmark-derived Allen-Cahn / heat-diffusion surrogate",
    }
    return {
        "phase95_gate": _write_json(tmp_path / "phase95_gate.json", phase95_gate),
        "phase95_candidate_design": _write_json(
            tmp_path / "phase95_candidate_design.json", phase95_design
        ),
    }


def test_phase96_local_smoke_opens_transfer_design_only(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(tmp_path, tmp_path / "out", paths=_paths(tmp_path))

    gate = manifest["gate"]
    assert manifest["phase"] == 96
    assert gate["status"] == "local_smoke_positive_transfer_design_only"
    assert gate["phase97_transfer_design_allowed"] is True
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert gate["transfer_design_signals"] == 1
    assert gate["positive_mechanisms"] == ["fixed_green_function_features"]

    target = json.loads((tmp_path / manifest["outputs"]["target_manifest"]).read_text())
    assert target["target_id"] == "phase96_pfhub_style_heat_source_v1"
    assert target["selection_policy"].startswith("validation-only")
    assert "not AM-Bench evidence" in target["not_a_claim"]

    with (tmp_path / manifest["outputs"]["metric_table"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        metric_rows = list(csv.DictReader(handle))
    assert len(metric_rows) == 5
    fixed = next(row for row in metric_rows if row["method_id"] == "fixed_green_function_features")
    vanilla = next(row for row in metric_rows if row["method_id"] == "vanilla_deterministic_surrogate")
    adaptive = next(row for row in metric_rows if row["method_id"] == "bayesian_adaptive_collocation")
    random_row = next(row for row in metric_rows if row["method_id"] == "random_collocation_same_budget")

    assert fixed["validation_gate_pass"] == "true"
    assert fixed["test_audit_pass"] == "true"
    assert float(fixed["test_rmse"]) < float(vanilla["test_rmse"])
    assert float(fixed["test_hot_q90_rmse"]) < float(vanilla["test_hot_q90_rmse"])
    assert float(fixed["test_gradient_q90_rmse"]) < float(vanilla["test_gradient_q90_rmse"])

    assert adaptive["test_audit_pass"] == "false"
    assert float(adaptive["test_rmse"]) > float(random_row["test_rmse"])
    assert float(adaptive["test_hot_q90_rmse"]) < float(random_row["test_hot_q90_rmse"])

    with (tmp_path / manifest["outputs"]["mechanism_decision_table"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        mechanism_rows = list(csv.DictReader(handle))
    assert len(mechanism_rows) == 2
    fixed_decision = next(row for row in mechanism_rows if row["mechanism"] == "fixed_green_function_features")
    adaptive_decision = next(
        row for row in mechanism_rows if row["mechanism"] == "bayesian_adaptive_collocation"
    )
    assert fixed_decision["transfer_design_signal"] == "true"
    assert adaptive_decision["transfer_design_signal"] == "false"


def test_phase96_blocks_if_phase95_does_not_allow_smoke(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        tmp_path, tmp_path / "out", paths=_paths(tmp_path, phase96_allowed=False)
    )

    gate = manifest["gate"]
    assert gate["status"] == "blocked_by_phase95_design_gate"
    assert gate["phase97_transfer_design_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert manifest["counts"]["metric_rows"] == 0
    assert manifest["counts"]["mechanism_rows"] == 0
