from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase171_hidden_closure_low_budget_smoke.py")
    spec = importlib.util.spec_from_file_location("phase171_hidden_closure_smoke", script)
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


def _control_rows() -> list[dict[str, object]]:
    names = [
        "posterior_only_calibrated_bayesian_no_neural",
        "grid_least_squares_source_closure_control",
        "no_closure_source_control",
        "uniform_grid_pinn_control",
        "data_only_tiny_mlp_no_residual",
        "wrong_source_prior_control",
        "failure_sampler_retrain_block",
        "seed_stability_control",
    ]
    return [
        {
            "control_id": f"P170-CTRL-{idx:03d}",
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
    phase170_status: str | None = None,
    phase171_allowed: bool = True,
) -> dict[str, Path]:
    phase170_gate = {
        "status": phase170_status
        or "phase170_hidden_closure_mechanism_smoke_design_ready_phase171_low_budget_smoke",
        "phase171_low_budget_hidden_closure_smoke_allowed": phase171_allowed,
        "phase170_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
    }
    return {
        "phase170_gate": _write_json(tmp_path / "p170/gate.json", phase170_gate),
        "phase170_control_table": _write_csv(
            tmp_path / "p170/control.csv",
            _control_rows(),
        ),
    }


def test_phase171_builds_positive_numpy_smoke_gate(tmp_path: Path, monkeypatch):
    module = _load_module()
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path),
    )

    gate = manifest["gate"]
    assert gate["status"] == (
        "phase171_hidden_closure_low_budget_smoke_ready_phase172_trainable_design"
    )
    assert gate["selected_variant"] == "calibrated_hidden_source_closure_parameter_head"
    assert gate["best_control_variant"] == "posterior_only_calibrated_bayesian_no_neural"
    assert gate["posterior_validation_closure_gain"] > 0
    assert gate["posterior_test_closure_gain"] > 0
    assert gate["phase172_trainable_hidden_closure_design_allowed"] is True
    assert gate["phase171_model_mechanism_allowed"] is False
    assert gate["phase171_model_training_allowed"] is False
    assert gate["phase172_training_allowed_now"] is False
    assert gate["bayesian_pinn_training_allowed_now"] is False
    assert gate["adaptive_sampling_training_allowed_now"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False

    variants = (tmp_path / "out/phase171_variant_table.csv").read_text(encoding="utf-8")
    assert "failure_sampler_retrain_block" in variants
    assert "false" in variants

    markdown = (tmp_path / "out/phase171_hidden_closure_low_budget_smoke.md").read_text(
        encoding="utf-8"
    )
    assert "NumPy-only" in markdown
    assert "|  |  |" not in markdown


def test_phase171_closes_when_phase170_not_ready(tmp_path: Path, monkeypatch):
    module = _load_module()
    original_run_smoke = module.run_smoke
    monkeypatch.setattr(
        module,
        "run_smoke",
        lambda: original_run_smoke(seeds=(171,), center_grid_size=24, width_grid_size=22),
    )
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(
            tmp_path,
            phase170_status="phase170_hidden_closure_mechanism_smoke_design_incomplete",
            phase171_allowed=False,
        ),
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase171_hidden_closure_low_budget_smoke_closed_no_stable_mechanism_gain"
    assert "phase170_gate_not_ready" in gate["blocking_audits"]
    assert gate["phase172_trainable_hidden_closure_design_allowed"] is False
    assert gate["phase171_model_training_allowed"] is False
    assert gate["a100_80gb_request_now"] is False


def test_phase171_blocks_if_required_controls_missing(tmp_path: Path, monkeypatch):
    module = _load_module()
    original_run_smoke = module.run_smoke
    monkeypatch.setattr(
        module,
        "run_smoke",
        lambda: original_run_smoke(seeds=(171,), center_grid_size=24, width_grid_size=22),
    )
    paths = _paths(tmp_path)
    _write_csv(tmp_path / "p170/control.csv", _control_rows()[:3])
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=paths,
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase171_hidden_closure_low_budget_smoke_closed_no_stable_mechanism_gain"
    assert "phase170_control_contract_missing" in gate["blocking_audits"]
    assert gate["phase172_trainable_hidden_closure_design_allowed"] is False
