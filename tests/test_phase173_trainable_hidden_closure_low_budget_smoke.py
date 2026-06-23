from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase173_trainable_hidden_closure_low_budget_smoke.py")
    spec = importlib.util.spec_from_file_location("phase173_trainable_hidden_closure", script)
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


def _phase172_controls() -> list[dict[str, object]]:
    names = [
        "phase171_numpy_closure_head",
        "posterior_only_calibrated_bayesian_no_neural",
        "grid_least_squares_source_closure_control",
        "no_closure_source_control",
        "data_only_tiny_control",
        "wrong_source_prior_control",
        "uniform_grid_pinn_control",
        "failure_sampler_retrain_block",
        "seed_stability_control",
    ]
    return [
        {
            "control_id": f"P172-CTRL-{idx:03d}",
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
    phase172_status: str | None = None,
    phase173_allowed: bool = True,
    phase171_status: str | None = None,
    phase172_design_allowed: bool = True,
) -> dict[str, Path]:
    phase172_gate = {
        "status": phase172_status
        or "phase172_trainable_hidden_closure_smoke_design_ready_phase173_low_budget_trainable_smoke",
        "phase173_low_budget_trainable_smoke_allowed": phase173_allowed,
        "phase172_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
    }
    phase171_gate = {
        "status": phase171_status
        or "phase171_hidden_closure_low_budget_smoke_ready_phase172_trainable_design",
        "phase172_trainable_hidden_closure_design_allowed": phase172_design_allowed,
        "phase171_model_training_allowed": False,
    }
    return {
        "phase172_gate": _write_json(tmp_path / "p172/gate.json", phase172_gate),
        "phase172_control_table": _write_csv(
            tmp_path / "p172/control.csv",
            _phase172_controls(),
        ),
        "phase171_gate": _write_json(tmp_path / "p171/gate.json", phase171_gate),
    }


def _fast_smoke(run_smoke, seeds=(171, 172, 173)):
    return run_smoke(
        seeds=seeds,
        center_grid_size=18,
        width_grid_size=16,
        max_latent_rounds=20,
    )


def test_phase173_builds_positive_trainable_smoke_gate(tmp_path: Path, monkeypatch):
    module = _load_module()
    original_run_smoke = module.run_smoke
    monkeypatch.setattr(module, "run_smoke", lambda: _fast_smoke(original_run_smoke))

    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path),
    )

    gate = manifest["gate"]
    assert gate["status"] == (
        "phase173_trainable_hidden_closure_low_budget_smoke_ready_phase174_low_capacity_hidden_closure_design"
    )
    assert gate["selected_variant"] == "tiny_explicit_latent_hidden_closure_smoke"
    assert gate["validation_score_gain_vs_best_control"] > 0.005
    assert gate["phase171_validation_score_gain"] > 0.01
    assert gate["phase171_test_score_gain"] > 0.01
    assert gate["phase171_validation_closure_gain"] > 0.005
    assert gate["seed_stability_pass_rate"] == 1.0
    assert gate["phase174_low_capacity_hidden_closure_design_allowed"] is True
    assert gate["phase173_model_mechanism_allowed"] is False
    assert gate["phase173_model_training_allowed"] is False
    assert gate["phase174_training_allowed_now"] is False
    assert gate["bayesian_pinn_training_allowed_now"] is False
    assert gate["adaptive_sampling_training_allowed_now"] is False
    assert gate["gcn_pinn_training_allowed_now"] is False
    assert gate["cnn_operator_training_allowed_now"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert manifest["counts"]["training_audit_rows"] > 0

    variants = (tmp_path / "out/phase173_variant_table.csv").read_text(encoding="utf-8")
    assert "failure_sampler_retrain_block" in variants
    assert "tiny_explicit_latent_hidden_closure_smoke" in variants

    markdown = (
        tmp_path / "out/phase173_trainable_hidden_closure_low_budget_smoke.md"
    ).read_text(encoding="utf-8")
    assert "tiny synthetic trainable-latent smoke" in markdown
    assert "|  |  |" not in markdown

def test_phase173_closes_when_phase172_not_ready(tmp_path: Path, monkeypatch):
    module = _load_module()
    original_run_smoke = module.run_smoke
    monkeypatch.setattr(
        module,
        "run_smoke",
        lambda: _fast_smoke(original_run_smoke, seeds=(171,)),
    )

    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(
            tmp_path,
            phase172_status="phase172_trainable_hidden_closure_smoke_design_incomplete",
            phase173_allowed=False,
        ),
    )

    gate = manifest["gate"]
    assert gate["status"] == (
        "phase173_trainable_hidden_closure_low_budget_smoke_closed_no_stable_trainable_gain"
    )
    assert "phase172_gate_not_ready" in gate["blocking_audits"]
    assert gate["phase174_low_capacity_hidden_closure_design_allowed"] is False
    assert gate["phase173_model_training_allowed"] is False
    assert gate["a100_80gb_request_now"] is False


def test_phase173_blocks_if_required_controls_missing(tmp_path: Path, monkeypatch):
    module = _load_module()
    original_run_smoke = module.run_smoke
    monkeypatch.setattr(
        module,
        "run_smoke",
        lambda: _fast_smoke(original_run_smoke, seeds=(171,)),
    )
    paths = _paths(tmp_path)
    _write_csv(tmp_path / "p172/control.csv", _phase172_controls()[:3])

    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=paths,
    )

    gate = manifest["gate"]
    assert gate["status"] == (
        "phase173_trainable_hidden_closure_low_budget_smoke_closed_no_stable_trainable_gain"
    )
    assert "phase172_control_contract_missing" in gate["blocking_audits"]
    assert gate["phase174_low_capacity_hidden_closure_design_allowed"] is False
