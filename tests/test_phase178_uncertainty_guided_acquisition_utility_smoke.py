from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path(
        "scripts/server/build_phase178_uncertainty_guided_acquisition_utility_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("phase178_acquisition_smoke", script)
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


def _phase177_acquisition_rows() -> list[dict[str, object]]:
    return [
        {"acquisition_id": f"P177-ACQ-{idx:03d}", "policy": f"policy_{idx}"}
        for idx in range(1, 7)
    ]


def _phase177_control_rows() -> list[dict[str, object]]:
    return [
        {"control_id": f"P177-CTRL-{idx:03d}", "control_name": f"control_{idx}"}
        for idx in range(1, 11)
    ]


def _paths(tmp_path: Path, *, phase177_ready: bool = True) -> dict[str, Path]:
    return {
        "phase177_gate": _write_json(
            tmp_path / "p177/gate.json",
            {
                "status": "phase177_uncertainty_guided_latent_acquisition_design_ready_phase178_no_training_smoke"
                if phase177_ready
                else "phase177_uncertainty_guided_latent_acquisition_design_incomplete",
                "phase178_no_training_acquisition_smoke_allowed": phase177_ready,
                "phase177_model_training_allowed": False,
                "phase178_training_allowed_now": False,
                "a100_80gb_request_now": False,
            },
        ),
        "phase177_acquisition_table": _write_csv(
            tmp_path / "p177/acquisition.csv",
            _phase177_acquisition_rows(),
        ),
        "phase177_control_table": _write_csv(
            tmp_path / "p177/control.csv",
            _phase177_control_rows(),
        ),
    }


def _smoke_payload(module, *, candidate_wins: bool) -> dict[str, list[dict[str, object]]]:
    policy_rows = module.build_policy_rows()
    policies = [row["policy_id"] for row in policy_rows]
    family = {row["policy_id"]: row["family"] for row in policy_rows}
    best_candidate = "posterior_entropy_reduction_candidate"
    best_control = "uniform_budget_control"
    case_metric_rows = []
    summary_rows = []
    seed_summary_rows = []
    for split in ("val", "test"):
        for policy_id in policies:
            if policy_id == best_candidate:
                utility = 0.06 if candidate_wins else -0.04
                closure_gain = 0.012 if candidate_wins else -0.010
                param_gain = 0.04 if candidate_wins else -0.03
            elif policy_id == best_control:
                utility = 0.02
                closure_gain = 0.005
                param_gain = 0.015
            else:
                utility = 0.0
                closure_gain = 0.0
                param_gain = 0.0
            summary_rows.append(
                {
                    "policy_id": policy_id,
                    "family": family[policy_id],
                    "split": split,
                    "seed_count": 3,
                    "case_count": 9,
                    "posterior_trace_contraction_mean": 0.01,
                    "parameter_error_gain_mean": param_gain,
                    "closure_abs_error_gain_mean": closure_gain,
                    "parameter_error_after_mean": 0.3,
                    "closure_abs_error_after_mean": 0.03,
                    "duplicate_fraction_mean": 0.0,
                    "boundary_fraction_mean": 0.1,
                    "utility_score_mean": utility,
                    "utility_score_std": 0.0,
                }
            )
            for seed in (178, 179, 180):
                seed_summary_rows.append(
                    {
                        "seed": seed,
                        "policy_id": policy_id,
                        "split": split,
                        "case_count": 3,
                        "posterior_trace_contraction_mean": 0.01,
                        "parameter_error_gain_mean": param_gain,
                        "closure_abs_error_gain_mean": closure_gain,
                        "utility_score_mean": utility,
                    }
                )
    for seed in (178, 179, 180):
        for policy_id in policies:
            case_metric_rows.append(
                {
                    "seed": seed,
                    "case_id": f"P169-CASE-{seed}",
                    "split": "val",
                    "policy_id": policy_id,
                    "family": family[policy_id],
                    "selected_count": 12,
                    "posterior_trace_before": 0.3,
                    "posterior_trace_after": 0.2,
                    "posterior_trace_contraction": 0.1,
                    "parameter_error_before": 0.4,
                    "parameter_error_after": 0.3,
                    "parameter_error_gain": 0.1,
                    "closure_abs_error_before": 0.04,
                    "closure_abs_error_after": 0.03,
                    "closure_abs_error_gain": 0.01,
                    "duplicate_fraction": 0.0,
                    "boundary_fraction": 0.0,
                    "utility_score": 0.01,
                }
            )
    return {
        "policy_rows": policy_rows,
        "case_metric_rows": case_metric_rows,
        "summary_rows": summary_rows,
        "seed_summary_rows": seed_summary_rows,
    }


def test_phase178_closes_when_controls_win_validation(tmp_path: Path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "run_smoke", lambda: _smoke_payload(module, candidate_wins=False))
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path),
    )

    gate = manifest["gate"]
    assert gate["status"] == (
        "phase178_uncertainty_guided_acquisition_smoke_closed_no_guarded_acquisition_gain"
    )
    assert gate["selected_policy"] == "uniform_budget_control"
    assert "validation_selected_control_policy" in gate["blocking_audits"]
    assert gate["phase179_training_design_allowed"] is False
    assert gate["phase178_model_training_allowed"] is False
    assert gate["phase179_training_allowed_now"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False

    markdown = (
        tmp_path / "out/phase178_uncertainty_guided_acquisition_utility_smoke.md"
    ).read_text(encoding="utf-8")
    assert "same-budget uniform/random/no-new control wins" in markdown
    assert "|  |  |" not in markdown


def test_phase178_can_open_training_design_only_when_candidate_wins(tmp_path: Path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "run_smoke", lambda: _smoke_payload(module, candidate_wins=True))
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path),
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase178_uncertainty_guided_acquisition_smoke_ready_phase179_training_design"
    assert gate["selected_policy"] == "posterior_entropy_reduction_candidate"
    assert gate["phase179_training_design_allowed"] is True
    assert gate["phase178_model_training_allowed"] is False
    assert gate["phase179_training_allowed_now"] is False
    assert gate["bayesian_pinn_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False


def test_phase178_incomplete_if_phase177_not_ready(tmp_path: Path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "run_smoke", lambda: _smoke_payload(module, candidate_wins=True))
    manifest = module.build_package(
        root=tmp_path,
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path, phase177_ready=False),
    )

    gate = manifest["gate"]
    assert gate["status"] == (
        "phase178_uncertainty_guided_acquisition_smoke_closed_no_guarded_acquisition_gain"
    )
    assert "phase177_gate_not_ready" in gate["blocking_audits"]
    assert gate["phase179_training_design_allowed"] is False
    assert gate["phase178_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
