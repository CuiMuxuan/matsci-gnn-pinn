from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase99_registered_surrogate_smoke_gate.py")
    spec = importlib.util.spec_from_file_location("phase99_gate", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _phase98_gate(*, allowed: bool = True) -> dict:
    return {
        "status": "registered_surrogate_unlocked_no_a100" if allowed else "blocked_no_unlock_candidate",
        "phase99_local_smoke_allowed": allowed,
        "preferred_phase99_candidate": "phase98_generated_pfhub_registered_surrogate_v1"
        if allowed
        else "none",
        "am_bench_transfer_unlocked": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
    }


def _surrogate_card() -> dict:
    return {
        "candidate_id": "phase98_generated_pfhub_registered_surrogate_v1",
        "source_target_id": "phase96_pfhub_style_heat_source_v1",
        "dataset_family": "generated_registered_surrogate",
        "registration_story": "exact analytic registration",
        "source_manifest": {
            "generator": "scripts/server/build_phase96_pfhub_local_smoke_gate.py",
            "alpha": 0.04,
            "source_x0": 0.28,
            "source_velocity": 0.42,
            "source_sigma2": 0.018,
        },
        "split_plan": {
            "train_grid": {"nx": 19, "nt": 15},
            "validation_grid": {"nx": 23, "nt": 17},
            "test_grid": {"nx": 41, "nt": 31},
            "selection_rule": "candidate and hyperparameter selection must use train/validation only",
        },
        "baseline_plan": [
            "low_order_interpolation",
            "vanilla_deterministic_surrogate",
            "fixed_green_function_features",
            "random_collocation_same_budget",
        ],
        "not_a_claim": ["not AM-Bench evidence", "not permission for A100 training"],
    }


def _paths(tmp_path: Path, *, allowed: bool = True) -> dict[str, Path]:
    return {
        "phase98_gate": _write_json(tmp_path / "phase98_gate.json", _phase98_gate(allowed=allowed)),
        "phase98_surrogate_card": _write_json(tmp_path / "phase98_card.json", _surrogate_card()),
    }


def test_phase99_registered_surrogate_smoke_allows_local_package_only(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(tmp_path, tmp_path / "out", paths=_paths(tmp_path))

    gate = manifest["gate"]
    assert manifest["phase"] == 99
    assert gate["status"] == "local_surrogate_positive_with_focused_baseline_boundary"
    assert gate["phase100_local_mechanism_package_allowed"] is True
    assert gate["am_bench_transfer_unlocked"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert gate["full_grid_baseline_pass"] is True
    assert gate["same_budget_boundary_failures"] >= 1

    with (tmp_path / manifest["outputs"]["metric_table"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        metric_rows = list(csv.DictReader(handle))
    assert len(metric_rows) == 5
    candidate = next(
        row for row in metric_rows if row["method_id"] == "fixed_green_function_features_full"
    )
    vanilla = next(
        row for row in metric_rows if row["method_id"] == "vanilla_deterministic_surrogate_full"
    )
    random_budget = next(
        row for row in metric_rows if row["method_id"] == "random_collocation_same_budget"
    )
    assert float(candidate["test_rmse"]) < float(vanilla["test_rmse"])
    assert float(candidate["test_pde_residual_rmse"]) < float(vanilla["test_pde_residual_rmse"])
    assert float(candidate["test_rmse"]) < float(random_budget["test_rmse"])

    with (tmp_path / manifest["outputs"]["comparison_table"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        comparison_rows = list(csv.DictReader(handle))
    assert len(comparison_rows) == 12
    same_budget_failures = [
        row
        for row in comparison_rows
        if row["scope"] == "same-budget boundary audit" and row["pass"] == "false"
    ]
    assert len(same_budget_failures) == gate["same_budget_boundary_failures"]
    assert {row["metric"] for row in same_budget_failures} == {"gradient_q90_rmse"}
    assert all(
        row["pass"] == "true"
        for row in comparison_rows
        if row["scope"] == "full-grid baselines"
    )


def test_phase99_blocks_when_phase98_does_not_allow_smoke(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        tmp_path,
        tmp_path / "out",
        paths=_paths(tmp_path, allowed=False),
    )

    gate = manifest["gate"]
    assert gate["status"] == "blocked_by_phase98_unlock_gate"
    assert gate["phase100_local_mechanism_package_allowed"] is False
    assert gate["am_bench_transfer_unlocked"] is False
    assert gate["a100_training_allowed_now"] is False
    assert manifest["counts"]["metric_rows"] == 0
    assert manifest["counts"]["comparison_rows"] == 0
