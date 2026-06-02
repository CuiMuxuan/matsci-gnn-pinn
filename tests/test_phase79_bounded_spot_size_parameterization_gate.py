from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase79_bounded_spot_size_parameterization_gate.py")
    spec = importlib.util.spec_from_file_location("phase79_gate", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _metric(candidate: float, best: float, no_process: float) -> dict:
    return {
        "candidate": candidate,
        "best_strong_baseline": best,
        "best_strong_baseline_method": "mean",
        "no_process": no_process,
        "delta_vs_best_strong": candidate - best,
        "delta_vs_no_process": candidate - no_process,
        "beats_best_strong_baseline": candidate < best,
        "beats_no_process": candidate < no_process,
    }


def _summary(*, density_debt: bool = False, floor_fails: bool = False) -> dict:
    datasets = []
    for label in ("broad12", "broad21"):
        candidate = 155.0 if floor_fails and label == "broad21" else 140.0
        best = 150.0
        if density_debt and label == "broad21":
            candidate = 165.0
        datasets.append(
            {
                "label": label,
                "split": "spot_size",
                "aggregate_gate": {
                    "metrics": {
                        metric: _metric(candidate, best, 230.0)
                        for metric in ("rmse", "hot_q90_rmse", "gradient_q90_rmse")
                    },
                },
            }
        )
    return {"datasets": datasets}


def _paths(
    tmp_path: Path,
    *,
    density_debt: bool = True,
    floor_fails: bool = False,
    phase69_open: bool = False,
    phase75_open: bool = False,
) -> dict[str, Path]:
    return {
        "phase55_fixed": _write_json(
            tmp_path / "phase55_fixed.json",
            _summary(density_debt=False, floor_fails=floor_fails),
        ),
        "phase58_density": _write_json(
            tmp_path / "phase58_density.json",
            _summary(density_debt=density_debt, floor_fails=False),
        ),
        "phase69_gate": _write_json(
            tmp_path / "phase69_gate.json",
            {"open_for_seed7_a100_gate": phase69_open},
        ),
        "phase75_manifest": _write_json(
            tmp_path / "phase75_manifest.json",
            {"gate_status": {"phase76_seed7_allowed": phase75_open}},
        ),
    }


def test_phase79_requires_local_surrogate_for_current_density_debt(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        tmp_path,
        tmp_path / "out",
        paths=_paths(tmp_path, density_debt=True, phase69_open=False),
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 79
    assert gate["status"] == "local_surrogate_required_before_a100"
    assert gate["a100_seed7_allowed"] is False
    assert gate["local_surrogate_allowed"] is True
    assert gate["density_debt_row_count"] == 3
    assert gate["candidate_a_phase69_open"] is False
    assert gate["a100_80gb_request_now"] is False

    table_path = tmp_path / manifest["outputs"]["gate_table"]
    with table_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert any(row["status"] == "density_debt_exceeds_floor_margin" for row in rows)


def test_phase79_opens_a100_only_when_margin_and_prior_gates_pass(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        tmp_path,
        tmp_path / "out",
        paths=_paths(tmp_path, density_debt=False, phase69_open=True),
    )

    gate = manifest["gate"]
    assert gate["status"] == "opened_for_phase76_seed7"
    assert gate["a100_seed7_allowed"] is True
    assert gate["local_surrogate_allowed"] is False
    assert gate["density_debt_row_count"] == 0
    assert gate["candidate_a_phase69_open"] is True


def test_phase79_blocks_when_fixed_floor_does_not_preserve_margin(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        tmp_path,
        tmp_path / "out",
        paths=_paths(tmp_path, floor_fails=True, density_debt=False, phase69_open=True),
    )

    gate = manifest["gate"]
    assert gate["status"] == "blocked_no_safe_margin"
    assert gate["fixed_floor_preserved"] is False
    assert gate["a100_seed7_allowed"] is False
    assert gate["local_surrogate_allowed"] is False
