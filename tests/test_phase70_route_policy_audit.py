from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase70_route_policy_audit.py")
    spec = importlib.util.spec_from_file_location("phase70_route_policy", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _main_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in ("broad12", "broad21"):
        for metric in ("Test RMSE", "Hot q90 RMSE", "Gradient q90 RMSE"):
            rows.append(
                {
                    "dataset": dataset,
                    "split": "spot_size",
                    "route": "film/global_standard",
                    "metric": metric,
                    "broad_process_v1_mean": 100.0,
                    "broad_process_v1_std": 1.0,
                    "no_process_mean": 200.0,
                    "no_process_std": 1.0,
                    "best_strong_baseline": 150.0,
                    "best_strong_baseline_method": "mean",
                    "delta_vs_best_strong": -50.0,
                    "delta_vs_no_process": -100.0,
                    "n_seeds": 3,
                    "gate": "seed_robust_transfer_positive",
                }
            )
    return rows


def _route_rows(include_process_positive: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {
            "dataset": "broad12",
            "split": "line",
            "classification": "paper_claim_positive",
            "route": "none/none",
            "claim_use": "route guard / no-process fallback evidence",
            "metrics_summary": "strong-baseline positive through fallback",
            "notes": "do not claim process conditioning",
        },
        {
            "dataset": "broad21",
            "split": "laser_power",
            "classification": "route_guard_positive",
            "route": "concat/global_standard",
            "claim_use": "route-guard-only boundary evidence",
            "metrics_summary": "candidate trails mean",
            "notes": "strong baseline remains better",
        },
    ]
    if include_process_positive:
        rows.append(
            {
                "dataset": "broad12",
                "split": "laser_power",
                "classification": "paper_claim_positive",
                "route": "concat/global_standard",
                "claim_use": "process-conditioned route evidence",
                "metrics_summary": "beats strong baseline",
                "notes": "validation-visible process route",
            }
        )
    return rows


def _stress_rows(include_blocker: bool = True) -> list[dict[str, object]]:
    rows = [
        {
            "scenario": "stronger_baseline_stress",
            "dataset": "broad12",
            "split": "spot_size",
            "metric": "Test RMSE",
            "status": "pass",
            "candidate": 100.0,
            "comparator": "mean: 150",
            "delta_vs_comparator": -50.0,
            "selected_variant": "",
            "manuscript_use": "supports floor",
            "evidence": "stress.json",
        }
    ]
    if include_blocker:
        rows.append(
            {
                "scenario": "residual_upper_bound_gate",
                "dataset": "broad21_density",
                "split": "test",
                "metric": "Test RMSE",
                "status": "blocks_model_expansion",
                "candidate": 153.0,
                "comparator": "mean: 139",
                "delta_vs_comparator": 14.0,
                "selected_variant": "blend:broad_process_v1->mean:alpha=1",
                "manuscript_use": "do not expand",
                "evidence": "upper.json",
            }
        )
    return rows


def _paths(
    tmp_path: Path,
    *,
    include_process_positive: bool = False,
    include_upper_blocker: bool = True,
    candidate_a_open: bool = False,
) -> dict[str, Path]:
    main = _write_csv(tmp_path / "phase60/main.csv", _main_rows())
    route = _write_csv(tmp_path / "phase60/route.csv", _route_rows(include_process_positive))
    stress = _write_csv(tmp_path / "phase60/stress.csv", _stress_rows(include_upper_blocker))
    upper = _write_json(
        tmp_path / "phase59/upper.json",
        {
            "uses_test_for_selection": False,
            "decision": {
                "selected_beats_reference_rmse": not include_upper_blocker,
                "selected_variant": "identity"
                if not include_upper_blocker
                else "blend:broad_process_v1->mean:alpha=1",
            },
        },
    )
    phase68 = _write_json(
        tmp_path / "phase68/manifest.json",
        {"current_decision": {"trainable_model_opened": False}},
    )
    phase69 = _write_json(
        tmp_path / "phase69/gate.json",
        {
            "open_for_seed7_a100_gate": candidate_a_open,
            "status": "opened_for_seed7_a100_gate" if candidate_a_open else "paused_no_training_signal",
        },
    )
    return {
        "phase60_main": main,
        "phase60_route": route,
        "phase60_stress": stress,
        "phase59_upper": upper,
        "phase68_manifest": phase68,
        "phase69_gate": phase69,
    }


def test_phase70_blocks_route_policy_with_boundary_and_upper_bound(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(tmp_path, tmp_path / "out", paths=_paths(tmp_path))

    gate = manifest["candidate_b_gate"]
    assert manifest["phase"] == 70
    assert gate["status"] == "blocked_no_validation_visible_route_policy_signal"
    assert gate["open_low_capacity_policy_gate"] is False
    assert gate["preserves_spot_size_floor"] is True
    assert gate["boundary_blocker_count"] >= 1
    assert gate["phase59_upper_blocks_route_policy"] is True
    assert gate["candidate_a_open_for_seed7"] is False
    with (tmp_path / manifest["outputs"]["audit_table"]).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert any(row["status"] == "route_boundary_no_policy_signal" for row in rows)
    assert any(row["status"] == "blocks_density_route_policy" for row in rows)


def test_phase70_opens_only_when_process_signal_upper_bound_and_candidate_a_gate_pass(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        tmp_path,
        tmp_path / "out",
        paths=_paths(
            tmp_path,
            include_process_positive=True,
            include_upper_blocker=False,
            candidate_a_open=True,
        ),
    )

    gate = manifest["candidate_b_gate"]
    assert gate["status"] == "opened_for_low_capacity_policy_gate"
    assert gate["open_low_capacity_policy_gate"] is True
    assert gate["process_route_signal_count"] == 1
    assert gate["phase59_upper_blocks_route_policy"] is False
    assert gate["candidate_a_open_for_seed7"] is True
