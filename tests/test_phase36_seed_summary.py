from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path


def _load_summary_module():
    script = Path("scripts/server/summarize_phase36_process_graph_seed_check.py")
    spec = importlib.util.spec_from_file_location("phase36_seed_summary", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _short_test_root(name: str) -> Path:
    root = Path(".pytest_tmp") / name
    if root.exists():
        shutil.rmtree(root)
    return root


def _metrics(rmse: float, graph: bool = False) -> dict:
    payload = {
        "split_metrics": {
            "test": {
                "metrics": {"rmse": rmse, "mae": rmse / 2, "relative_l2": 1.0},
                "region_metrics": {
                    "hot_q90": {"rmse": rmse + 10.0},
                    "gradient_q90": {"rmse": rmse + 20.0},
                },
            }
        },
        "input_features": {
            "enabled": graph,
            "count": 0,
            "effective_columns": [],
            "conditioning_profile": {
                "profile": "broad_process_v1",
                "group_key": "spot_size_um",
                "selected": {"conditioning_mode": "film", "feature_normalization": "global_standard"},
                "effective": {
                    "conditioning_mode": "film",
                    "feature_columns": ["laser_power_W", "scan_speed_mm_s", "spot_size_um"],
                },
            },
        },
    }
    if graph:
        payload["input_features"].update(
            {
                "count": 7,
                "effective_columns": [
                    "laser_power_W",
                    "scan_speed_mm_s",
                    "spot_size_um",
                    "process_graph_rbf_0",
                    "process_graph_rbf_1",
                    "process_graph_rbf_2",
                    "process_graph_rbf_3",
                ],
                "process_graph_features": {
                    "enabled": True,
                    "mode": "rbf",
                    "columns": ["laser_power_W", "scan_speed_mm_s", "spot_size_um"],
                    "fit_scope": "global",
                    "requested_anchor_count": 4,
                    "anchor_count": 4,
                    "source_unique_nodes": 4,
                    "length_scale": 1.0,
                    "feature_names": [
                        "process_graph_rbf_0",
                        "process_graph_rbf_1",
                        "process_graph_rbf_2",
                        "process_graph_rbf_3",
                    ],
                },
            }
        )
    return payload


def test_phase36_seed_summary_collects_seed7_and_paired_seed_paths(tmp_path: Path):
    summary = _load_summary_module()
    root = _short_test_root("p36a")
    split = "spot_size"
    base_run_id = summary._base_run_id(split, 12, "process_round_robin")
    graph_run_id = base_run_id.replace("broad_process_profile", "pg_rbf_global")

    _write_json(
        root / "outputs/runs" / f"{base_run_id}_macro_pinn_minmax_broad_process_profile_v1" / "metrics.json",
        _metrics(100.0),
    )
    _write_json(
        root / "outputs/runs" / f"{base_run_id}_seed1_macro_pinn_minmax_broad_process_profile_v1" / "metrics.json",
        _metrics(110.0),
    )
    _write_json(
        root / "outputs/runs" / f"{graph_run_id}_macro_pinn_minmax_pg_rbf_global_v1" / "metrics.json",
        _metrics(90.0, graph=True),
    )
    _write_json(
        root / "outputs/runs" / f"{base_run_id}_seed1_macro_pinn_minmax_pg_rbf_global_v1" / "metrics.json",
        _metrics(95.0, graph=True),
    )

    payload = summary.collect_summary(
        root,
        [split],
        [7, 1],
        ["pg_rbf_global"],
        12,
        "process_round_robin",
    )

    baseline = payload["aggregate"][split]["broad_process_v1"]
    graph = payload["aggregate"][split]["pg_rbf_global"]
    delta = payload["deltas_vs_broad_process_v1"][split]["pg_rbf_global"]
    graph_rows = [row for row in payload["rows"] if row["method"] == "pg_rbf_global"]

    assert baseline["n"] == 2
    assert baseline["rmse_mean"] == 105.0
    assert graph["n"] == 2
    assert graph["rmse_mean"] == 92.5
    assert graph["hot_q90_rmse_mean"] == 102.5
    assert delta["rmse_mean_delta"] == -12.5
    assert delta["hot_q90_rmse_mean_delta"] == -12.5
    assert graph_rows[0]["process_graph_anchor_count"] == 4
    assert graph_rows[0]["process_graph_fit_scope"] == "global"
    assert graph_rows[0]["input_effective_columns"][-1] == "process_graph_rbf_3"


def test_phase36_seed_summary_marks_missing_expected_metrics(tmp_path: Path):
    summary = _load_summary_module()
    root = _short_test_root("p36b")
    split = "laser_power"
    base_run_id = summary._base_run_id(split, 12, "process_round_robin")
    _write_json(
        root / "outputs/runs" / f"{base_run_id}_seed1_macro_pinn_minmax_broad_process_profile_v1" / "metrics.json",
        _metrics(120.0),
    )

    payload = summary.collect_summary(
        root,
        [split],
        [1],
        ["pg_rbf_global"],
        12,
        "process_round_robin",
    )

    rows = payload["rows"]
    assert rows[0]["missing"] is False
    assert rows[1]["missing"] is True
    assert payload["aggregate"][split]["broad_process_v1"]["n"] == 1
    assert payload["aggregate"][split]["pg_rbf_global"]["n"] == 0
