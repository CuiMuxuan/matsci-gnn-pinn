from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_summary_module():
    script = Path("scripts/server/summarize_phase30_broad_process_selector_smoke.py")
    spec = importlib.util.spec_from_file_location("phase30_summary", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _manifest(n_rows: int, max_frames: int, max_points_per_frame: int) -> dict:
    return {
        "target": "temperature_C",
        "n_rows": n_rows,
        "dataset_paths": ["ThermalData/Line_0_1/Signal", "ThermalData/Line_1_1_1/Signal"],
        "metadata": {
            "dataset_selection": {
                "explicit_datasets": [],
                "dataset_regex": "ThermalData/Line_.*?/Signal$",
                "dataset_limit": 12,
                "dataset_order": "process_round_robin",
            },
            "sampling": {
                "frame_start": 0,
                "frame_step": 20,
                "max_frames": max_frames,
                "row_start": 0,
                "row_step": 8,
                "max_rows": 80,
                "col_start": 0,
                "col_step": 8,
                "max_cols": 38,
                "min_signal": 100,
                "selection": {
                    "mode": "balanced_hot_gradient",
                    "active_target": "temperature_C",
                    "hot_quantile": 0.9,
                    "gradient_quantile": 0.9,
                    "background_fraction": 0.15,
                    "max_points_per_frame": max_points_per_frame,
                },
            },
            "process_groups": {
                "scan_speed": {"scan_speed_mm_s=960": n_rows},
            },
        },
    }


def _split(n_rows: int) -> dict:
    return {
        "n_rows": n_rows,
        "n_groups": 1,
        "group_key": "scan_speed_mm_s",
        "strategy": "scan_speed_mm_s_order",
        "group_order": ["scan_speed_mm_s=960"],
        "group_splits": {"train": ["scan_speed_mm_s=960"], "val": [], "test": []},
        "rows_per_group": {"scan_speed_mm_s=960": n_rows},
        "splits": {"train": list(range(n_rows)), "val": [], "test": []},
    }


def _metrics(rmse: float) -> dict:
    return {
        "split_metrics": {
            "test": {
                "metrics": {"rmse": rmse, "mae": rmse / 2, "relative_l2": 1.0},
                "region_metrics": {
                    "hot_q90": {"rmse": rmse + 1},
                    "gradient_q90": {"rmse": rmse + 2},
                },
            }
        },
        "input_features": {
            "enabled": False,
            "count": 0,
            "conditioning_profile": {
                "profile": "broad_process_v1",
                "group_key": "scan_speed_mm_s",
                "selected": {"conditioning_mode": "none", "feature_normalization": "none"},
                "effective": {"conditioning_mode": "concat", "feature_columns": []},
            },
        },
    }


def test_phase30_summary_marks_mismatched_tiny_smoke_as_incomparable(tmp_path: Path):
    summary = _load_summary_module()
    split = "scan_speed"
    baseline_id = summary._run_id(split, 12, "process_round_robin", "process_axis_profile")
    broad_id = summary._run_id(split, 12, "process_round_robin", "broad_process_profile")

    _write_json(tmp_path / "outputs/data_audits" / f"{baseline_id}_manifest.json", _manifest(1200, 30, 96))
    _write_json(tmp_path / "outputs/data_splits" / f"{baseline_id}_split.json", _split(1200))
    _write_json(
        tmp_path / "outputs/baselines" / f"{baseline_id}_mean_constant_regions_q90.json",
        _metrics(100.0),
    )
    _write_json(
        tmp_path / "outputs/runs" / f"{baseline_id}_macro_pinn_minmax_no_process_v1" / "metrics.json",
        _metrics(90.0),
    )
    _write_json(
        tmp_path / "outputs/runs" / f"{baseline_id}_macro_pinn_minmax_process_axis_profile_v1" / "metrics.json",
        _metrics(80.0),
    )
    _write_json(tmp_path / "outputs/data_audits" / f"{broad_id}_manifest.json", _manifest(24, 2, 32))
    _write_json(tmp_path / "outputs/data_splits" / f"{broad_id}_split.json", _split(24))
    _write_json(
        tmp_path / "outputs/runs" / f"{broad_id}_macro_pinn_minmax_broad_process_profile_v1" / "metrics.json",
        _metrics(70.0),
    )

    payload = summary.collect_rows(tmp_path, (split,), 12, "process_round_robin")
    broad_row = payload["splits"][split]["methods"]["broad_process_v1"]

    assert payload["splits"][split]["all_methods_comparable"] is False
    assert broad_row["comparison_status"] == "incomparable"
    assert broad_row["comparison_reason"] == "manifest n_rows differs from reference"


def test_phase30_summary_accepts_matching_full_profile_artifacts(tmp_path: Path):
    summary = _load_summary_module()
    split = "scan_speed"
    baseline_id = summary._run_id(split, 12, "process_round_robin", "process_axis_profile")
    broad_id = summary._run_id(split, 12, "process_round_robin", "broad_process_profile")

    manifest = _manifest(1200, 30, 96)
    split_payload = _split(1200)
    _write_json(tmp_path / "outputs/data_audits" / f"{baseline_id}_manifest.json", manifest)
    _write_json(tmp_path / "outputs/data_splits" / f"{baseline_id}_split.json", split_payload)
    _write_json(tmp_path / "outputs/data_audits" / f"{broad_id}_manifest.json", manifest)
    _write_json(tmp_path / "outputs/data_splits" / f"{broad_id}_split.json", split_payload)
    for method, tag in summary.BASELINE_TAGS:
        _write_json(
            tmp_path / "outputs/baselines" / f"{baseline_id}_{tag}_regions_q90.json",
            _metrics(100.0),
        )
    _write_json(
        tmp_path / "outputs/runs" / f"{baseline_id}_macro_pinn_minmax_no_process_v1" / "metrics.json",
        _metrics(90.0),
    )
    _write_json(
        tmp_path / "outputs/runs" / f"{baseline_id}_macro_pinn_minmax_process_axis_profile_v1" / "metrics.json",
        _metrics(80.0),
    )
    _write_json(
        tmp_path / "outputs/runs" / f"{broad_id}_macro_pinn_minmax_broad_process_profile_v1" / "metrics.json",
        _metrics(70.0),
    )

    payload = summary.collect_rows(tmp_path, (split,), 12, "process_round_robin")
    broad_row = payload["splits"][split]["methods"]["broad_process_v1"]

    assert payload["splits"][split]["all_methods_comparable"] is True
    assert broad_row["comparison_status"] == "comparable"
    assert broad_row["rmse"] == 70.0
