from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_summary_module():
    script = Path("scripts/server/summarize_phase54_process_route_claim_boundary.py")
    spec = importlib.util.spec_from_file_location("phase54_summary", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _method(rmse: float | None, hot: float | None, gradient: float | None) -> dict:
    return {
        "comparison_status": "comparable",
        "rmse": rmse,
        "hot_q90_rmse": hot,
        "gradient_q90_rmse": gradient,
    }


def _broad(
    rmse: float | None,
    hot: float | None,
    gradient: float | None,
    selected: str = "film",
    norm: str = "global_standard",
) -> dict:
    row = _method(rmse, hot, gradient)
    row.update(
        {
            "selected_conditioning_mode": selected,
            "selected_feature_normalization": norm,
            "effective_conditioning_mode": selected,
            "effective_feature_columns": ["laser_power_W", "scan_speed_mm_s", "spot_size_um"]
            if selected != "none"
            else [],
        }
    )
    return row


def _split_payload(
    broad: dict,
    mean: dict | None = None,
    no_process: dict | None = None,
    process_axis: dict | None = None,
) -> dict:
    return {
        "all_methods_comparable": True,
        "methods": {
            "mean": mean or _method(100.0, 100.0, 100.0),
            "knn_coords": _method(110.0, 110.0, 110.0),
            "knn_process": _method(111.0, 111.0, 111.0),
            "extra_trees_coords": _method(112.0, 112.0, 112.0),
            "extra_trees_process": _method(113.0, 113.0, 113.0),
            "no_process": no_process or _method(150.0, 150.0, 150.0),
            "process_axis_v1": process_axis or _method(120.0, 120.0, 120.0),
            "broad_process_v1": broad,
        },
    }


def _write_input(path: Path, split_payload: dict, dataset_limit: int = 12) -> None:
    payload = {
        "dataset_limit": dataset_limit,
        "dataset_order": "process_round_robin",
        "pinn_methods": ["no_process", "process_axis_v1", "broad_process_v1"],
        "splits": {"spot_size": split_payload},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_phase54_classifies_full_strong_baseline_positive(tmp_path: Path):
    summary = _load_summary_module()
    path = tmp_path / "broad12.json"
    _write_input(path, _split_payload(_broad(90.0, 91.0, 92.0)))

    payload = summary.collect_summary([path])
    row = payload["datasets"][0]["splits"]["spot_size"]

    assert row["classification"] == "paper_claim_positive"
    assert row["metrics"]["rmse"]["candidate_delta_vs_best_strong"] == -10.0
    assert payload["claim_boundary"]["paper_claim_positive"] == ["broad12:spot_size"]


def test_phase54_classifies_route_guard_when_no_process_improves_but_mean_wins(tmp_path: Path):
    summary = _load_summary_module()
    path = tmp_path / "broad21.json"
    _write_input(
        path,
        _split_payload(
            _broad(130.0, 130.0, 130.0, selected="concat"),
            mean=_method(100.0, 100.0, 100.0),
            no_process=_method(170.0, 180.0, 175.0),
            process_axis=_method(130.0, 130.0, 130.0),
        ),
        dataset_limit=21,
    )

    payload = summary.collect_summary([path])
    row = payload["datasets"][0]["splits"]["spot_size"]

    assert row["classification"] == "route_guard_positive"
    assert row["metrics"]["hot_q90_rmse"]["candidate_beats_no_process"] is True
    assert row["metrics"]["hot_q90_rmse"]["candidate_beats_best_strong_baseline"] is False
    assert payload["claim_boundary"]["route_guard_positive"] == ["broad21:spot_size"]


def test_phase54_keeps_missing_hot_metric_incomplete_even_if_available_metrics_win(tmp_path: Path):
    summary = _load_summary_module()
    path = tmp_path / "broad21.json"
    _write_input(
        path,
        _split_payload(
            _broad(90.0, None, 92.0),
            mean=_method(100.0, None, 100.0),
            no_process=_method(150.0, None, 150.0),
            process_axis=_method(90.0, None, 92.0),
        ),
        dataset_limit=21,
    )

    payload = summary.collect_summary([path])
    row = payload["datasets"][0]["splits"]["spot_size"]

    assert row["classification"] == "incomplete_metric"
    assert row["available_metric_classification"] == "paper_claim_positive"
    assert row["missing_metrics"] == ["hot_q90_rmse"]
    assert payload["claim_boundary"]["incomplete_metric"] == ["broad21:spot_size"]


def test_phase54_marks_incomparable_input_without_promoting_claim(tmp_path: Path):
    summary = _load_summary_module()
    row = summary.classify_split(
        "laser_power",
        {
            "all_methods_comparable": False,
            "methods": {"broad_process_v1": _broad(80.0, 80.0, 80.0)},
        },
    )

    assert row["classification"] == "incomparable"
    assert "Manifest/split comparability failed" in row["notes"][0]
