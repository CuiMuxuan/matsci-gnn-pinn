from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_summary_module():
    script = Path("scripts/server/summarize_phase55_spot_size_seed_check.py")
    spec = importlib.util.spec_from_file_location("phase55_summary", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metrics(rmse: float, hot: float, gradient: float, *, route: bool = False) -> dict:
    payload = {
        "split_metrics": {
            "test": {
                "metrics": {"rmse": rmse},
                "region_metrics": {
                    "hot_q90": {"metrics": {"rmse": hot}},
                    "gradient_q90": {"metrics": {"rmse": gradient}},
                },
            }
        }
    }
    if route:
        payload["input_features"] = {
            "conditioning_profile": {
                "profile": "broad_process_v1",
                "selected": {
                    "conditioning_mode": "film",
                    "feature_normalization": "global_standard",
                },
                "effective": {
                    "conditioning_mode": "film",
                    "feature_columns": ["laser_power_W", "scan_speed_mm_s", "spot_size_um"],
                },
            }
        }
    return payload


def test_phase55_collects_seed7_and_seed_specific_paths_with_transfer_gate(tmp_path: Path):
    module = _load_summary_module()
    seeds = [7, 1, 2]
    for dataset_limit in (12, 21):
        run_id = module._run_id("spot_size", dataset_limit, "process_round_robin", "broad_process_profile")
        for method, tag, values in (
            ("mean", "mean_constant", (150.0, 250.0, 230.0)),
            ("knn_coords", "knn_coords", (180.0, 280.0, 260.0)),
            ("knn_process", "knn_process", (181.0, 281.0, 261.0)),
            ("extra_trees_coords", "extra_trees_coords", (182.0, 282.0, 262.0)),
            ("extra_trees_process", "extra_trees_process", (183.0, 283.0, 263.0)),
        ):
            _write_json(
                tmp_path / "outputs" / "baselines" / f"{run_id}_{tag}_regions_q90.json",
                _metrics(*values),
            )
        for seed in seeds:
            broad_path = module._seed_metrics_path(tmp_path, run_id, seed, "broad_process_profile")
            no_process_path = module._seed_metrics_path(tmp_path, run_id, seed, "no_process")
            _write_json(broad_path, _metrics(140.0 + seed * 0.1, 160.0 + seed, 170.0 + seed, route=True))
            _write_json(no_process_path, _metrics(210.0 + seed, 360.0 + seed, 330.0 + seed))

    payload = module.collect_summary(
        root=tmp_path,
        dataset_limits=[12, 21],
        dataset_order="process_round_robin",
        split="spot_size",
        seeds=seeds,
    )

    assert payload["transfer_gate"]["paper_claim_status"] == "seed_robust_transfer_positive"
    assert payload["transfer_gate"]["paired_seed_transfer_positive"] is True
    for dataset in payload["datasets"]:
        assert dataset["aggregate_gate"]["pass"] is True
        assert dataset["paired_seed_gate"]["pass"] is True
        assert dataset["route"]["selected_conditioning_mode"] == "film"
        assert dataset["aggregates"]["broad_process_v1"]["n"] == 3
        assert dataset["aggregates"]["no_process"]["n"] == 3


def test_phase55_distinguishes_aggregate_positive_from_per_seed_failure(tmp_path: Path):
    module = _load_summary_module()
    run_id = module._run_id("spot_size", 12, "process_round_robin", "broad_process_profile")
    for _, tag in module.STRONG_BASELINES:
        _write_json(
            tmp_path / "outputs" / "baselines" / f"{run_id}_{tag}_regions_q90.json",
            _metrics(150.0, 250.0, 230.0),
        )
    for seed, rmse in ((7, 130.0), (1, 130.0), (2, 160.0)):
        _write_json(
            module._seed_metrics_path(tmp_path, run_id, seed, "broad_process_profile"),
            _metrics(rmse, 160.0, 170.0, route=True),
        )
        _write_json(
            module._seed_metrics_path(tmp_path, run_id, seed, "no_process"),
            _metrics(210.0, 360.0, 330.0),
        )

    payload = module.collect_summary(
        root=tmp_path,
        dataset_limits=[12],
        dataset_order="process_round_robin",
        split="spot_size",
        seeds=[7, 1, 2],
    )

    assert payload["transfer_gate"]["paper_claim_status"] == "aggregate_positive_seed_mixed"
    assert payload["datasets"][0]["aggregate_gate"]["pass"] is True
    assert payload["datasets"][0]["paired_seed_gate"]["pass"] is False
    assert payload["datasets"][0]["paired_seed_gate"]["seeds"][2]["pass"] is False
