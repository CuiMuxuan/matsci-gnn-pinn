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


def _fourier_metrics(rmse: float) -> dict:
    payload = _metrics(rmse)
    payload["spacetime_encoding"] = {
        "encoding": "fourier",
        "fourier_bands": 4,
        "input_dim": 27,
    }
    return payload


def _residual_metrics(rmse: float) -> dict:
    payload = _metrics(rmse)
    payload["residual_correction"] = {
        "enabled": True,
        "mode": "mlp",
        "input_dim": 6,
        "hidden_dim": 32,
        "layers": 1,
        "scale": 0.1,
        "start_step": 100,
        "lr": 5e-4,
        "parameter_count": 49,
        "last_layer_zero_initialized": True,
    }
    return payload


def _region_weighted_metrics(rmse: float) -> dict:
    payload = _metrics(rmse)
    payload["data_loss_weighting"] = {
        "enabled": True,
        "mode": "hot_gradient",
        "fit_scope": "train",
        "normalization": "sum_weights",
        "train_points": 800,
        "selected_points": 120,
        "selected_fraction": 0.15,
        "region_weight": 2.0,
        "hot_quantile": 0.9,
        "gradient_quantile": 0.9,
        "weight_sum": 920.0,
        "mean_weight": 1.15,
        "selectors": {},
    }
    return payload


def _group_balanced_metrics(rmse: float) -> dict:
    payload = _metrics(rmse)
    payload["data_loss_group_balance"] = {
        "enabled": True,
        "mode": "inverse_frequency",
        "fit_scope": "train",
        "normalization": "blend_with_uniform",
        "column": "process_condition",
        "strength": 1.0,
        "train_points": 800,
        "group_count": 2,
        "group_sizes": {
            "laser_power_W=245__scan_speed_mm_s=800__spot_size_um=49": 600,
            "laser_power_W=285__scan_speed_mm_s=960__spot_size_um=67": 200,
        },
        "group_weights": {
            "laser_power_W=245__scan_speed_mm_s=800__spot_size_um=49": 2 / 3,
            "laser_power_W=285__scan_speed_mm_s=960__spot_size_um=67": 2.0,
        },
        "weight_sum": 800.0,
        "mean_weight": 1.0,
    }
    payload["data_loss_objective"] = {
        "enabled": True,
        "region_component_enabled": False,
        "group_balance_component_enabled": True,
        "weight_sum": 800.0,
        "mean_weight": 1.0,
    }
    return payload


def _process_graph_metrics(rmse: float) -> dict:
    payload = _metrics(rmse)
    payload["input_features"].update(
        {
            "enabled": True,
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
                "fit_scope": "train",
                "requested_anchor_count": 4,
                "anchor_count": 4,
                "source_unique_nodes": 7,
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


def _target_residual_metrics(rmse: float) -> dict:
    payload = _metrics(rmse)
    payload["target_normalization"] = {
        "enabled": True,
        "mean": 0.0,
        "std": 12.5,
        "target_space": "residual",
    }
    payload["target_residual_baseline"] = {
        "enabled": True,
        "strategy": "extra_trees",
        "feature_columns": [
            "x",
            "y",
            "t",
            "laser_power_W",
            "scan_speed_mm_s",
            "spot_size_um",
        ],
        "fit_split": "train",
        "fit_points": 800,
        "n_neighbors": None,
        "n_estimators": 80,
        "random_state": 7,
        "train_residual_mean": 0.0,
        "train_residual_rmse": 18.25,
    }
    return payload


def _residual_backbone_metrics(rmse: float) -> dict:
    payload = _metrics(rmse)
    payload["backbone"] = {
        "mode": "residual",
        "residual_scale": 0.5,
        "hidden_dim": 128,
        "layers": 4,
        "parameter_count": 51201,
        "implementation": "MacroPINN",
    }
    return payload


def _output_affine_metrics(rmse: float) -> dict:
    payload = _metrics(rmse)
    payload["output_affine"] = {
        "enabled": True,
        "mode": "linear",
        "input_dim": 3,
        "scale": 0.5,
        "lr": 0.001,
        "parameter_count": 8,
        "identity_initialized": True,
    }
    return payload


def _prediction_anchor_metrics(rmse: float) -> dict:
    payload = _metrics(rmse)
    payload["prediction_anchor"] = {
        "enabled": True,
        "weight": 0.05,
        "target_space": "normalized_training_target",
        "loss": "mean(prediction ** 2)",
    }
    return payload


def _process_encoder_metrics(rmse: float) -> dict:
    payload = _derived_process_metrics(rmse)
    payload["process_encoder"] = {
        "enabled": True,
        "mode": "linear",
        "input_dim": 7,
        "output_dim": 3,
        "identity_initialized": True,
        "parameter_count": 24,
    }
    return payload


def _derived_process_metrics(rmse: float) -> dict:
    payload = _metrics(rmse)
    feature_names = [
        "line_energy_J_per_mm",
        "energy_density_proxy_J_per_mm_um",
        "energy_density_area_proxy_J_per_mm_um2",
        "dwell_time_ms",
    ]
    payload["input_features"].update(
        {
            "enabled": True,
            "columns": ["laser_power_W", "scan_speed_mm_s", "spot_size_um"],
            "effective_columns": [
                "laser_power_W",
                "scan_speed_mm_s",
                "spot_size_um",
                *feature_names,
            ],
            "count": 7,
            "derived_process_features": {
                "enabled": True,
                "mode": "am_energy_v1",
                "source_columns": ["laser_power_W", "scan_speed_mm_s", "spot_size_um"],
                "feature_names": feature_names,
            },
        }
    )
    return payload


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


def test_phase30_summary_can_include_broad_process_v2_artifacts(tmp_path: Path):
    summary = _load_summary_module()
    split = "line"
    baseline_id = summary._run_id(split, 21, "process_round_robin", "process_axis_profile")
    broad_v2_id = summary._run_id(split, 21, "process_round_robin", "broad_process_profile_v2")

    manifest = _manifest(2100, 30, 96)
    split_payload = _split(2100)
    split_payload["group_key"] = "line_id"
    _write_json(tmp_path / "outputs/data_audits" / f"{baseline_id}_manifest.json", manifest)
    _write_json(tmp_path / "outputs/data_splits" / f"{baseline_id}_split.json", split_payload)
    _write_json(tmp_path / "outputs/data_audits" / f"{broad_v2_id}_manifest.json", manifest)
    _write_json(tmp_path / "outputs/data_splits" / f"{broad_v2_id}_split.json", split_payload)
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
    broad_v2_metrics = _metrics(69.0)
    broad_v2_metrics["input_features"]["conditioning_profile"]["profile"] = "broad_process_v2"
    broad_v2_metrics["input_features"]["conditioning_profile"]["group_key"] = "line_id"
    broad_v2_metrics["input_features"]["conditioning_profile"]["selected"] = {
        "conditioning_mode": "concat",
        "feature_normalization": "same",
    }
    broad_v2_metrics["input_features"]["conditioning_profile"]["effective"] = {
        "conditioning_mode": "concat",
        "feature_columns": ["laser_power_W", "scan_speed_mm_s", "spot_size_um"],
    }
    _write_json(
        tmp_path / "outputs/runs" / f"{broad_v2_id}_macro_pinn_minmax_broad_process_profile_v2_v1" / "metrics.json",
        broad_v2_metrics,
    )

    payload = summary.collect_rows(
        tmp_path,
        (split,),
        21,
        "process_round_robin",
        (*summary.DEFAULT_PINN_SPECS, summary.BROAD_PROCESS_V2_SPEC),
    )
    broad_v2_row = payload["splits"][split]["methods"]["broad_process_v2"]

    assert payload["splits"][split]["all_methods_comparable"] is False
    assert payload["pinn_methods"] == [
        "no_process",
        "process_axis_v1",
        "broad_process_v1",
        "broad_process_v2",
    ]
    assert broad_v2_row["comparison_status"] == "comparable"
    assert broad_v2_row["rmse"] == 69.0
    assert broad_v2_row["selected_conditioning_mode"] == "concat"
    assert broad_v2_row["selected_feature_normalization"] == "same"
    assert broad_v2_row["effective_feature_columns"] == ["laser_power_W", "scan_speed_mm_s", "spot_size_um"]


def test_phase30_summary_can_include_broad_process_fourier_artifacts(tmp_path: Path):
    summary = _load_summary_module()
    split = "spot_size"
    baseline_id = summary._run_id(split, 12, "process_round_robin", "process_axis_profile")
    fourier_id = summary._run_id(split, 12, "process_round_robin", "broad_process_fourier")

    manifest = _manifest(1200, 30, 96)
    split_payload = _split(1200)
    split_payload["group_key"] = "spot_size_um"
    _write_json(tmp_path / "outputs/data_audits" / f"{baseline_id}_manifest.json", manifest)
    _write_json(tmp_path / "outputs/data_splits" / f"{baseline_id}_split.json", split_payload)
    _write_json(tmp_path / "outputs/data_audits" / f"{fourier_id}_manifest.json", manifest)
    _write_json(tmp_path / "outputs/data_splits" / f"{fourier_id}_split.json", split_payload)
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
        tmp_path / "outputs/runs" / f"{fourier_id}_macro_pinn_minmax_broad_process_fourier_v1" / "metrics.json",
        _fourier_metrics(68.0),
    )

    payload = summary.collect_rows(
        tmp_path,
        (split,),
        12,
        "process_round_robin",
        (*summary.DEFAULT_PINN_SPECS, summary.BROAD_PROCESS_FOURIER_SPEC),
    )
    fourier_row = payload["splits"][split]["methods"]["broad_process_fourier"]

    assert payload["pinn_methods"] == [
        "no_process",
        "process_axis_v1",
        "broad_process_v1",
        "broad_process_fourier",
    ]
    assert fourier_row["comparison_status"] == "comparable"
    assert fourier_row["rmse"] == 68.0
    assert fourier_row["spacetime_encoding"] == "fourier"
    assert fourier_row["spacetime_fourier_bands"] == 4
    assert fourier_row["spacetime_input_dim"] == 27


def test_phase30_summary_can_include_broad_process_residual_artifacts(tmp_path: Path):
    summary = _load_summary_module()
    split = "spot_size"
    baseline_id = summary._run_id(split, 12, "process_round_robin", "process_axis_profile")
    residual_id = summary._run_id(split, 12, "process_round_robin", "broad_residual_mlp")

    manifest = _manifest(1200, 30, 96)
    split_payload = _split(1200)
    split_payload["group_key"] = "spot_size_um"
    _write_json(tmp_path / "outputs/data_audits" / f"{baseline_id}_manifest.json", manifest)
    _write_json(tmp_path / "outputs/data_splits" / f"{baseline_id}_split.json", split_payload)
    _write_json(tmp_path / "outputs/data_audits" / f"{residual_id}_manifest.json", manifest)
    _write_json(tmp_path / "outputs/data_splits" / f"{residual_id}_split.json", split_payload)
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
        tmp_path / "outputs/runs" / f"{residual_id}_macro_pinn_minmax_broad_residual_mlp_v1" / "metrics.json",
        _residual_metrics(67.0),
    )

    payload = summary.collect_rows(
        tmp_path,
        (split,),
        12,
        "process_round_robin",
        (*summary.DEFAULT_PINN_SPECS, summary.BROAD_PROCESS_RESIDUAL_SPEC),
    )
    residual_row = payload["splits"][split]["methods"]["broad_residual_mlp"]

    assert payload["pinn_methods"] == [
        "no_process",
        "process_axis_v1",
        "broad_process_v1",
        "broad_residual_mlp",
    ]
    assert residual_row["comparison_status"] == "comparable"
    assert residual_row["rmse"] == 67.0
    assert residual_row["residual_correction_enabled"] is True
    assert residual_row["residual_correction_mode"] == "mlp"
    assert residual_row["residual_correction_scale"] == 0.1
    assert residual_row["residual_correction_start_step"] == 100


def test_phase30_summary_can_include_broad_region_weighted_artifacts(tmp_path: Path):
    summary = _load_summary_module()
    split = "spot_size"
    tag = "rw2"
    baseline_id = summary._run_id(split, 12, "process_round_robin", "process_axis_profile")
    weighted_id = summary._run_id(split, 12, "process_round_robin", tag)

    manifest = _manifest(1200, 30, 96)
    split_payload = _split(1200)
    split_payload["group_key"] = "spot_size_um"
    _write_json(tmp_path / "outputs/data_audits" / f"{baseline_id}_manifest.json", manifest)
    _write_json(tmp_path / "outputs/data_splits" / f"{baseline_id}_split.json", split_payload)
    _write_json(tmp_path / "outputs/data_audits" / f"{weighted_id}_manifest.json", manifest)
    _write_json(tmp_path / "outputs/data_splits" / f"{weighted_id}_split.json", split_payload)
    for method, baseline_tag in summary.BASELINE_TAGS:
        _write_json(
            tmp_path / "outputs/baselines" / f"{baseline_id}_{baseline_tag}_regions_q90.json",
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
        tmp_path / "outputs/runs" / f"{weighted_id}_macro_pinn_minmax_{tag}_v1" / "metrics.json",
        _region_weighted_metrics(66.0),
    )

    payload = summary.collect_rows(
        tmp_path,
        (split,),
        12,
        "process_round_robin",
        (*summary.DEFAULT_PINN_SPECS, (tag, tag, tag)),
    )
    weighted_row = payload["splits"][split]["methods"][tag]

    assert payload["pinn_methods"] == [
        "no_process",
        "process_axis_v1",
        "broad_process_v1",
        tag,
    ]
    assert weighted_row["comparison_status"] == "comparable"
    assert weighted_row["rmse"] == 66.0
    assert weighted_row["data_loss_weighting_enabled"] is True
    assert weighted_row["data_loss_weighting_mode"] == "hot_gradient"
    assert weighted_row["data_loss_region_weight"] == 2.0
    assert weighted_row["data_loss_weighted_points"] == 120


def test_phase30_summary_can_include_broad_group_balance_artifacts(tmp_path: Path):
    summary = _load_summary_module()
    split = "laser_power"
    tag = "group_bal"
    baseline_id = summary._run_id(split, 12, "process_round_robin", "process_axis_profile")
    balanced_id = summary._run_id(split, 12, "process_round_robin", tag)

    manifest = _manifest(1200, 30, 96)
    split_payload = _split(1200)
    split_payload["group_key"] = "laser_power_W"
    _write_json(tmp_path / "outputs/data_audits" / f"{baseline_id}_manifest.json", manifest)
    _write_json(tmp_path / "outputs/data_splits" / f"{baseline_id}_split.json", split_payload)
    _write_json(tmp_path / "outputs/data_audits" / f"{balanced_id}_manifest.json", manifest)
    _write_json(tmp_path / "outputs/data_splits" / f"{balanced_id}_split.json", split_payload)
    for method, baseline_tag in summary.BASELINE_TAGS:
        _write_json(
            tmp_path / "outputs/baselines" / f"{baseline_id}_{baseline_tag}_regions_q90.json",
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
        tmp_path / "outputs/runs" / f"{balanced_id}_macro_pinn_minmax_{tag}_v1" / "metrics.json",
        _group_balanced_metrics(66.0),
    )

    payload = summary.collect_rows(
        tmp_path,
        (split,),
        12,
        "process_round_robin",
        (*summary.DEFAULT_PINN_SPECS, (tag, tag, tag)),
    )
    balanced_row = payload["splits"][split]["methods"][tag]

    assert payload["pinn_methods"] == [
        "no_process",
        "process_axis_v1",
        "broad_process_v1",
        tag,
    ]
    assert balanced_row["comparison_status"] == "comparable"
    assert balanced_row["rmse"] == 66.0
    assert balanced_row["data_loss_group_balance_enabled"] is True
    assert balanced_row["data_loss_group_balance_column"] == "process_condition"
    assert balanced_row["data_loss_group_balance_strength"] == 1.0
    assert balanced_row["data_loss_group_balance_groups"] == 2
    assert balanced_row["data_loss_group_balance_weight_sum"] == 800.0
    assert balanced_row["data_loss_objective_enabled"] is True
    assert balanced_row["data_loss_objective_weight_sum"] == 800.0


def test_phase30_summary_can_include_broad_process_graph_rbf_artifacts(tmp_path: Path):
    summary = _load_summary_module()
    split = "spot_size"
    baseline_id = summary._run_id(split, 12, "process_round_robin", "process_axis_profile")
    graph_id = summary._run_id(split, 12, "process_round_robin", "pg_rbf")

    manifest = _manifest(1200, 30, 96)
    split_payload = _split(1200)
    split_payload["group_key"] = "spot_size_um"
    _write_json(tmp_path / "outputs/data_audits" / f"{baseline_id}_manifest.json", manifest)
    _write_json(tmp_path / "outputs/data_splits" / f"{baseline_id}_split.json", split_payload)
    _write_json(tmp_path / "outputs/data_audits" / f"{graph_id}_manifest.json", manifest)
    _write_json(tmp_path / "outputs/data_splits" / f"{graph_id}_split.json", split_payload)
    for method, baseline_tag in summary.BASELINE_TAGS:
        _write_json(
            tmp_path / "outputs/baselines" / f"{baseline_id}_{baseline_tag}_regions_q90.json",
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
        tmp_path / "outputs/runs" / f"{graph_id}_macro_pinn_minmax_pg_rbf_v1" / "metrics.json",
        _process_graph_metrics(65.0),
    )

    payload = summary.collect_rows(
        tmp_path,
        (split,),
        12,
        "process_round_robin",
        (*summary.DEFAULT_PINN_SPECS, summary.BROAD_PROCESS_GRAPH_RBF_SPEC),
    )
    graph_row = payload["splits"][split]["methods"]["broad_process_graph_rbf"]

    assert payload["pinn_methods"] == [
        "no_process",
        "process_axis_v1",
        "broad_process_v1",
        "broad_process_graph_rbf",
    ]
    assert graph_row["comparison_status"] == "comparable"
    assert graph_row["rmse"] == 65.0
    assert graph_row["input_feature_count"] == 7
    assert graph_row["process_graph_enabled"] is True
    assert graph_row["process_graph_mode"] == "rbf"
    assert graph_row["process_graph_anchor_count"] == 4
    assert graph_row["process_graph_fit_scope"] == "train"
    assert graph_row["process_graph_length_scale"] == 1.0
    assert graph_row["input_effective_columns"][-1] == "process_graph_rbf_3"


def test_phase30_summary_can_include_broad_target_residual_artifacts(tmp_path: Path):
    summary = _load_summary_module()
    split = "laser_power"
    baseline_id = summary._run_id(split, 12, "process_round_robin", "process_axis_profile")
    residual_id = summary._run_id(split, 12, "process_round_robin", "target_resid_et")

    manifest = _manifest(1200, 30, 96)
    split_payload = _split(1200)
    split_payload["group_key"] = "laser_power_W"
    _write_json(tmp_path / "outputs/data_audits" / f"{baseline_id}_manifest.json", manifest)
    _write_json(tmp_path / "outputs/data_splits" / f"{baseline_id}_split.json", split_payload)
    _write_json(tmp_path / "outputs/data_audits" / f"{residual_id}_manifest.json", manifest)
    _write_json(tmp_path / "outputs/data_splits" / f"{residual_id}_split.json", split_payload)
    for method, baseline_tag in summary.BASELINE_TAGS:
        _write_json(
            tmp_path / "outputs/baselines" / f"{baseline_id}_{baseline_tag}_regions_q90.json",
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
        tmp_path / "outputs/runs" / f"{residual_id}_macro_pinn_minmax_target_resid_et_v1" / "metrics.json",
        _target_residual_metrics(64.0),
    )

    payload = summary.collect_rows(
        tmp_path,
        (split,),
        12,
        "process_round_robin",
        (*summary.DEFAULT_PINN_SPECS, summary.BROAD_TARGET_RESIDUAL_SPEC),
    )
    residual_row = payload["splits"][split]["methods"]["broad_target_residual"]

    assert payload["pinn_methods"] == [
        "no_process",
        "process_axis_v1",
        "broad_process_v1",
        "broad_target_residual",
    ]
    assert residual_row["comparison_status"] == "comparable"
    assert residual_row["rmse"] == 64.0
    assert residual_row["target_space"] == "residual"
    assert residual_row["target_residual_enabled"] is True
    assert residual_row["target_residual_strategy"] == "extra_trees"
    assert residual_row["target_residual_fit_points"] == 800
    assert residual_row["target_residual_train_rmse"] == 18.25
    assert residual_row["target_residual_feature_columns"][-1] == "spot_size_um"


def test_phase30_summary_can_include_broad_residual_backbone_artifacts(tmp_path: Path):
    summary = _load_summary_module()
    split = "spot_size"
    baseline_id = summary._run_id(split, 12, "process_round_robin", "process_axis_profile")
    residual_backbone_id = summary._run_id(split, 12, "process_round_robin", "res_backbone")

    manifest = _manifest(1200, 30, 96)
    split_payload = _split(1200)
    split_payload["group_key"] = "spot_size_um"
    _write_json(tmp_path / "outputs/data_audits" / f"{baseline_id}_manifest.json", manifest)
    _write_json(tmp_path / "outputs/data_splits" / f"{baseline_id}_split.json", split_payload)
    _write_json(tmp_path / "outputs/data_audits" / f"{residual_backbone_id}_manifest.json", manifest)
    _write_json(tmp_path / "outputs/data_splits" / f"{residual_backbone_id}_split.json", split_payload)
    for method, baseline_tag in summary.BASELINE_TAGS:
        _write_json(
            tmp_path / "outputs/baselines" / f"{baseline_id}_{baseline_tag}_regions_q90.json",
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
        tmp_path / "outputs/runs" / f"{residual_backbone_id}_macro_pinn_minmax_res_backbone_v1" / "metrics.json",
        _residual_backbone_metrics(63.0),
    )

    payload = summary.collect_rows(
        tmp_path,
        (split,),
        12,
        "process_round_robin",
        (*summary.DEFAULT_PINN_SPECS, summary.BROAD_RESIDUAL_BACKBONE_SPEC),
    )
    row = payload["splits"][split]["methods"]["broad_residual_backbone"]

    assert payload["pinn_methods"] == [
        "no_process",
        "process_axis_v1",
        "broad_process_v1",
        "broad_residual_backbone",
    ]
    assert row["comparison_status"] == "comparable"
    assert row["rmse"] == 63.0
    assert row["backbone_mode"] == "residual"
    assert row["backbone_residual_scale"] == 0.5
    assert row["backbone_parameter_count"] == 51201


def test_phase30_summary_can_include_broad_output_affine_artifacts(tmp_path: Path):
    summary = _load_summary_module()
    split = "laser_power"
    baseline_id = summary._run_id(split, 12, "process_round_robin", "process_axis_profile")
    output_affine_id = summary._run_id(split, 12, "process_round_robin", "out_affine")

    manifest = _manifest(1200, 30, 96)
    split_payload = _split(1200)
    split_payload["group_key"] = "laser_power_W"
    _write_json(tmp_path / "outputs/data_audits" / f"{baseline_id}_manifest.json", manifest)
    _write_json(tmp_path / "outputs/data_splits" / f"{baseline_id}_split.json", split_payload)
    _write_json(tmp_path / "outputs/data_audits" / f"{output_affine_id}_manifest.json", manifest)
    _write_json(tmp_path / "outputs/data_splits" / f"{output_affine_id}_split.json", split_payload)
    for method, baseline_tag in summary.BASELINE_TAGS:
        _write_json(
            tmp_path / "outputs/baselines" / f"{baseline_id}_{baseline_tag}_regions_q90.json",
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
        tmp_path / "outputs/runs" / f"{output_affine_id}_macro_pinn_minmax_out_affine_v1" / "metrics.json",
        _output_affine_metrics(62.0),
    )

    payload = summary.collect_rows(
        tmp_path,
        (split,),
        12,
        "process_round_robin",
        (*summary.DEFAULT_PINN_SPECS, summary.BROAD_OUTPUT_AFFINE_SPEC),
    )
    row = payload["splits"][split]["methods"]["broad_output_affine"]

    assert payload["pinn_methods"] == [
        "no_process",
        "process_axis_v1",
        "broad_process_v1",
        "broad_output_affine",
    ]
    assert row["comparison_status"] == "comparable"
    assert row["rmse"] == 62.0
    assert row["output_affine_enabled"] is True
    assert row["output_affine_mode"] == "linear"
    assert row["output_affine_scale"] == 0.5
    assert row["output_affine_input_dim"] == 3


def test_phase30_summary_can_include_broad_prediction_anchor_artifacts(tmp_path: Path):
    summary = _load_summary_module()
    split = "laser_power"
    baseline_id = summary._run_id(split, 12, "process_round_robin", "process_axis_profile")
    prediction_anchor_id = summary._run_id(split, 12, "process_round_robin", "pred_anchor")

    manifest = _manifest(1200, 30, 96)
    split_payload = _split(1200)
    split_payload["group_key"] = "laser_power_W"
    _write_json(tmp_path / "outputs/data_audits" / f"{baseline_id}_manifest.json", manifest)
    _write_json(tmp_path / "outputs/data_splits" / f"{baseline_id}_split.json", split_payload)
    _write_json(tmp_path / "outputs/data_audits" / f"{prediction_anchor_id}_manifest.json", manifest)
    _write_json(tmp_path / "outputs/data_splits" / f"{prediction_anchor_id}_split.json", split_payload)
    for method, baseline_tag in summary.BASELINE_TAGS:
        _write_json(
            tmp_path / "outputs/baselines" / f"{baseline_id}_{baseline_tag}_regions_q90.json",
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
        tmp_path / "outputs/runs" / f"{prediction_anchor_id}_macro_pinn_minmax_pred_anchor_v1" / "metrics.json",
        _prediction_anchor_metrics(63.0),
    )

    payload = summary.collect_rows(
        tmp_path,
        (split,),
        12,
        "process_round_robin",
        (*summary.DEFAULT_PINN_SPECS, summary.BROAD_PREDICTION_ANCHOR_SPEC),
    )
    row = payload["splits"][split]["methods"]["broad_prediction_anchor"]

    assert payload["pinn_methods"] == [
        "no_process",
        "process_axis_v1",
        "broad_process_v1",
        "broad_prediction_anchor",
    ]
    assert row["comparison_status"] == "comparable"
    assert row["rmse"] == 63.0
    assert row["prediction_anchor_enabled"] is True
    assert row["prediction_anchor_weight"] == 0.05
    assert row["prediction_anchor_target_space"] == "normalized_training_target"


def test_phase30_summary_can_include_broad_process_encoder_artifacts(tmp_path: Path):
    summary = _load_summary_module()
    split = "laser_power"
    baseline_id = summary._run_id(split, 12, "process_round_robin", "process_axis_profile")
    encoder_id = summary._run_id(split, 12, "process_round_robin", "proc_enc")

    manifest = _manifest(1200, 30, 96)
    split_payload = _split(1200)
    split_payload["group_key"] = "laser_power_W"
    _write_json(tmp_path / "outputs/data_audits" / f"{baseline_id}_manifest.json", manifest)
    _write_json(tmp_path / "outputs/data_splits" / f"{baseline_id}_split.json", split_payload)
    _write_json(tmp_path / "outputs/data_audits" / f"{encoder_id}_manifest.json", manifest)
    _write_json(tmp_path / "outputs/data_splits" / f"{encoder_id}_split.json", split_payload)
    for method, baseline_tag in summary.BASELINE_TAGS:
        _write_json(
            tmp_path / "outputs/baselines" / f"{baseline_id}_{baseline_tag}_regions_q90.json",
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
        tmp_path / "outputs/runs" / f"{encoder_id}_macro_pinn_minmax_proc_enc_v1" / "metrics.json",
        _process_encoder_metrics(61.0),
    )

    payload = summary.collect_rows(
        tmp_path,
        (split,),
        12,
        "process_round_robin",
        (*summary.DEFAULT_PINN_SPECS, summary.BROAD_PROCESS_ENCODER_SPEC),
    )
    row = payload["splits"][split]["methods"]["broad_process_encoder"]

    assert payload["pinn_methods"] == [
        "no_process",
        "process_axis_v1",
        "broad_process_v1",
        "broad_process_encoder",
    ]
    assert row["comparison_status"] == "comparable"
    assert row["rmse"] == 61.0
    assert row["process_encoder_enabled"] is True
    assert row["process_encoder_mode"] == "linear"
    assert row["process_encoder_input_dim"] == 7
    assert row["process_encoder_output_dim"] == 3
    assert row["process_encoder_identity_initialized"] is True


def test_phase30_summary_can_include_broad_derived_process_artifacts(tmp_path: Path):
    summary = _load_summary_module()
    split = "laser_power"
    baseline_id = summary._run_id(split, 12, "process_round_robin", "process_axis_profile")
    derived_id = summary._run_id(split, 12, "process_round_robin", "phys_proc")

    manifest = _manifest(1200, 30, 96)
    split_payload = _split(1200)
    split_payload["group_key"] = "laser_power_W"
    _write_json(tmp_path / "outputs/data_audits" / f"{baseline_id}_manifest.json", manifest)
    _write_json(tmp_path / "outputs/data_splits" / f"{baseline_id}_split.json", split_payload)
    _write_json(tmp_path / "outputs/data_audits" / f"{derived_id}_manifest.json", manifest)
    _write_json(tmp_path / "outputs/data_splits" / f"{derived_id}_split.json", split_payload)
    for method, baseline_tag in summary.BASELINE_TAGS:
        _write_json(
            tmp_path / "outputs/baselines" / f"{baseline_id}_{baseline_tag}_regions_q90.json",
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
        tmp_path / "outputs/runs" / f"{derived_id}_macro_pinn_minmax_phys_proc_v1" / "metrics.json",
        _derived_process_metrics(61.0),
    )

    payload = summary.collect_rows(
        tmp_path,
        (split,),
        12,
        "process_round_robin",
        (*summary.DEFAULT_PINN_SPECS, summary.BROAD_DERIVED_PROCESS_SPEC),
    )
    row = payload["splits"][split]["methods"]["broad_derived_process"]

    assert payload["pinn_methods"] == [
        "no_process",
        "process_axis_v1",
        "broad_process_v1",
        "broad_derived_process",
    ]
    assert row["comparison_status"] == "comparable"
    assert row["rmse"] == 61.0
    assert row["derived_process_enabled"] is True
    assert row["derived_process_mode"] == "am_energy_v1"
    assert row["derived_process_feature_names"][-1] == "dwell_time_ms"
    assert row["input_effective_columns"][-1] == "dwell_time_ms"
