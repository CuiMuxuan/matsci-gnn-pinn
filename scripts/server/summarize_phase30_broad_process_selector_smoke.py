#!/usr/bin/env python3
"""Compare Phase 29 and Phase 30 broad process-profile smoke artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SPLITS = ("line", "laser_power", "scan_speed", "spot_size", "process")
BASELINE_TAGS = (
    ("mean", "mean_constant"),
    ("knn_coords", "knn_coords"),
    ("knn_process", "knn_process"),
    ("extra_trees_coords", "extra_trees_coords"),
    ("extra_trees_process", "extra_trees_process"),
)
DEFAULT_PINN_SPECS = (
    (
        "no_process",
        "process_axis_profile",
        "no_process",
    ),
    (
        "process_axis_v1",
        "process_axis_profile",
        "process_axis_profile",
    ),
    (
        "broad_process_v1",
        "broad_process_profile",
        "broad_process_profile",
    ),
)
BROAD_PROCESS_V2_SPEC = (
    "broad_process_v2",
    "broad_process_profile_v2",
    "broad_process_profile_v2",
)
BROAD_PROCESS_FOURIER_SPEC = (
    "broad_process_fourier",
    "broad_process_fourier",
    "broad_process_fourier",
)
BROAD_PROCESS_RESIDUAL_SPEC = (
    "broad_residual_mlp",
    "broad_residual_mlp",
    "broad_residual_mlp",
)
DEFAULT_BROAD_PROCESS_GRAPH_RBF_TAG = "pg_rbf"
BROAD_PROCESS_GRAPH_RBF_SPEC = (
    "broad_process_graph_rbf",
    DEFAULT_BROAD_PROCESS_GRAPH_RBF_TAG,
    DEFAULT_BROAD_PROCESS_GRAPH_RBF_TAG,
)
DEFAULT_BROAD_TARGET_RESIDUAL_TAG = "target_resid_et"
BROAD_TARGET_RESIDUAL_SPEC = (
    "broad_target_residual",
    DEFAULT_BROAD_TARGET_RESIDUAL_TAG,
    DEFAULT_BROAD_TARGET_RESIDUAL_TAG,
)
DEFAULT_BROAD_RESIDUAL_BACKBONE_TAG = "res_backbone"
BROAD_RESIDUAL_BACKBONE_SPEC = (
    "broad_residual_backbone",
    DEFAULT_BROAD_RESIDUAL_BACKBONE_TAG,
    DEFAULT_BROAD_RESIDUAL_BACKBONE_TAG,
)
DEFAULT_BROAD_OUTPUT_AFFINE_TAG = "out_affine"
BROAD_OUTPUT_AFFINE_SPEC = (
    "broad_output_affine",
    DEFAULT_BROAD_OUTPUT_AFFINE_TAG,
    DEFAULT_BROAD_OUTPUT_AFFINE_TAG,
)
DEFAULT_BROAD_PREDICTION_ANCHOR_TAG = "pred_anchor"
BROAD_PREDICTION_ANCHOR_SPEC = (
    "broad_prediction_anchor",
    DEFAULT_BROAD_PREDICTION_ANCHOR_TAG,
    DEFAULT_BROAD_PREDICTION_ANCHOR_TAG,
)
DEFAULT_BROAD_PROCESS_ENCODER_TAG = "proc_enc"
BROAD_PROCESS_ENCODER_SPEC = (
    "broad_process_encoder",
    DEFAULT_BROAD_PROCESS_ENCODER_TAG,
    DEFAULT_BROAD_PROCESS_ENCODER_TAG,
)
DEFAULT_BROAD_PROCESS_GROUP_BALANCE_TAG = "group_bal"
BROAD_PROCESS_GROUP_BALANCE_SPEC = (
    "broad_process_group_balance",
    DEFAULT_BROAD_PROCESS_GROUP_BALANCE_TAG,
    DEFAULT_BROAD_PROCESS_GROUP_BALANCE_TAG,
)
DEFAULT_BROAD_DERIVED_PROCESS_TAG = "phys_proc"
BROAD_DERIVED_PROCESS_SPEC = (
    "broad_derived_process",
    DEFAULT_BROAD_DERIVED_PROCESS_TAG,
    DEFAULT_BROAD_DERIVED_PROCESS_TAG,
)
DEFAULT_BROAD_REGION_WEIGHTED_TAG = "rw2"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _test_metrics(data: dict[str, Any]) -> dict[str, Any]:
    results = data.get("results") or []
    if results and isinstance(results[0], dict):
        data = results[0]
    split_metrics = data.get("split_metrics") or {}
    if isinstance(split_metrics, dict) and isinstance(split_metrics.get("test"), dict):
        return split_metrics["test"]
    metrics = data.get("metrics") or {}
    if isinstance(metrics, dict) and isinstance(metrics.get("test"), dict):
        return metrics["test"]
    return data


def _metric_value(metrics: dict[str, Any], name: str) -> float | None:
    primary_metrics = metrics.get("metrics") or metrics
    value = primary_metrics.get(name)
    return float(value) if value is not None else None


def _region_rmse(metrics: dict[str, Any], name: str) -> float | None:
    regions = metrics.get("region_metrics") or metrics.get("regions") or {}
    region = regions.get(name) or {}
    region_metrics = region.get("metrics") or region
    value = region_metrics.get("rmse")
    return float(value) if value is not None else None


def _metric_fields(metrics: dict[str, Any]) -> dict[str, float | None]:
    return {
        "rmse": _metric_value(metrics, "rmse"),
        "mae": _metric_value(metrics, "mae"),
        "relative_l2": _metric_value(metrics, "relative_l2"),
        "hot_q90_rmse": _region_rmse(metrics, "hot_q90"),
        "gradient_q90_rmse": _region_rmse(metrics, "gradient_q90"),
    }


def _run_id(split: str, dataset_limit: int, dataset_order: str, profile_tag: str) -> str:
    return (
        "ambench_multiline_process_temperature_"
        f"broad{dataset_limit}_{dataset_order}_{split}_{profile_tag}_smoke_a100_sxm4_40gb_v1"
    )


def _manifest_path(root: Path, run_id: str) -> Path:
    return root / "outputs" / "data_audits" / f"{run_id}_manifest.json"


def _split_path(root: Path, run_id: str) -> Path:
    return root / "outputs" / "data_splits" / f"{run_id}_split.json"


def _baseline_run_id(split: str, dataset_limit: int, dataset_order: str) -> str:
    return _run_id(
        split=split,
        dataset_limit=dataset_limit,
        dataset_order=dataset_order,
        profile_tag="process_axis_profile",
    )


def _manifest_signature(manifest: dict[str, Any]) -> dict[str, Any]:
    metadata = manifest.get("metadata") or {}
    sampling = metadata.get("sampling") or {}
    selection = sampling.get("selection") or {}
    return {
        "target": manifest.get("target"),
        "n_rows": manifest.get("n_rows"),
        "dataset_paths": manifest.get("dataset_paths"),
        "dataset_selection": metadata.get("dataset_selection"),
        "sampling": {
            "frame_start": sampling.get("frame_start"),
            "frame_step": sampling.get("frame_step"),
            "max_frames": sampling.get("max_frames"),
            "row_start": sampling.get("row_start"),
            "row_step": sampling.get("row_step"),
            "max_rows": sampling.get("max_rows"),
            "col_start": sampling.get("col_start"),
            "col_step": sampling.get("col_step"),
            "max_cols": sampling.get("max_cols"),
            "min_signal": sampling.get("min_signal"),
            "selection": {
                "mode": selection.get("mode"),
                "active_target": selection.get("active_target"),
                "hot_quantile": selection.get("hot_quantile"),
                "gradient_quantile": selection.get("gradient_quantile"),
                "background_fraction": selection.get("background_fraction"),
                "max_points_per_frame": selection.get("max_points_per_frame"),
            },
        },
        "process_groups": metadata.get("process_groups"),
    }


def _split_signature(split: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_rows": split.get("n_rows"),
        "n_groups": split.get("n_groups"),
        "group_key": split.get("group_key"),
        "strategy": split.get("strategy"),
        "group_order": split.get("group_order"),
        "group_splits": split.get("group_splits"),
        "rows_per_group": split.get("rows_per_group"),
        "splits": split.get("splits"),
    }


def _comparison_signature(root: Path, run_id: str) -> dict[str, Any] | None:
    manifest_file = _manifest_path(root, run_id)
    split_file = _split_path(root, run_id)
    if not manifest_file.exists() or not split_file.exists():
        return None
    return {
        "manifest": _manifest_signature(_read_json(manifest_file)),
        "split": _split_signature(_read_json(split_file)),
    }


def _comparison_reason(
    reference: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> str | None:
    if reference is None:
        return "reference manifest/split is missing"
    if candidate is None:
        return "candidate manifest/split is missing"
    if reference == candidate:
        return None
    ref_manifest = reference.get("manifest") or {}
    cand_manifest = candidate.get("manifest") or {}
    for key in ("target", "n_rows", "dataset_paths", "dataset_selection", "sampling", "process_groups"):
        if ref_manifest.get(key) != cand_manifest.get(key):
            return f"manifest {key} differs from reference"
    ref_split = reference.get("split") or {}
    cand_split = candidate.get("split") or {}
    for key in ("n_rows", "n_groups", "group_key", "strategy", "group_order", "group_splits", "rows_per_group", "splits"):
        if ref_split.get(key) != cand_split.get(key):
            return f"split {key} differs from reference"
    return "manifest/split signature differs from reference"


def _collect_profile_metadata(data: dict[str, Any]) -> dict[str, Any]:
    features = data.get("input_features") or {}
    profile = features.get("conditioning_profile") or {}
    selected = profile.get("selected") or {}
    effective = profile.get("effective") or {}
    process_graph = features.get("process_graph_features") or {}
    derived_process = features.get("derived_process_features") or {}
    spacetime_encoding = data.get("spacetime_encoding") or {}
    residual_correction = data.get("residual_correction") or {}
    data_loss_weighting = data.get("data_loss_weighting") or {}
    data_loss_group_balance = data.get("data_loss_group_balance") or {}
    data_loss_objective = data.get("data_loss_objective") or {}
    target_residual = data.get("target_residual_baseline") or {}
    target_normalization = data.get("target_normalization") or {}
    backbone = data.get("backbone") or {}
    process_encoder = data.get("process_encoder") or {}
    output_affine = data.get("output_affine") or {}
    prediction_anchor = data.get("prediction_anchor") or {}
    return {
        "input_features_enabled": features.get("enabled"),
        "input_feature_count": features.get("count"),
        "input_effective_columns": features.get("effective_columns"),
        "conditioning_mode": features.get("conditioning_mode"),
        "feature_normalization": (features.get("normalization") or {}).get("mode"),
        "process_graph_enabled": process_graph.get("enabled"),
        "process_graph_mode": process_graph.get("mode"),
        "process_graph_anchor_count": process_graph.get("anchor_count"),
        "process_graph_fit_scope": process_graph.get("fit_scope"),
        "process_graph_length_scale": process_graph.get("length_scale"),
        "derived_process_enabled": derived_process.get("enabled"),
        "derived_process_mode": derived_process.get("mode"),
        "derived_process_feature_names": derived_process.get("feature_names"),
        "derived_process_source_columns": derived_process.get("source_columns"),
        "spacetime_encoding": spacetime_encoding.get("encoding"),
        "spacetime_fourier_bands": spacetime_encoding.get("fourier_bands"),
        "spacetime_input_dim": spacetime_encoding.get("input_dim"),
        "conditioning_profile": profile.get("profile"),
        "conditioning_profile_group_key": profile.get("group_key"),
        "selected_conditioning_mode": selected.get("conditioning_mode"),
        "selected_feature_normalization": selected.get("feature_normalization"),
        "effective_conditioning_mode": effective.get("conditioning_mode"),
        "effective_feature_columns": effective.get("feature_columns"),
        "residual_correction_enabled": residual_correction.get("enabled"),
        "residual_correction_mode": residual_correction.get("mode"),
        "residual_correction_scale": residual_correction.get("scale"),
        "residual_correction_start_step": residual_correction.get("start_step"),
        "data_loss_weighting_enabled": data_loss_weighting.get("enabled"),
        "data_loss_weighting_mode": data_loss_weighting.get("mode"),
        "data_loss_region_weight": data_loss_weighting.get("region_weight"),
        "data_loss_weighted_points": data_loss_weighting.get("selected_points"),
        "data_loss_group_balance_enabled": data_loss_group_balance.get("enabled"),
        "data_loss_group_balance_column": data_loss_group_balance.get("column"),
        "data_loss_group_balance_strength": data_loss_group_balance.get("strength"),
        "data_loss_group_balance_groups": data_loss_group_balance.get("group_count"),
        "data_loss_group_balance_weight_sum": data_loss_group_balance.get("weight_sum"),
        "data_loss_objective_enabled": data_loss_objective.get("enabled"),
        "data_loss_objective_weight_sum": data_loss_objective.get("weight_sum"),
        "data_loss_objective_mean_weight": data_loss_objective.get("mean_weight"),
        "target_space": target_normalization.get("target_space"),
        "target_residual_enabled": target_residual.get("enabled"),
        "target_residual_strategy": target_residual.get("strategy"),
        "target_residual_feature_columns": target_residual.get("feature_columns"),
        "target_residual_fit_points": target_residual.get("fit_points"),
        "target_residual_train_rmse": target_residual.get("train_residual_rmse"),
        "backbone_mode": backbone.get("mode"),
        "backbone_residual_scale": backbone.get("residual_scale"),
        "backbone_parameter_count": backbone.get("parameter_count"),
        "process_encoder_enabled": process_encoder.get("enabled"),
        "process_encoder_mode": process_encoder.get("mode"),
        "process_encoder_input_dim": process_encoder.get("input_dim"),
        "process_encoder_output_dim": process_encoder.get("output_dim"),
        "process_encoder_identity_initialized": process_encoder.get("identity_initialized"),
        "process_encoder_parameter_count": process_encoder.get("parameter_count"),
        "output_affine_enabled": output_affine.get("enabled"),
        "output_affine_mode": output_affine.get("mode"),
        "output_affine_scale": output_affine.get("scale"),
        "output_affine_input_dim": output_affine.get("input_dim"),
        "prediction_anchor_enabled": prediction_anchor.get("enabled"),
        "prediction_anchor_weight": prediction_anchor.get("weight"),
        "prediction_anchor_target_space": prediction_anchor.get("target_space"),
    }


def collect_rows(
    root: Path,
    splits: tuple[str, ...],
    dataset_limit: int,
    dataset_order: str,
    pinn_specs: tuple[tuple[str, str, str], ...] = DEFAULT_PINN_SPECS,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "dataset_limit": dataset_limit,
        "dataset_order": dataset_order,
        "pinn_methods": [method for method, _, _ in pinn_specs],
        "splits": {},
    }
    for split in splits:
        rows: dict[str, Any] = {"methods": {}}
        baseline_id = _baseline_run_id(split, dataset_limit, dataset_order)
        reference_signature = _comparison_signature(root, baseline_id)
        manifest_path = _manifest_path(root, baseline_id)
        if manifest_path.exists():
            manifest = _read_json(manifest_path)
            rows["manifest"] = {
                "path": str(manifest_path),
                "n_rows": manifest.get("n_rows"),
                "dataset_paths": manifest.get("dataset_paths"),
                "dataset_selection": (manifest.get("metadata") or {}).get("dataset_selection"),
                "sampling": (manifest.get("metadata") or {}).get("sampling"),
                "process_groups": (manifest.get("metadata") or {}).get("process_groups"),
            }
        rows["reference_run_id"] = baseline_id
        rows["reference_signature_available"] = reference_signature is not None
        rows["all_methods_comparable"] = reference_signature is not None
        for method, tag in BASELINE_TAGS:
            path = root / "outputs" / "baselines" / f"{baseline_id}_{tag}_regions_q90.json"
            row = {"path": str(path), "comparison_status": "comparable"}
            if path.exists():
                row.update(_metric_fields(_test_metrics(_read_json(path))))
            else:
                row["missing"] = True
                rows["all_methods_comparable"] = False
            rows["methods"][method] = row
        for method, profile_tag, run_tag in pinn_specs:
            run_id = _run_id(split, dataset_limit, dataset_order, profile_tag)
            path = root / "outputs" / "runs" / f"{run_id}_macro_pinn_minmax_{run_tag}_v1" / "metrics.json"
            candidate_signature = _comparison_signature(root, run_id)
            comparison_reason = _comparison_reason(reference_signature, candidate_signature)
            comparison_status = "comparable" if comparison_reason is None else "incomparable"
            row = {
                "path": str(path),
                "run_id": run_id,
                "comparison_status": comparison_status,
            }
            if comparison_reason:
                row["comparison_reason"] = comparison_reason
                rows["all_methods_comparable"] = False
            if path.exists():
                data = _read_json(path)
                row.update(_metric_fields(_test_metrics(data)))
                row.update(_collect_profile_metadata(data))
            else:
                row["missing"] = True
                rows["all_methods_comparable"] = False
            rows["methods"][method] = row
        summary["splits"][split] = rows
    return summary


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def print_metrics_markdown(summary: dict[str, Any]) -> None:
    print(
        "| split | method | status | test RMSE | hot q90 RMSE | gradient q90 RMSE | "
        "spacetime | backbone | process encoder | output affine | prediction anchor | graph | target residual | selected | effective features |"
    )
    print("|---|---|---|---:|---:|---:|---|---|---|---|---|---|---|---|---|")
    pinn_methods = tuple(summary.get("pinn_methods") or ("no_process", "process_axis_v1", "broad_process_v1"))
    method_order = (
        "mean",
        *pinn_methods,
        "extra_trees_process",
        "knn_process",
        "extra_trees_coords",
        "knn_coords",
    )
    for split, split_rows in summary["splits"].items():
        for method in method_order:
            row = split_rows["methods"].get(method, {})
            if row.get("missing"):
                print(
                    f"| {split} | {method} | MISSING |  |  |  |  |  |  |  |  |  |  |  | {row.get('path', '')} |"
                )
                continue
            status = str(row.get("comparison_status") or "comparable")
            if status != "comparable":
                reason = row.get("comparison_reason", "")
                print(
                    f"| {split} | {method} | INCOMPARABLE: {reason} |  |  |  |  |  |  |  |  |  |  |  | {row.get('path', '')} |"
                )
                continue
            selected = ""
            if row.get("selected_conditioning_mode"):
                selected = "{}/{}".format(
                    row.get("selected_conditioning_mode"),
                    row.get("selected_feature_normalization"),
                )
            spacetime = row.get("spacetime_encoding") or ""
            if row.get("spacetime_fourier_bands"):
                spacetime = f"{spacetime}/{row.get('spacetime_fourier_bands')}"
            backbone = ""
            if row.get("backbone_mode"):
                backbone = str(row.get("backbone_mode"))
                if row.get("backbone_residual_scale") is not None:
                    backbone = f"{backbone}@{row.get('backbone_residual_scale')}"
            process_encoder = ""
            if row.get("process_encoder_enabled"):
                process_encoder = "{}:{}->{}".format(
                    row.get("process_encoder_mode"),
                    row.get("process_encoder_input_dim"),
                    row.get("process_encoder_output_dim"),
                )
            output_affine = ""
            if row.get("output_affine_enabled"):
                output_affine = "{}:{}@{}".format(
                    row.get("output_affine_mode"),
                    row.get("output_affine_input_dim"),
                    row.get("output_affine_scale"),
                )
            prediction_anchor = ""
            if row.get("prediction_anchor_enabled"):
                prediction_anchor = "{}@{}".format(
                    _fmt(row.get("prediction_anchor_weight")),
                    row.get("prediction_anchor_target_space"),
                )
            graph = ""
            if row.get("process_graph_enabled"):
                graph = "{}:{}@{}".format(
                    row.get("process_graph_mode"),
                    row.get("process_graph_anchor_count"),
                    row.get("process_graph_fit_scope"),
                )
            target_residual = ""
            if row.get("target_residual_enabled"):
                target_residual = "{}@{} rmse={}".format(
                    row.get("target_residual_strategy"),
                    row.get("target_residual_fit_points"),
                    _fmt(row.get("target_residual_train_rmse")),
                )
            features = row.get("input_effective_columns") or row.get("effective_feature_columns")
            print(
                (
                    "| {split} | {method} | {status} | {rmse} | {hot} | {grad} | "
                    "{spacetime} | {backbone} | {process_encoder} | {output_affine} | {prediction_anchor} | {graph} | {target_residual} | {selected} | {features} |"
                ).format(
                    split=split,
                    method=method,
                    status=status,
                    rmse=_fmt(row.get("rmse")),
                    hot=_fmt(row.get("hot_q90_rmse")),
                    grad=_fmt(row.get("gradient_q90_rmse")),
                    spacetime=spacetime,
                    backbone=backbone,
                    process_encoder=process_encoder,
                    output_affine=output_affine,
                    prediction_anchor=prediction_anchor,
                    graph=graph,
                    target_residual=target_residual,
                    selected=selected,
                    features=_fmt(features),
                )
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Project root containing outputs/")
    parser.add_argument("--dataset-limit", type=int, default=12)
    parser.add_argument("--dataset-order", default="process_round_robin")
    parser.add_argument("--split", action="append", choices=DEFAULT_SPLITS)
    parser.add_argument("--json-output", help="Optional path to write JSON summary")
    parser.add_argument(
        "--require-comparable",
        action="store_true",
        help="Exit non-zero if any requested method has a missing or mismatched manifest/split signature.",
    )
    parser.add_argument(
        "--include-broad-process-v2",
        action="store_true",
        help=(
            "Also summarize the Phase 32 broad_process_v2 artifacts. "
            "Expected run/profile tag is broad_process_profile_v2."
        ),
    )
    parser.add_argument(
        "--include-broad-process-fourier",
        action="store_true",
        help=(
            "Also summarize the Phase 33 broad_process_fourier artifacts. "
            "These use broad_process_v1 routing with fixed Fourier spacetime features."
        ),
    )
    parser.add_argument(
        "--include-broad-process-residual",
        action="store_true",
        help=(
            "Also summarize the Phase 34 broad_residual_mlp artifacts. "
            "These use broad_process_v1 routing plus a weak learned residual correction head."
        ),
    )
    parser.add_argument(
        "--include-broad-region-weighted",
        action="store_true",
        help=(
            "Also summarize a Phase 35 region-weighted data-loss artifact. "
            "The expected run/profile tag defaults to rw2."
        ),
    )
    parser.add_argument(
        "--include-broad-process-graph-rbf",
        action="store_true",
        help=(
            "Also summarize the Phase 36 broad_process_graph_rbf artifacts. "
            "These use broad_process_v1 routing plus process-neighborhood RBF graph features."
        ),
    )
    parser.add_argument(
        "--include-broad-target-residual",
        action="store_true",
        help=(
            "Also summarize the Phase 37 broad_target_residual artifacts. "
            "These train Macro PINN on target residuals after fitting a train-split baseline."
        ),
    )
    parser.add_argument(
        "--include-broad-residual-backbone",
        action="store_true",
        help=(
            "Also summarize the Phase 38 broad_residual_backbone artifacts. "
            "These use broad_process_v1 routing with a residual Macro PINN backbone."
        ),
    )
    parser.add_argument(
        "--include-broad-output-affine",
        action="store_true",
        help=(
            "Also summarize the Phase 39 broad_output_affine artifacts. "
            "These use broad_process_v1 routing with a process-conditioned output affine calibration head."
        ),
    )
    parser.add_argument(
        "--include-broad-prediction-anchor",
        action="store_true",
        help=(
            "Also summarize the Phase 42 broad_prediction_anchor artifacts. "
            "These use broad_process_v1 routing with a prediction-space anchor regularizer."
        ),
    )
    parser.add_argument(
        "--include-broad-process-encoder",
        action="store_true",
        help=(
            "Also summarize the Phase 43 broad_process_encoder artifacts. "
            "These use broad_process_v1 routing with a learned low-dimensional process encoder."
        ),
    )
    parser.add_argument(
        "--include-broad-process-group-balance",
        action="store_true",
        help=(
            "Also summarize the Phase 44 broad_process_group_balance artifacts. "
            "These use broad_process_v1 routing with a group-balanced supervised objective."
        ),
    )
    parser.add_argument(
        "--include-broad-derived-process",
        action="store_true",
        help=(
            "Also summarize Phase 41 broad_derived_process artifacts. "
            "These use broad_process_v1 routing plus deterministic AM process-derived input features."
        ),
    )
    parser.add_argument(
        "--broad-region-weighted-tag",
        default=DEFAULT_BROAD_REGION_WEIGHTED_TAG,
        help="Run/profile tag used with --include-broad-region-weighted.",
    )
    parser.add_argument(
        "--broad-process-graph-rbf-tag",
        default=DEFAULT_BROAD_PROCESS_GRAPH_RBF_TAG,
        help="Run/profile tag used with --include-broad-process-graph-rbf.",
    )
    parser.add_argument(
        "--broad-target-residual-tag",
        default=DEFAULT_BROAD_TARGET_RESIDUAL_TAG,
        help="Run/profile tag used with --include-broad-target-residual.",
    )
    parser.add_argument(
        "--broad-residual-backbone-tag",
        default=DEFAULT_BROAD_RESIDUAL_BACKBONE_TAG,
        help="Run/profile tag used with --include-broad-residual-backbone.",
    )
    parser.add_argument(
        "--broad-output-affine-tag",
        default=DEFAULT_BROAD_OUTPUT_AFFINE_TAG,
        help="Run/profile tag used with --include-broad-output-affine.",
    )
    parser.add_argument(
        "--broad-prediction-anchor-tag",
        default=DEFAULT_BROAD_PREDICTION_ANCHOR_TAG,
        help="Run/profile tag used with --include-broad-prediction-anchor.",
    )
    parser.add_argument(
        "--broad-process-encoder-tag",
        default=DEFAULT_BROAD_PROCESS_ENCODER_TAG,
        help="Run/profile tag used with --include-broad-process-encoder.",
    )
    parser.add_argument(
        "--group-balance-tag",
        default=DEFAULT_BROAD_PROCESS_GROUP_BALANCE_TAG,
        help="Run/profile tag used with --include-broad-process-group-balance.",
    )
    parser.add_argument(
        "--broad-derived-process-tag",
        default=DEFAULT_BROAD_DERIVED_PROCESS_TAG,
        help="Run/profile tag used with --include-broad-derived-process.",
    )
    args = parser.parse_args()

    splits = tuple(args.split) if args.split else DEFAULT_SPLITS
    pinn_specs = DEFAULT_PINN_SPECS
    if args.include_broad_process_v2:
        pinn_specs = (*pinn_specs, BROAD_PROCESS_V2_SPEC)
    if args.include_broad_process_fourier:
        pinn_specs = (*pinn_specs, BROAD_PROCESS_FOURIER_SPEC)
    if args.include_broad_process_residual:
        pinn_specs = (*pinn_specs, BROAD_PROCESS_RESIDUAL_SPEC)
    if args.include_broad_region_weighted:
        tag = args.broad_region_weighted_tag
        pinn_specs = (*pinn_specs, (tag, tag, tag))
    if args.include_broad_process_graph_rbf:
        tag = args.broad_process_graph_rbf_tag
        pinn_specs = (*pinn_specs, ("broad_process_graph_rbf", tag, tag))
    if args.include_broad_target_residual:
        tag = args.broad_target_residual_tag
        pinn_specs = (*pinn_specs, ("broad_target_residual", tag, tag))
    if args.include_broad_residual_backbone:
        tag = args.broad_residual_backbone_tag
        pinn_specs = (*pinn_specs, ("broad_residual_backbone", tag, tag))
    if args.include_broad_output_affine:
        tag = args.broad_output_affine_tag
        pinn_specs = (*pinn_specs, ("broad_output_affine", tag, tag))
    if args.include_broad_prediction_anchor:
        tag = args.broad_prediction_anchor_tag
        pinn_specs = (*pinn_specs, ("broad_prediction_anchor", tag, tag))
    if args.include_broad_process_encoder:
        tag = args.broad_process_encoder_tag
        pinn_specs = (*pinn_specs, ("broad_process_encoder", tag, tag))
    if args.include_broad_process_group_balance:
        tag = args.group_balance_tag
        pinn_specs = (*pinn_specs, ("broad_process_group_balance", tag, tag))
    if args.include_broad_derived_process:
        tag = args.broad_derived_process_tag
        pinn_specs = (*pinn_specs, ("broad_derived_process", tag, tag))
    summary = collect_rows(Path(args.root), splits, args.dataset_limit, args.dataset_order, pinn_specs)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print_metrics_markdown(summary)
    if args.require_comparable and any(
        not split_rows.get("all_methods_comparable")
        for split_rows in summary["splits"].values()
    ):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
