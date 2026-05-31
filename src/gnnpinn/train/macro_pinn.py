"""Train a minimal macro PINN on a local field table."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import platform
from pathlib import Path
from typing import Any

from gnnpinn.data.loaders import load_field_table
from gnnpinn.data.splits import load_split_manifest, split_indices
from gnnpinn.eval.metrics import mae, relative_l2, rmse
from gnnpinn.eval.regions import _spatial_gradient_scores, region_metric_tables
from gnnpinn.models.closure import (
    CoordinateRBFGraphConfig,
    CoordinateRBFGraphFeatureProvider,
    RealMicroGraphFeatureConfig,
    RealMicroGraphFeatureProvider,
    RealMicroRegionEmbeddingFeatureConfig,
    RealMicroRegionEmbeddingFeatureProvider,
    RealMicroRegionFeatureConfig,
    RealMicroRegionFeatureProvider,
    SparseLibrary,
    SparseLibraryConfig,
    ToyStaticGraphConfig,
    ToyStaticGraphEmbeddingProvider,
    export_linear_library_expression,
    expression_to_string,
    graph_feature_names,
    l1_sparsity,
)
from gnnpinn.models.pinn import MLP, MLPConfig, MacroPINN
from gnnpinn.physics.heat import HeatEquationParams, transient_heat_residual


AM_ENERGY_DERIVED_FEATURE_NAMES = (
    "line_energy_J_per_mm",
    "energy_density_proxy_J_per_mm_um",
    "energy_density_area_proxy_J_per_mm_um2",
    "dwell_time_ms",
)
AM_ENERGY_SOURCE_COLUMNS = ("laser_power_W", "scan_speed_mm_s", "spot_size_um")


def _torch() -> Any:
    import torch

    return torch


def sample_to_tensors(sample: Any, target: str, device: str = "cpu") -> tuple[Any, Any, Any]:
    torch = _torch()
    coords = torch.tensor(sample.coordinates, dtype=torch.float32, device=device)
    time = torch.tensor(sample.time, dtype=torch.float32, device=device).reshape(-1, 1)
    values = torch.tensor(sample.require_observation(target), dtype=torch.float32, device=device).reshape(-1, 1)
    return coords, time, values


def _input_feature_tensor(sample: Any, feature_columns: list[str], device: str) -> Any | None:
    if not feature_columns:
        return None
    torch = _torch()
    row_metadata = sample.metadata.get("row_metadata", {})
    rows: list[list[float]] = []
    for row_index in range(sample.n_points):
        row: list[float] = []
        for column in feature_columns:
            row.append(_row_metadata_float(row_metadata, column, row_index, "Input feature column"))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32, device=device)


def _row_metadata_float(
    row_metadata: dict[str, Any],
    column: str,
    row_index: int,
    label: str,
) -> float:
    if column not in row_metadata:
        raise ValueError(
            f"{label} {column!r} was not found in field-table row metadata. "
            "Use metadata/process columns such as laser_power_W, scan_speed_mm_s, or spot_size_um."
        )
    value = row_metadata[column][row_index]
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} {column!r} contains a non-numeric value at row {row_index}: {value!r}"
        ) from exc


def _group_balance_scalar_value(
    row_metadata: dict[str, Any],
    column: str,
    row_index: int,
    label: str,
) -> str:
    if column not in row_metadata:
        raise ValueError(
            f"{label} {column!r} was not found in field-table row metadata. "
            "Use metadata/process columns such as laser_power_W, scan_speed_mm_s, or spot_size_um."
        )
    value = row_metadata[column][row_index]
    if value in {None, ""}:
        raise ValueError(f"{label} {column!r} is empty at row {row_index}")
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def _group_balance_value(sample: Any, column: str, row_index: int) -> str:
    row_metadata = sample.metadata.get("row_metadata", {})
    if column == "process_condition":
        return "__".join(
            [
                f"{source_column}="
                f"{_group_balance_scalar_value(row_metadata, source_column, row_index, 'Process group balance source column')}"
                for source_column in AM_ENERGY_SOURCE_COLUMNS
            ]
        )
    return _group_balance_scalar_value(row_metadata, column, row_index, "Data-loss group balance column")


def _derived_process_feature_tensor(
    sample: Any,
    args: argparse.Namespace,
    device: str,
) -> tuple[Any | None, dict[str, Any]]:
    mode = args.input_derived_process_features
    if mode == "none":
        return None, {"enabled": False, "mode": "none"}
    if mode != "am_energy_v1":
        raise ValueError(f"Unsupported derived process feature mode: {mode}")
    torch = _torch()
    row_metadata = sample.metadata.get("row_metadata", {})
    rows: list[list[float]] = []
    for row_index in range(sample.n_points):
        laser_power = _row_metadata_float(row_metadata, "laser_power_W", row_index, "Derived process source column")
        scan_speed = _row_metadata_float(row_metadata, "scan_speed_mm_s", row_index, "Derived process source column")
        spot_size = _row_metadata_float(row_metadata, "spot_size_um", row_index, "Derived process source column")
        if scan_speed == 0.0:
            raise ValueError("Derived process feature scan_speed_mm_s must be non-zero")
        if spot_size == 0.0:
            raise ValueError("Derived process feature spot_size_um must be non-zero")
        rows.append(
            [
                laser_power / scan_speed,
                laser_power / (scan_speed * spot_size),
                laser_power / (scan_speed * spot_size * spot_size),
                spot_size / scan_speed,
            ]
        )
    payload = {
        "enabled": True,
        "mode": mode,
        "source_columns": list(AM_ENERGY_SOURCE_COLUMNS),
        "feature_names": list(AM_ENERGY_DERIVED_FEATURE_NAMES),
        "formulas": {
            "line_energy_J_per_mm": "laser_power_W / scan_speed_mm_s",
            "energy_density_proxy_J_per_mm_um": "laser_power_W / (scan_speed_mm_s * spot_size_um)",
            "energy_density_area_proxy_J_per_mm_um2": (
                "laser_power_W / (scan_speed_mm_s * spot_size_um * spot_size_um)"
            ),
            "dwell_time_ms": "spot_size_um / scan_speed_mm_s",
        },
    }
    return torch.tensor(rows, dtype=torch.float32, device=device), payload


def _process_graph_feature_tensor(
    sample: Any,
    args: argparse.Namespace,
    train_indices: list[int],
    device: str,
) -> tuple[Any | None, dict[str, Any]]:
    if args.process_graph_feature_mode == "none":
        return None, {"enabled": False, "mode": "none"}
    if args.process_graph_feature_mode != "rbf":
        raise ValueError(f"Unsupported process graph feature mode: {args.process_graph_feature_mode}")
    if args.process_graph_feature_count <= 0:
        raise ValueError("--process-graph-feature-count must be positive")
    if args.process_graph_length_scale <= 0.0:
        raise ValueError("--process-graph-length-scale must be positive")
    columns = list(args.process_graph_feature_columns or args.input_feature_columns)
    if not columns:
        return None, {
            "enabled": False,
            "mode": args.process_graph_feature_mode,
            "reason": "no process graph feature columns are active after profile resolution",
        }
    torch = _torch()
    source = _input_feature_tensor(sample, columns, device)
    if source is None:
        raise ValueError("--process-graph-feature-mode rbf could not build source process features")
    if args.process_graph_fit_scope == "train":
        if not train_indices:
            raise ValueError("--process-graph-feature-mode rbf requires non-empty train indices")
        fit_index = _index_tensor(train_indices, device)
    elif args.process_graph_fit_scope == "global":
        fit_index = torch.arange(source.shape[0], dtype=torch.long, device=device)
    else:
        raise ValueError(f"Unsupported process graph fit scope: {args.process_graph_fit_scope}")

    fit_values = source[fit_index]
    center = fit_values.mean(dim=0, keepdim=True)
    scale = fit_values.std(dim=0, unbiased=False, keepdim=True)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    normalized = (source - center) / scale
    fit_normalized = normalized[fit_index]

    unique_keys: set[tuple[float, ...]] = set()
    unique_rows: list[Any] = []
    for row in fit_normalized.detach().cpu().tolist():
        key = tuple(round(float(value), 10) for value in row)
        if key not in unique_keys:
            unique_keys.add(key)
            unique_rows.append(row)
    if not unique_rows:
        raise ValueError("--process-graph-feature-mode rbf found no unique process nodes")

    requested_count = int(args.process_graph_feature_count)
    if len(unique_rows) <= requested_count:
        anchor_rows = unique_rows
    elif requested_count == 1:
        anchor_rows = [unique_rows[0]]
    else:
        anchor_indices = [
            min(len(unique_rows) - 1, int(math.floor(index * len(unique_rows) / requested_count)))
            for index in range(requested_count)
        ]
        anchor_rows = [unique_rows[index] for index in anchor_indices]
    anchors = torch.tensor(anchor_rows, dtype=torch.float32, device=device)
    deltas = normalized[:, None, :] - anchors[None, :, :]
    squared_distances = torch.sum(deltas * deltas, dim=-1)
    features = torch.exp(-squared_distances / (2.0 * args.process_graph_length_scale**2))
    feature_names = [f"process_graph_rbf_{index}" for index in range(features.shape[-1])]
    payload = {
        "enabled": True,
        "mode": args.process_graph_feature_mode,
        "columns": columns,
        "fit_scope": args.process_graph_fit_scope,
        "requested_anchor_count": requested_count,
        "anchor_count": int(features.shape[-1]),
        "source_unique_nodes": len(unique_rows),
        "length_scale": args.process_graph_length_scale,
        "feature_names": feature_names,
        "center": center.detach().cpu().reshape(-1).tolist(),
        "scale": scale.detach().cpu().reshape(-1).tolist(),
        "anchors": anchors.detach().cpu().tolist(),
    }
    return features, payload


def _input_feature_payload(
    feature_columns: list[str],
    normalization: dict[str, Any] | None,
    conditioning_mode: str,
    film_strength: float,
    route_film_prior: float,
    route_trainable: bool,
    route_summary: dict[str, Any] | None = None,
    conditioning_profile: dict[str, Any] | None = None,
    derived_process_features: dict[str, Any] | None = None,
    process_graph_features: dict[str, Any] | None = None,
    effective_feature_count: int | None = None,
) -> dict[str, Any]:
    derived_payload = derived_process_features or {"enabled": False, "mode": "none"}
    derived_names = (
        list(derived_payload.get("feature_names") or []) if derived_payload.get("enabled") else []
    )
    process_graph_payload = process_graph_features or {"enabled": False, "mode": "none"}
    process_graph_names = (
        list(process_graph_payload.get("feature_names") or []) if process_graph_payload.get("enabled") else []
    )
    return {
        "enabled": (
            bool(feature_columns)
            or bool(derived_payload.get("enabled"))
            or bool(process_graph_payload.get("enabled"))
        ),
        "columns": feature_columns,
        "effective_columns": [*feature_columns, *derived_names, *process_graph_names],
        "count": effective_feature_count if effective_feature_count is not None else len(feature_columns),
        "normalization": normalization,
        "conditioning_mode": conditioning_mode,
        "film_strength": film_strength,
        "derived_process_features": derived_payload,
        "process_graph_features": process_graph_payload,
        "route": {
            "enabled": route_summary is not None,
            "film_prior": route_film_prior,
            "trainable": route_trainable,
            "summary": route_summary,
        },
        "conditioning_profile": conditioning_profile or {"enabled": False, "profile": "none"},
    }


def _spacetime_encoding_payload(args: argparse.Namespace, model: Any) -> dict[str, Any]:
    return {
        "encoding": args.spacetime_encoding,
        "fourier_bands": args.spacetime_fourier_bands,
        "input_dim": int(model.spacetime_dim),
    }


def _backbone_payload(args: argparse.Namespace, model: Any) -> dict[str, Any]:
    return {
        "mode": args.backbone_mode,
        "residual_scale": args.backbone_residual_scale if args.backbone_mode == "residual" else None,
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "implementation": type(model).__name__,
    }


def _process_encoder_payload(args: argparse.Namespace, model: Any, input_dim: int) -> dict[str, Any]:
    enabled = args.input_process_encoder_mode != "none"
    encoder = getattr(model, "process_encoder", None)
    parameter_count = int(sum(parameter.numel() for parameter in encoder.parameters())) if encoder is not None else 0
    return {
        "enabled": enabled,
        "mode": args.input_process_encoder_mode,
        "input_dim": input_dim if enabled else None,
        "output_dim": int(getattr(model, "process_encoder_dim", 0)) if enabled else None,
        "identity_initialized": bool(getattr(model, "process_encoder_identity_initialized", False)) if enabled else False,
        "parameter_count": parameter_count,
    }


def _residual_correction_input(coords: Any, time: Any, params: Any | None = None) -> Any:
    torch = _torch()
    if time.ndim == 1:
        time = time[:, None]
    tensors = [coords, time]
    if params is not None:
        tensors.append(params)
    return torch.cat(tensors, dim=-1)


def _zero_initialize_last_linear(module: Any) -> None:
    torch = _torch()
    for layer in reversed(list(module.modules())):
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.zeros_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            return
    raise ValueError("Residual correction module has no Linear layer to initialize")


def _build_residual_correction(args: argparse.Namespace, input_dim: int) -> Any | None:
    if args.residual_correction_mode == "none":
        return None
    if args.residual_correction_mode != "mlp":
        raise ValueError(f"Unsupported residual correction mode: {args.residual_correction_mode}")
    if args.residual_correction_hidden_dim <= 0:
        raise ValueError("residual_correction_hidden_dim must be positive")
    if args.residual_correction_layers < 1:
        raise ValueError("residual_correction_layers must be >= 1")
    if args.residual_correction_scale < 0.0:
        raise ValueError("residual_correction_scale must be non-negative")
    if args.residual_correction_start_step < 0:
        raise ValueError("residual_correction_start_step must be non-negative")
    module = MLP(
        MLPConfig(
            input_dim=input_dim,
            output_dim=1,
            hidden_dim=args.residual_correction_hidden_dim,
            num_hidden_layers=args.residual_correction_layers,
            activation=args.activation,
        )
    )
    _zero_initialize_last_linear(module)
    return module.to(args.device)


def _apply_residual_correction(
    base_prediction: Any,
    residual_correction: Any | None,
    args: argparse.Namespace,
    coords: Any,
    time: Any,
    params: Any | None = None,
    *,
    active: bool,
) -> Any:
    if residual_correction is None or not active:
        return base_prediction
    residual_input = _residual_correction_input(coords, time, params)
    return base_prediction + args.residual_correction_scale * residual_correction(residual_input)


def _residual_correction_payload(
    args: argparse.Namespace,
    residual_correction: Any | None,
    input_dim: int,
) -> dict[str, Any]:
    enabled = residual_correction is not None
    lr = args.residual_correction_lr if args.residual_correction_lr is not None else args.lr
    parameter_count = (
        int(sum(parameter.numel() for parameter in residual_correction.parameters()))
        if residual_correction is not None
        else 0
    )
    return {
        "enabled": enabled,
        "mode": args.residual_correction_mode,
        "input_dim": input_dim if enabled else None,
        "hidden_dim": args.residual_correction_hidden_dim if enabled else None,
        "layers": args.residual_correction_layers if enabled else None,
        "scale": args.residual_correction_scale if enabled else None,
        "start_step": args.residual_correction_start_step if enabled else None,
        "lr": lr if enabled else None,
        "parameter_count": parameter_count,
        "last_layer_zero_initialized": enabled,
    }


def _build_output_affine(args: argparse.Namespace, input_dim: int) -> Any | None:
    if args.output_affine_mode == "none":
        return None
    if args.output_affine_mode != "linear":
        raise ValueError(f"Unsupported output affine mode: {args.output_affine_mode}")
    if input_dim <= 0:
        raise ValueError("--output-affine-mode linear requires active input/process features")
    if args.output_affine_scale < 0.0:
        raise ValueError("--output-affine-scale must be non-negative")
    torch = _torch()
    module = torch.nn.Linear(input_dim, 2)
    torch.nn.init.zeros_(module.weight)
    torch.nn.init.zeros_(module.bias)
    return module.to(args.device)


def _apply_output_affine(
    prediction: Any,
    output_affine: Any | None,
    args: argparse.Namespace,
    params: Any | None,
) -> Any:
    if output_affine is None:
        return prediction
    if params is None:
        raise ValueError("Output affine calibration requires process/input features")
    gamma, beta = output_affine(params).chunk(2, dim=-1)
    return prediction * (1.0 + args.output_affine_scale * gamma) + args.output_affine_scale * beta


def _output_affine_payload(
    args: argparse.Namespace,
    output_affine: Any | None,
    input_dim: int,
) -> dict[str, Any]:
    enabled = output_affine is not None
    lr = args.output_affine_lr if args.output_affine_lr is not None else args.lr
    parameter_count = (
        int(sum(parameter.numel() for parameter in output_affine.parameters()))
        if output_affine is not None
        else 0
    )
    return {
        "enabled": enabled,
        "mode": args.output_affine_mode,
        "input_dim": input_dim if enabled else None,
        "scale": args.output_affine_scale if enabled else None,
        "lr": lr if enabled else None,
        "parameter_count": parameter_count,
        "identity_initialized": enabled,
    }


def _prediction_anchor_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "enabled": args.prediction_anchor_weight > 0.0,
        "weight": args.prediction_anchor_weight,
        "target_space": "normalized_training_target" if args.normalize_target else "training_target",
        "loss": "mean(prediction ** 2)",
    }


def _resolve_input_conditioning_profile(
    args: argparse.Namespace,
    split_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    profile = args.input_conditioning_profile
    if profile == "none":
        return {"enabled": False, "profile": profile}
    if split_manifest is None:
        raise ValueError("--input-conditioning-profile requires --split-manifest")
    group_key = split_manifest.get("group_key")
    if not group_key:
        raise ValueError("--input-conditioning-profile requires a grouped split manifest with group_key")

    profile_v1 = {
        "line_id": {
            "conditioning_mode": "concat",
            "feature_normalization": "same",
            "reason": "line holdout keeps the strongest neural route from Phase 24: concat with train-fitted minmax.",
        },
        "scan_speed_mm_s": {
            "conditioning_mode": "concat",
            "feature_normalization": "global_standard",
            "reason": "scan-speed holdout was strongest with concat plus global process-feature standardization.",
        },
        "spot_size_um": {
            "conditioning_mode": "film",
            "feature_normalization": "global_standard",
            "reason": "spot-size holdout was repaired by FiLM plus global process-feature standardization.",
        },
        "laser_power_W": {
            "conditioning_mode": "concat",
            "feature_normalization": "global_standard",
            "reason": "laser-power holdout uses the conservative global-standard concat route from process-axis diagnostics.",
        },
        "process_condition": {
            "conditioning_mode": "concat",
            "feature_normalization": "same",
            "reason": "full process holdout follows the line-like train-minmax concat route after Phase 28 validation.",
        },
    }
    broad_process_v1 = {
        "line_id": {
            "conditioning_mode": "none",
            "feature_normalization": "none",
            "feature_columns": [],
            "reason": "broad12 line holdout favored no-process Macro PINN over forced process conditioning.",
        },
        "scan_speed_mm_s": {
            "conditioning_mode": "none",
            "feature_normalization": "none",
            "feature_columns": [],
            "reason": "broad12 scan-speed holdout degraded under the old concat/global-standard profile.",
        },
        "spot_size_um": {
            "conditioning_mode": "film",
            "feature_normalization": "global_standard",
            "reason": "broad12 spot-size holdout retained the strong FiLM/global-standard gain.",
        },
        "laser_power_W": {
            "conditioning_mode": "concat",
            "feature_normalization": "global_standard",
            "reason": "broad12 laser-power holdout still improved no-process Macro PINN with concat/global-standard.",
        },
        "process_condition": {
            "conditioning_mode": "none",
            "feature_normalization": "none",
            "feature_columns": [],
            "reason": "broad12 full-process holdout degraded under the line-like concat/same profile.",
        },
    }
    broad_process_v2 = {
        **broad_process_v1,
        "line_id": {
            "conditioning_mode": "concat",
            "feature_normalization": "same",
            "reason": "broad21 line holdout slightly favored the old concat/same route over conservative no-process.",
        },
    }
    profiles = {
        "process_axis_v1": profile_v1,
        "broad_process_v1": broad_process_v1,
        "broad_process_v2": broad_process_v2,
    }
    if profile not in profiles:
        raise ValueError(f"Unsupported input conditioning profile: {profile}")
    profile_routes = profiles[profile]
    if group_key not in profile_routes:
        raise ValueError(f"Unsupported group_key {group_key!r} for input conditioning profile {profile!r}")

    selected = profile_routes[group_key]
    requested = {
        "conditioning_mode": args.input_conditioning_mode,
        "feature_normalization": args.input_feature_normalization,
        "feature_columns": list(args.input_feature_columns),
        "derived_process_features": args.input_derived_process_features,
    }
    if "feature_columns" in selected:
        args.input_feature_columns = list(selected["feature_columns"])
        if not args.input_feature_columns:
            args.input_derived_process_features = "none"
    if selected["conditioning_mode"] == "none":
        args.input_conditioning_mode = "concat"
        args.input_feature_normalization = "none"
        args.input_derived_process_features = "none"
    else:
        args.input_conditioning_mode = selected["conditioning_mode"]
        args.input_feature_normalization = selected["feature_normalization"]
    return {
        "enabled": True,
        "profile": profile,
        "group_key": group_key,
        "requested": requested,
        "selected": selected,
        "effective": {
            "conditioning_mode": args.input_conditioning_mode,
            "feature_normalization": args.input_feature_normalization,
            "feature_columns": list(args.input_feature_columns),
            "derived_process_features": args.input_derived_process_features,
        },
    }


def _model_forward(model: Any, coords: Any, time: Any, params: Any | None = None) -> Any:
    if params is None:
        return model(coords, time)
    return model(coords, time, params)


def _index_tensor(indices: list[int], device: str) -> Any:
    torch = _torch()
    return torch.tensor(indices, dtype=torch.long, device=device)


def _normalize_feature_tensor(tensor: Any, train_index: Any, mode: str) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    fit_index = train_index
    fit_scope = "train"
    stat_mode = mode
    if mode == "global_standard":
        fit_index = torch.arange(tensor.shape[0], dtype=torch.long, device=tensor.device)
        fit_scope = "global"
        stat_mode = "standard"
    elif mode == "global_minmax":
        fit_index = torch.arange(tensor.shape[0], dtype=torch.long, device=tensor.device)
        fit_scope = "global"
        stat_mode = "minmax"
    fit_values = tensor[fit_index]
    stats: dict[str, Any] = {"mode": mode, "fit_scope": fit_scope}
    if mode == "none":
        stats["applied"] = False
        return tensor, stats
    if stat_mode == "standard":
        center = fit_values.mean(dim=0, keepdim=True)
        scale = fit_values.std(dim=0, unbiased=False, keepdim=True)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        normalized = (tensor - center) / scale
        stats.update(
            {
                "applied": True,
                "center": center.detach().cpu().reshape(-1).tolist(),
                "scale": scale.detach().cpu().reshape(-1).tolist(),
            }
        )
        return normalized, stats
    if stat_mode == "minmax":
        minimum = fit_values.min(dim=0, keepdim=True).values
        maximum = fit_values.max(dim=0, keepdim=True).values
        scale = maximum - minimum
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        normalized = (tensor - minimum) / scale
        stats.update(
            {
                "applied": True,
                "minimum": minimum.detach().cpu().reshape(-1).tolist(),
                "maximum": maximum.detach().cpu().reshape(-1).tolist(),
                "scale": scale.detach().cpu().reshape(-1).tolist(),
            }
        )
        return normalized, stats
    raise ValueError(f"Unsupported input normalization mode: {mode}")


def _baseline_feature_matrix(sample: Any, feature_columns: list[str] | None) -> tuple[list[list[float]], list[str]]:
    default_columns = list(sample.metadata.get("coordinate_columns") or [])
    time_column = sample.metadata.get("time_column")
    if time_column:
        default_columns.append(time_column)
    columns = list(feature_columns or default_columns)
    if not columns:
        raise ValueError("Target residual baselines require at least one feature column")

    coord_columns = list(sample.metadata.get("coordinate_columns") or [])
    row_metadata = sample.metadata.get("row_metadata", {})
    rows: list[list[float]] = []
    for row_index in range(sample.n_points):
        row: list[float] = []
        for column in columns:
            if column in coord_columns:
                row.append(float(sample.coordinates[row_index][coord_columns.index(column)]))
            elif time_column and column == time_column:
                row.append(float(sample.time[row_index]))
            elif column in sample.observations:
                row.append(float(sample.observations[column][row_index]))
            elif column in row_metadata:
                value = row_metadata[column][row_index]
                try:
                    row.append(float(value))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Target residual baseline feature column {column!r} contains a non-numeric value "
                        f"at row {row_index}: {value!r}"
                    ) from exc
            else:
                raise ValueError(f"Target residual baseline feature column not found: {column}")
        rows.append(row)
    return rows, columns


def _target_residual_baseline_predictions(
    sample: Any,
    target: str,
    train_indices: list[int],
    args: argparse.Namespace,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    if args.target_residual_baseline == "none":
        predictions = torch.zeros((sample.n_points, 1), dtype=torch.float32, device=device)
        return predictions, {"enabled": False, "strategy": "none"}

    if args.pde_weight > 0:
        raise ValueError("--target-residual-baseline is currently supported only for data-only training")
    if not train_indices:
        raise ValueError("--target-residual-baseline requires a non-empty train split")

    y_true = [float(value) for value in sample.require_observation(target)]
    if args.target_residual_baseline == "mean":
        mean_value = sum(y_true[index] for index in train_indices) / len(train_indices)
        predictions_list = [mean_value for _ in y_true]
        feature_columns: list[str] = []
    else:
        feature_matrix, feature_columns = _baseline_feature_matrix(
            sample,
            args.target_residual_baseline_feature_columns,
        )
        x_fit = [feature_matrix[index] for index in train_indices]
        y_fit = [y_true[index] for index in train_indices]
        if args.target_residual_baseline == "knn":
            from sklearn.neighbors import KNeighborsRegressor
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            model = make_pipeline(
                StandardScaler(),
                KNeighborsRegressor(
                    n_neighbors=max(1, min(args.target_residual_baseline_n_neighbors, len(train_indices)))
                ),
            )
        elif args.target_residual_baseline == "extra_trees":
            from sklearn.ensemble import ExtraTreesRegressor

            model = ExtraTreesRegressor(
                n_estimators=args.target_residual_baseline_n_estimators,
                random_state=args.target_residual_baseline_random_state,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unsupported target residual baseline: {args.target_residual_baseline}")
        model.fit(x_fit, y_fit)
        predictions_list = [float(value) for value in model.predict(feature_matrix)]

    predictions = torch.tensor(predictions_list, dtype=torch.float32, device=device).reshape(-1, 1)
    residuals = [truth - pred for truth, pred in zip(y_true, predictions_list)]
    train_residuals = [residuals[index] for index in train_indices]
    payload = {
        "enabled": True,
        "strategy": args.target_residual_baseline,
        "feature_columns": feature_columns,
        "fit_split": args.train_split,
        "fit_points": len(train_indices),
        "n_neighbors": (
            args.target_residual_baseline_n_neighbors if args.target_residual_baseline == "knn" else None
        ),
        "n_estimators": (
            args.target_residual_baseline_n_estimators if args.target_residual_baseline == "extra_trees" else None
        ),
        "random_state": (
            args.target_residual_baseline_random_state
            if args.target_residual_baseline == "extra_trees"
            else None
        ),
        "train_residual_mean": sum(train_residuals) / len(train_residuals),
        "train_residual_rmse": math.sqrt(sum(value * value for value in train_residuals) / len(train_residuals)),
    }
    return predictions, payload


def _metric_payload(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    metrics: dict[str, float] = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
    }
    try:
        metrics["relative_l2"] = relative_l2(y_true, y_pred)
    except ValueError:
        pass
    return metrics


def _jsonable_config(args: argparse.Namespace) -> dict[str, Any]:
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
        elif isinstance(value, list):
            config[key] = [str(item) if isinstance(item, Path) else item for item in value]
    return config


def _optimizer_payload(
    args: argparse.Namespace,
    closure_coefficients: Any | None,
    residual_correction: Any | None = None,
    output_affine: Any | None = None,
) -> dict[str, Any]:
    closure_lr = args.closure_lr if args.closure_lr is not None else args.lr
    residual_lr = args.residual_correction_lr if args.residual_correction_lr is not None else args.lr
    output_affine_lr = args.output_affine_lr if args.output_affine_lr is not None else args.lr
    return {
        "backbone_lr": args.lr,
        "closure_lr": closure_lr if closure_coefficients is not None else None,
        "closure_lr_overridden": args.closure_lr is not None,
        "residual_correction_lr": residual_lr if residual_correction is not None else None,
        "residual_correction_lr_overridden": args.residual_correction_lr is not None,
        "output_affine_lr": output_affine_lr if output_affine is not None else None,
        "output_affine_lr_overridden": args.output_affine_lr is not None,
        "freeze_backbone_after_closure_start": args.freeze_backbone_after_closure_start,
        "closure_graph_lr": args.closure_graph_lr,
    }


def _closure_feature_tensor(
    feature_names: list[str],
    pred_field: Any,
    coords: Any,
    time: Any,
    graph_embedding: Any | None = None,
    graph_features: Any | None = None,
) -> Any:
    torch = _torch()
    columns: list[Any] = []
    for name in feature_names:
        normalized_name = name.lower()
        if normalized_name in {"t", "time"}:
            columns.append(time[:, 0] if time.ndim > 1 else time)
        elif normalized_name == "x":
            columns.append(coords[:, 0])
        elif normalized_name == "y":
            if coords.shape[-1] < 2:
                raise ValueError("Closure feature 'y' requires at least two coordinate dimensions")
            columns.append(coords[:, 1])
        elif normalized_name == "z":
            if coords.shape[-1] < 3:
                raise ValueError("Closure feature 'z' requires at least three coordinate dimensions")
            columns.append(coords[:, 2])
        elif normalized_name in {"t_field", "temperature", "temperature_c"}:
            columns.append(pred_field)
        elif name == "T":
            columns.append(pred_field)
        elif normalized_name.startswith("g") and normalized_name[1:].isdigit():
            if graph_embedding is None and graph_features is None:
                raise ValueError(f"Closure feature '{name}' requires graph conditioning")
            graph_index = int(normalized_name[1:])
            if graph_features is not None:
                if graph_index >= graph_features.shape[-1]:
                    raise ValueError(
                        f"Closure feature '{name}' requires graph feature dimension > {graph_index}"
                    )
                columns.append(graph_features[:, graph_index])
                continue
            flattened_graph = graph_embedding.reshape(-1)
            if graph_index >= flattened_graph.numel():
                raise ValueError(
                    f"Closure feature '{name}' requires graph embedding dimension > {graph_index}"
                )
            columns.append(torch.ones_like(pred_field) * flattened_graph[graph_index])
        else:
            raise ValueError(f"Unsupported closure feature: {name}")
    if not columns:
        raise ValueError("At least one closure feature is required when closure is enabled")
    return torch.stack(columns, dim=-1)


def _closure_expression(term_names: list[str], coefficients: list[float], threshold: float) -> str:
    expression = export_linear_library_expression(
        term_names=term_names,
        coefficients=coefficients,
        threshold=threshold,
    )
    return expression_to_string(expression)


def _closure_payload(
    closure_mode: str,
    closure_library: Any | None,
    closure_coefficients: Any | None,
    source_values: Any | None,
    threshold: float,
    graph_conditioning: dict[str, Any] | None = None,
    graph_gate: float | None = None,
    graph_l1_weight: float = 0.0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"mode": closure_mode}
    if graph_conditioning is not None:
        payload["graph_conditioning"] = graph_conditioning
        payload["graph_gate"] = graph_gate
        payload["graph_l1_weight"] = graph_l1_weight
    if closure_library is None or closure_coefficients is None:
        payload["enabled"] = False
        return payload

    coefficients = closure_coefficients.detach().cpu().reshape(-1).tolist()
    payload.update(
        {
            "enabled": True,
            "term_names": closure_library.term_names,
            "coefficients": coefficients,
            "threshold": threshold,
            "expression": _closure_expression(
                term_names=closure_library.term_names,
                coefficients=coefficients,
                threshold=threshold,
            ),
        }
    )
    if source_values is not None:
        values = source_values.detach().cpu()
        payload["source_summary"] = {
            "mean": float(values.mean()),
            "std": float(values.std(unbiased=False)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return payload


def _split_closure_source(
    closure_matrix: Any,
    coefficients: Any,
    term_names: list[str],
) -> tuple[Any, Any, Any, Any]:
    torch = _torch()
    graph_mask_values = [name.lower().startswith("g") for name in term_names]
    graph_mask = torch.tensor(graph_mask_values, dtype=torch.bool, device=closure_matrix.device)
    base_mask = ~graph_mask
    base_source = closure_matrix[:, base_mask] @ coefficients[base_mask] if bool(base_mask.any()) else torch.zeros(
        closure_matrix.shape[0],
        dtype=closure_matrix.dtype,
        device=closure_matrix.device,
    )
    graph_source = closure_matrix[:, graph_mask] @ coefficients[graph_mask] if bool(graph_mask.any()) else torch.zeros(
        closure_matrix.shape[0],
        dtype=closure_matrix.dtype,
        device=closure_matrix.device,
    )
    return base_source, graph_source, base_mask, graph_mask


def _residual_sample_indices(
    train_indices: list[int],
    candidate_indices: list[int],
    step: int,
    sample_size: int | None,
    seed: int,
) -> list[int]:
    population = candidate_indices or train_indices
    if sample_size is None:
        return population
    if sample_size >= len(population):
        return population
    if sample_size <= 0:
        raise ValueError("residual_sample_size must be positive when provided")
    rng = __import__("random").Random(seed + step)
    return sorted(rng.sample(population, sample_size))


def _quantile_threshold(values: list[float], quantile: float) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")
    if not values:
        raise ValueError("cannot compute quantile of empty values")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _residual_candidate_indices(
    sample: Any,
    target: str,
    train_indices: list[int],
    mode: str,
    hot_quantile: float,
    gradient_quantile: float,
) -> list[int]:
    if mode == "random":
        return train_indices
    if mode not in {"hot", "gradient", "hot_gradient"}:
        raise ValueError(f"Unsupported residual sampling mode: {mode}")
    selected, _ = _region_candidate_indices(
        sample=sample,
        target=target,
        train_indices=train_indices,
        mode=mode,
        hot_quantile=hot_quantile,
        gradient_quantile=gradient_quantile,
    )
    return selected


def _region_candidate_indices(
    sample: Any,
    target: str,
    train_indices: list[int],
    mode: str,
    hot_quantile: float,
    gradient_quantile: float,
    gradient_reference_indices: list[int] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    if mode not in {"hot", "gradient", "hot_gradient"}:
        raise ValueError(f"Unsupported region candidate mode: {mode}")
    target_values = sample.require_observation(target)
    selected: set[int] = set()
    selectors: dict[str, Any] = {}
    if mode in {"hot", "hot_gradient"}:
        threshold = _quantile_threshold([target_values[index] for index in train_indices], hot_quantile)
        hot_indices = [index for index in train_indices if float(target_values[index]) >= threshold]
        selected.update(hot_indices)
        selectors["hot"] = {
            "target": target,
            "quantile": hot_quantile,
            "threshold": threshold,
            "n_points": len(hot_indices),
        }
    if mode in {"gradient", "hot_gradient"}:
        gradient_scores = (
            _spatial_gradient_scores_for_indices(sample, target_values, gradient_reference_indices)
            if gradient_reference_indices is not None
            else _spatial_gradient_scores(sample, target_values)
        )
        threshold = _quantile_threshold([gradient_scores[index] for index in train_indices], gradient_quantile)
        gradient_indices = [index for index in train_indices if float(gradient_scores[index]) >= threshold]
        selected.update(gradient_indices)
        selectors["gradient"] = {
            "target": target,
            "quantile": gradient_quantile,
            "threshold": threshold,
            "n_points": len(gradient_indices),
            "score_scope": "provided_indices" if gradient_reference_indices is not None else "full_sample",
        }
    selected_indices = sorted(selected)
    selectors["fallback_to_all_train"] = not selected_indices
    return selected_indices or train_indices, selectors


def _spatial_gradient_scores_for_indices(sample: Any, values: list[float], active_indices: list[int]) -> list[float]:
    row_values = sample.observations.get("row_index")
    col_values = sample.observations.get("col_index")
    if row_values is None or col_values is None:
        coord_columns = list(sample.metadata.get("coordinate_columns") or [])
        if "y" in coord_columns and "x" in coord_columns:
            row_pos = coord_columns.index("y")
            col_pos = coord_columns.index("x")
            row_values = [row[row_pos] for row in sample.coordinates]
            col_values = [row[col_pos] for row in sample.coordinates]
        else:
            return [0.0 for _ in range(sample.n_points)]

    frame_values = sample.observations.get("frame_index")
    frames = frame_values if frame_values is not None else [0.0 for _ in range(sample.n_points)]
    groups: dict[float, list[int]] = {}
    for index in active_indices:
        groups.setdefault(float(frames[index]), []).append(index)

    scores = [0.0 for _ in range(sample.n_points)]
    for group_indices in groups.values():
        rows = sorted({float(row_values[index]) for index in group_indices})
        cols = sorted({float(col_values[index]) for index in group_indices})
        row_neighbors = _axis_neighbors(rows)
        col_neighbors = _axis_neighbors(cols)
        by_position = {
            (float(row_values[index]), float(col_values[index])): index
            for index in group_indices
        }
        for index in group_indices:
            row = float(row_values[index])
            col = float(col_values[index])
            local_scores: list[float] = []
            for neighbor_row in row_neighbors.get(row, []):
                neighbor = by_position.get((neighbor_row, col))
                if neighbor is not None:
                    distance = abs(neighbor_row - row) or 1.0
                    local_scores.append(abs(float(values[index]) - float(values[neighbor])) / distance)
            for neighbor_col in col_neighbors.get(col, []):
                neighbor = by_position.get((row, neighbor_col))
                if neighbor is not None:
                    distance = abs(neighbor_col - col) or 1.0
                    local_scores.append(abs(float(values[index]) - float(values[neighbor])) / distance)
            if local_scores:
                scores[index] = max(local_scores)
    return scores


def _axis_neighbors(values: list[float]) -> dict[float, list[float]]:
    neighbors: dict[float, list[float]] = {}
    for position, value in enumerate(values):
        current: list[float] = []
        if position > 0:
            current.append(values[position - 1])
        if position + 1 < len(values):
            current.append(values[position + 1])
        neighbors[value] = current
    return neighbors


def _data_loss_weight_tensor(
    sample: Any,
    target: str,
    train_indices: list[int],
    args: argparse.Namespace,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    if args.data_loss_weighting == "none":
        weights = torch.ones((len(train_indices), 1), dtype=torch.float32, device=device)
        return weights, {
            "enabled": False,
            "mode": "none",
            "fit_scope": "train",
            "normalization": "sum_weights",
            "train_points": len(train_indices),
            "selected_points": 0,
            "selected_fraction": 0.0,
            "region_weight": args.data_loss_region_weight,
            "hot_quantile": args.data_loss_hot_quantile,
            "gradient_quantile": args.data_loss_gradient_quantile,
            "weight_sum": float(weights.sum().detach().cpu()),
            "mean_weight": float(weights.mean().detach().cpu()) if len(train_indices) else 0.0,
            "selectors": {},
        }
    if args.data_loss_region_weight <= 0.0:
        raise ValueError("data_loss_region_weight must be positive")
    selected_indices, selectors = _region_candidate_indices(
        sample=sample,
        target=target,
        train_indices=train_indices,
        mode=args.data_loss_weighting,
        hot_quantile=args.data_loss_hot_quantile,
        gradient_quantile=args.data_loss_gradient_quantile,
        gradient_reference_indices=train_indices,
    )
    selected = set(selected_indices)
    weights_list = [
        args.data_loss_region_weight if index in selected else 1.0
        for index in train_indices
    ]
    weights = torch.tensor(weights_list, dtype=torch.float32, device=device).reshape(-1, 1)
    selected_points = sum(1 for index in train_indices if index in selected)
    return weights, {
        "enabled": True,
        "mode": args.data_loss_weighting,
        "fit_scope": "train",
        "normalization": "sum_weights",
        "train_points": len(train_indices),
        "selected_points": selected_points,
        "selected_fraction": selected_points / len(train_indices) if train_indices else 0.0,
        "region_weight": args.data_loss_region_weight,
        "hot_quantile": args.data_loss_hot_quantile,
        "gradient_quantile": args.data_loss_gradient_quantile,
        "weight_sum": float(weights.sum().detach().cpu()),
        "mean_weight": float(weights.mean().detach().cpu()) if len(train_indices) else 0.0,
        "selectors": selectors,
    }


def _group_balance_weight_tensor(
    sample: Any,
    train_indices: list[int],
    args: argparse.Namespace,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    column = args.data_loss_group_balance_column.strip()
    strength = args.data_loss_group_balance_strength
    if strength < 0.0 or strength > 1.0:
        raise ValueError("data_loss_group_balance_strength must be between 0 and 1")
    if not column or strength == 0.0 or not train_indices:
        weights = torch.ones((len(train_indices), 1), dtype=torch.float32, device=device)
        return weights, {
            "enabled": False,
            "mode": "none",
            "fit_scope": "train",
            "normalization": "blend_with_uniform",
            "column": column or None,
            "strength": strength,
            "train_points": len(train_indices),
            "group_count": 0,
            "group_sizes": {},
            "group_weights": {},
            "weight_sum": float(weights.sum().detach().cpu()),
            "mean_weight": float(weights.mean().detach().cpu()) if len(train_indices) else 0.0,
        }
    group_values = [_group_balance_value(sample, column, index) for index in train_indices]
    group_counts = Counter(group_values)
    group_count = len(group_counts)
    balanced_group_weights = {
        group: len(train_indices) / (group_count * count)
        for group, count in group_counts.items()
    }
    weights_list = [
        (1.0 - strength) + strength * balanced_group_weights[group]
        for group in group_values
    ]
    weights = torch.tensor(weights_list, dtype=torch.float32, device=device).reshape(-1, 1)
    return weights, {
        "enabled": True,
        "mode": "inverse_frequency",
        "fit_scope": "train",
        "normalization": "blend_with_uniform",
        "column": column,
        "strength": strength,
        "train_points": len(train_indices),
        "group_count": group_count,
        "group_sizes": dict(sorted((group, int(count)) for group, count in group_counts.items())),
        "group_weights": dict(sorted((group, float(weight)) for group, weight in balanced_group_weights.items())),
        "weight_sum": float(weights.sum().detach().cpu()),
        "mean_weight": float(weights.mean().detach().cpu()) if len(train_indices) else 0.0,
    }


def _build_graph_conditioning(
    args: argparse.Namespace,
    device: str,
    state_dim: int | None = None,
) -> tuple[Any | None, dict[str, Any] | None]:
    if args.closure_graph_mode == "none":
        return None, None
    if args.closure_graph_mode == "coordinate_rbf":
        graph_state_dim = args.closure_graph_state_dim or state_dim
        if graph_state_dim is None:
            raise ValueError("closure_graph_state_dim could not be inferred")
        provider = CoordinateRBFGraphFeatureProvider(
            CoordinateRBFGraphConfig(
                state_dim=graph_state_dim,
                embedding_dim=args.closure_graph_embedding_dim,
                length_scale=args.closure_graph_length_scale,
                normalize=not args.no_closure_graph_normalize,
            )
        ).to(device)
        payload = {
            "enabled": True,
            "mode": args.closure_graph_mode,
            "trainable": False,
            "metadata": provider.metadata(),
        }
        return provider, payload
    if args.closure_graph_mode == "real_micro":
        if args.closure_graph_features is None:
            raise ValueError("--closure-graph-features is required for real_micro graph conditioning")
        provider = RealMicroGraphFeatureProvider(
            RealMicroGraphFeatureConfig(
                graph_features=str(args.closure_graph_features),
                sample_id=args.closure_graph_sample_id,
                embedding_dim=args.closure_graph_embedding_dim,
                normalize=not args.no_closure_graph_normalize,
            )
        ).to(device)
        payload = {
            "enabled": True,
            "mode": args.closure_graph_mode,
            "trainable": False,
            "selection": {
                "sample_id": args.closure_graph_sample_id,
                "sample_id_column": args.closure_graph_sample_id_column,
            },
            "metadata": provider.metadata(),
        }
        return provider, payload
    if args.closure_graph_mode == "real_micro_region":
        if args.closure_graph_features is None:
            raise ValueError("--closure-graph-features is required for real_micro_region graph conditioning")
        provider = RealMicroRegionFeatureProvider(
            RealMicroRegionFeatureConfig(
                graph_features=str(args.closure_graph_features),
                sample_id=args.closure_graph_sample_id,
                embedding_dim=args.closure_graph_embedding_dim,
                normalize=not args.no_closure_graph_normalize,
                row_source=args.closure_graph_region_row_source,
                col_source=args.closure_graph_region_col_source,
                flip_row=args.closure_graph_region_flip_row,
                flip_col=args.closure_graph_region_flip_col,
                selection=args.closure_graph_region_selection,
                inverse_distance_epsilon=args.closure_graph_region_inverse_distance_epsilon,
            )
        ).to(device)
        payload = {
            "enabled": True,
            "mode": args.closure_graph_mode,
            "trainable": False,
            "selection": {
                "sample_id": args.closure_graph_sample_id,
                "sample_id_column": args.closure_graph_sample_id_column,
            },
            "metadata": provider.metadata(),
        }
        return provider, payload
    if args.closure_graph_mode == "real_micro_region_embedding":
        if args.closure_graph_features is None:
            raise ValueError("--closure-graph-features is required for real_micro_region_embedding graph conditioning")
        provider = RealMicroRegionEmbeddingFeatureProvider(
            RealMicroRegionEmbeddingFeatureConfig(
                graph_features=str(args.closure_graph_features),
                sample_id=args.closure_graph_sample_id,
                embedding_dim=args.closure_graph_embedding_dim,
                normalize=not args.no_closure_graph_normalize,
                row_source=args.closure_graph_region_row_source,
                col_source=args.closure_graph_region_col_source,
                flip_row=args.closure_graph_region_flip_row,
                flip_col=args.closure_graph_region_flip_col,
                selection=args.closure_graph_region_selection,
                inverse_distance_epsilon=args.closure_graph_region_inverse_distance_epsilon,
            )
        ).to(device)
        payload = {
            "enabled": True,
            "mode": args.closure_graph_mode,
            "trainable": False,
            "selection": {
                "sample_id": args.closure_graph_sample_id,
                "sample_id_column": args.closure_graph_sample_id_column,
            },
            "metadata": provider.metadata(),
        }
        return provider, payload
    if args.closure_graph_mode != "toy_static":
        raise ValueError(f"Unsupported closure graph mode: {args.closure_graph_mode}")
    provider = ToyStaticGraphEmbeddingProvider(
        ToyStaticGraphConfig(
            node_feature_dim=args.closure_graph_node_dim,
            hidden_dim=args.closure_graph_hidden_dim,
            embedding_dim=args.closure_graph_embedding_dim,
            steps=args.closure_graph_steps,
            seed=args.closure_graph_seed,
        )
    ).to(device)
    if not args.closure_graph_trainable:
        for parameter in provider.parameters():
            parameter.requires_grad_(False)
    payload = {
        "enabled": True,
        "mode": args.closure_graph_mode,
        "trainable": args.closure_graph_trainable,
        "metadata": provider.metadata(),
    }
    return provider, payload


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch = _torch()
    torch.manual_seed(args.seed)
    if args.closure_mode == "sparse_linear" and not args.closure_features:
        args.closure_features = ["T", "x", "y", "t"]
    if args.closure_mode == "sparse_linear" and args.closure_graph_mode != "none":
        for feature_name in graph_feature_names(args.closure_graph_embedding_dim):
            if feature_name not in args.closure_features:
                args.closure_features.append(feature_name)
    sample = load_field_table(args.table, observation_columns=[args.target])
    if args.closure_graph_sample_id_column and args.closure_graph_mode not in {
        "real_micro",
        "real_micro_region",
        "real_micro_region_embedding",
    }:
        raise ValueError(
            "--closure-graph-sample-id-column is only supported with "
            "--closure-graph-mode real_micro, real_micro_region, or real_micro_region_embedding"
        )
    closure_graph_sample_ids = None
    if args.closure_graph_sample_id_column:
        row_metadata = sample.metadata.get("row_metadata", {})
        if args.closure_graph_sample_id_column not in row_metadata:
            raise ValueError(
                f"Field table has no row metadata column {args.closure_graph_sample_id_column!r}; "
                "add a micro sample-id column or use --closure-graph-sample-id for a fixed record."
            )
        closure_graph_sample_ids = [str(value) for value in row_metadata[args.closure_graph_sample_id_column]]
    split_manifest = load_split_manifest(args.split_manifest) if args.split_manifest else None
    input_conditioning_profile = _resolve_input_conditioning_profile(args, split_manifest)
    train_indices = (
        split_indices(split_manifest, args.train_split)
        if split_manifest
        else list(range(sample.n_points))
    )
    coords, time, target = sample_to_tensors(sample, args.target, args.device)
    input_features = _input_feature_tensor(sample, args.input_feature_columns, args.device)
    derived_process_features, derived_process_feature_payload = _derived_process_feature_tensor(
        sample,
        args,
        args.device,
    )
    if derived_process_features is not None:
        input_features = (
            torch.cat([input_features, derived_process_features], dim=-1)
            if input_features is not None
            else derived_process_features
        )
    process_graph_features, process_graph_feature_payload = _process_graph_feature_tensor(
        sample,
        args,
        train_indices,
        args.device,
    )
    if process_graph_features is not None:
        input_features = (
            torch.cat([input_features, process_graph_features], dim=-1)
            if input_features is not None
            else process_graph_features
        )
    residual_candidate_indices = _residual_candidate_indices(
        sample=sample,
        target=args.target,
        train_indices=train_indices,
        mode=args.residual_sampling_mode,
        hot_quantile=args.residual_hot_quantile,
        gradient_quantile=args.residual_gradient_quantile,
    )
    train_index = _index_tensor(train_indices, args.device)
    coords, coord_normalization = _normalize_feature_tensor(coords, train_index, args.input_normalization)
    time, time_normalization = _normalize_feature_tensor(time, train_index, args.input_normalization)
    input_feature_normalization = None
    if input_features is not None:
        input_feature_normalization_mode = (
            args.input_normalization
            if args.input_feature_normalization == "same"
            else args.input_feature_normalization
        )
        input_features, input_feature_normalization = _normalize_feature_tensor(
            input_features,
            train_index,
            input_feature_normalization_mode,
        )
    target_residual_baseline, target_residual_baseline_payload = _target_residual_baseline_predictions(
        sample,
        args.target,
        train_indices,
        args,
        args.device,
    )
    target_for_training = target - target_residual_baseline
    target_mean = target_for_training[train_index].mean()
    target_std = target_for_training[train_index].std(unbiased=False)
    if float(target_std.detach().cpu()) == 0.0:
        target_std = torch.ones_like(target_std)
    train_target = (target_for_training - target_mean) / target_std if args.normalize_target else target_for_training
    data_loss_region_weights, data_loss_weighting = _data_loss_weight_tensor(
        sample=sample,
        target=args.target,
        train_indices=train_indices,
        args=args,
        device=args.device,
    )
    data_loss_group_balance_weights, data_loss_group_balance = _group_balance_weight_tensor(
        sample=sample,
        train_indices=train_indices,
        args=args,
        device=args.device,
    )
    data_loss_weights = data_loss_region_weights * data_loss_group_balance_weights
    data_loss_objective = {
        "enabled": bool(data_loss_weighting["enabled"] or data_loss_group_balance["enabled"]),
        "region_component_enabled": bool(data_loss_weighting["enabled"]),
        "group_balance_component_enabled": bool(data_loss_group_balance["enabled"]),
        "weight_sum": float(data_loss_weights.sum().detach().cpu()),
        "mean_weight": float(data_loss_weights.mean().detach().cpu()) if len(train_indices) else 0.0,
    }

    model = MacroPINN(
        coord_dim=coords.shape[-1],
        field_dim=1,
        param_dim=input_features.shape[-1] if input_features is not None else 0,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.layers,
        activation=args.activation,
        conditioning_mode=args.input_conditioning_mode,
        film_strength=args.input_film_strength,
        route_film_prior=args.input_route_film_prior,
        route_trainable=args.input_route_trainable,
        spacetime_encoding=args.spacetime_encoding,
        spacetime_fourier_bands=args.spacetime_fourier_bands,
        backbone_mode=args.backbone_mode,
        backbone_residual_scale=args.backbone_residual_scale,
        process_encoder_mode=args.input_process_encoder_mode,
        process_encoder_dim=args.input_process_encoder_dim,
    ).to(args.device)
    process_encoder_input_dim = int(input_features.shape[-1]) if input_features is not None else 0
    residual_correction_input_dim = coords.shape[-1] + time.reshape(time.shape[0], -1).shape[-1]
    if input_features is not None:
        residual_correction_input_dim += input_features.shape[-1]
    residual_correction = _build_residual_correction(args, residual_correction_input_dim)
    output_affine_input_dim = int(input_features.shape[-1]) if input_features is not None else 0
    output_affine = _build_output_affine(args, output_affine_input_dim)
    closure_library = None
    closure_coefficients = None
    graph_state_dim = coords.shape[-1] + time.reshape(time.shape[0], -1).shape[-1]
    graph_provider, graph_conditioning = _build_graph_conditioning(args, args.device, state_dim=graph_state_dim)
    if args.closure_mode == "sparse_linear":
        closure_library = SparseLibrary(
            SparseLibraryConfig(
                feature_names=tuple(args.closure_features),
                polynomial_order=args.closure_polynomial_order,
                include_bias=args.closure_include_bias,
                include_linear=True,
            )
        )
        closure_coefficients = torch.nn.Parameter(
            torch.zeros(len(closure_library.term_names), dtype=target.dtype, device=target.device)
        )
    elif args.closure_mode != "none":
        raise ValueError(f"Unsupported closure mode: {args.closure_mode}")

    closure_lr = args.closure_lr if args.closure_lr is not None else args.lr
    model_parameters = list(model.parameters())
    parameter_groups: list[dict[str, Any]] = [{"params": model_parameters, "lr": args.lr}]
    if residual_correction is not None:
        residual_lr = args.residual_correction_lr if args.residual_correction_lr is not None else args.lr
        parameter_groups.append({"params": list(residual_correction.parameters()), "lr": residual_lr})
    if output_affine is not None:
        output_affine_lr = args.output_affine_lr if args.output_affine_lr is not None else args.lr
        parameter_groups.append({"params": list(output_affine.parameters()), "lr": output_affine_lr})
    graph_parameters = (
        [parameter for parameter in graph_provider.parameters() if parameter.requires_grad]
        if graph_provider is not None
        else []
    )
    if graph_parameters:
        parameter_groups.append({"params": graph_parameters, "lr": args.closure_graph_lr or args.lr})
    if closure_coefficients is not None:
        parameter_groups.append({"params": [closure_coefficients], "lr": closure_lr})
    optimizer = torch.optim.Adam(parameter_groups)

    history: list[dict[str, float]] = []
    last_train_source = None
    backbone_frozen = False
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        closure_stage_active = step >= args.closure_start_step
        needs_residual = args.pde_weight > 0 and closure_stage_active
        if (
            args.freeze_backbone_after_closure_start
            and closure_coefficients is not None
            and needs_residual
            and not backbone_frozen
        ):
            for parameter in model_parameters:
                parameter.requires_grad_(False)
            backbone_frozen = True
        residual_indices = _residual_sample_indices(
            train_indices=train_indices,
            candidate_indices=residual_candidate_indices,
            step=step,
            sample_size=args.residual_sample_size if needs_residual else None,
            seed=args.residual_sampling_seed,
        )
        residual_index = _index_tensor(residual_indices, args.device)
        coords_data = coords[train_index].detach().clone()
        time_data = time[train_index].detach().clone()
        input_features_data = input_features[train_index].detach().clone() if input_features is not None else None
        pred = _model_forward(model, coords_data, time_data, input_features_data)
        residual_correction_active = step >= args.residual_correction_start_step
        pred = _apply_residual_correction(
            pred,
            residual_correction,
            args,
            coords_data,
            time_data,
            input_features_data,
            active=residual_correction_active,
        )
        pred = _apply_output_affine(pred, output_affine, args, input_features_data)
        pred_for_loss = pred
        baseline_data = target_residual_baseline[train_index]
        if args.normalize_target:
            pred_physical = pred * target_std + target_mean + baseline_data
        else:
            pred_physical = pred + baseline_data
        data_error = (pred_for_loss - train_target[train_index]) ** 2
        data_loss = torch.sum(data_error * data_loss_weights) / data_loss_weights.sum()
        pde_loss = torch.zeros((), dtype=target.dtype, device=target.device)
        closure_loss = torch.zeros((), dtype=target.dtype, device=target.device)
        prediction_anchor_loss = torch.zeros((), dtype=target.dtype, device=target.device)
        if args.prediction_anchor_weight > 0.0:
            prediction_anchor_loss = torch.mean(pred_for_loss**2)
        closure_source = None
        if args.pde_weight > 0:
            coords_residual = coords[residual_index].detach().clone().requires_grad_(True)
            time_residual = time[residual_index].detach().clone().requires_grad_(True)
            input_features_residual = (
                input_features[residual_index].detach().clone() if input_features is not None else None
            )
            pred_residual = _model_forward(model, coords_residual, time_residual, input_features_residual)
            pred_residual = _apply_residual_correction(
                pred_residual,
                residual_correction,
                args,
                coords_residual,
                time_residual,
                input_features_residual,
                active=residual_correction_active,
            )
            pred_residual = _apply_output_affine(pred_residual, output_affine, args, input_features_residual)
            baseline_residual = target_residual_baseline[residual_index]
            if args.normalize_target:
                pred_residual_physical = pred_residual * target_std + target_mean + baseline_residual
            else:
                pred_residual_physical = pred_residual + baseline_residual
            residual_field = pred_residual_physical[:, 0] if args.pde_field == "physical" else pred_residual[:, 0]
            if closure_library is not None and closure_coefficients is not None:
                graph_embedding = None
                graph_features = None
                if graph_provider is not None and args.closure_graph_mode == "toy_static":
                    graph_embedding = graph_provider()
                elif graph_provider is not None and args.closure_graph_mode == "coordinate_rbf":
                    graph_features = graph_provider(coords_residual, time_residual)
                elif graph_provider is not None and args.closure_graph_mode in {
                    "real_micro",
                    "real_micro_region",
                    "real_micro_region_embedding",
                }:
                    residual_sample_ids = (
                        [closure_graph_sample_ids[index] for index in residual_indices]
                        if closure_graph_sample_ids is not None
                        else None
                    )
                    graph_features = graph_provider(coords_residual, time_residual, sample_ids=residual_sample_ids)
                closure_features = _closure_feature_tensor(
                    feature_names=args.closure_features,
                    pred_field=residual_field,
                    coords=coords_residual,
                    time=time_residual,
                    graph_embedding=graph_embedding,
                    graph_features=graph_features,
                )
                closure_matrix = closure_library.transform(closure_features)
                base_source, graph_source, base_mask, graph_mask = _split_closure_source(
                    closure_matrix=closure_matrix,
                    coefficients=closure_coefficients,
                    term_names=closure_library.term_names,
                )
                closure_source = base_source + args.closure_graph_gate * graph_source
                last_train_source = closure_source
                closure_loss = l1_sparsity(closure_coefficients[base_mask], weight=args.closure_l1_weight)
                if bool(graph_mask.any()):
                    closure_loss = closure_loss + l1_sparsity(
                        closure_coefficients[graph_mask],
                        weight=args.closure_graph_l1_weight,
                    )
            else:
                closure_source = args.source
            residual = transient_heat_residual(
                residual_field,
                coords_residual,
                time_residual,
                params=HeatEquationParams(rho_cp=args.rho_cp, conductivity=args.conductivity),
                source=closure_source,
            )
            pde_loss = torch.mean(residual**2)
        loss = (
            data_loss
            + args.prediction_anchor_weight * prediction_anchor_loss
            + args.pde_weight * pde_loss
            + closure_loss
        )
        loss.backward()
        optimizer.step()
        if step == 0 or step == args.steps - 1 or (args.log_every and (step + 1) % args.log_every == 0):
            history.append(
                {
                    "step": float(step + 1),
                    "loss": float(loss.detach().cpu()),
                    "data_loss": float(data_loss.detach().cpu()),
                    "prediction_anchor_loss": float(prediction_anchor_loss.detach().cpu()),
                    "prediction_anchor_enabled": bool(args.prediction_anchor_weight > 0.0),
                    "pde_loss": float(pde_loss.detach().cpu()),
                    "closure_loss": float(closure_loss.detach().cpu()),
                    "residual_points": float(len(residual_indices) if needs_residual else 0),
                    "residual_candidates": float(len(residual_candidate_indices) if needs_residual else 0),
                    "closure_stage_active": bool(closure_stage_active),
                    "backbone_frozen": bool(backbone_frozen),
                    "residual_correction_active": bool(residual_correction is not None and residual_correction_active),
                    "output_affine_enabled": bool(output_affine is not None),
                    "data_loss_region_weighting_enabled": bool(data_loss_weighting["enabled"]),
                    "data_loss_region_weighted_points": float(data_loss_weighting["selected_points"]),
                    "data_loss_mean_weight": float(data_loss_weighting["mean_weight"]),
                    "data_loss_group_balance_enabled": bool(data_loss_group_balance["enabled"]),
                    "data_loss_group_balance_column": data_loss_group_balance.get("column"),
                    "data_loss_group_balance_strength": float(data_loss_group_balance.get("strength") or 0.0),
                    "data_loss_group_balance_groups": float(data_loss_group_balance.get("group_count") or 0),
                    "data_loss_objective_enabled": bool(data_loss_objective["enabled"]),
                }
            )

    with torch.no_grad():
        pred_tensor = _model_forward(model, coords, time, input_features)
        pred_tensor = _apply_residual_correction(
            pred_tensor,
            residual_correction,
            args,
            coords,
            time,
            input_features,
            active=True,
        )
        pred_tensor = _apply_output_affine(pred_tensor, output_affine, args, input_features)
        if args.normalize_target:
            pred_tensor = pred_tensor * target_std + target_mean
        pred_tensor = pred_tensor + target_residual_baseline
        pred = pred_tensor.detach().cpu().reshape(-1).tolist()
        input_route_summary = (
            model.routing_summary(input_features)
            if input_features is not None and args.input_conditioning_mode == "routed"
            else None
        )
    y_true = target.detach().cpu().reshape(-1).tolist()
    metrics = _metric_payload(y_true, pred)
    split_metrics = None
    if split_manifest:
        split_metrics = {}
        for split_name in split_manifest["splits"]:
            indices = split_indices(split_manifest, split_name)
            split_metrics[split_name] = {
                "n_points": len(indices),
                "metrics": _metric_payload(
                    [y_true[index] for index in indices],
                    [pred[index] for index in indices],
                ),
            }
            regions = region_metric_tables(
                sample,
                target=args.target,
                y_pred=pred,
                indices=indices,
                hot_quantiles=args.hot_quantiles,
                gradient_quantiles=args.gradient_quantiles,
            )
            if regions:
                split_metrics[split_name]["region_metrics"] = regions

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "sample_id": sample.sample_id,
        "target": args.target,
        "n_points": sample.n_points,
        "train_points": len(train_indices),
        "metrics": metrics,
        "region_metrics": region_metric_tables(
            sample,
            target=args.target,
            y_pred=pred,
            hot_quantiles=args.hot_quantiles,
            gradient_quantiles=args.gradient_quantiles,
        ),
        "split_metrics": split_metrics,
        "history": history,
        "config": _jsonable_config(args),
        "split_manifest": str(args.split_manifest) if args.split_manifest else None,
        "train_split": args.train_split if split_manifest else None,
        "target_normalization": {
            "enabled": args.normalize_target,
            "mean": float(target_mean.detach().cpu()),
            "std": float(target_std.detach().cpu()),
            "target_space": (
                "residual"
                if target_residual_baseline_payload.get("enabled")
                else "target"
            ),
        },
        "target_residual_baseline": target_residual_baseline_payload,
        "data_loss_weighting": data_loss_weighting,
        "data_loss_group_balance": data_loss_group_balance,
        "data_loss_objective": data_loss_objective,
        "prediction_anchor": _prediction_anchor_payload(args),
        "input_normalization": {
            "mode": args.input_normalization,
            "coordinate_columns": sample.metadata.get("coordinate_columns"),
            "time_column": sample.metadata.get("time_column"),
            "coordinates": coord_normalization,
            "time": time_normalization,
        },
        "input_features": _input_feature_payload(
            args.input_feature_columns,
            input_feature_normalization,
            args.input_conditioning_mode,
            args.input_film_strength,
            args.input_route_film_prior,
            args.input_route_trainable,
            input_route_summary,
            input_conditioning_profile,
            derived_process_feature_payload,
            process_graph_feature_payload,
            int(input_features.shape[-1]) if input_features is not None else 0,
        ),
        "spacetime_encoding": _spacetime_encoding_payload(args, model),
        "backbone": _backbone_payload(args, model),
        "process_encoder": _process_encoder_payload(args, model, process_encoder_input_dim),
        "residual_correction": _residual_correction_payload(
            args,
            residual_correction,
            residual_correction_input_dim,
        ),
        "output_affine": _output_affine_payload(
            args,
            output_affine,
            output_affine_input_dim,
        ),
        "pde": {
            "field": args.pde_field,
            "weight": args.pde_weight,
            "rho_cp": args.rho_cp,
            "conductivity": args.conductivity,
            "source": args.source,
            "closure_start_step": args.closure_start_step,
            "residual_sample_size": args.residual_sample_size,
            "residual_sampling_mode": args.residual_sampling_mode,
            "residual_sampling_seed": args.residual_sampling_seed,
            "residual_hot_quantile": args.residual_hot_quantile,
            "residual_gradient_quantile": args.residual_gradient_quantile,
            "residual_candidate_points": len(residual_candidate_indices),
        },
        "optimizer": _optimizer_payload(args, closure_coefficients, residual_correction, output_affine),
        "closure": _closure_payload(
            closure_mode=args.closure_mode,
            closure_library=closure_library,
            closure_coefficients=closure_coefficients,
            source_values=last_train_source,
            threshold=args.closure_threshold,
            graph_conditioning=graph_conditioning,
            graph_gate=args.closure_graph_gate if graph_conditioning is not None else None,
            graph_l1_weight=args.closure_graph_l1_weight,
        ),
    }
    metrics_path = args.output_dir / "metrics.json"
    checkpoint_path = args.output_dir / "checkpoint.pt"
    manifest_path = args.output_dir / "artifact_manifest.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "residual_correction_state_dict": (
                residual_correction.state_dict() if residual_correction is not None else None
            ),
            "output_affine_state_dict": (
                output_affine.state_dict() if output_affine is not None else None
            ),
            "metadata": {
                "coord_dim": coords.shape[-1],
                "target": args.target,
                "sample_id": sample.sample_id,
                "closure": metrics_payload["closure"],
                "optimizer": metrics_payload["optimizer"],
                "param_dim": int(input_features.shape[-1]) if input_features is not None else 0,
                "input_features": metrics_payload["input_features"],
                "spacetime_encoding": metrics_payload["spacetime_encoding"],
                "backbone": metrics_payload["backbone"],
                "process_encoder": metrics_payload["process_encoder"],
                "residual_correction": metrics_payload["residual_correction"],
                "output_affine": metrics_payload["output_affine"],
                "data_loss_weighting": metrics_payload["data_loss_weighting"],
                "prediction_anchor": metrics_payload["prediction_anchor"],
                "target_residual_baseline": metrics_payload["target_residual_baseline"],
            },
        },
        checkpoint_path,
    )
    manifest = {
        "run_type": "macro_pinn",
        "artifacts": {
            "metrics": str(metrics_path),
            "checkpoint": str(checkpoint_path),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": args.device,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {metrics_path}")
    print(f"Wrote: {checkpoint_path}")
    print(f"Wrote: {manifest_path}")
    print(json.dumps(metrics, indent=2))
    return metrics_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True, type=Path, help="CSV/JSON field table.")
    parser.add_argument("--target", required=True, help="Observation column to train against.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Run output directory.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--closure-lr",
        type=float,
        help="Optional learning rate for sparse closure coefficients; defaults to --lr.",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--activation", default="tanh")
    parser.add_argument(
        "--backbone-mode",
        choices=["mlp", "residual"],
        default="mlp",
        help=(
            "Internal Macro PINN backbone. mlp preserves the original fully connected network; "
            "residual adds same-width hidden residual transitions inside the backbone."
        ),
    )
    parser.add_argument(
        "--backbone-residual-scale",
        type=float,
        default=1.0,
        help="Residual branch multiplier used when --backbone-mode=residual.",
    )
    parser.add_argument(
        "--spacetime-encoding",
        choices=["raw", "fourier"],
        default="raw",
        help=(
            "Coordinate/time representation used by Macro PINN. raw preserves the original x/y[/z]/t inputs; "
            "fourier appends fixed multi-scale sin/cos features."
        ),
    )
    parser.add_argument(
        "--spacetime-fourier-bands",
        type=int,
        default=4,
        help="Number of power-of-two Fourier bands used when --spacetime-encoding=fourier.",
    )
    parser.add_argument(
        "--residual-correction-mode",
        choices=["none", "mlp"],
        default="none",
        help=(
            "Optional weak learned correction added to the Macro PINN output. "
            "mlp uses normalized coordinates/time and any process features as residual inputs."
        ),
    )
    parser.add_argument(
        "--residual-correction-hidden-dim",
        type=int,
        default=32,
        help="Hidden width for --residual-correction-mode=mlp.",
    )
    parser.add_argument(
        "--residual-correction-layers",
        type=int,
        default=1,
        help="Number of hidden layers for --residual-correction-mode=mlp.",
    )
    parser.add_argument(
        "--residual-correction-scale",
        type=float,
        default=0.1,
        help="Multiplier for the learned residual correction before it is added to the base Macro PINN output.",
    )
    parser.add_argument(
        "--residual-correction-lr",
        type=float,
        help="Optional learning rate for residual-correction parameters; defaults to --lr.",
    )
    parser.add_argument(
        "--residual-correction-start-step",
        type=int,
        default=0,
        help="Training step at which the learned residual correction starts contributing to predictions.",
    )
    parser.add_argument(
        "--output-affine-mode",
        choices=["none", "linear"],
        default="none",
        help=(
            "Optional process-conditioned output calibration. linear learns gamma/beta from active input "
            "features and applies prediction <- (1 + scale * gamma) * prediction + scale * beta."
        ),
    )
    parser.add_argument(
        "--output-affine-scale",
        type=float,
        default=1.0,
        help="Multiplier for gamma/beta produced by --output-affine-mode=linear.",
    )
    parser.add_argument(
        "--output-affine-lr",
        type=float,
        help="Optional learning rate for output-affine parameters; defaults to --lr.",
    )
    parser.add_argument(
        "--prediction-anchor-weight",
        type=float,
        default=0.0,
        help=(
            "Optional L2 penalty on Macro PINN predictions in the training target space. "
            "With default target normalization this anchors predictions toward the train-target mean."
        ),
    )
    parser.add_argument("--pde-weight", type=float, default=0.0)
    parser.add_argument(
        "--pde-field",
        choices=["physical", "normalized"],
        default="physical",
        help="Field scale used inside PDE residual. normalized uses the model output before target denormalization.",
    )
    parser.add_argument("--rho-cp", type=float, default=1.0)
    parser.add_argument("--conductivity", type=float, default=1.0)
    parser.add_argument("--source", type=float, default=0.0)
    parser.add_argument(
        "--residual-sample-size",
        type=int,
        help="Optional number of train points sampled per step for PDE/closure residual loss.",
    )
    parser.add_argument(
        "--closure-start-step",
        type=int,
        default=0,
        help="Training step at which PDE/closure residual loss becomes active.",
    )
    parser.add_argument(
        "--residual-sampling-seed",
        type=int,
        default=1337,
        help="Base seed for deterministic per-step residual point sampling.",
    )
    parser.add_argument(
        "--residual-sampling-mode",
        choices=["random", "hot", "gradient", "hot_gradient"],
        default="random",
        help="Candidate pool for residual point sampling.",
    )
    parser.add_argument(
        "--residual-hot-quantile",
        type=float,
        default=0.9,
        help="Train-split target quantile used by hot residual sampling modes.",
    )
    parser.add_argument(
        "--residual-gradient-quantile",
        type=float,
        default=0.9,
        help="Train-split gradient-score quantile used by gradient residual sampling modes.",
    )
    parser.add_argument(
        "--data-loss-weighting",
        choices=["none", "hot", "gradient", "hot_gradient"],
        default="none",
        help=(
            "Optional train-split region weighting for the supervised data loss. "
            "hot uses high target values, gradient uses high spatial-gradient scores, and hot_gradient uses their union."
        ),
    )
    parser.add_argument(
        "--data-loss-hot-quantile",
        type=float,
        default=0.9,
        help="Train-split target quantile used by hot data-loss weighting modes.",
    )
    parser.add_argument(
        "--data-loss-gradient-quantile",
        type=float,
        default=0.9,
        help="Train-split gradient-score quantile used by gradient data-loss weighting modes.",
    )
    parser.add_argument(
        "--data-loss-region-weight",
        type=float,
        default=2.0,
        help="Loss multiplier applied to train points selected by --data-loss-weighting.",
    )
    parser.add_argument(
        "--data-loss-group-balance-column",
        default="",
        help=(
            "Optional row-metadata column used to balance the supervised data loss across process groups. "
            "Use process_condition to balance on the full laser_power/scan_speed/spot_size combination."
        ),
    )
    parser.add_argument(
        "--data-loss-group-balance-strength",
        type=float,
        default=1.0,
        help=(
            "Blend strength for data-loss group balancing. 0 disables the group balance, 1 applies full "
            "inverse-frequency balancing while preserving a unit mean weight."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--split-manifest", type=Path, help="Optional JSON split manifest.")
    parser.add_argument("--train-split", default="train", help="Split used for optimization when split manifest is provided.")
    parser.add_argument(
        "--target-residual-baseline",
        choices=["none", "mean", "knn", "extra_trees"],
        default="none",
        help=(
            "Optional strong baseline fit on the train split. When enabled, Macro PINN trains on "
            "target - baseline_prediction and adds the baseline back for metrics."
        ),
    )
    parser.add_argument(
        "--target-residual-baseline-feature-column",
        action="append",
        dest="target_residual_baseline_feature_columns",
        default=[],
        help=(
            "Feature column used by knn/extra_trees target residual baselines. Defaults to coordinates plus time. "
            "Can include row metadata such as laser_power_W, scan_speed_mm_s, and spot_size_um."
        ),
    )
    parser.add_argument(
        "--target-residual-baseline-n-neighbors",
        type=int,
        default=8,
        help="k for --target-residual-baseline=knn.",
    )
    parser.add_argument(
        "--target-residual-baseline-n-estimators",
        type=int,
        default=200,
        help="Number of trees for --target-residual-baseline=extra_trees.",
    )
    parser.add_argument(
        "--target-residual-baseline-random-state",
        type=int,
        default=7,
        help="Random state for --target-residual-baseline=extra_trees.",
    )
    parser.add_argument(
        "--input-normalization",
        default="none",
        choices=["none", "minmax", "standard"],
        help="Normalize coordinate and time inputs using statistics fitted on the train split.",
    )
    parser.add_argument(
        "--input-feature-column",
        action="append",
        dest="input_feature_columns",
        default=[],
        help=(
            "Additional numeric row-metadata column to append to Macro PINN inputs. "
            "Use this for process conditioning such as laser_power_W, scan_speed_mm_s, or spot_size_um."
        ),
    )
    parser.add_argument(
        "--input-derived-process-features",
        choices=["none", "am_energy_v1"],
        default="none",
        help=(
            "Append deterministic AM process features derived from laser_power_W, scan_speed_mm_s, "
            "and spot_size_um. am_energy_v1 adds line-energy, energy-density proxy, "
            "area-normalized energy proxy, and dwell-time features."
        ),
    )
    parser.add_argument(
        "--input-process-encoder-mode",
        choices=["none", "linear"],
        default="none",
        help=(
            "Optional trainable encoder applied to active input/process features before Macro PINN conditioning. "
            "linear is identity-initialized on the leading features so it starts near the route-guard representation."
        ),
    )
    parser.add_argument(
        "--input-process-encoder-dim",
        type=int,
        default=0,
        help=(
            "Output dimension for --input-process-encoder-mode. "
            "The default 0 preserves the active input/process feature count."
        ),
    )
    parser.add_argument(
        "--input-feature-normalization",
        choices=["same", "none", "minmax", "standard", "global_minmax", "global_standard"],
        default="same",
        help=(
            "Normalization for --input-feature-column values. same reuses --input-normalization. "
            "global_* fits feature statistics on the full field table, useful for known process-design scalars."
        ),
    )
    parser.add_argument(
        "--input-conditioning-mode",
        choices=["concat", "film", "concat_film", "routed"],
        default="concat",
        help=(
            "How additional input-feature columns condition Macro PINN. "
            "concat appends them to coordinates/time; film uses them to modulate hidden coordinate/time layers; "
            "concat_film does both; routed trains concat and FiLM experts and gates their outputs from process features."
        ),
    )
    parser.add_argument(
        "--input-film-strength",
        type=float,
        default=1.0,
        help=(
            "Multiplier for FiLM gamma/beta modulation in film and concat_film modes. "
            "The default 1.0 preserves previous FiLM behavior; smaller values make FiLM a limited correction."
        ),
    )
    parser.add_argument(
        "--input-route-film-prior",
        type=float,
        default=0.5,
        help=(
            "Initial FiLM-expert gate weight for --input-conditioning-mode routed. "
            "Use values below 0.5 for concat-favored axes and above 0.5 for FiLM-favored axes."
        ),
    )
    parser.add_argument(
        "--freeze-input-route",
        action="store_false",
        dest="input_route_trainable",
        help="Keep the routed concat/FiLM gate fixed at its initialized process-feature prior.",
    )
    parser.add_argument(
        "--input-conditioning-profile",
        choices=["none", "process_axis_v1", "broad_process_v1", "broad_process_v2"],
        default="none",
        help=(
            "Optional split-aware conditioning profile. process_axis_v1 reads the split manifest group_key "
            "and selects the Phase 25 best-known route for line, scan-speed, spot-size, laser-power, or full-process holdouts. "
            "broad_process_v1 can also fall back to no process features on broad-data splits where process conditioning degraded. "
            "broad_process_v2 is the same conservative broad-data profile except line_id uses concat/same."
        ),
    )
    parser.add_argument(
        "--process-graph-feature-mode",
        choices=["none", "rbf"],
        default="none",
        help=(
            "Optional structured process-neighborhood features appended to Macro PINN input features. "
            "rbf maps each row's process metadata to RBF similarities against process anchors."
        ),
    )
    parser.add_argument(
        "--process-graph-feature-column",
        action="append",
        dest="process_graph_feature_columns",
        default=[],
        help=(
            "Numeric row-metadata column used to build process-neighborhood graph features. "
            "Defaults to active --input-feature-column values after any conditioning profile is resolved."
        ),
    )
    parser.add_argument(
        "--process-graph-feature-count",
        type=int,
        default=4,
        help="Maximum number of process RBF anchor features when --process-graph-feature-mode=rbf.",
    )
    parser.add_argument(
        "--process-graph-length-scale",
        type=float,
        default=1.0,
        help="RBF length scale in standardized process-feature space.",
    )
    parser.add_argument(
        "--process-graph-fit-scope",
        choices=["train", "global"],
        default="train",
        help="Rows used to fit process graph feature standardization and anchors.",
    )
    parser.add_argument(
        "--hot-quantile",
        action="append",
        type=float,
        dest="hot_quantiles",
        help="Report metrics on target values above this split-local quantile, e.g. 0.9.",
    )
    parser.add_argument(
        "--gradient-quantile",
        action="append",
        type=float,
        dest="gradient_quantiles",
        help="Report metrics on spatial-gradient scores above this split-local quantile, e.g. 0.9.",
    )
    parser.add_argument(
        "--no-normalize-target",
        action="store_false",
        dest="normalize_target",
        help="Disable target normalization during data-loss training.",
    )
    parser.add_argument(
        "--closure-mode",
        choices=["none", "sparse_linear"],
        default="none",
        help="Optional learnable closure/source term used by the PDE residual.",
    )
    parser.add_argument(
        "--closure-feature",
        action="append",
        dest="closure_features",
        default=[],
        help="Closure feature name. Can repeat; supported first-stage names include T, x, y, z, and t.",
    )
    parser.add_argument("--closure-polynomial-order", type=int, default=1)
    parser.add_argument("--closure-l1-weight", type=float, default=0.0)
    parser.add_argument("--closure-threshold", type=float, default=0.0)
    parser.add_argument(
        "--closure-graph-gate",
        type=float,
        default=1.0,
        help="Multiplier for graph-conditioned closure terms; values below 1 make graph source a small correction.",
    )
    parser.add_argument(
        "--closure-graph-l1-weight",
        type=float,
        default=0.0,
        help="Optional separate L1 penalty for graph-conditioned closure coefficients.",
    )
    parser.add_argument(
        "--closure-graph-mode",
        choices=[
            "none",
            "toy_static",
            "coordinate_rbf",
            "real_micro",
            "real_micro_region",
            "real_micro_region_embedding",
        ],
        default="none",
        help=(
            "Optional graph-conditioned closure features. coordinate_rbf creates per-point anchor features; "
            "real_micro_region selects local patch features from AM-Bench micrograph grids; "
            "real_micro_region_embedding selects fixed low-dimensional patch embeddings."
        ),
    )
    parser.add_argument("--closure-graph-embedding-dim", type=int, default=2)
    parser.add_argument(
        "--closure-graph-state-dim",
        type=int,
        default=0,
        help="State dimension for coordinate_rbf; default 0 infers coordinate dimension plus time dimension.",
    )
    parser.add_argument("--closure-graph-length-scale", type=float, default=0.35)
    parser.add_argument("--closure-graph-node-dim", type=int, default=2)
    parser.add_argument("--closure-graph-hidden-dim", type=int, default=16)
    parser.add_argument("--closure-graph-steps", type=int, default=1)
    parser.add_argument("--closure-graph-seed", type=int, default=2026)
    parser.add_argument(
        "--closure-graph-features",
        type=Path,
        help="JSONL graph feature table used by real_micro graph conditioning.",
    )
    parser.add_argument(
        "--closure-graph-sample-id",
        help="Sample id selected from --closure-graph-features for real_micro graph conditioning.",
    )
    parser.add_argument(
        "--closure-graph-sample-id-column",
        help=(
            "Field-table metadata column containing per-row sample ids for real_micro graph conditioning. "
            "Use this with a panel-level graph feature JSONL to avoid broadcasting one micro graph record."
        ),
    )
    parser.add_argument(
        "--no-closure-graph-normalize",
        action="store_true",
        help="Disable normalization of coordinate_rbf graph features across anchors.",
    )
    parser.add_argument(
        "--closure-graph-region-row-source",
        choices=["x", "y"],
        default="y",
        help="Coordinate source used as micrograph row for real_micro_region.",
    )
    parser.add_argument(
        "--closure-graph-region-col-source",
        choices=["x", "y"],
        default="x",
        help="Coordinate source used as micrograph column for real_micro_region.",
    )
    parser.add_argument(
        "--closure-graph-region-flip-row",
        action="store_true",
        help="Map real_micro_region query row to 1 - row after source selection.",
    )
    parser.add_argument(
        "--closure-graph-region-flip-col",
        action="store_true",
        help="Map real_micro_region query column to 1 - column after source selection.",
    )
    parser.add_argument(
        "--closure-graph-region-selection",
        choices=["nearest", "inverse_distance"],
        default="nearest",
        help="Patch selection rule for real_micro_region graph conditioning.",
    )
    parser.add_argument(
        "--closure-graph-region-inverse-distance-epsilon",
        type=float,
        default=1e-6,
        help="Positive epsilon for inverse-distance real_micro_region interpolation.",
    )
    parser.add_argument(
        "--closure-graph-lr",
        type=float,
        help="Optional learning rate for trainable graph-conditioned closure modules.",
    )
    parser.add_argument(
        "--train-closure-graph",
        action="store_true",
        dest="closure_graph_trainable",
        help="Train graph-conditioned closure provider parameters when supported.",
    )
    parser.add_argument(
        "--freeze-backbone-after-closure-start",
        action="store_true",
        help="Freeze Macro PINN parameters once closure/PDE residual training starts.",
    )
    parser.add_argument(
        "--no-closure-bias",
        action="store_false",
        dest="closure_include_bias",
        help="Disable the constant term in sparse closure libraries.",
    )
    parser.set_defaults(normalize_target=True)
    parser.set_defaults(closure_include_bias=True)
    parser.set_defaults(closure_graph_trainable=False)
    parser.set_defaults(input_route_trainable=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
