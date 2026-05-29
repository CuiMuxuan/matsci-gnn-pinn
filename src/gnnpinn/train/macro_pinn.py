"""Train a minimal macro PINN on a local field table."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

from gnnpinn.data.loaders import load_field_table
from gnnpinn.data.splits import load_split_manifest, split_indices
from gnnpinn.eval.metrics import mae, relative_l2, rmse
from gnnpinn.eval.regions import region_metric_tables
from gnnpinn.models.closure import (
    SparseLibrary,
    SparseLibraryConfig,
    export_linear_library_expression,
    expression_to_string,
    l1_sparsity,
)
from gnnpinn.models.pinn import MacroPINN
from gnnpinn.physics.heat import HeatEquationParams, transient_heat_residual


def _torch() -> Any:
    import torch

    return torch


def sample_to_tensors(sample: Any, target: str, device: str = "cpu") -> tuple[Any, Any, Any]:
    torch = _torch()
    coords = torch.tensor(sample.coordinates, dtype=torch.float32, device=device)
    time = torch.tensor(sample.time, dtype=torch.float32, device=device).reshape(-1, 1)
    values = torch.tensor(sample.require_observation(target), dtype=torch.float32, device=device).reshape(-1, 1)
    return coords, time, values


def _index_tensor(indices: list[int], device: str) -> Any:
    torch = _torch()
    return torch.tensor(indices, dtype=torch.long, device=device)


def _normalize_feature_tensor(tensor: Any, train_index: Any, mode: str) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    train_values = tensor[train_index]
    stats: dict[str, Any] = {"mode": mode}
    if mode == "none":
        stats["applied"] = False
        return tensor, stats
    if mode == "standard":
        center = train_values.mean(dim=0, keepdim=True)
        scale = train_values.std(dim=0, unbiased=False, keepdim=True)
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
    if mode == "minmax":
        minimum = train_values.min(dim=0, keepdim=True).values
        maximum = train_values.max(dim=0, keepdim=True).values
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
    return config


def _closure_feature_tensor(
    feature_names: list[str],
    pred_field: Any,
    coords: Any,
    time: Any,
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
) -> dict[str, Any]:
    payload: dict[str, Any] = {"mode": closure_mode}
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


def _residual_sample_indices(
    train_indices: list[int],
    step: int,
    sample_size: int | None,
    seed: int,
) -> list[int]:
    if sample_size is None or sample_size >= len(train_indices):
        return train_indices
    if sample_size <= 0:
        raise ValueError("residual_sample_size must be positive when provided")
    rng = __import__("random").Random(seed + step)
    return sorted(rng.sample(train_indices, sample_size))


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch = _torch()
    torch.manual_seed(args.seed)
    if args.closure_mode == "sparse_linear" and not args.closure_features:
        args.closure_features = ["T", "x", "y", "t"]
    sample = load_field_table(args.table, observation_columns=[args.target])
    coords, time, target = sample_to_tensors(sample, args.target, args.device)
    split_manifest = load_split_manifest(args.split_manifest) if args.split_manifest else None
    train_indices = (
        split_indices(split_manifest, args.train_split)
        if split_manifest
        else list(range(sample.n_points))
    )
    train_index = _index_tensor(train_indices, args.device)
    coords, coord_normalization = _normalize_feature_tensor(coords, train_index, args.input_normalization)
    time, time_normalization = _normalize_feature_tensor(time, train_index, args.input_normalization)
    target_mean = target[train_index].mean()
    target_std = target[train_index].std(unbiased=False)
    if float(target_std.detach().cpu()) == 0.0:
        target_std = torch.ones_like(target_std)
    train_target = (target - target_mean) / target_std if args.normalize_target else target

    model = MacroPINN(
        coord_dim=coords.shape[-1],
        field_dim=1,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.layers,
        activation=args.activation,
    ).to(args.device)
    closure_library = None
    closure_coefficients = None
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

    parameters = list(model.parameters())
    if closure_coefficients is not None:
        parameters.append(closure_coefficients)
    optimizer = torch.optim.Adam(parameters, lr=args.lr)

    history: list[dict[str, float]] = []
    last_train_source = None
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        needs_residual = args.pde_weight > 0
        residual_indices = _residual_sample_indices(
            train_indices=train_indices,
            step=step,
            sample_size=args.residual_sample_size if needs_residual else None,
            seed=args.residual_sampling_seed,
        )
        residual_index = _index_tensor(residual_indices, args.device)
        coords_data = coords[train_index].detach().clone()
        time_data = time[train_index].detach().clone()
        pred = model(coords_data, time_data)
        pred_for_loss = pred
        if args.normalize_target:
            pred_physical = pred * target_std + target_mean
        else:
            pred_physical = pred
        data_loss = torch.mean((pred_for_loss - train_target[train_index]) ** 2)
        pde_loss = torch.zeros((), dtype=target.dtype, device=target.device)
        closure_loss = torch.zeros((), dtype=target.dtype, device=target.device)
        closure_source = None
        if args.pde_weight > 0:
            coords_residual = coords[residual_index].detach().clone().requires_grad_(True)
            time_residual = time[residual_index].detach().clone().requires_grad_(True)
            pred_residual = model(coords_residual, time_residual)
            if args.normalize_target:
                pred_residual_physical = pred_residual * target_std + target_mean
            else:
                pred_residual_physical = pred_residual
            residual_field = pred_residual_physical[:, 0] if args.pde_field == "physical" else pred_residual[:, 0]
            if closure_library is not None and closure_coefficients is not None:
                closure_features = _closure_feature_tensor(
                    feature_names=args.closure_features,
                    pred_field=residual_field,
                    coords=coords_residual,
                    time=time_residual,
                )
                closure_matrix = closure_library.transform(closure_features)
                closure_source = closure_matrix @ closure_coefficients
                last_train_source = closure_source
                closure_loss = l1_sparsity(closure_coefficients, weight=args.closure_l1_weight)
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
        loss = data_loss + args.pde_weight * pde_loss + closure_loss
        loss.backward()
        optimizer.step()
        if step == 0 or step == args.steps - 1 or (args.log_every and (step + 1) % args.log_every == 0):
            history.append(
                {
                    "step": float(step + 1),
                    "loss": float(loss.detach().cpu()),
                    "data_loss": float(data_loss.detach().cpu()),
                    "pde_loss": float(pde_loss.detach().cpu()),
                    "closure_loss": float(closure_loss.detach().cpu()),
                    "residual_points": float(len(residual_indices) if needs_residual else 0),
                }
            )

    with torch.no_grad():
        pred_tensor = model(coords, time)
        if args.normalize_target:
            pred_tensor = pred_tensor * target_std + target_mean
        pred = pred_tensor.detach().cpu().reshape(-1).tolist()
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
        },
        "input_normalization": {
            "mode": args.input_normalization,
            "coordinate_columns": sample.metadata.get("coordinate_columns"),
            "time_column": sample.metadata.get("time_column"),
            "coordinates": coord_normalization,
            "time": time_normalization,
        },
        "pde": {
            "field": args.pde_field,
            "weight": args.pde_weight,
            "rho_cp": args.rho_cp,
            "conductivity": args.conductivity,
            "source": args.source,
            "residual_sample_size": args.residual_sample_size,
            "residual_sampling_seed": args.residual_sampling_seed,
        },
        "closure": _closure_payload(
            closure_mode=args.closure_mode,
            closure_library=closure_library,
            closure_coefficients=closure_coefficients,
            source_values=last_train_source,
            threshold=args.closure_threshold,
        ),
    }
    metrics_path = args.output_dir / "metrics.json"
    checkpoint_path = args.output_dir / "checkpoint.pt"
    manifest_path = args.output_dir / "artifact_manifest.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": {
                "coord_dim": coords.shape[-1],
                "target": args.target,
                "sample_id": sample.sample_id,
                "closure": metrics_payload["closure"],
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
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--activation", default="tanh")
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
        "--residual-sampling-seed",
        type=int,
        default=1337,
        help="Base seed for deterministic per-step residual point sampling.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--split-manifest", type=Path, help="Optional JSON split manifest.")
    parser.add_argument("--train-split", default="train", help="Split used for optimization when split manifest is provided.")
    parser.add_argument(
        "--input-normalization",
        default="none",
        choices=["none", "minmax", "standard"],
        help="Normalize coordinate and time inputs using statistics fitted on the train split.",
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
        "--no-closure-bias",
        action="store_false",
        dest="closure_include_bias",
        help="Disable the constant term in sparse closure libraries.",
    )
    parser.set_defaults(normalize_target=True)
    parser.set_defaults(closure_include_bias=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
