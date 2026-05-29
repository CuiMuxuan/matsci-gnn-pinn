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


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch = _torch()
    torch.manual_seed(args.seed)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict[str, float]] = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        coords_step = coords[train_index].detach().clone().requires_grad_(args.pde_weight > 0)
        time_step = time[train_index].detach().clone().requires_grad_(args.pde_weight > 0)
        pred = model(coords_step, time_step)
        pred_for_loss = pred
        if args.normalize_target:
            pred_physical = pred * target_std + target_mean
        else:
            pred_physical = pred
        data_loss = torch.mean((pred_for_loss - train_target[train_index]) ** 2)
        pde_loss = torch.zeros((), dtype=target.dtype, device=target.device)
        if args.pde_weight > 0:
            residual = transient_heat_residual(
                pred_physical[:, 0],
                coords_step,
                time_step,
                params=HeatEquationParams(rho_cp=args.rho_cp, conductivity=args.conductivity),
                source=args.source,
            )
            pde_loss = torch.mean(residual**2)
        loss = data_loss + args.pde_weight * pde_loss
        loss.backward()
        optimizer.step()
        if step == 0 or step == args.steps - 1 or (args.log_every and (step + 1) % args.log_every == 0):
            history.append(
                {
                    "step": float(step + 1),
                    "loss": float(loss.detach().cpu()),
                    "data_loss": float(data_loss.detach().cpu()),
                    "pde_loss": float(pde_loss.detach().cpu()),
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "sample_id": sample.sample_id,
        "target": args.target,
        "n_points": sample.n_points,
        "train_points": len(train_indices),
        "metrics": metrics,
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
    parser.add_argument("--rho-cp", type=float, default=1.0)
    parser.add_argument("--conductivity", type=float, default=1.0)
    parser.add_argument("--source", type=float, default=0.0)
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
        "--no-normalize-target",
        action="store_false",
        dest="normalize_target",
        help="Disable target normalization during data-loss training.",
    )
    parser.set_defaults(normalize_target=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
