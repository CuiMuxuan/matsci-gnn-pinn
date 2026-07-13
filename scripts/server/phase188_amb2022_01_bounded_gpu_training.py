#!/usr/bin/env python3
"""Run the fixed Phase 187 six-run B6/B7/B8 neural comparison once."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATASET = Path(
    os.environ.get(
        "AMB2022_01_PHASE182_DATASET",
        "/root/matsci-gnn-pinn-data/derived/ambench/2022_3d_build/AMB2022-01/"
        "phase182/phase182_layer_space_dataset.h5",
    )
)
DEFAULT_KERNEL = Path(
    os.environ.get(
        "AMB2022_01_PHASE186_KERNEL",
        "/root/matsci-gnn-pinn-data/derived/ambench/2022_3d_build/AMB2022-01/"
        "phase186/causal_heat_kernel_features.h5",
    )
)
DEFAULT_PHASE187 = Path(
    os.environ.get(
        "AMB2022_01_PHASE187_DESIGN",
        "/root/matsci-gnn-pinn-ops/phase187_candidate_model_design.json",
    )
)
TARGETS = ("tam_s", "scr_C_per_s")
TARGET_DATASETS = {"tam_s": "target_tam_s", "scr_C_per_s": "target_scr_C_per_s"}
SPLITS = {"train": 0, "val": 1, "test": 2}
SEEDS = (1871, 1872, 1873)
HIDDEN_WIDTHS = (64, 32)
BATCH_SIZE = 4096
MAX_EPOCHS = 100
PATIENCE = 12
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MONOTONIC_WEIGHT = 0.01


def _torch() -> Any:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("PyTorch is required for Phase 188") from exc
    return torch, nn, DataLoader, TensorDataset


def _h5py() -> Any:
    try:
        import h5py
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("h5py is required for Phase 188") from exc
    return h5py


def _json_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def _read_phase187(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    gate = payload.get("gate", {})
    expected = "phase187_candidate_model_design_ready_phase188_bounded_gpu_training"
    if gate.get("status") != expected or not gate.get("phase188_bounded_gpu_training_allowed"):
        raise ValueError(f"Phase 187 does not permit bounded GPU training: {gate.get('status')!r}")
    return payload


class PreparedData:
    def __init__(
        self,
        *,
        features: np.ndarray,
        targets: np.ndarray,
        masks: np.ndarray,
        split_codes: np.ndarray,
        feature_names: tuple[str, ...],
        target_means: np.ndarray,
        target_scales: np.ndarray,
        target_hot_q90: np.ndarray,
        feature_means: np.ndarray,
        feature_scales: np.ndarray,
        energy_feature_index: int,
    ) -> None:
        self.features = features
        self.targets = targets
        self.masks = masks
        self.split_codes = split_codes
        self.feature_names = feature_names
        self.target_means = target_means
        self.target_scales = target_scales
        self.target_hot_q90 = target_hot_q90
        self.feature_means = feature_means
        self.feature_scales = feature_scales
        self.energy_feature_index = energy_feature_index


def prepare_data(dataset_path: Path, kernel_path: Path, *, use_kernel: bool) -> PreparedData:
    h5py = _h5py()
    with h5py.File(dataset_path, "r") as handle:
        base_features = handle["features"][...].astype(np.float32)
        feature_names = list(_json_attr(handle.attrs["feature_names"]))
        targets = np.column_stack([handle[TARGET_DATASETS[name]][...] for name in TARGETS]).astype(np.float32)
        masks = handle["target_valid_mask"][...].astype(bool)
        split_codes = handle["primary_split"][...]
    if use_kernel:
        with h5py.File(kernel_path, "r") as handle:
            kernel = handle["causal_heat_kernel_features"][...].astype(np.float32)
            alpha_values = _json_attr(handle.attrs["kernel_alphas_mm2_s"])
        if len(kernel) != len(base_features):
            raise ValueError("Phase 186 kernel rows do not align with Phase 182 rows")
        base_features = np.column_stack([base_features, kernel]).astype(np.float32)
        feature_names.extend([f"causal_heat_kernel_alpha_{float(value):g}_mm2_s" for value in alpha_values])
    if not np.isfinite(base_features).all():
        raise ValueError("Feature matrix has non-finite values")
    train_rows = split_codes == SPLITS["train"]
    feature_means = base_features[train_rows].mean(axis=0)
    feature_scales = base_features[train_rows].std(axis=0)
    feature_scales[feature_scales <= 1e-12] = 1.0
    target_means = np.zeros(len(TARGETS), dtype=np.float32)
    target_scales = np.ones(len(TARGETS), dtype=np.float32)
    target_hot_q90 = np.zeros(len(TARGETS), dtype=np.float32)
    for index in range(len(TARGETS)):
        values = targets[train_rows & masks[:, index], index]
        if len(values) == 0 or not np.isfinite(values).all():
            raise ValueError(f"Training target {TARGETS[index]} has no finite B6 observations")
        target_means[index] = float(values.mean())
        target_scales[index] = max(float(values.std()), 1e-12)
        target_hot_q90[index] = float(np.quantile(values, 0.9))
    standardized_features = (base_features - feature_means) / feature_scales
    standardized_targets = (targets - target_means) / target_scales
    standardized_targets[~masks] = 0.0
    return PreparedData(
        features=standardized_features.astype(np.float32),
        targets=standardized_targets.astype(np.float32),
        masks=masks,
        split_codes=split_codes,
        feature_names=tuple(feature_names),
        target_means=target_means,
        target_scales=target_scales,
        target_hot_q90=target_hot_q90,
        feature_means=feature_means,
        feature_scales=feature_scales,
        energy_feature_index=feature_names.index("laser_energy_density_J_mm2"),
    )


def _make_model(input_size: int) -> Any:
    torch, nn, _, _ = _torch()
    return nn.Sequential(
        nn.Linear(input_size, HIDDEN_WIDTHS[0]),
        nn.SiLU(),
        nn.Linear(HIDDEN_WIDTHS[0], HIDDEN_WIDTHS[1]),
        nn.SiLU(),
        nn.Linear(HIDDEN_WIDTHS[1], len(TARGETS)),
    )


def masked_mse(prediction: Any, target: Any, mask: Any) -> Any:
    squared = (prediction - target).square() * mask
    return squared.sum() / mask.sum().clamp_min(1.0)


def _predict(model: Any, features: np.ndarray, device: Any, batch_size: int = 16384) -> np.ndarray:
    torch, _, _, _ = _torch()
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _metrics_for_split(
    *,
    predictions_standard: np.ndarray,
    data: PreparedData,
    split_name: str,
    variant_id: str,
    seed: int,
    best_epoch: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_rows = data.split_codes == SPLITS[split_name]
    predictions = predictions_standard * data.target_scales + data.target_means
    targets = data.targets * data.target_scales + data.target_means
    for target_index, target_name in enumerate(TARGETS):
        mask = split_rows & data.masks[:, target_index]
        y = targets[mask, target_index].astype(np.float64)
        pred = predictions[mask, target_index].astype(np.float64)
        residual = y - pred
        rmse = float(math.sqrt(np.mean(residual**2)))
        mae = float(np.mean(np.abs(residual)))
        denominator = float(np.sum((y - y.mean()) ** 2))
        r2 = float(1.0 - np.sum(residual**2) / denominator) if denominator > 0.0 else 0.0
        hot = y >= float(data.target_hot_q90[target_index])
        hot_rmse = float(math.sqrt(np.mean(residual[hot] ** 2))) if np.any(hot) else float("nan")
        rows.append(
            {
                "variant_id": variant_id,
                "seed": seed,
                "best_epoch": best_epoch,
                "target": target_name,
                "split": split_name,
                "n_rows": int(np.sum(mask)),
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "nrmse_train_std": rmse / float(data.target_scales[target_index]),
                "hot_q90_rmse": hot_rmse,
            }
        )
    return rows


def _monotonic_violation_fraction(model: Any, data: PreparedData, split_name: str, device: Any) -> float:
    torch, _, _, _ = _torch()
    rows = np.flatnonzero(data.split_codes == SPLITS[split_name])
    if not len(rows):
        return float("nan")
    violations: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(rows), 8192):
        batch = torch.from_numpy(data.features[rows[start : start + 8192]]).to(device).requires_grad_(True)
        output = model(batch)[:, 0].sum()
        gradient = torch.autograd.grad(output, batch)[0][:, data.energy_feature_index]
        violations.append((gradient < 0.0).detach().cpu().numpy())
    return float(np.mean(np.concatenate(violations)))


def train_variant(
    *,
    data: PreparedData,
    variant_id: str,
    seed: int,
    device: Any,
    checkpoint_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch, _, DataLoader, TensorDataset = _torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    use_monotonic = variant_id == "physics_regularized_history_mlp"
    train_rows = np.flatnonzero(data.split_codes == SPLITS["train"])
    dataset = TensorDataset(
        torch.from_numpy(data.features[train_rows]),
        torch.from_numpy(data.targets[train_rows]),
        torch.from_numpy(data.masks[train_rows].astype(np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator, drop_last=False)
    model = _make_model(data.features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    val_rows = np.flatnonzero(data.split_codes == SPLITS["val"])
    val_x = torch.from_numpy(data.features[val_rows]).to(device)
    val_y = torch.from_numpy(data.targets[val_rows]).to(device)
    val_mask = torch.from_numpy(data.masks[val_rows].astype(np.float32)).to(device)
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    no_improvement = 0
    train_loss_last = float("nan")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0
        for x_batch, y_batch, mask_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)
            if use_monotonic:
                x_batch = x_batch.detach().requires_grad_(True)
            prediction = model(x_batch)
            loss = masked_mse(prediction, y_batch, mask_batch)
            if use_monotonic:
                tam_gradient = torch.autograd.grad(prediction[:, 0].sum(), x_batch, create_graph=True)[0][
                    :, data.energy_feature_index
                ]
                loss = loss + MONOTONIC_WEIGHT * torch.relu(-tam_gradient).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            batch_count += 1
        train_loss_last = running_loss / max(1, batch_count)
        model.eval()
        with torch.no_grad():
            val_prediction = model(val_x)
            val_score = float(masked_mse(val_prediction, val_y, val_mask).cpu())
        if val_score < best_score - 1e-8:
            best_score = val_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("No validation checkpoint was captured")
    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "variant_id": variant_id,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_masked_mse": best_score,
            "feature_names": list(data.feature_names),
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    all_prediction = _predict(model, data.features, device)
    metric_rows = []
    for split_name in SPLITS:
        metric_rows.extend(
            _metrics_for_split(
                predictions_standard=all_prediction,
                data=data,
                split_name=split_name,
                variant_id=variant_id,
                seed=seed,
                best_epoch=best_epoch,
            )
        )
    audit = {
        "variant_id": variant_id,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_masked_mse": best_score,
        "final_train_loss": train_loss_last,
        "monotonic_violation_fraction_val": _monotonic_violation_fraction(model, data, "val", device)
        if use_monotonic
        else None,
        "monotonic_violation_fraction_test": _monotonic_violation_fraction(model, data, "test", device)
        if use_monotonic
        else None,
        "checkpoint": str(checkpoint_path),
    }
    return metric_rows, audit


def build_gate(metric_rows: list[dict[str, Any]], audits: list[dict[str, Any]]) -> dict[str, Any]:
    blockers: list[str] = []
    comparison: dict[str, dict[str, float]] = {}
    expected_seeds = set(SEEDS)
    for target_name in TARGETS:
        comparison[target_name] = {}
        for split_name in ("val", "test"):
            values = {}
            for variant_id in ("small_data_only_mlp", "physics_regularized_history_mlp"):
                subset = [
                    float(row["rmse"])
                    for row in metric_rows
                    if row["variant_id"] == variant_id and row["target"] == target_name and row["split"] == split_name
                ]
                observed_seeds = {
                    int(row["seed"])
                    for row in metric_rows
                    if row["variant_id"] == variant_id and row["target"] == target_name and row["split"] == split_name
                }
                if len(subset) != len(SEEDS) or observed_seeds != expected_seeds:
                    blockers.append(f"{target_name}_{split_name}_{variant_id}_seed_count")
                    continue
                values[variant_id] = float(np.mean(subset))
            if len(values) == 2:
                comparison[target_name][f"{split_name}_rmse_gain_candidate_vs_data_only"] = (
                    values["small_data_only_mlp"] - values["physics_regularized_history_mlp"]
                )
        candidate_audits = [row for row in audits if row["variant_id"] == "physics_regularized_history_mlp"]
        audit_seeds = {int(row["seed"]) for row in candidate_audits}
        if len(candidate_audits) != len(SEEDS) or audit_seeds != expected_seeds:
            blockers.append(f"{target_name}_candidate_audit_seed_count")
    violation_values = [
        float(row["monotonic_violation_fraction_test"])
        for row in audits
        if row["variant_id"] == "physics_regularized_history_mlp" and row["monotonic_violation_fraction_test"] is not None
    ]
    if len(violation_values) != len(SEEDS):
        blockers.append("candidate_monotonic_audit_missing")
    elif float(np.mean(violation_values)) > 0.25:
        blockers.append("candidate_excessive_monotonic_violations")
    ready = not blockers
    return {
        "status": (
            "phase188_bounded_gpu_training_complete_phase189_replicate_review"
            if ready
            else "phase188_bounded_gpu_training_incomplete_or_unstable"
        ),
        "training_complete": ready,
        "phase189_replicate_review_allowed": ready,
        "model_training_allowed": False,
        "a800_training_allowed_now": False,
        "candidate_vs_data_only_rmse_gains": comparison,
        "candidate_monotonic_violation_fraction_test_mean": float(np.mean(violation_values)) if violation_values else None,
        "blocking_audits": blockers,
        "next_action": (
            "review replicate stability, controls, and held-build error before deciding whether a paper-facing route survives"
            if ready
            else "inspect failed seeds or constraint behavior; do not tune after B8 observation"
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(
    *,
    dataset_path: Path,
    kernel_path: Path,
    phase187_path: Path,
    checkpoint_dir: Path,
    device_name: str | None = None,
) -> dict[str, Any]:
    _read_phase187(phase187_path)
    torch, _, _, _ = _torch()
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Phase 188 is authorized for the GPU instance; CUDA is required")
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for variant_id, use_kernel in (("small_data_only_mlp", False), ("physics_regularized_history_mlp", True)):
        data = prepare_data(dataset_path, kernel_path, use_kernel=use_kernel)
        for seed in SEEDS:
            metric_rows, audit = train_variant(
                data=data,
                variant_id=variant_id,
                seed=seed,
                device=device,
                checkpoint_path=checkpoint_dir / f"{variant_id}_seed{seed}.pt",
            )
            rows.extend(metric_rows)
            audits.append(audit)
    return {
        "phase": 188,
        "objective": "fixed_b6_fit_b7_selection_b8_held_build_neural_comparison",
        "dataset": str(dataset_path),
        "kernel_features": str(kernel_path),
        "phase187_design": str(phase187_path),
        "device": str(device),
        "seeds": list(SEEDS),
        "training_contract": {
            "hidden_widths": list(HIDDEN_WIDTHS),
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "tam_monotonic_weight": MONOTONIC_WEIGHT,
        },
        "metrics": rows,
        "training_audits": audits,
        "gate": build_gate(rows, audits),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--kernel", type=Path, default=DEFAULT_KERNEL)
    parser.add_argument("--phase187", type=Path, default=DEFAULT_PHASE187)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    payload = run(
        dataset_path=args.dataset,
        kernel_path=args.kernel,
        phase187_path=args.phase187,
        checkpoint_dir=args.checkpoint_dir,
        device_name=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(
        args.metrics_output,
        payload["metrics"],
        ("variant_id", "seed", "best_epoch", "target", "split", "n_rows", "rmse", "mae", "r2", "nrmse_train_std", "hot_q90_rmse"),
    )
    _write_csv(
        args.audit_output,
        payload["training_audits"],
        ("variant_id", "seed", "best_epoch", "best_validation_masked_mse", "final_train_loss", "monotonic_violation_fraction_val", "monotonic_violation_fraction_test", "checkpoint"),
    )
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
