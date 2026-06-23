#!/usr/bin/env python3
"""Run Phase 167 low-budget synthetic PINN smoke.

Phase 167 is the first tiny training smoke after the Phase 166 design gate. It
uses a synthetic 1D moving-heat-source task only. It does not read AM-Bench,
NIST AMMT, or other raw datasets, and it does not justify A100-SXM4-80GB.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np


DEFAULT_OUTPUT_DIR = Path("docs/results/phase167_low_budget_pinn_smoke")

PHASE_INPUTS = {
    "phase166_gate": Path(
        "docs/results/phase166_low_budget_pinn_smoke_design_gate/"
        "phase166_low_budget_pinn_smoke_design_gate.json"
    ),
}

VARIANT_FIELDS = (
    "variant_id",
    "family",
    "sampler_id",
    "residual_weight",
    "source_mode",
    "is_control",
    "description",
)

RUN_FIELDS = (
    "variant_id",
    "seed",
    "split",
    "steps",
    "sensor_count",
    "collocation_count",
    "temperature_rmse",
    "hot_q90_rmse",
    "gradient_q90_rmse",
    "residual_rmse",
    "selection_score",
)

SUMMARY_FIELDS = (
    "variant_id",
    "family",
    "sampler_id",
    "split",
    "seed_count",
    "temperature_rmse_mean",
    "temperature_rmse_std",
    "hot_q90_rmse_mean",
    "gradient_q90_rmse_mean",
    "residual_rmse_mean",
    "selection_score_mean",
    "selection_score_std",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 10)
    if isinstance(value, dict):
        return {key: _stable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stable(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(_stable(payload), indent=2, sort_keys=True) + "\n")


def _csv_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{round(value, 10):.10g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_stable(value), sort_keys=True)
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def _display_path(path: Path, root: Path | None = None) -> str:
    if root is None:
        return str(path).replace("\\", "/")
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _is_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def true_temperature_np(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    center = 0.18 + 0.58 * t + 0.025 * np.sin(2.0 * math.pi * t)
    width = 0.048 + 0.018 * t
    gaussian = np.exp(-0.5 * ((x - center) / width) ** 2)
    return 0.12 + 0.05 * x + 0.08 * t + 0.92 * gaussian


def gradient_score_np(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    center = 0.18 + 0.58 * t + 0.025 * np.sin(2.0 * math.pi * t)
    width = 0.048 + 0.018 * t
    temp = true_temperature_np(x, t)
    return np.abs(temp * (x - center) / np.maximum(width**2, 1e-8))


def collocation_field() -> dict[str, np.ndarray]:
    x_values = np.linspace(0.0, 1.0, 78)
    t_values = np.linspace(0.0, 1.0, 54)
    t_grid, x_grid = np.meshgrid(t_values, x_values, indexing="ij")
    x = x_grid.ravel()
    t = t_grid.ravel()
    temp = true_temperature_np(x, t)
    gradient = gradient_score_np(x, t)
    boundary = ((x <= 0.04) | (x >= 0.96) | (t <= 0.04) | (t >= 0.96)).astype(float)
    source_like = temp - np.quantile(temp, 0.10)
    residual_indicator = np.abs(source_like) * (1.0 + 0.15 * gradient / np.max(gradient))
    return {
        "x": x,
        "t": t,
        "temperature": temp,
        "gradient": gradient,
        "boundary": boundary,
        "residual_indicator": residual_indicator,
    }


def _top_mask(values: np.ndarray, quantile: float = 0.90) -> np.ndarray:
    return values >= float(np.quantile(values, quantile))


def _uniform_grid_indices(field: dict[str, np.ndarray], budget: int) -> np.ndarray:
    x = field["x"]
    t = field["t"]
    side_t = max(2, int(round(math.sqrt(budget * 0.7))))
    side_x = max(2, int(math.ceil(budget / side_t)))
    target_t = np.linspace(0.0, 1.0, side_t)
    target_x = np.linspace(0.0, 1.0, side_x)
    chosen: list[int] = []
    used: set[int] = set()
    for tv in target_t:
        for xv in target_x:
            score = (x - xv) ** 2 + (t - tv) ** 2
            for index in np.argsort(score):
                idx = int(index)
                if idx not in used:
                    used.add(idx)
                    chosen.append(idx)
                    break
            if len(chosen) >= budget:
                return np.asarray(sorted(chosen), dtype=int)
    return np.asarray(sorted(chosen[:budget]), dtype=int)


def _failure_informed_indices(
    field: dict[str, np.ndarray],
    *,
    budget: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    residual = field["residual_indicator"] / np.max(field["residual_indicator"])
    gradient = field["gradient"] / np.max(field["gradient"])
    hot = field["temperature"] / np.max(field["temperature"])
    boundary = field["boundary"]
    selected: set[int] = set()
    quotas = [
        (_top_mask(field["residual_indicator"], 0.90), residual + 0.2 * gradient, int(0.36 * budget)),
        (_top_mask(field["temperature"], 0.90), hot + 0.2 * residual, int(0.22 * budget)),
        (_top_mask(field["gradient"], 0.90), gradient + 0.2 * residual, int(0.22 * budget)),
        (boundary > 0, boundary + 0.05, int(0.10 * budget)),
    ]
    for mask, weights, count in quotas:
        candidates = np.where(mask)[0]
        available = np.asarray([idx for idx in candidates if int(idx) not in selected], dtype=int)
        if len(available) == 0 or count <= 0:
            continue
        local_weights = weights[available] + 1e-12
        take = min(count, len(available))
        chosen = rng.choice(
            available,
            size=take,
            replace=False,
            p=local_weights / float(np.sum(local_weights)),
        )
        selected.update(int(idx) for idx in chosen)
    fill_count = budget - len(selected)
    if fill_count > 0:
        weights = residual + gradient + hot + 0.08 * boundary + 0.02
        available = np.asarray([idx for idx in range(len(weights)) if idx not in selected], dtype=int)
        local_weights = weights[available]
        chosen = rng.choice(
            available,
            size=fill_count,
            replace=False,
            p=local_weights / float(np.sum(local_weights)),
        )
        selected.update(int(idx) for idx in chosen)
    return np.asarray(sorted(selected), dtype=int)


def sensor_points(seed: int, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    candidates = collocation_field()
    temp = candidates["temperature"]
    hot_mask = temp >= np.quantile(temp, 0.88)
    non_hot = np.where(~hot_mask)[0]
    hot = np.where(hot_mask)[0]
    hot_count = max(8, count // 7)
    non_hot_count = count - hot_count
    selected = [
        *rng.choice(non_hot, size=non_hot_count, replace=False).tolist(),
        *rng.choice(hot, size=hot_count, replace=False).tolist(),
    ]
    selected = np.asarray(sorted(int(idx) for idx in selected), dtype=int)
    y = candidates["temperature"][selected] + rng.normal(0.0, 0.006, size=len(selected))
    return candidates["x"][selected], candidates["t"][selected], y


def grid_points(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if split == "val":
        x_values = np.linspace(0.015, 0.985, 38)
        t_values = np.linspace(0.025, 0.975, 28)
    elif split == "test":
        x_values = np.linspace(0.006, 0.994, 41)
        t_values = np.linspace(0.012, 0.988, 31)
    else:
        raise ValueError(f"Unknown split: {split}")
    t_grid, x_grid = np.meshgrid(t_values, x_values, indexing="ij")
    x = x_grid.ravel()
    t = t_grid.ravel()
    return x, t, true_temperature_np(x, t), gradient_score_np(x, t)


def build_variant_rows() -> list[dict[str, Any]]:
    return [
        {
            "variant_id": "data_only_tiny_mlp_no_residual",
            "family": "control",
            "sampler_id": "none",
            "residual_weight": 0.0,
            "source_mode": "none",
            "is_control": True,
            "description": "tiny MLP fit to sparse sensors without PDE residual",
        },
        {
            "variant_id": "uniform_grid_pinn",
            "family": "control",
            "sampler_id": "uniform_grid_control",
            "residual_weight": 0.04,
            "source_mode": "true_source",
            "is_control": True,
            "description": "same tiny PINN budget with uniform collocation points",
        },
        {
            "variant_id": "wrong_prior_failure_sampler_control",
            "family": "control",
            "sampler_id": "failure_informed_hot_gradient",
            "residual_weight": 0.04,
            "source_mode": "wrong_source_prior",
            "is_control": True,
            "description": "failure-informed points with deliberately shifted heat-source prior",
        },
        {
            "variant_id": "failure_informed_hot_gradient_pinn",
            "family": "adaptive_candidate",
            "sampler_id": "failure_informed_hot_gradient",
            "residual_weight": 0.04,
            "source_mode": "true_source",
            "is_control": False,
            "description": "Phase 165 sampler with the same tiny PINN and true synthetic source",
        },
    ]


def _source_params(source_mode: str) -> dict[str, float]:
    if source_mode == "wrong_source_prior":
        return {"center_shift": -0.055, "width_scale": 1.34, "amplitude_scale": 0.78}
    return {"center_shift": 0.0, "width_scale": 1.0, "amplitude_scale": 1.0}


def _import_torch() -> Any:
    import torch

    return torch


def _true_temperature_torch(torch: Any, x: Any, t: Any, *, source_mode: str = "true_source") -> Any:
    params = _source_params(source_mode)
    center = 0.18 + 0.58 * t + 0.025 * torch.sin(2.0 * math.pi * t) + params["center_shift"]
    width = (0.048 + 0.018 * t) * params["width_scale"]
    gaussian = torch.exp(-0.5 * ((x - center) / width) ** 2)
    return 0.12 + 0.05 * x + 0.08 * t + params["amplitude_scale"] * 0.92 * gaussian


def _source_torch(torch: Any, x: Any, t: Any, *, diffusivity: float, source_mode: str) -> Any:
    xt = torch.cat([x, t], dim=1).detach().clone().requires_grad_(True)
    xx = xt[:, :1]
    tt = xt[:, 1:2]
    temp = _true_temperature_torch(torch, xx, tt, source_mode=source_mode)
    grad = torch.autograd.grad(
        temp,
        xt,
        grad_outputs=torch.ones_like(temp),
        create_graph=True,
        retain_graph=True,
    )[0]
    temp_t = grad[:, 1:2]
    temp_x = grad[:, :1]
    grad_x = torch.autograd.grad(
        temp_x,
        xt,
        grad_outputs=torch.ones_like(temp_x),
        create_graph=True,
        retain_graph=True,
    )[0]
    temp_xx = grad_x[:, :1]
    return (temp_t - diffusivity * temp_xx).detach()


def _model_residual(torch: Any, model: Any, x: Any, t: Any, source: Any, *, diffusivity: float) -> Any:
    xt = torch.cat([x, t], dim=1).detach().clone().requires_grad_(True)
    pred = model(xt)
    grad = torch.autograd.grad(
        pred,
        xt,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
        retain_graph=True,
    )[0]
    pred_t = grad[:, 1:2]
    pred_x = grad[:, :1]
    grad_x = torch.autograd.grad(
        pred_x,
        xt,
        grad_outputs=torch.ones_like(pred_x),
        create_graph=True,
        retain_graph=True,
    )[0]
    pred_xx = grad_x[:, :1]
    return pred_t - diffusivity * pred_xx - source


def _rmse_np(pred: np.ndarray, true: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean((pred - true) ** 2))))


def _selection_score(row: dict[str, float]) -> float:
    return (
        row["temperature_rmse"]
        + 0.25 * row["hot_q90_rmse"]
        + 0.15 * row["gradient_q90_rmse"]
        + 0.015 * row["residual_rmse"]
    )


def train_one_variant(
    variant: dict[str, Any],
    *,
    seed: int,
    steps: int,
    sensor_count: int,
    collocation_count: int,
    device: str,
) -> list[dict[str, Any]]:
    torch = _import_torch()
    from gnnpinn.models.pinn.coordinate_networks import MLP, MLPConfig

    torch.manual_seed(seed)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    x_train, t_train, y_train = sensor_points(seed=seed, count=sensor_count)
    field = collocation_field()
    if variant["sampler_id"] == "uniform_grid_control":
        indices = _uniform_grid_indices(field, collocation_count)
    elif variant["sampler_id"] == "failure_informed_hot_gradient":
        indices = _failure_informed_indices(field, budget=collocation_count, seed=seed)
    else:
        indices = np.asarray([], dtype=int)

    model = MLP(
        MLPConfig(input_dim=2, output_dim=1, hidden_dim=32, num_hidden_layers=2, activation="tanh")
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0025)
    mse = torch.nn.MSELoss()
    x_tensor = torch.tensor(x_train[:, None], dtype=torch.float32, device=device)
    t_tensor = torch.tensor(t_train[:, None], dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_train[:, None], dtype=torch.float32, device=device)
    train_input = torch.cat([x_tensor, t_tensor], dim=1)
    if len(indices):
        colloc_x = torch.tensor(field["x"][indices, None], dtype=torch.float32, device=device)
        colloc_t = torch.tensor(field["t"][indices, None], dtype=torch.float32, device=device)
        source = _source_torch(
            torch,
            colloc_x,
            colloc_t,
            diffusivity=0.034,
            source_mode=str(variant["source_mode"]),
        )
    else:
        colloc_x = colloc_t = source = None

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(train_input)
        loss = mse(prediction, y_tensor)
        if float(variant["residual_weight"]) > 0.0 and colloc_x is not None and colloc_t is not None:
            residual = _model_residual(
                torch,
                model,
                colloc_x,
                colloc_t,
                source,
                diffusivity=0.034,
            )
            loss = loss + float(variant["residual_weight"]) * torch.mean(residual**2)
        loss.backward()
        optimizer.step()

    rows: list[dict[str, Any]] = []
    for split in ("val", "test"):
        x_eval, t_eval, y_eval, grad_eval = grid_points(split)
        eval_input = torch.tensor(
            np.column_stack([x_eval, t_eval]),
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            pred = model(eval_input).detach().cpu().numpy().reshape(-1)
        eval_x = torch.tensor(x_eval[:, None], dtype=torch.float32, device=device)
        eval_t = torch.tensor(t_eval[:, None], dtype=torch.float32, device=device)
        eval_source = _source_torch(
            torch,
            eval_x,
            eval_t,
            diffusivity=0.034,
            source_mode="true_source",
        )
        residual = _model_residual(
            torch,
            model,
            eval_x,
            eval_t,
            eval_source,
            diffusivity=0.034,
        )
        residual_rmse = float(
            math.sqrt(float(torch.mean(residual.detach().cpu() ** 2).item()))
        )
        hot_mask = y_eval >= float(np.quantile(y_eval, 0.90))
        gradient_mask = grad_eval >= float(np.quantile(grad_eval, 0.90))
        metric_payload = {
            "temperature_rmse": _rmse_np(pred, y_eval),
            "hot_q90_rmse": _rmse_np(pred[hot_mask], y_eval[hot_mask]),
            "gradient_q90_rmse": _rmse_np(pred[gradient_mask], y_eval[gradient_mask]),
            "residual_rmse": residual_rmse,
        }
        rows.append(
            {
                "variant_id": variant["variant_id"],
                "seed": seed,
                "split": split,
                "steps": steps,
                "sensor_count": sensor_count,
                "collocation_count": len(indices),
                **metric_payload,
                "selection_score": _selection_score(metric_payload),
            }
        )
    return rows


def build_summary_rows(
    variant_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    variants = {row["variant_id"]: row for row in variant_rows}
    output: list[dict[str, Any]] = []
    for variant_id in sorted({row["variant_id"] for row in run_rows}):
        for split in ("val", "test"):
            subset = [row for row in run_rows if row["variant_id"] == variant_id and row["split"] == split]
            if not subset:
                continue
            variant = variants[variant_id]
            output.append(
                {
                    "variant_id": variant_id,
                    "family": variant["family"],
                    "sampler_id": variant["sampler_id"],
                    "split": split,
                    "seed_count": len(subset),
                    "temperature_rmse_mean": mean(row["temperature_rmse"] for row in subset),
                    "temperature_rmse_std": pstdev(row["temperature_rmse"] for row in subset),
                    "hot_q90_rmse_mean": mean(row["hot_q90_rmse"] for row in subset),
                    "gradient_q90_rmse_mean": mean(row["gradient_q90_rmse"] for row in subset),
                    "residual_rmse_mean": mean(row["residual_rmse"] for row in subset),
                    "selection_score_mean": mean(row["selection_score"] for row in subset),
                    "selection_score_std": pstdev(row["selection_score"] for row in subset),
                }
            )
    return output


def _summary_lookup(rows: list[dict[str, Any]], variant_id: str, split: str) -> dict[str, Any]:
    for row in rows:
        if row["variant_id"] == variant_id and row["split"] == split:
            return row
    raise KeyError((variant_id, split))


def build_gate(
    *,
    phase166_gate: dict[str, Any],
    variant_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    local_torch_blocked: bool,
) -> dict[str, Any]:
    phase166_ready = (
        phase166_gate.get("status")
        == "phase166_low_budget_pinn_smoke_design_ready_phase167_local_smoke"
        and _is_true(phase166_gate.get("phase167_local_low_budget_pinn_smoke_allowed"))
    )
    val_rows = [row for row in summary_rows if row["split"] == "val"]
    selected = min(val_rows, key=lambda row: float(row["selection_score_mean"]))
    control_rows = [row for row in val_rows if row["family"] == "control"]
    best_control = min(control_rows, key=lambda row: float(row["selection_score_mean"]))
    selected_test = _summary_lookup(summary_rows, selected["variant_id"], "test")
    best_control_test = _summary_lookup(summary_rows, best_control["variant_id"], "test")
    adaptive_selected = selected["family"] == "adaptive_candidate"
    validation_relative_gain = (
        (float(best_control["selection_score_mean"]) - float(selected["selection_score_mean"]))
        / max(float(best_control["selection_score_mean"]), 1e-12)
    )
    test_relative_gain = (
        (float(best_control_test["selection_score_mean"]) - float(selected_test["selection_score_mean"]))
        / max(float(best_control_test["selection_score_mean"]), 1e-12)
    )
    test_reversal_ratio = float(selected_test["selection_score_mean"]) / max(
        float(best_control_test["selection_score_mean"]),
        1e-12,
    )
    stability_ok = float(selected["selection_score_std"]) <= 0.06
    variant_count_ok = len(variant_rows) >= 4
    pass_gate = (
        phase166_ready
        and adaptive_selected
        and validation_relative_gain >= 0.015
        and test_reversal_ratio <= 1.05
        and stability_ok
        and variant_count_ok
    )
    blockers: list[str] = []
    if not phase166_ready:
        blockers.append("phase166_gate_not_ready")
    if not adaptive_selected:
        blockers.append("validation_selected_control_variant")
    if validation_relative_gain < 0.015:
        blockers.append("validation_gain_vs_best_control")
    if test_reversal_ratio > 1.05:
        blockers.append("test_reversal_vs_best_control")
    if not stability_ok:
        blockers.append("seed_stability_guard")
    if not variant_count_ok:
        blockers.append("missing_required_controls")
    return {
        "status": (
            "phase167_low_budget_pinn_smoke_ready_phase168_focused_review"
            if pass_gate
            else "phase167_low_budget_pinn_smoke_closed_no_stable_model_gain"
        ),
        "selected_variant": selected["variant_id"],
        "best_control_variant": best_control["variant_id"],
        "validation_relative_gain_vs_best_control": validation_relative_gain,
        "test_relative_gain_vs_best_control": test_relative_gain,
        "test_reversal_ratio_vs_best_control": test_reversal_ratio,
        "selected_validation_score_mean": selected["selection_score_mean"],
        "selected_test_score_mean": selected_test["selection_score_mean"],
        "selected_validation_score_std": selected["selection_score_std"],
        "local_torch_blocked": local_torch_blocked,
        "phase167_low_budget_training_executed": True,
        "phase168_focused_review_allowed": bool(pass_gate),
        "phase167_model_mechanism_allowed": False,
        "phase167_model_claim_allowed": False,
        "phase167_bayesian_pinn_training_executed": False,
        "bayesian_pinn_training_allowed_now": False,
        "adaptive_sampling_training_allowed_now": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "blocking_audits": blockers,
        "next_action": (
            "enter Phase 168 focused low-budget validation design; still no AM-Bench claim"
            if pass_gate
            else "close or redesign the tiny PINN smoke before further model work"
        ),
    }


def build_markdown(
    *,
    gate: dict[str, Any],
    summary_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Phase 167 Low-Budget PINN Smoke",
        "",
        "## Gate",
        f"- Status: `{gate['status']}`",
        f"- Selected variant: `{gate['selected_variant']}`",
        f"- Best control variant: `{gate['best_control_variant']}`",
        f"- Validation relative gain vs best control: `{_csv_value(gate['validation_relative_gain_vs_best_control'])}`",
        f"- Test reversal ratio vs best control: `{_csv_value(gate['test_reversal_ratio_vs_best_control'])}`",
        f"- Phase 168 focused review allowed: `{_csv_value(gate['phase168_focused_review_allowed'])}`",
        f"- Model claim allowed: `{_csv_value(gate['phase167_model_claim_allowed'])}`",
        f"- A100 training allowed now: `{_csv_value(gate['a100_training_allowed_now'])}`",
        f"- A100-SXM4-80GB request now: `{_csv_value(gate['a100_80gb_request_now'])}`",
        "",
        "## Interpretation",
        (
            "This is a tiny synthetic training smoke. It can only support a later "
            "focused validation phase if the adaptive sampler PINN beats equal-budget "
            "controls. It does not support AM-Bench, Bayesian PINN, GCN, CNN/operator, "
            "or A100-SXM4-80GB claims."
        ),
        "",
        "## Summary Metrics",
        *_markdown_table(summary_rows, SUMMARY_FIELDS),
        "",
    ]
    return "\n".join(lines)


def _markdown_table(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[str]:
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    body = [
        "| " + " | ".join(_csv_value(row.get(field, "")) for field in fields)
        + " |"
        for row in rows
    ]
    return [header, sep, *body]


def build_package(
    *,
    root: Path,
    output_dir: Path,
    phase_inputs: dict[str, Path],
    steps: int,
    seeds: tuple[int, ...],
    sensor_count: int,
    collocation_count: int,
    device: str,
    local_torch_blocked: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    output_dir = output_dir if output_dir.is_absolute() else root / output_dir
    resolved = {
        name: path if path.is_absolute() else root / path for name, path in phase_inputs.items()
    }
    phase166_gate = _read_json(resolved["phase166_gate"])
    variant_rows = build_variant_rows()
    run_rows: list[dict[str, Any]] = []
    for variant in variant_rows:
        for seed in seeds:
            run_rows.extend(
                train_one_variant(
                    variant,
                    seed=seed,
                    steps=steps,
                    sensor_count=sensor_count,
                    collocation_count=collocation_count,
                    device=device,
                )
            )
    summary_rows = build_summary_rows(variant_rows, run_rows)
    gate = build_gate(
        phase166_gate=phase166_gate,
        variant_rows=variant_rows,
        summary_rows=summary_rows,
        local_torch_blocked=local_torch_blocked,
    )

    variant_path = output_dir / "phase167_variant_table.csv"
    run_path = output_dir / "phase167_run_metric_table.csv"
    summary_path = output_dir / "phase167_variant_summary_table.csv"
    gate_path = output_dir / "phase167_low_budget_pinn_smoke_gate.json"
    markdown_path = output_dir / "phase167_low_budget_pinn_smoke.md"
    manifest_path = output_dir / "phase167_low_budget_pinn_smoke_manifest.json"

    _write_csv(variant_path, variant_rows, VARIANT_FIELDS)
    _write_csv(run_path, run_rows, RUN_FIELDS)
    _write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    _write_json(gate_path, gate)
    with markdown_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(build_markdown(gate=gate, summary_rows=summary_rows))

    manifest = {
        "phase": 167,
        "description": "low-budget synthetic PINN smoke with adaptive sampler control",
        "inputs": {name: _display_path(path, root) for name, path in resolved.items()},
        "outputs": {
            "variant_table": _display_path(variant_path, root),
            "run_metric_table": _display_path(run_path, root),
            "summary_table": _display_path(summary_path, root),
            "gate": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "settings": {
            "steps": steps,
            "seeds": list(seeds),
            "sensor_count": sensor_count,
            "collocation_count": collocation_count,
            "device": device,
            "raw_data_used": False,
        },
        "counts": {
            "variant_rows": len(variant_rows),
            "run_rows": len(run_rows),
            "summary_rows": len(summary_rows),
        },
        "gate": gate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--steps", type=int, default=650)
    parser.add_argument("--seed", action="append", type=int, dest="seeds", default=[])
    parser.add_argument("--sensor-count", type=int, default=72)
    parser.add_argument("--collocation-count", type=int, default=192)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--local-torch-blocked", action="store_true")
    for name, default in PHASE_INPUTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=default)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(args.seeds or [167, 168, 169])
    phase_inputs = {name: getattr(args, name) for name in PHASE_INPUTS}
    manifest = build_package(
        root=args.root,
        output_dir=args.output_dir,
        phase_inputs=phase_inputs,
        steps=args.steps,
        seeds=seeds,
        sensor_count=args.sensor_count,
        collocation_count=args.collocation_count,
        device=args.device,
        local_torch_blocked=args.local_torch_blocked,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
