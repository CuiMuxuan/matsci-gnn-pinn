#!/usr/bin/env python3
"""Build Phase 160 low-capacity mechanism gate for UCI concrete strength.

This phase tests interpretable concrete-mechanism feature families against the
Phase 159 HistGradientBoosting strong-baseline guard. It is a no-training gate:
a positive result may only open a later focused validation step.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_PHASE159_DIR = Path("docs/results/phase159_uci_concrete_focused_review")
DEFAULT_RAW_PATH = Path(
    "data/raw/external/phase158_uci_concrete/concrete_compressive_strength.zip"
)
DEFAULT_OUTPUT_DIR = Path("docs/results/phase160_uci_concrete_low_capacity_mechanism_gate")

MIN_RELATIVE_PHASE159_VAL_GAIN = 0.01
MAX_PHASE159_TEST_REVERSAL_RATIO = 1.05
MAX_LOW_CAPACITY_FEATURES = 36


def _load_phase158_module():
    script = Path(__file__).with_name("build_phase158_uci_concrete_baseline_gate.py")
    spec = importlib.util.spec_from_file_location("phase158_helpers", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Phase 158 helper module from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


p158 = _load_phase158_module()


class ModelSpec:
    def __init__(self, model: str, alpha: float | None = None, l1_ratio: float | None = None) -> None:
        self.model = model
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    @property
    def label(self) -> str:
        parts = [self.model]
        if self.alpha is not None:
            parts.append(f"alpha={self.alpha:g}")
        if self.l1_ratio is not None:
            parts.append(f"l1_ratio={self.l1_ratio:g}")
        return ";".join(parts)


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("ordinary_least_squares"),
    *(ModelSpec("ridge", alpha=value) for value in (1e-4, 1e-2, 1.0, 10.0, 100.0)),
    *(ModelSpec("lasso", alpha=value) for value in (1e-4, 1e-3, 1e-2, 0.1, 1.0)),
    *(
        ModelSpec("elastic_net", alpha=alpha, l1_ratio=l1_ratio)
        for alpha in (1e-3, 1e-2, 0.1, 1.0)
        for l1_ratio in (0.2, 0.5, 0.8)
    ),
    *(ModelSpec("huber", alpha=value) for value in (1e-5, 1e-3, 0.1)),
)

MECHANISM_PROFILES: dict[str, tuple[str, ...]] = {
    "abrams_age_core": (
        "water_binder_ratio",
        "phase160_inverse_water_binder_ratio",
        "phase160_effective_water_binder_ratio",
        "phase160_water_cement_ratio",
        "log_age_day",
        "sqrt_age_day",
        "phase160_log_age_squared",
        "phase160_water_binder_squared",
        "phase160_water_binder_age_interaction",
        "phase160_effective_wb_age_interaction",
    ),
    "binder_composition_core": (
        "binder_kg_m3",
        "scm_kg_m3",
        "cement_binder_ratio",
        "slag_binder_ratio",
        "fly_ash_binder_ratio",
        "scm_fraction",
        "has_slag",
        "has_fly_ash",
        "phase160_effective_binder_kg_m3",
        "phase160_cement_age_interaction",
        "phase160_slag_age_interaction",
        "phase160_fly_ash_age_interaction",
        "phase160_scm_age_interaction",
    ),
    "paste_aggregate_packing": (
        "aggregate_kg_m3",
        "paste_kg_m3",
        "total_mix_kg_m3",
        "aggregate_binder_ratio",
        "coarse_fine_ratio",
        "superplasticizer_binder_ratio",
        "phase160_paste_aggregate_ratio",
        "phase160_water_paste_ratio",
        "phase160_binder_total_fraction",
        "phase160_water_total_fraction",
        "phase160_aggregate_total_fraction",
        "phase160_superplasticizer_water_interaction",
    ),
    "hydration_interactions": (
        "water_binder_ratio",
        "phase160_effective_water_binder_ratio",
        "superplasticizer_binder_ratio",
        "cement_binder_ratio",
        "slag_binder_ratio",
        "fly_ash_binder_ratio",
        "scm_fraction",
        "log_age_day",
        "sqrt_age_day",
        "phase160_water_binder_age_interaction",
        "phase160_effective_wb_age_interaction",
        "phase160_cement_age_interaction",
        "phase160_slag_age_interaction",
        "phase160_fly_ash_age_interaction",
        "phase160_scm_age_interaction",
        "phase160_binder_age_log_product",
        "phase160_wb_scm_age_interaction",
    ),
    "mechanism_compact": (
        "water_binder_ratio",
        "phase160_effective_water_binder_ratio",
        "log_age_day",
        "sqrt_age_day",
        "binder_kg_m3",
        "cement_binder_ratio",
        "slag_binder_ratio",
        "fly_ash_binder_ratio",
        "scm_fraction",
        "superplasticizer_binder_ratio",
        "aggregate_binder_ratio",
        "coarse_fine_ratio",
        "phase160_water_binder_age_interaction",
        "phase160_scm_age_interaction",
        "phase160_paste_aggregate_ratio",
        "phase160_water_paste_ratio",
    ),
    "mechanism_full_low_capacity": (
        "cement_kg_m3",
        "blast_furnace_slag_kg_m3",
        "fly_ash_kg_m3",
        "water_kg_m3",
        "superplasticizer_kg_m3",
        "coarse_aggregate_kg_m3",
        "fine_aggregate_kg_m3",
        "age_day",
        "binder_kg_m3",
        "scm_kg_m3",
        "aggregate_kg_m3",
        "paste_kg_m3",
        "total_mix_kg_m3",
        "water_binder_ratio",
        "superplasticizer_binder_ratio",
        "slag_binder_ratio",
        "fly_ash_binder_ratio",
        "cement_binder_ratio",
        "aggregate_binder_ratio",
        "coarse_fine_ratio",
        "log_age_day",
        "sqrt_age_day",
        "has_slag",
        "has_fly_ash",
        "has_superplasticizer",
        "scm_fraction",
        "phase160_effective_binder_kg_m3",
        "phase160_effective_water_binder_ratio",
        "phase160_water_cement_ratio",
        "phase160_paste_aggregate_ratio",
        "phase160_water_paste_ratio",
        "phase160_water_binder_age_interaction",
        "phase160_scm_age_interaction",
        "phase160_cement_age_interaction",
        "phase160_slag_age_interaction",
        "phase160_fly_ash_age_interaction",
    ),
}

FEATURE_EXPRESSIONS = {
    "phase160_inverse_water_binder_ratio": "1 / water_binder_ratio",
    "phase160_effective_binder_kg_m3": "cement + 0.75 * slag + 0.55 * fly_ash",
    "phase160_effective_water_binder_ratio": "water / effective_binder",
    "phase160_water_cement_ratio": "water / cement",
    "phase160_cement_water_ratio": "cement / water",
    "phase160_log_age_squared": "log1p(age_day)^2",
    "phase160_water_binder_squared": "water_binder_ratio^2",
    "phase160_water_binder_age_interaction": "water_binder_ratio * log1p(age_day)",
    "phase160_effective_wb_age_interaction": "effective_water_binder_ratio * log1p(age_day)",
    "phase160_cement_age_interaction": "cement_binder_ratio * log1p(age_day)",
    "phase160_slag_age_interaction": "slag_binder_ratio * log1p(age_day)",
    "phase160_fly_ash_age_interaction": "fly_ash_binder_ratio * log1p(age_day)",
    "phase160_scm_age_interaction": "scm_fraction * log1p(age_day)",
    "phase160_binder_age_log_product": "binder_kg_m3 * log1p(age_day)",
    "phase160_wb_scm_age_interaction": "water_binder_ratio * scm_fraction * log1p(age_day)",
    "phase160_paste_aggregate_ratio": "paste_kg_m3 / aggregate_kg_m3",
    "phase160_water_paste_ratio": "water_kg_m3 / paste_kg_m3",
    "phase160_binder_total_fraction": "binder_kg_m3 / total_mix_kg_m3",
    "phase160_water_total_fraction": "water_kg_m3 / total_mix_kg_m3",
    "phase160_aggregate_total_fraction": "aggregate_kg_m3 / total_mix_kg_m3",
    "phase160_coarse_total_fraction": "coarse_aggregate_kg_m3 / total_mix_kg_m3",
    "phase160_fine_total_fraction": "fine_aggregate_kg_m3 / total_mix_kg_m3",
    "phase160_superplasticizer_water_interaction": "superplasticizer_binder_ratio * water_binder_ratio",
}

FEATURE_SCHEMA_FIELDS = (
    "feature",
    "source",
    "mechanism_role",
    "expression",
    "profiles",
    "min",
    "max",
    "std",
)
METRIC_FIELDS = (
    "target",
    "profile",
    "model",
    "model_label",
    "alpha",
    "l1_ratio",
    "split",
    "n_rows",
    "feature_count",
    "rmse",
    "mae",
    "r2",
)
CANDIDATE_FIELDS = (
    "target",
    "profile",
    "model",
    "model_label",
    "alpha",
    "l1_ratio",
    "feature_count",
    "train_rmse",
    "val_rmse",
    "test_rmse",
    "phase159_guard_profile",
    "phase159_guard_method",
    "phase159_guard_val_rmse",
    "phase159_guard_test_rmse",
    "val_gain_vs_phase159",
    "relative_val_gain_vs_phase159",
    "test_gain_vs_phase159",
    "test_reversal_ratio_vs_phase159",
    "clears_phase159_validation_guard",
    "clears_phase159_test_guard",
    "selected_low_capacity",
)
COEFFICIENT_FIELDS = (
    "target",
    "selected_low_capacity",
    "profile",
    "model",
    "model_label",
    "feature",
    "coefficient_standardized",
    "coefficient_raw_scale",
    "abs_coefficient_standardized",
    "coefficient_rank",
    "intercept_raw_scale",
)
AUDIT_FIELDS = (
    "audit",
    "status",
    "severity",
    "value",
    "threshold",
    "reason",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _csv_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.12g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _display_path(path: Path, root: Path | None = None) -> str:
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    safe = denominator.replace(0.0, np.nan)
    return (numerator / safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_mechanism_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    out = df.copy()
    cement = out["cement_kg_m3"]
    slag = out["blast_furnace_slag_kg_m3"]
    fly_ash = out["fly_ash_kg_m3"]
    water = out["water_kg_m3"]
    aggregate = out["aggregate_kg_m3"]
    paste = out["paste_kg_m3"]
    total = out["total_mix_kg_m3"]
    log_age = out["log_age_day"]
    effective_binder = cement + 0.75 * slag + 0.55 * fly_ash

    out["phase160_inverse_water_binder_ratio"] = _safe_divide(
        pd.Series(np.ones(len(out), dtype=float), index=out.index),
        out["water_binder_ratio"],
    )
    out["phase160_effective_binder_kg_m3"] = effective_binder
    out["phase160_effective_water_binder_ratio"] = _safe_divide(water, effective_binder)
    out["phase160_water_cement_ratio"] = _safe_divide(water, cement)
    out["phase160_cement_water_ratio"] = _safe_divide(cement, water)
    out["phase160_log_age_squared"] = log_age * log_age
    out["phase160_water_binder_squared"] = out["water_binder_ratio"] * out["water_binder_ratio"]
    out["phase160_water_binder_age_interaction"] = out["water_binder_ratio"] * log_age
    out["phase160_effective_wb_age_interaction"] = (
        out["phase160_effective_water_binder_ratio"] * log_age
    )
    out["phase160_cement_age_interaction"] = out["cement_binder_ratio"] * log_age
    out["phase160_slag_age_interaction"] = out["slag_binder_ratio"] * log_age
    out["phase160_fly_ash_age_interaction"] = out["fly_ash_binder_ratio"] * log_age
    out["phase160_scm_age_interaction"] = out["scm_fraction"] * log_age
    out["phase160_binder_age_log_product"] = out["binder_kg_m3"] * log_age
    out["phase160_wb_scm_age_interaction"] = (
        out["water_binder_ratio"] * out["scm_fraction"] * log_age
    )
    out["phase160_paste_aggregate_ratio"] = _safe_divide(paste, aggregate)
    out["phase160_water_paste_ratio"] = _safe_divide(water, paste)
    out["phase160_binder_total_fraction"] = _safe_divide(out["binder_kg_m3"], total)
    out["phase160_water_total_fraction"] = _safe_divide(water, total)
    out["phase160_aggregate_total_fraction"] = _safe_divide(aggregate, total)
    out["phase160_coarse_total_fraction"] = _safe_divide(out["coarse_aggregate_kg_m3"], total)
    out["phase160_fine_total_fraction"] = _safe_divide(out["fine_aggregate_kg_m3"], total)
    out["phase160_superplasticizer_water_interaction"] = (
        out["superplasticizer_binder_ratio"] * out["water_binder_ratio"]
    )

    feature_to_profiles: dict[str, list[str]] = {}
    for profile, features in MECHANISM_PROFILES.items():
        for feature in features[:MAX_LOW_CAPACITY_FEATURES]:
            feature_to_profiles.setdefault(feature, []).append(profile)

    rows: list[dict[str, Any]] = []
    for feature, profiles in sorted(feature_to_profiles.items()):
        values = (
            pd.to_numeric(out[feature], errors="coerce").fillna(0.0)
            if feature in out.columns
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "feature": feature,
                "source": "phase158_feature" if not feature.startswith("phase160_") else "phase160_derived",
                "mechanism_role": "low_capacity_concrete_mechanism_proxy",
                "expression": FEATURE_EXPRESSIONS.get(feature, feature),
                "profiles": ";".join(profiles),
                "min": float(values.min()) if len(values) else None,
                "max": float(values.max()) if len(values) else None,
                "std": float(values.std()) if len(values) else None,
            }
        )
    return out, rows


def _model(spec: ModelSpec):
    if spec.model == "ordinary_least_squares":
        estimator = LinearRegression()
    elif spec.model == "ridge":
        estimator = Ridge(alpha=float(spec.alpha), random_state=160)
    elif spec.model == "lasso":
        estimator = Lasso(alpha=float(spec.alpha), max_iter=20000, random_state=160)
    elif spec.model == "elastic_net":
        estimator = ElasticNet(
            alpha=float(spec.alpha),
            l1_ratio=float(spec.l1_ratio),
            max_iter=20000,
            random_state=160,
        )
    elif spec.model == "huber":
        estimator = HuberRegressor(alpha=float(spec.alpha), max_iter=1000)
    else:
        raise ValueError(f"Unknown model: {spec.model}")
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


def _assignments_from_phase159_split(phase159_dir: Path, df: pd.DataFrame) -> list[str]:
    split_manifest = _read_json(phase159_dir / "phase159_uci_concrete_split_manifest.json")
    registered = split_manifest["splits"]["phase158_registered_mix_design"]
    split_groups = {
        split: set(str(value) for value in registered["split_groups"][split])
        for split in ("train", "val", "test")
    }
    assignments: list[str] = []
    for group in df["mix_design_key"].fillna("unknown").astype(str):
        if group in split_groups["train"]:
            assignments.append("train")
        elif group in split_groups["val"]:
            assignments.append("val")
        elif group in split_groups["test"]:
            assignments.append("test")
        else:
            raise KeyError(f"Mix-design group missing from Phase 159 split manifest: {group}")
    return assignments


def _split_indices(assignments: list[str]) -> dict[str, np.ndarray]:
    return {
        split: np.array([idx for idx, label in enumerate(assignments) if label == split], dtype=int)
        for split in ("train", "val", "test")
    }


def evaluate_candidates(
    df: pd.DataFrame,
    assignments: list[str],
    *,
    phase159_gate: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    y = df[p158.TARGET_COLUMN].to_numpy(dtype=float)
    split_indices = _split_indices(assignments)
    guard_val = float(phase159_gate["registered_replay_validation_rmse"])
    guard_test = float(phase159_gate["registered_replay_test_rmse"])
    metric_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    fitted: list[dict[str, Any]] = []

    for profile, requested_features in MECHANISM_PROFILES.items():
        features = [feature for feature in requested_features if feature in df.columns]
        if len(features) > MAX_LOW_CAPACITY_FEATURES:
            features = features[:MAX_LOW_CAPACITY_FEATURES]
        if not features:
            continue
        x = df[features].to_numpy(dtype=float)
        for spec in MODEL_SPECS:
            model = _model(spec)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(x[split_indices["train"]], y[split_indices["train"]])
            split_metrics: dict[str, dict[str, float]] = {}
            for split, indices in split_indices.items():
                pred = model.predict(x[indices])
                metrics = _metrics(y[indices], pred)
                split_metrics[split] = metrics
                metric_rows.append(
                    {
                        "target": p158.TARGET_COLUMN,
                        "profile": profile,
                        "model": spec.model,
                        "model_label": spec.label,
                        "alpha": spec.alpha,
                        "l1_ratio": spec.l1_ratio,
                        "split": split,
                        "n_rows": int(len(indices)),
                        "feature_count": len(features),
                        **metrics,
                    }
                )
            val_rmse = split_metrics["val"]["rmse"]
            test_rmse = split_metrics["test"]["rmse"]
            val_gain = guard_val - val_rmse
            test_gain = guard_test - test_rmse
            relative_val_gain = val_gain / guard_val if guard_val else 0.0
            reversal_ratio = test_rmse / guard_test if guard_test else float("inf")
            clears_val = relative_val_gain >= MIN_RELATIVE_PHASE159_VAL_GAIN
            clears_test = reversal_ratio <= MAX_PHASE159_TEST_REVERSAL_RATIO
            row = {
                "target": p158.TARGET_COLUMN,
                "profile": profile,
                "model": spec.model,
                "model_label": spec.label,
                "alpha": spec.alpha,
                "l1_ratio": spec.l1_ratio,
                "feature_count": len(features),
                "train_rmse": split_metrics["train"]["rmse"],
                "val_rmse": val_rmse,
                "test_rmse": test_rmse,
                "phase159_guard_profile": phase159_gate.get("registered_replay_profile"),
                "phase159_guard_method": phase159_gate.get("registered_replay_method"),
                "phase159_guard_val_rmse": guard_val,
                "phase159_guard_test_rmse": guard_test,
                "val_gain_vs_phase159": val_gain,
                "relative_val_gain_vs_phase159": relative_val_gain,
                "test_gain_vs_phase159": test_gain,
                "test_reversal_ratio_vs_phase159": reversal_ratio,
                "clears_phase159_validation_guard": clears_val,
                "clears_phase159_test_guard": clears_test,
                "selected_low_capacity": False,
            }
            candidate_rows.append(row)
            fitted.append({"row": row, "model": model, "features": features})

    if not candidate_rows:
        return metric_rows, candidate_rows, []
    selected_index = min(
        range(len(candidate_rows)),
        key=lambda index: (
            candidate_rows[index]["val_rmse"],
            candidate_rows[index]["test_reversal_ratio_vs_phase159"],
            candidate_rows[index]["feature_count"],
        ),
    )
    candidate_rows[selected_index]["selected_low_capacity"] = True
    coefficient_rows = _coefficient_rows(
        fitted[selected_index]["model"],
        fitted[selected_index]["features"],
        candidate_rows[selected_index],
    )
    return metric_rows, candidate_rows, coefficient_rows


def _coefficient_rows(model, features: list[str], selected: dict[str, Any]) -> list[dict[str, Any]]:
    estimator = model.steps[-1][1]
    if not hasattr(estimator, "coef_"):
        return []
    scaler = model.named_steps["standardscaler"]
    coef = np.asarray(estimator.coef_, dtype=float).reshape(-1)
    scale = np.asarray(scaler.scale_, dtype=float)
    mean = np.asarray(scaler.mean_, dtype=float)
    raw_coef = coef / np.where(scale == 0.0, 1.0, scale)
    intercept = float(getattr(estimator, "intercept_", 0.0))
    raw_intercept = float(intercept - np.sum(coef * mean / np.where(scale == 0.0, 1.0, scale)))
    order = np.argsort(-np.abs(coef))
    ranks = {int(index): rank + 1 for rank, index in enumerate(order)}
    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(features):
        rows.append(
            {
                "target": selected["target"],
                "selected_low_capacity": True,
                "profile": selected["profile"],
                "model": selected["model"],
                "model_label": selected["model_label"],
                "feature": feature,
                "coefficient_standardized": float(coef[index]),
                "coefficient_raw_scale": float(raw_coef[index]),
                "abs_coefficient_standardized": float(abs(coef[index])),
                "coefficient_rank": ranks[index],
                "intercept_raw_scale": raw_intercept,
            }
        )
    return rows


def _audit_row(
    *,
    audit: str,
    status: str,
    severity: str,
    value: Any,
    threshold: Any,
    reason: str,
) -> dict[str, Any]:
    return {
        "audit": audit,
        "status": status,
        "severity": severity,
        "value": value,
        "threshold": threshold,
        "reason": reason,
    }


def build_audit_rows(
    *,
    phase159_gate: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    phase159_ready = (
        phase159_gate.get("status")
        == "phase159_uci_concrete_focused_review_ready_low_capacity_mechanism_gate"
        and bool(phase159_gate.get("phase159_model_mechanism_allowed"))
    )
    rows.append(
        _audit_row(
            audit="phase159_gate_consistency",
            status="pass" if phase159_ready else "block",
            severity="blocking" if not phase159_ready else "info",
            value=phase159_gate.get("status"),
            threshold="phase159_uci_concrete_focused_review_ready_low_capacity_mechanism_gate",
            reason="Phase 160 requires Phase 159 to allow a no-training mechanism gate",
        )
    )
    selected = next((row for row in candidate_rows if row.get("selected_low_capacity")), None)
    clears_val = bool(selected and selected.get("clears_phase159_validation_guard"))
    rows.append(
        _audit_row(
            audit="phase159_validation_guard_gain",
            status="pass" if clears_val else "block",
            severity="blocking" if not clears_val else "info",
            value=selected.get("relative_val_gain_vs_phase159") if selected else None,
            threshold=MIN_RELATIVE_PHASE159_VAL_GAIN,
            reason="selected low-capacity mechanism must beat the Phase 159 HGB guard on validation",
        )
    )
    clears_test = bool(selected and selected.get("clears_phase159_test_guard"))
    rows.append(
        _audit_row(
            audit="phase159_test_reversal_guard",
            status="pass" if clears_test else "block",
            severity="blocking" if not clears_test else "info",
            value=selected.get("test_reversal_ratio_vs_phase159") if selected else None,
            threshold=MAX_PHASE159_TEST_REVERSAL_RATIO,
            reason="selected low-capacity mechanism must not substantially reverse on test",
        )
    )
    feature_ok = bool(selected and int(selected.get("feature_count") or 0) <= MAX_LOW_CAPACITY_FEATURES)
    rows.append(
        _audit_row(
            audit="low_capacity_feature_budget",
            status="pass" if feature_ok else "block",
            severity="blocking" if not feature_ok else "info",
            value=selected.get("feature_count") if selected else None,
            threshold=MAX_LOW_CAPACITY_FEATURES,
            reason="mechanism candidate must stay low-capacity and interpretable",
        )
    )
    return rows


def build_gate(
    *,
    phase159_gate: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = next((row for row in candidate_rows if row.get("selected_low_capacity")), None)
    blocking = [row for row in audit_rows if row["severity"] == "blocking" and row["status"] == "block"]
    focused_allowed = not blocking
    status = (
        "phase160_uci_concrete_low_capacity_mechanism_ready_focused_validation"
        if focused_allowed
        else "phase160_uci_concrete_low_capacity_mechanism_closed_no_guarded_gain"
    )
    return {
        "status": status,
        "selected_target": p158.TARGET_COLUMN,
        "phase159_guard_profile": phase159_gate.get("registered_replay_profile"),
        "phase159_guard_method": phase159_gate.get("registered_replay_method"),
        "phase159_guard_validation_rmse": phase159_gate.get("registered_replay_validation_rmse"),
        "phase159_guard_test_rmse": phase159_gate.get("registered_replay_test_rmse"),
        "candidate_rows": len(candidate_rows),
        "selected_low_capacity_profile": selected.get("profile") if selected else None,
        "selected_low_capacity_model": selected.get("model") if selected else None,
        "selected_low_capacity_model_label": selected.get("model_label") if selected else None,
        "selected_low_capacity_feature_count": selected.get("feature_count") if selected else None,
        "selected_low_capacity_validation_rmse": selected.get("val_rmse") if selected else None,
        "selected_low_capacity_test_rmse": selected.get("test_rmse") if selected else None,
        "relative_validation_gain_vs_phase159": selected.get("relative_val_gain_vs_phase159") if selected else None,
        "test_reversal_ratio_vs_phase159": selected.get("test_reversal_ratio_vs_phase159") if selected else None,
        "blocking_audits": [row["audit"] for row in blocking],
        "phase160_focused_validation_allowed": focused_allowed,
        "phase160_model_mechanism_allowed": focused_allowed,
        "phase160_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": (
            "run Phase 161 focused validation before any neural training"
            if focused_allowed
            else "close Phase 160 low-capacity concrete mechanism route as diagnostic; do not train"
        ),
    }


def _markdown_table(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[str]:
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    body = [
        "| " + " | ".join(_csv_value(row.get(field)) for field in fields)
        + " |"
        for row in rows
    ]
    return [header, sep, *body]


def build_markdown(
    *,
    gate: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
) -> str:
    top_candidates = sorted(candidate_rows, key=lambda row: row["val_rmse"])[:8]
    candidate_fields = (
        "profile",
        "model_label",
        "feature_count",
        "val_rmse",
        "test_rmse",
        "relative_val_gain_vs_phase159",
        "test_reversal_ratio_vs_phase159",
        "selected_low_capacity",
    )
    audit_fields = ("audit", "status", "severity", "value", "threshold", "reason")
    lines = [
        "# Phase 160 UCI Concrete Low-Capacity Mechanism Gate",
        "",
        "## Gate",
        f"- Status: `{gate['status']}`",
        f"- Selected profile: `{gate['selected_low_capacity_profile']}`",
        f"- Selected model: `{gate['selected_low_capacity_model_label']}`",
        f"- Focused validation allowed: `{_csv_value(gate['phase160_focused_validation_allowed'])}`",
        f"- Model training allowed: `{_csv_value(gate['phase160_model_training_allowed'])}`",
        f"- A100 training allowed now: `{_csv_value(gate['a100_training_allowed_now'])}`",
        f"- A100-SXM4-80GB request now: `{_csv_value(gate['a100_80gb_request_now'])}`",
        "",
        "## Interpretation",
        (
            "This no-training gate asks whether interpretable concrete mechanism "
            "features can beat the Phase 159 HistGradientBoosting guard. A closure "
            "means the concrete source remains a robust source-level diagnostic, not "
            "a second-paper mechanism or model claim."
        ),
        "",
        "## Top Candidates",
        *_markdown_table(top_candidates, candidate_fields),
        "",
        "## Audits",
        *_markdown_table(audit_rows, audit_fields),
        "",
    ]
    return "\n".join(lines)


def _raw_path_from_phase159_manifest(phase159_dir: Path, root: Path) -> Path | None:
    manifest_path = phase159_dir / "phase159_uci_concrete_focused_review_manifest.json"
    if not manifest_path.exists():
        return None
    raw = _read_json(manifest_path).get("inputs", {}).get("raw_path")
    if not raw:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate if candidate.exists() else None


def build_package(
    *,
    root: Path,
    phase159_dir: Path,
    raw_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    root = root.resolve()
    phase159_dir = phase159_dir if phase159_dir.is_absolute() else root / phase159_dir
    raw_path = raw_path if raw_path.is_absolute() else root / raw_path
    output_dir = output_dir if output_dir.is_absolute() else root / output_dir

    if not raw_path.exists():
        manifest_raw_path = _raw_path_from_phase159_manifest(phase159_dir, root)
        if manifest_raw_path is not None:
            raw_path = manifest_raw_path

    phase159_gate = _read_json(phase159_dir / "phase159_uci_concrete_focused_review_gate.json")
    df = p158.load_concrete_table(raw_path)
    df, feature_rows = add_mechanism_features(df)
    assignments = _assignments_from_phase159_split(phase159_dir, df)
    metric_rows, candidate_rows, coefficient_rows = evaluate_candidates(
        df,
        assignments,
        phase159_gate=phase159_gate,
    )
    audit_rows = build_audit_rows(phase159_gate=phase159_gate, candidate_rows=candidate_rows)
    gate = build_gate(phase159_gate=phase159_gate, candidate_rows=candidate_rows, audit_rows=audit_rows)

    feature_schema_path = output_dir / "phase160_uci_concrete_feature_schema.csv"
    metric_table_path = output_dir / "phase160_uci_concrete_metric_table.csv"
    candidate_table_path = output_dir / "phase160_uci_concrete_candidate_table.csv"
    coefficient_table_path = output_dir / "phase160_uci_concrete_coefficient_table.csv"
    audit_table_path = output_dir / "phase160_uci_concrete_audit_table.csv"
    gate_path = output_dir / "phase160_uci_concrete_low_capacity_mechanism_gate.json"
    markdown_path = output_dir / "phase160_uci_concrete_low_capacity_mechanism_gate.md"
    manifest_path = output_dir / "phase160_uci_concrete_low_capacity_mechanism_manifest.json"

    _write_csv(feature_schema_path, feature_rows, FEATURE_SCHEMA_FIELDS)
    _write_csv(metric_table_path, metric_rows, METRIC_FIELDS)
    _write_csv(candidate_table_path, candidate_rows, CANDIDATE_FIELDS)
    _write_csv(coefficient_table_path, coefficient_rows, COEFFICIENT_FIELDS)
    _write_csv(audit_table_path, audit_rows, AUDIT_FIELDS)
    _write_json(gate_path, gate)
    with markdown_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(build_markdown(gate=gate, candidate_rows=candidate_rows, audit_rows=audit_rows))
    manifest = {
        "phase": 160,
        "description": "no-training low-capacity mechanism gate for UCI concrete strength",
        "inputs": {
            "phase159_dir": _display_path(phase159_dir, root),
            "raw_path": _display_path(raw_path, root),
        },
        "outputs": {
            "feature_schema": _display_path(feature_schema_path, root),
            "metric_table": _display_path(metric_table_path, root),
            "candidate_table": _display_path(candidate_table_path, root),
            "coefficient_table": _display_path(coefficient_table_path, root),
            "audit_table": _display_path(audit_table_path, root),
            "gate": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "field_rows": int(len(df)),
            "feature_rows": len(feature_rows),
            "candidate_rows": len(candidate_rows),
            "metric_rows": len(metric_rows),
            "coefficient_rows": len(coefficient_rows),
            "audit_rows": len(audit_rows),
        },
        "gate": gate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--phase159-dir", type=Path, default=DEFAULT_PHASE159_DIR)
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_package(
        root=args.root,
        phase159_dir=args.phase159_dir,
        raw_path=args.raw_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
