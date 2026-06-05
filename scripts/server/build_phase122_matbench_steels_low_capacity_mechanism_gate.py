#!/usr/bin/env python3
"""Build Phase 122 low-capacity mechanism gate for Matbench steels.

This phase consumes only the small Phase 120/121 Matbench steels artifacts. It
tests interpretable linear alloy-mechanism descriptors against the Phase 121
ExtraTrees strong-baseline guard. It does not train a neural model or open A100
training.
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


DEFAULT_PHASE120_DIR = Path("docs/results/phase120_matbench_steels_baseline_gate")
DEFAULT_PHASE121_DIR = Path("docs/results/phase121_matbench_steels_focused_review")
DEFAULT_OUTPUT_DIR = Path("docs/results/phase122_matbench_steels_low_capacity_mechanism_gate")
MIN_SPLIT_ROWS = 20
MIN_RELATIVE_PHASE121_VAL_GAIN = 0.01
MAX_PHASE121_TEST_REVERSAL_RATIO = 1.05
MAX_LOW_CAPACITY_FEATURES = 34


def _load_phase120_module():
    script = Path(__file__).with_name("build_phase120_matbench_steels_baseline_gate.py")
    spec = importlib.util.spec_from_file_location("phase120_helpers", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Phase 120 helper module from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase120 = _load_phase120_module()

ELEMENTS = tuple(phase120.ELEMENTS)
MINOR_ELEMENTS = ("C", "Mn", "Si", "V", "N", "Nb", "W", "Al", "Ti")
KEY_STRENGTH_ELEMENTS = ("C", "Ni", "Co", "Cr", "Mo", "V", "Nb", "W", "Al", "Ti")
PAIR_FEATURES = (
    ("C", "Cr", "carbon_chromium_precipitation"),
    ("C", "Mo", "carbon_molybdenum_precipitation"),
    ("C", "V", "carbon_vanadium_precipitation"),
    ("C", "Nb", "carbon_niobium_precipitation"),
    ("C", "W", "carbon_tungsten_precipitation"),
    ("C", "Ti", "carbon_titanium_precipitation"),
    ("Al", "Ti", "aluminum_titanium_precipitation"),
    ("Ni", "Ti", "nickel_titanium_precipitation"),
    ("Mo", "Ni", "molybdenum_nickel_solid_solution"),
    ("Cr", "Ni", "chromium_nickel_balance"),
    ("Co", "Ni", "cobalt_nickel_balance"),
    ("Cr", "Mo", "chromium_molybdenum_carbide_family"),
    ("C", "Ni", "carbon_nickel_interaction"),
    ("C", "Co", "carbon_cobalt_interaction"),
)
SOLID_SOLUTION_WEIGHTS = {
    "Ni": 1.00,
    "Co": 0.90,
    "Cr": 0.70,
    "Mo": 1.10,
    "Mn": 0.40,
    "Si": 0.30,
}


MECHANISM_PROFILES: dict[str, tuple[str, ...]] = {
    "minor_element_linear": tuple(f"frac_{element}" for element in MINOR_ELEMENTS),
    "all_element_linear": tuple(f"frac_{element}" for element in ELEMENTS),
    "solid_solution_proxy": (
        "non_fe_fraction",
        "transition_fraction",
        "light_fraction",
        "refractory_fraction",
        "entropy_fraction",
        "phase122_solid_solution_weighted",
        "phase122_gamma_stabilizer_fraction",
        "phase122_carbide_former_fraction",
        "phase122_refractory_to_transition",
        "phase122_light_to_non_fe",
    ),
    "precipitation_proxy": (
        "frac_C",
        "frac_N",
        "frac_Al",
        "frac_Ti",
        "phase122_C_x_Cr",
        "phase122_C_x_Mo",
        "phase122_C_x_V",
        "phase122_C_x_Nb",
        "phase122_C_x_W",
        "phase122_C_x_Ti",
        "phase122_Al_x_Ti",
        "phase122_Ni_x_Ti",
        "phase122_precipitation_carbon_sum",
        "phase122_carbide_former_fraction",
    ),
    "transition_interaction_proxy": (
        "frac_Ni",
        "frac_Co",
        "frac_Cr",
        "frac_Mo",
        "frac_Mn",
        "phase122_solid_solution_weighted",
        "phase122_gamma_stabilizer_fraction",
        "phase122_carbide_former_fraction",
        "phase122_Mo_x_Ni",
        "phase122_Cr_x_Ni",
        "phase122_Co_x_Ni",
        "phase122_Cr_x_Mo",
    ),
    "mechanism_compact": (
        *(f"frac_{element}" for element in KEY_STRENGTH_ELEMENTS),
        "non_fe_fraction",
        "entropy_fraction",
        "phase122_solid_solution_weighted",
        "phase122_precipitation_carbon_sum",
        "phase122_gamma_stabilizer_fraction",
        "phase122_carbide_former_fraction",
        "phase122_light_transition_interaction",
        "phase122_refractory_transition_interaction",
        "phase122_gamma_carbide_interaction",
    ),
    "mechanism_interaction_full": (
        *(f"frac_{element}" for element in ELEMENTS),
        *(f"phase122_{left}_x_{right}" for left, right, _ in PAIR_FEATURES),
        "phase122_solid_solution_weighted",
        "phase122_precipitation_carbon_sum",
        "phase122_gamma_stabilizer_fraction",
        "phase122_carbide_former_fraction",
        "phase122_light_transition_interaction",
        "phase122_refractory_transition_interaction",
    ),
}


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
    *(ModelSpec("ridge", alpha=value) for value in (1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0)),
    *(ModelSpec("lasso", alpha=value) for value in (1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0)),
    *(
        ModelSpec("elastic_net", alpha=alpha, l1_ratio=l1_ratio)
        for alpha in (1e-4, 1e-3, 1e-2, 0.1, 1.0)
        for l1_ratio in (0.2, 0.5, 0.8)
    ),
    *(ModelSpec("huber", alpha=value) for value in (1e-5, 1e-3, 0.1)),
)


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
    "nrmse_train_std",
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
    "phase121_guard_profile",
    "phase121_guard_method",
    "phase121_guard_val_rmse",
    "phase121_guard_test_rmse",
    "phase120_selected_profile",
    "phase120_selected_method",
    "phase120_selected_val_rmse",
    "phase120_selected_test_rmse",
    "val_gain_vs_phase121",
    "relative_val_gain_vs_phase121",
    "test_gain_vs_phase121",
    "test_reversal_ratio_vs_phase121",
    "val_gain_vs_phase120",
    "test_gain_vs_phase120",
    "clears_phase121_validation_guard",
    "clears_phase121_test_guard",
    "selected_low_capacity",
)
COEFFICIENT_FIELDS = (
    "target",
    "selected_low_capacity",
    "profile",
    "model",
    "model_label",
    "alpha",
    "l1_ratio",
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
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.9f}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
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


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _bounded_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.astype(float) / denominator.astype(float).clip(lower=1e-12)


def build_mechanism_feature_table(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    out = df.copy()
    feature_rows: dict[str, dict[str, Any]] = {}

    def add_feature(name: str, values: Any, *, role: str, expression: str, source: str = "phase122") -> None:
        out[name] = pd.Series(values, index=out.index, dtype=float)
        numeric = pd.to_numeric(out[name], errors="coerce")
        feature_rows[name] = {
            "feature": name,
            "source": source,
            "mechanism_role": role,
            "expression": expression,
            "profiles": [],
            "min": float(numeric.min()) if numeric.notna().any() else None,
            "max": float(numeric.max()) if numeric.notna().any() else None,
            "std": float(numeric.std()) if numeric.notna().sum() > 1 else None,
        }

    for element in ELEMENTS:
        name = f"frac_{element}"
        numeric = pd.to_numeric(out[name], errors="coerce")
        feature_rows[name] = {
            "feature": name,
            "source": "phase120",
            "mechanism_role": "element_fraction",
            "expression": f"normalized atomic fraction of {element}",
            "profiles": [],
            "min": float(numeric.min()) if numeric.notna().any() else None,
            "max": float(numeric.max()) if numeric.notna().any() else None,
            "std": float(numeric.std()) if numeric.notna().sum() > 1 else None,
        }
    for name in ("non_fe_fraction", "transition_fraction", "light_fraction", "refractory_fraction", "entropy_fraction"):
        numeric = pd.to_numeric(out[name], errors="coerce")
        feature_rows[name] = {
            "feature": name,
            "source": "phase120",
            "mechanism_role": "composition_descriptor",
            "expression": name,
            "profiles": [],
            "min": float(numeric.min()) if numeric.notna().any() else None,
            "max": float(numeric.max()) if numeric.notna().any() else None,
            "std": float(numeric.std()) if numeric.notna().sum() > 1 else None,
        }

    for left, right, role in PAIR_FEATURES:
        add_feature(
            f"phase122_{left}_x_{right}",
            out[f"frac_{left}"] * out[f"frac_{right}"],
            role=role,
            expression=f"frac_{left} * frac_{right}",
        )
    add_feature(
        "phase122_solid_solution_weighted",
        sum(weight * out[f"frac_{element}"] for element, weight in SOLID_SOLUTION_WEIGHTS.items()),
        role="solid_solution_proxy",
        expression=" + ".join(f"{weight:g}*frac_{element}" for element, weight in SOLID_SOLUTION_WEIGHTS.items()),
    )
    carbide_terms = ("Cr", "Mo", "V", "Nb", "W", "Ti")
    add_feature(
        "phase122_precipitation_carbon_sum",
        sum(out["frac_C"] * out[f"frac_{element}"] for element in carbide_terms),
        role="precipitation_proxy",
        expression="sum(frac_C * carbide-former fractions)",
    )
    add_feature(
        "phase122_gamma_stabilizer_fraction",
        out["frac_Ni"] + out["frac_Co"] + out["frac_Mn"] + out["frac_N"],
        role="phase_balance_proxy",
        expression="frac_Ni + frac_Co + frac_Mn + frac_N",
    )
    add_feature(
        "phase122_carbide_former_fraction",
        sum(out[f"frac_{element}"] for element in carbide_terms),
        role="carbide_former_proxy",
        expression="frac_Cr + frac_Mo + frac_V + frac_Nb + frac_W + frac_Ti",
    )
    add_feature(
        "phase122_light_transition_interaction",
        out["light_fraction"] * out["transition_fraction"],
        role="element_family_interaction",
        expression="light_fraction * transition_fraction",
    )
    add_feature(
        "phase122_refractory_transition_interaction",
        out["refractory_fraction"] * out["transition_fraction"],
        role="element_family_interaction",
        expression="refractory_fraction * transition_fraction",
    )
    add_feature(
        "phase122_gamma_carbide_interaction",
        out["phase122_gamma_stabilizer_fraction"] * out["phase122_carbide_former_fraction"],
        role="element_family_interaction",
        expression="gamma_stabilizer_fraction * carbide_former_fraction",
    )
    add_feature(
        "phase122_entropy_non_fe_interaction",
        out["entropy_fraction"] * out["non_fe_fraction"],
        role="complexity_interaction",
        expression="entropy_fraction * non_fe_fraction",
    )
    add_feature(
        "phase122_refractory_to_transition",
        _bounded_ratio(out["refractory_fraction"], out["transition_fraction"]),
        role="bounded_ratio_proxy",
        expression="refractory_fraction / max(transition_fraction, 1e-12)",
    )
    add_feature(
        "phase122_light_to_non_fe",
        _bounded_ratio(out["light_fraction"], out["non_fe_fraction"]),
        role="bounded_ratio_proxy",
        expression="light_fraction / max(non_fe_fraction, 1e-12)",
    )

    for profile_name, features in MECHANISM_PROFILES.items():
        for feature in features:
            if feature not in feature_rows:
                raise ValueError(f"Profile {profile_name} references missing feature {feature}")
            feature_rows[feature]["profiles"].append(profile_name)
    rows = []
    for feature in sorted(feature_rows):
        row = dict(feature_rows[feature])
        row["profiles"] = ";".join(row["profiles"])
        rows.append(row)
    return out, rows


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, train_std: float) -> dict[str, Any]:
    residual = y_true - y_pred
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / denom) if denom > 0 else 0.0
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "nrmse_train_std": rmse / train_std if train_std > 0 else None,
    }


def _fit_predict_linear(
    spec: ModelSpec,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_all: np.ndarray,
    feature_names: tuple[str, ...],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if spec.model == "ordinary_least_squares":
        estimator = LinearRegression()
    elif spec.model == "ridge":
        estimator = Ridge(alpha=float(spec.alpha), random_state=122)
    elif spec.model == "lasso":
        estimator = Lasso(alpha=float(spec.alpha), max_iter=20_000, random_state=122)
    elif spec.model == "elastic_net":
        estimator = ElasticNet(
            alpha=float(spec.alpha),
            l1_ratio=float(spec.l1_ratio),
            max_iter=20_000,
            random_state=122,
        )
    elif spec.model == "huber":
        estimator = HuberRegressor(alpha=float(spec.alpha), max_iter=1000)
    else:
        raise ValueError(f"Unsupported low-capacity model: {spec.model}")

    pipe = make_pipeline(StandardScaler(), estimator)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        pipe.fit(x_train, y_train)
    pred = np.asarray(pipe.predict(x_all), dtype=float)
    scaler = pipe.steps[0][1]
    fitted = pipe.steps[-1][1]
    coef = getattr(fitted, "coef_", np.zeros(len(feature_names), dtype=float))
    coef = np.asarray(coef, dtype=float).reshape(-1)
    if len(coef) != len(feature_names):
        coef = np.resize(coef, len(feature_names))
    scale = np.asarray(getattr(scaler, "scale_", np.ones(len(feature_names))), dtype=float)
    mean = np.asarray(getattr(scaler, "mean_", np.zeros(len(feature_names))), dtype=float)
    raw_coef = coef / np.where(scale == 0.0, 1.0, scale)
    intercept = float(getattr(fitted, "intercept_", 0.0))
    raw_intercept = float(intercept - np.sum(coef * mean / np.where(scale == 0.0, 1.0, scale)))
    ranks = np.argsort(-np.abs(coef))
    rank_by_index = {int(index): rank + 1 for rank, index in enumerate(ranks)}
    coefficient_rows = [
        {
            "feature": feature,
            "coefficient_standardized": float(coef[index]),
            "coefficient_raw_scale": float(raw_coef[index]),
            "abs_coefficient_standardized": float(abs(coef[index])),
            "coefficient_rank": rank_by_index[index],
            "intercept_raw_scale": raw_intercept,
        }
        for index, feature in enumerate(feature_names)
    ]
    return pred, coefficient_rows


def _guard_from_phase121(
    *,
    phase120_gate: dict[str, Any],
    phase121_gate: dict[str, Any],
    phase121_split_rows: pd.DataFrame,
) -> dict[str, Any]:
    phase121_val = _safe_float(phase121_gate.get("original_best_admissible_val_rmse"))
    phase121_test = _safe_float(phase121_gate.get("original_best_admissible_test_rmse"))
    if phase121_val is None or phase121_test is None:
        original = phase121_split_rows[phase121_split_rows["split_id"].astype(str) == "phase120_registered_split"]
        if not original.empty:
            row = original.iloc[0].to_dict()
            phase121_val = phase121_val if phase121_val is not None else _safe_float(row.get("best_admissible_val_rmse"))
            phase121_test = phase121_test if phase121_test is not None else _safe_float(row.get("best_admissible_test_rmse"))
    if phase121_val is None or phase121_test is None:
        raise ValueError("Could not determine Phase 121 strong-baseline guard RMSE values")
    return {
        "phase121_guard_profile": phase121_gate.get("original_best_admissible_profile"),
        "phase121_guard_method": phase121_gate.get("original_best_admissible_method"),
        "phase121_guard_val_rmse": phase121_val,
        "phase121_guard_test_rmse": phase121_test,
        "phase120_selected_profile": phase120_gate.get("selected_profile"),
        "phase120_selected_method": phase120_gate.get("selected_method"),
        "phase120_selected_val_rmse": _safe_float(phase120_gate.get("selected_validation_rmse")),
        "phase120_selected_test_rmse": _safe_float(phase120_gate.get("selected_test_rmse")),
    }


def evaluate_low_capacity_models(
    df: pd.DataFrame,
    *,
    target: str,
    split_manifest: dict[str, Any],
    guard: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    splits = split_manifest["splits"]
    counts = {split: len(splits[split]) for split in ("train", "val", "test")}
    if min(counts.values()) < MIN_SPLIT_ROWS:
        raise ValueError(f"Split below minimum row count {MIN_SPLIT_ROWS}: {counts}")

    y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
    train_idx = splits["train"]
    train_std = float(np.std(y[train_idx])) if train_idx else 0.0
    metric_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []

    phase121_val = float(guard["phase121_guard_val_rmse"])
    phase121_test = float(guard["phase121_guard_test_rmse"])
    phase120_val = guard.get("phase120_selected_val_rmse")
    phase120_test = guard.get("phase120_selected_test_rmse")

    for profile_name, feature_names in MECHANISM_PROFILES.items():
        if len(feature_names) > MAX_LOW_CAPACITY_FEATURES:
            raise ValueError(f"Profile {profile_name} has {len(feature_names)} features, above low-capacity cap")
        x_all = df[list(feature_names)].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        x_train = x_all[train_idx]
        y_train = y[train_idx]
        for spec in MODEL_SPECS:
            pred, coef_base = _fit_predict_linear(spec, x_train, y_train, x_all, feature_names)
            split_metrics = {
                split: _metrics(y[splits[split]], pred[splits[split]], train_std)
                for split in ("train", "val", "test")
            }
            for split in ("train", "val", "test"):
                metric_rows.append(
                    {
                        "target": target,
                        "profile": profile_name,
                        "model": spec.model,
                        "model_label": spec.label,
                        "alpha": spec.alpha,
                        "l1_ratio": spec.l1_ratio,
                        "split": split,
                        "n_rows": counts[split],
                        "feature_count": len(feature_names),
                        **split_metrics[split],
                    }
                )
            val_rmse = float(split_metrics["val"]["rmse"])
            test_rmse = float(split_metrics["test"]["rmse"])
            row = {
                "target": target,
                "profile": profile_name,
                "model": spec.model,
                "model_label": spec.label,
                "alpha": spec.alpha,
                "l1_ratio": spec.l1_ratio,
                "feature_count": len(feature_names),
                "train_rmse": split_metrics["train"]["rmse"],
                "val_rmse": val_rmse,
                "test_rmse": test_rmse,
                **guard,
                "val_gain_vs_phase121": phase121_val - val_rmse,
                "relative_val_gain_vs_phase121": (phase121_val - val_rmse) / phase121_val if phase121_val > 0 else None,
                "test_gain_vs_phase121": phase121_test - test_rmse,
                "test_reversal_ratio_vs_phase121": test_rmse / phase121_test if phase121_test > 0 else None,
                "val_gain_vs_phase120": phase120_val - val_rmse if phase120_val is not None else None,
                "test_gain_vs_phase120": phase120_test - test_rmse if phase120_test is not None else None,
                "clears_phase121_validation_guard": (
                    (phase121_val - val_rmse) / phase121_val >= MIN_RELATIVE_PHASE121_VAL_GAIN
                    if phase121_val > 0
                    else False
                ),
                "clears_phase121_test_guard": (
                    test_rmse <= phase121_test * MAX_PHASE121_TEST_REVERSAL_RATIO if phase121_test > 0 else False
                ),
                "selected_low_capacity": False,
            }
            candidate_rows.append(row)
            for coefficient in coef_base:
                coefficient_rows.append(
                    {
                        "target": target,
                        "selected_low_capacity": False,
                        "profile": profile_name,
                        "model": spec.model,
                        "model_label": spec.label,
                        "alpha": spec.alpha,
                        "l1_ratio": spec.l1_ratio,
                        **coefficient,
                    }
                )

    selected = min(candidate_rows, key=lambda row: float(row["val_rmse"]))
    selected["selected_low_capacity"] = True
    for row in coefficient_rows:
        if (
            row["profile"] == selected["profile"]
            and row["model_label"] == selected["model_label"]
        ):
            row["selected_low_capacity"] = True
    return metric_rows, candidate_rows, coefficient_rows


def build_audit_rows(
    *,
    phase120_gate: dict[str, Any],
    phase121_gate: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = next(row for row in candidate_rows if _is_true(row["selected_low_capacity"]))
    phase121_ready = (
        phase121_gate.get("status")
        == "phase121_matbench_steels_focused_review_ready_low_capacity_mechanism_gate"
    )
    phase121_mechanism_allowed = _is_true(phase121_gate.get("phase121_model_mechanism_allowed"))
    phase121_training_locked = not _is_true(phase121_gate.get("phase121_model_training_allowed"))
    phase120_ready = phase120_gate.get("status") == "phase120_matbench_steels_gap_ready_focused_review"
    rows = [
        {
            "audit": "phase120_gate_status",
            "status": "pass" if phase120_ready else "block",
            "severity": "blocking" if not phase120_ready else "info",
            "value": phase120_gate.get("status"),
            "threshold": "phase120_matbench_steels_gap_ready_focused_review",
            "reason": "Phase 122 consumes the Phase 120 registered Matbench steels target table",
        },
        {
            "audit": "phase121_gate_status",
            "status": "pass" if phase121_ready else "block",
            "severity": "blocking" if not phase121_ready else "info",
            "value": phase121_gate.get("status"),
            "threshold": "phase121_matbench_steels_focused_review_ready_low_capacity_mechanism_gate",
            "reason": "low-capacity mechanism design requires the Phase 121 focused-review pass",
        },
        {
            "audit": "phase121_mechanism_allowed",
            "status": "pass" if phase121_mechanism_allowed else "block",
            "severity": "blocking" if not phase121_mechanism_allowed else "info",
            "value": phase121_gate.get("phase121_model_mechanism_allowed"),
            "threshold": True,
            "reason": "Phase 121 must explicitly allow only no-training mechanism design",
        },
        {
            "audit": "phase121_training_lock",
            "status": "pass" if phase121_training_locked else "block",
            "severity": "blocking" if not phase121_training_locked else "info",
            "value": phase121_gate.get("phase121_model_training_allowed"),
            "threshold": False,
            "reason": "Phase 122 must not inherit an open training flag",
        },
        {
            "audit": "low_capacity_feature_cap",
            "status": "pass" if int(selected["feature_count"]) <= MAX_LOW_CAPACITY_FEATURES else "block",
            "severity": "blocking" if int(selected["feature_count"]) > MAX_LOW_CAPACITY_FEATURES else "info",
            "value": selected["feature_count"],
            "threshold": MAX_LOW_CAPACITY_FEATURES,
            "reason": "selected mechanism must remain a low-capacity linear descriptor set",
        },
        {
            "audit": "phase121_validation_guard_gain",
            "status": "pass" if _is_true(selected["clears_phase121_validation_guard"]) else "block",
            "severity": "blocking" if not _is_true(selected["clears_phase121_validation_guard"]) else "info",
            "value": selected["relative_val_gain_vs_phase121"],
            "threshold": MIN_RELATIVE_PHASE121_VAL_GAIN,
            "reason": "validation-only selected low-capacity mechanism must beat the Phase 121 ExtraTrees guard",
        },
        {
            "audit": "phase121_test_reversal_guard",
            "status": "pass" if _is_true(selected["clears_phase121_test_guard"]) else "block",
            "severity": "blocking" if not _is_true(selected["clears_phase121_test_guard"]) else "info",
            "value": selected["test_reversal_ratio_vs_phase121"],
            "threshold": MAX_PHASE121_TEST_REVERSAL_RATIO,
            "reason": "validation gain must not reverse badly on the held-out test split",
        },
        {
            "audit": "phase120_selected_test_comparison",
            "status": "info",
            "severity": "info",
            "value": selected["test_gain_vs_phase120"],
            "threshold": "reported only",
            "reason": "Phase 120 all-element ExtraTrees remains a secondary comparison, not the Phase 122 selection target",
        },
    ]
    return rows


def build_gate(
    *,
    phase120_gate: dict[str, Any],
    phase121_gate: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = next(row for row in candidate_rows if _is_true(row["selected_low_capacity"]))
    blockers = [row for row in audit_rows if row["status"] == "block"]
    phase121_ready = (
        phase121_gate.get("status")
        == "phase121_matbench_steels_focused_review_ready_low_capacity_mechanism_gate"
    )
    if not phase121_ready:
        status = "phase122_matbench_steels_low_capacity_mechanism_blocked_by_phase121"
        mechanism_positive = False
        next_action = "complete or close Phase 121 before mechanism design"
    elif blockers:
        if any(row["audit"] == "phase121_test_reversal_guard" for row in blockers) and not any(
            row["audit"] == "phase121_validation_guard_gain" for row in blockers
        ):
            status = "phase122_matbench_steels_low_capacity_mechanism_closed_validation_test_reversal"
        else:
            status = "phase122_matbench_steels_low_capacity_mechanism_closed_no_guarded_gain"
        mechanism_positive = False
        next_action = "close the low-capacity mechanism branch as diagnostic; do not train"
    else:
        status = "phase122_matbench_steels_low_capacity_mechanism_ready_focused_validation"
        mechanism_positive = True
        next_action = "run a separate focused validation or symbolic mechanism review before any neural training"
    return {
        "status": status,
        "phase120_status": phase120_gate.get("status"),
        "phase121_status": phase121_gate.get("status"),
        "selected_target": selected["target"],
        "selected_low_capacity_profile": selected["profile"],
        "selected_low_capacity_model": selected["model"],
        "selected_low_capacity_model_label": selected["model_label"],
        "selected_low_capacity_alpha": selected["alpha"],
        "selected_low_capacity_l1_ratio": selected["l1_ratio"],
        "selected_low_capacity_feature_count": selected["feature_count"],
        "selected_low_capacity_val_rmse": selected["val_rmse"],
        "selected_low_capacity_test_rmse": selected["test_rmse"],
        "phase121_guard_profile": selected["phase121_guard_profile"],
        "phase121_guard_method": selected["phase121_guard_method"],
        "phase121_guard_val_rmse": selected["phase121_guard_val_rmse"],
        "phase121_guard_test_rmse": selected["phase121_guard_test_rmse"],
        "phase120_selected_profile": selected["phase120_selected_profile"],
        "phase120_selected_method": selected["phase120_selected_method"],
        "phase120_selected_val_rmse": selected["phase120_selected_val_rmse"],
        "phase120_selected_test_rmse": selected["phase120_selected_test_rmse"],
        "selected_val_gain_vs_phase121": selected["val_gain_vs_phase121"],
        "selected_relative_val_gain_vs_phase121": selected["relative_val_gain_vs_phase121"],
        "selected_test_gain_vs_phase121": selected["test_gain_vs_phase121"],
        "selected_test_reversal_ratio_vs_phase121": selected["test_reversal_ratio_vs_phase121"],
        "blocking_audit_rows": len(blockers),
        "blocking_audits": [row["audit"] for row in blockers],
        "candidate_rows": len(candidate_rows),
        "phase122_low_capacity_mechanism_positive": mechanism_positive,
        "phase122_focused_validation_allowed": mechanism_positive,
        "phase122_model_mechanism_allowed": mechanism_positive,
        "phase122_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": next_action,
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for _, key in columns:
            value = row.get(key)
            values.append(f"{value:.6g}" if isinstance(value, float) else str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def build_markdown(
    gate: dict[str, Any],
    audit_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    coefficient_rows: list[dict[str, Any]],
) -> str:
    selected = [row for row in candidate_rows if _is_true(row.get("selected_low_capacity"))]
    top_candidates = sorted(candidate_rows, key=lambda row: float(row["val_rmse"]))[:8]
    selected_coefficients = sorted(
        [row for row in coefficient_rows if _is_true(row.get("selected_low_capacity"))],
        key=lambda row: int(row["coefficient_rank"]),
    )[:12]
    blocking = [row for row in audit_rows if row["status"] == "block"]
    lines = [
        "# Phase 122 Matbench Steels Low-Capacity Mechanism Gate",
        "",
        f"- Status: `{gate['status']}`",
        f"- Selected low-capacity profile: `{gate['selected_low_capacity_profile']}`",
        f"- Selected model: `{gate['selected_low_capacity_model_label']}`",
        f"- Selected validation RMSE: `{gate['selected_low_capacity_val_rmse']:.6g}`",
        f"- Phase 121 guard validation RMSE: `{gate['phase121_guard_val_rmse']:.6g}`",
        f"- Focused validation allowed: `{gate['phase122_focused_validation_allowed']}`",
        f"- Model training allowed: `{gate['phase122_model_training_allowed']}`",
        f"- A100 training allowed now: `{gate['a100_training_allowed_now']}`",
        "",
        "## Selected Candidate",
        "",
        _markdown_table(
            selected,
            [
                ("Profile", "profile"),
                ("Model", "model_label"),
                ("Features", "feature_count"),
                ("Val RMSE", "val_rmse"),
                ("Test RMSE", "test_rmse"),
                ("Val gain vs Phase121", "relative_val_gain_vs_phase121"),
                ("Test ratio vs Phase121", "test_reversal_ratio_vs_phase121"),
            ],
        ),
        "",
        "## Blocking Audits",
        "",
        _markdown_table(blocking, [("Audit", "audit"), ("Value", "value"), ("Threshold", "threshold"), ("Reason", "reason")]),
        "",
        "## Top Validation Candidates",
        "",
        _markdown_table(
            top_candidates,
            [
                ("Profile", "profile"),
                ("Model", "model_label"),
                ("Val RMSE", "val_rmse"),
                ("Test RMSE", "test_rmse"),
                ("Features", "feature_count"),
            ],
        ),
        "",
        "## Selected Coefficients",
        "",
        _markdown_table(
            selected_coefficients,
            [
                ("Rank", "coefficient_rank"),
                ("Feature", "feature"),
                ("Std coef", "coefficient_standardized"),
                ("Raw coef", "coefficient_raw_scale"),
            ],
        ),
    ]
    return "\n".join(lines) + "\n"


def build_package(*, root: Path, phase120_dir: Path, phase121_dir: Path, output_dir: Path) -> dict[str, Any]:
    field_path = phase120_dir / "phase120_matbench_steels_field_table.csv"
    split_path = phase120_dir / "phase120_matbench_steels_split_manifest.json"
    phase120_gate_path = phase120_dir / "phase120_matbench_steels_gate.json"
    phase121_gate_path = phase121_dir / "phase121_matbench_steels_focused_review_gate.json"
    phase121_split_path = phase121_dir / "phase121_matbench_steels_split_sensitivity_table.csv"

    df = pd.read_csv(field_path)
    split_manifest = _read_json(split_path)
    phase120_gate = _read_json(phase120_gate_path)
    phase121_gate = _read_json(phase121_gate_path)
    phase121_split_rows = pd.read_csv(phase121_split_path)
    target = str(phase120_gate.get("selected_target") or phase121_gate.get("selected_target") or "yield_strength_mpa")

    mechanism_df, feature_schema_rows = build_mechanism_feature_table(df)
    guard = _guard_from_phase121(
        phase120_gate=phase120_gate,
        phase121_gate=phase121_gate,
        phase121_split_rows=phase121_split_rows,
    )
    metric_rows, candidate_rows, coefficient_rows = evaluate_low_capacity_models(
        mechanism_df,
        target=target,
        split_manifest=split_manifest,
        guard=guard,
    )
    audit_rows = build_audit_rows(
        phase120_gate=phase120_gate,
        phase121_gate=phase121_gate,
        candidate_rows=candidate_rows,
    )
    gate = build_gate(
        phase120_gate=phase120_gate,
        phase121_gate=phase121_gate,
        candidate_rows=candidate_rows,
        audit_rows=audit_rows,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_table_path = output_dir / "phase122_matbench_steels_mechanism_feature_table.csv"
    feature_schema_path = output_dir / "phase122_matbench_steels_mechanism_schema_table.csv"
    metric_path = output_dir / "phase122_matbench_steels_mechanism_metric_table.csv"
    candidate_path = output_dir / "phase122_matbench_steels_mechanism_candidate_table.csv"
    coefficient_path = output_dir / "phase122_matbench_steels_mechanism_coefficient_table.csv"
    audit_path = output_dir / "phase122_matbench_steels_mechanism_audit_table.csv"
    gate_path = output_dir / "phase122_matbench_steels_low_capacity_mechanism_gate.json"
    markdown_path = output_dir / "phase122_matbench_steels_low_capacity_mechanism.md"
    manifest_path = output_dir / "phase122_matbench_steels_low_capacity_mechanism_manifest.json"

    _write_csv(feature_table_path, mechanism_df.to_dict("records"), tuple(mechanism_df.columns))
    _write_csv(feature_schema_path, feature_schema_rows, FEATURE_SCHEMA_FIELDS)
    _write_csv(metric_path, metric_rows, METRIC_FIELDS)
    _write_csv(candidate_path, candidate_rows, CANDIDATE_FIELDS)
    _write_csv(coefficient_path, coefficient_rows, COEFFICIENT_FIELDS)
    _write_csv(audit_path, audit_rows, AUDIT_FIELDS)
    _write_json(gate_path, gate)
    markdown_path.write_text(build_markdown(gate, audit_rows, candidate_rows, coefficient_rows), encoding="utf-8")

    manifest = {
        "phase": 122,
        "objective": "matbench_steels_low_capacity_mechanism_gate_no_training",
        "inputs": {
            "phase120_dir": _display_path(phase120_dir, root),
            "phase121_dir": _display_path(phase121_dir, root),
            "field_table": _display_path(field_path, root),
            "split_manifest": _display_path(split_path, root),
            "phase120_gate": _display_path(phase120_gate_path, root),
            "phase121_gate": _display_path(phase121_gate_path, root),
            "phase121_split_sensitivity_table": _display_path(phase121_split_path, root),
        },
        "outputs": {
            "mechanism_feature_table": _display_path(feature_table_path, root),
            "mechanism_schema_table": _display_path(feature_schema_path, root),
            "mechanism_metric_table": _display_path(metric_path, root),
            "mechanism_candidate_table": _display_path(candidate_path, root),
            "mechanism_coefficient_table": _display_path(coefficient_path, root),
            "mechanism_audit_table": _display_path(audit_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "field_rows": int(len(df)),
            "feature_columns": len(mechanism_df.columns),
            "mechanism_schema_rows": len(feature_schema_rows),
            "metric_rows": len(metric_rows),
            "candidate_rows": len(candidate_rows),
            "coefficient_rows": len(coefficient_rows),
            "audit_rows": len(audit_rows),
            "blocking_audit_rows": gate["blocking_audit_rows"],
        },
        "gate": gate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--phase120-dir", type=Path, default=DEFAULT_PHASE120_DIR)
    parser.add_argument("--phase121-dir", type=Path, default=DEFAULT_PHASE121_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    phase120_dir = args.phase120_dir if args.phase120_dir.is_absolute() else root / args.phase120_dir
    phase121_dir = args.phase121_dir if args.phase121_dir.is_absolute() else root / args.phase121_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = build_package(root=root, phase120_dir=phase120_dir, phase121_dir=phase121_dir, output_dir=output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
