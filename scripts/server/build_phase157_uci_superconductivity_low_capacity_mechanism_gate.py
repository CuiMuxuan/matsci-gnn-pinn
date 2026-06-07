#!/usr/bin/env python3
"""Build Phase 157 low-capacity mechanism gate for UCI superconductivity.

This phase tests interpretable linear/sparse mechanism feature families against
the Phase 156 ExtraTrees strong-baseline guard. It is a no-training gate: a
positive result may only open a later focused validation step.
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


DEFAULT_PHASE156_DIR = Path("docs/results/phase156_uci_superconductivity_focused_review")
DEFAULT_RAW_PATH = Path(
    "data/raw/external/phase155_uci_superconductivity/superconductivty_data.zip"
)
DEFAULT_OUTPUT_DIR = Path("docs/results/phase157_uci_superconductivity_low_capacity_mechanism_gate")

MIN_RELATIVE_PHASE156_VAL_GAIN = 0.01
MAX_PHASE156_TEST_REVERSAL_RATIO = 1.05
MAX_LOW_CAPACITY_FEATURES = 36

SUPERCU_ELEMENTS = ("Cu", "O", "Ba", "Sr", "Ca", "Y", "Bi", "Tl", "Hg", "La")
FE_BASED_ELEMENTS = ("Fe", "As", "Se", "Te", "P", "F", "O", "Ba", "Sr", "Ca", "K")
HEAVY_ELEMENTS = ("Ba", "La", "Ce", "Yb", "Bi", "Tl", "Hg", "Pb", "Sn", "In")
LIGHT_DOPANTS = ("B", "C", "N", "O", "F", "P", "S")


def _load_module(name: str, filename: str):
    script = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


p155 = _load_module("phase155_helpers", "build_phase155_uci_superconductivity_baseline_gate.py")


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
    "composition_complexity_linear": (
        "number_of_elements",
        "max_element_fraction",
        "phase157_fraction_entropy",
        "phase157_oxide_fraction",
        "phase157_cuprate_fraction",
        "phase157_fe_pnictide_fraction",
        "phase157_heavy_fraction",
        "phase157_light_dopant_fraction",
    ),
    "cuprate_oxide_proxy": (
        *(f"frac_{element}" for element in SUPERCU_ELEMENTS),
        "phase157_oxide_fraction",
        "phase157_cuprate_fraction",
        "phase157_cu_o_interaction",
        "phase157_ba_sr_ca_fraction",
        "phase157_y_bi_tl_hg_fraction",
        "phase157_cuprate_layer_proxy",
    ),
    "iron_pnictide_chalcogenide_proxy": (
        *(f"frac_{element}" for element in FE_BASED_ELEMENTS),
        "phase157_fe_pnictide_fraction",
        "phase157_fe_chalcogen_fraction",
        "phase157_fe_as_interaction",
        "phase157_fe_se_te_interaction",
        "phase157_alkaline_earth_fe_pnictide_proxy",
    ),
    "weighted_property_linear": (
        "number_of_elements",
        "wtd_mean_atomic_mass",
        "wtd_std_atomic_mass",
        "wtd_mean_fie",
        "wtd_entropy_fie",
        "wtd_mean_atomic_radius",
        "wtd_std_atomic_radius",
        "wtd_mean_Density",
        "wtd_std_Density",
        "wtd_mean_ElectronAffinity",
        "wtd_std_ElectronAffinity",
        "wtd_mean_FusionHeat",
        "wtd_std_FusionHeat",
        "wtd_mean_ThermalConductivity",
        "wtd_std_ThermalConductivity",
        "wtd_mean_Valence",
        "wtd_std_Valence",
    ),
    "mechanism_compact": (
        "number_of_elements",
        "max_element_fraction",
        "phase157_fraction_entropy",
        "wtd_mean_atomic_mass",
        "wtd_std_atomic_mass",
        "wtd_mean_fie",
        "wtd_mean_atomic_radius",
        "wtd_mean_Density",
        "wtd_mean_ElectronAffinity",
        "wtd_mean_ThermalConductivity",
        "wtd_mean_Valence",
        "phase157_oxide_fraction",
        "phase157_cuprate_fraction",
        "phase157_fe_pnictide_fraction",
        "phase157_heavy_fraction",
        "phase157_cu_o_interaction",
        "phase157_fe_as_interaction",
        "phase157_fe_se_te_interaction",
    ),
    "mechanism_full_low_capacity": (
        "number_of_elements",
        "max_element_fraction",
        "phase157_fraction_entropy",
        "wtd_mean_atomic_mass",
        "wtd_std_atomic_mass",
        "wtd_mean_fie",
        "wtd_entropy_fie",
        "wtd_mean_atomic_radius",
        "wtd_std_atomic_radius",
        "wtd_mean_Density",
        "wtd_std_Density",
        "wtd_mean_ElectronAffinity",
        "wtd_std_ElectronAffinity",
        "wtd_mean_ThermalConductivity",
        "wtd_std_ThermalConductivity",
        "wtd_mean_Valence",
        "wtd_std_Valence",
        *(f"frac_{element}" for element in SUPERCU_ELEMENTS),
        "frac_Fe",
        "frac_As",
        "frac_Se",
        "frac_Te",
        "phase157_cu_o_interaction",
        "phase157_fe_as_interaction",
        "phase157_fe_se_te_interaction",
        "phase157_cuprate_layer_proxy",
        "phase157_alkaline_earth_fe_pnictide_proxy",
    ),
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
    "phase156_guard_profile",
    "phase156_guard_method",
    "phase156_guard_val_rmse",
    "phase156_guard_test_rmse",
    "val_gain_vs_phase156",
    "relative_val_gain_vs_phase156",
    "test_gain_vs_phase156",
    "test_reversal_ratio_vs_phase156",
    "clears_phase156_validation_guard",
    "clears_phase156_test_guard",
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


def _frac(df: pd.DataFrame, element: str) -> pd.Series:
    column = f"frac_{element}"
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return pd.Series(np.zeros(len(df), dtype=float), index=df.index)


def _sum_frac(df: pd.DataFrame, elements: tuple[str, ...]) -> pd.Series:
    total = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    for element in elements:
        total = total + _frac(df, element)
    return total


def add_mechanism_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    out = df.copy()
    frac_columns = [column for column in out.columns if column.startswith("frac_")]
    fraction_matrix = out[frac_columns].to_numpy(dtype=float) if frac_columns else np.zeros((len(out), 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.nansum(np.where(fraction_matrix > 0.0, fraction_matrix * np.log(fraction_matrix), 0.0), axis=1)
    out["phase157_fraction_entropy"] = entropy
    out["phase157_oxide_fraction"] = _frac(out, "O")
    out["phase157_cuprate_fraction"] = _sum_frac(out, SUPERCU_ELEMENTS)
    out["phase157_fe_pnictide_fraction"] = _sum_frac(out, ("Fe", "As", "P", "F", "O"))
    out["phase157_fe_chalcogen_fraction"] = _sum_frac(out, ("Fe", "Se", "Te"))
    out["phase157_heavy_fraction"] = _sum_frac(out, HEAVY_ELEMENTS)
    out["phase157_light_dopant_fraction"] = _sum_frac(out, LIGHT_DOPANTS)
    out["phase157_cu_o_interaction"] = _frac(out, "Cu") * _frac(out, "O")
    out["phase157_ba_sr_ca_fraction"] = _sum_frac(out, ("Ba", "Sr", "Ca"))
    out["phase157_y_bi_tl_hg_fraction"] = _sum_frac(out, ("Y", "Bi", "Tl", "Hg"))
    out["phase157_cuprate_layer_proxy"] = (
        _frac(out, "Cu")
        * _frac(out, "O")
        * (_sum_frac(out, ("Ba", "Sr", "Ca")) + _sum_frac(out, ("Y", "Bi", "Tl", "Hg")))
    )
    out["phase157_fe_as_interaction"] = _frac(out, "Fe") * _frac(out, "As")
    out["phase157_fe_se_te_interaction"] = _frac(out, "Fe") * (_frac(out, "Se") + _frac(out, "Te"))
    out["phase157_alkaline_earth_fe_pnictide_proxy"] = (
        _frac(out, "Fe") * (_frac(out, "As") + _frac(out, "P")) * _sum_frac(out, ("Ba", "Sr", "Ca"))
    )

    feature_to_profiles: dict[str, list[str]] = {}
    for profile, features in MECHANISM_PROFILES.items():
        for feature in features:
            feature_to_profiles.setdefault(feature, []).append(profile)

    rows: list[dict[str, Any]] = []
    for feature, profiles in sorted(feature_to_profiles.items()):
        values = pd.to_numeric(out[feature], errors="coerce").fillna(0.0) if feature in out.columns else pd.Series(dtype=float)
        rows.append(
            {
                "feature": feature,
                "source": "uci_feature" if not feature.startswith("phase157_") else "phase157_derived",
                "mechanism_role": "low_capacity_superconductivity_proxy",
                "expression": feature,
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
        estimator = Ridge(alpha=float(spec.alpha), random_state=157)
    elif spec.model == "lasso":
        estimator = Lasso(alpha=float(spec.alpha), max_iter=20000, random_state=157)
    elif spec.model == "elastic_net":
        estimator = ElasticNet(
            alpha=float(spec.alpha),
            l1_ratio=float(spec.l1_ratio),
            max_iter=20000,
            random_state=157,
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


def _assignments_from_phase156_split(phase156_dir: Path, df: pd.DataFrame) -> list[str]:
    split_manifest = _read_json(phase156_dir / "phase156_uci_superconductivity_split_manifest.json")
    registered = split_manifest["splits"]["phase155_registered_element_set"]
    split_groups = {
        split: set(str(value) for value in registered["split_groups"][split])
        for split in ("train", "val", "test")
    }
    assignments: list[str] = []
    for group in df["element_set_key"].fillna("unknown").astype(str):
        if group in split_groups["train"]:
            assignments.append("train")
        elif group in split_groups["val"]:
            assignments.append("val")
        elif group in split_groups["test"]:
            assignments.append("test")
        else:
            raise KeyError(f"Element-set group missing from Phase 156 split manifest: {group}")
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
    phase156_gate: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    y = df["target_critical_temp_K"].to_numpy(dtype=float)
    split_indices = _split_indices(assignments)
    guard_val = float(phase156_gate["registered_replay_validation_rmse"])
    guard_test = float(phase156_gate["registered_replay_test_rmse"])
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
                        "target": "target_critical_temp_K",
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
            clears_val = relative_val_gain >= MIN_RELATIVE_PHASE156_VAL_GAIN
            clears_test = reversal_ratio <= MAX_PHASE156_TEST_REVERSAL_RATIO
            row = {
                "target": "target_critical_temp_K",
                "profile": profile,
                "model": spec.model,
                "model_label": spec.label,
                "alpha": spec.alpha,
                "l1_ratio": spec.l1_ratio,
                "feature_count": len(features),
                "train_rmse": split_metrics["train"]["rmse"],
                "val_rmse": val_rmse,
                "test_rmse": test_rmse,
                "phase156_guard_profile": phase156_gate.get("registered_replay_profile"),
                "phase156_guard_method": phase156_gate.get("registered_replay_method"),
                "phase156_guard_val_rmse": guard_val,
                "phase156_guard_test_rmse": guard_test,
                "val_gain_vs_phase156": val_gain,
                "relative_val_gain_vs_phase156": relative_val_gain,
                "test_gain_vs_phase156": test_gain,
                "test_reversal_ratio_vs_phase156": reversal_ratio,
                "clears_phase156_validation_guard": clears_val,
                "clears_phase156_test_guard": clears_test,
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
            candidate_rows[index]["test_reversal_ratio_vs_phase156"],
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
    phase156_gate: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    phase156_ready = (
        phase156_gate.get("status")
        == "phase156_uci_superconductivity_focused_review_ready_low_capacity_mechanism_gate"
        and bool(phase156_gate.get("phase156_model_mechanism_allowed"))
    )
    rows.append(
        _audit_row(
            audit="phase156_gate_consistency",
            status="pass" if phase156_ready else "block",
            severity="blocking" if not phase156_ready else "info",
            value=phase156_gate.get("status"),
            threshold="phase156_uci_superconductivity_focused_review_ready_low_capacity_mechanism_gate",
            reason="Phase 157 requires Phase 156 to allow a no-training mechanism gate",
        )
    )
    selected = next((row for row in candidate_rows if row.get("selected_low_capacity")), None)
    clears_val = bool(selected and selected.get("clears_phase156_validation_guard"))
    rows.append(
        _audit_row(
            audit="phase156_validation_guard_gain",
            status="pass" if clears_val else "block",
            severity="blocking" if not clears_val else "info",
            value=selected.get("relative_val_gain_vs_phase156") if selected else None,
            threshold=MIN_RELATIVE_PHASE156_VAL_GAIN,
            reason="selected low-capacity mechanism must beat the Phase 156 ExtraTrees guard on validation",
        )
    )
    clears_test = bool(selected and selected.get("clears_phase156_test_guard"))
    rows.append(
        _audit_row(
            audit="phase156_test_reversal_guard",
            status="pass" if clears_test else "block",
            severity="blocking" if not clears_test else "info",
            value=selected.get("test_reversal_ratio_vs_phase156") if selected else None,
            threshold=MAX_PHASE156_TEST_REVERSAL_RATIO,
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
    phase156_gate: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = next((row for row in candidate_rows if row.get("selected_low_capacity")), None)
    blocking = [row for row in audit_rows if row["severity"] == "blocking" and row["status"] == "block"]
    focused_allowed = not blocking
    status = (
        "phase157_uci_superconductivity_low_capacity_mechanism_ready_focused_validation"
        if focused_allowed
        else "phase157_uci_superconductivity_low_capacity_mechanism_closed_no_guarded_gain"
    )
    return {
        "status": status,
        "selected_target": "target_critical_temp_K",
        "phase156_guard_profile": phase156_gate.get("registered_replay_profile"),
        "phase156_guard_method": phase156_gate.get("registered_replay_method"),
        "phase156_guard_validation_rmse": phase156_gate.get("registered_replay_validation_rmse"),
        "phase156_guard_test_rmse": phase156_gate.get("registered_replay_test_rmse"),
        "candidate_rows": len(candidate_rows),
        "selected_low_capacity_profile": selected.get("profile") if selected else None,
        "selected_low_capacity_model": selected.get("model") if selected else None,
        "selected_low_capacity_model_label": selected.get("model_label") if selected else None,
        "selected_low_capacity_feature_count": selected.get("feature_count") if selected else None,
        "selected_low_capacity_validation_rmse": selected.get("val_rmse") if selected else None,
        "selected_low_capacity_test_rmse": selected.get("test_rmse") if selected else None,
        "relative_validation_gain_vs_phase156": selected.get("relative_val_gain_vs_phase156") if selected else None,
        "test_reversal_ratio_vs_phase156": selected.get("test_reversal_ratio_vs_phase156") if selected else None,
        "blocking_audits": [row["audit"] for row in blocking],
        "phase157_focused_validation_allowed": focused_allowed,
        "phase157_model_mechanism_allowed": focused_allowed,
        "phase157_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": (
            "run Phase 158 focused validation before any neural training"
            if focused_allowed
            else "close Phase 157 low-capacity mechanism route as diagnostic; do not train"
        ),
    }


def _markdown_table(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[str]:
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    body = [
        "| " + " | ".join(_csv_value(row.get(field)) for field in fields) + " |"
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
        "relative_val_gain_vs_phase156",
        "test_reversal_ratio_vs_phase156",
        "selected_low_capacity",
    )
    audit_fields = ("audit", "status", "severity", "value", "threshold", "reason")
    lines = [
        "# Phase 157 UCI Superconductivity Low-Capacity Mechanism Gate",
        "",
        "## Gate",
        f"- Status: `{gate['status']}`",
        f"- Selected profile: `{gate['selected_low_capacity_profile']}`",
        f"- Selected model: `{gate['selected_low_capacity_model_label']}`",
        f"- Focused validation allowed: `{_csv_value(gate['phase157_focused_validation_allowed'])}`",
        f"- Model training allowed: `{_csv_value(gate['phase157_model_training_allowed'])}`",
        f"- A100 training allowed now: `{_csv_value(gate['a100_training_allowed_now'])}`",
        f"- A100-SXM4-80GB request now: `{_csv_value(gate['a100_80gb_request_now'])}`",
        "",
        "## Interpretation",
        (
            "This no-training gate asks whether an interpretable low-capacity mechanism "
            "can beat the Phase 156 ExtraTrees guard. A closure means the UCI source "
            "remains a strong baseline-positive diagnostic, not a model claim."
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


def build_package(
    *,
    root: Path,
    phase156_dir: Path,
    raw_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    root = root.resolve()
    phase156_dir = phase156_dir if phase156_dir.is_absolute() else root / phase156_dir
    raw_path = raw_path if raw_path.is_absolute() else root / raw_path
    output_dir = output_dir if output_dir.is_absolute() else root / output_dir

    phase156_gate = _read_json(phase156_dir / "phase156_uci_superconductivity_focused_review_gate.json")
    df = p155.load_superconductivity_table(raw_path)
    df, feature_rows = add_mechanism_features(df)
    assignments = _assignments_from_phase156_split(phase156_dir, df)
    metric_rows, candidate_rows, coefficient_rows = evaluate_candidates(
        df,
        assignments,
        phase156_gate=phase156_gate,
    )
    audit_rows = build_audit_rows(phase156_gate=phase156_gate, candidate_rows=candidate_rows)
    gate = build_gate(phase156_gate=phase156_gate, candidate_rows=candidate_rows, audit_rows=audit_rows)

    feature_schema_path = output_dir / "phase157_uci_superconductivity_feature_schema.csv"
    metric_table_path = output_dir / "phase157_uci_superconductivity_metric_table.csv"
    candidate_table_path = output_dir / "phase157_uci_superconductivity_candidate_table.csv"
    coefficient_table_path = output_dir / "phase157_uci_superconductivity_coefficient_table.csv"
    audit_table_path = output_dir / "phase157_uci_superconductivity_audit_table.csv"
    gate_path = output_dir / "phase157_uci_superconductivity_low_capacity_mechanism_gate.json"
    markdown_path = output_dir / "phase157_uci_superconductivity_low_capacity_mechanism_gate.md"
    manifest_path = output_dir / "phase157_uci_superconductivity_low_capacity_mechanism_manifest.json"

    _write_csv(feature_schema_path, feature_rows, FEATURE_SCHEMA_FIELDS)
    _write_csv(metric_table_path, metric_rows, METRIC_FIELDS)
    _write_csv(candidate_table_path, candidate_rows, CANDIDATE_FIELDS)
    _write_csv(coefficient_table_path, coefficient_rows, COEFFICIENT_FIELDS)
    _write_csv(audit_table_path, audit_rows, AUDIT_FIELDS)
    _write_json(gate_path, gate)
    with markdown_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(build_markdown(gate=gate, candidate_rows=candidate_rows, audit_rows=audit_rows))
    manifest = {
        "phase": 157,
        "description": "no-training low-capacity mechanism gate for UCI superconductivity",
        "inputs": {
            "phase156_dir": _display_path(phase156_dir, root),
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
    parser.add_argument("--phase156-dir", type=Path, default=DEFAULT_PHASE156_DIR)
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_package(
        root=args.root,
        phase156_dir=args.phase156_dir,
        raw_path=args.raw_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
