#!/usr/bin/env python3
"""Build Phase 125 low-capacity mechanism gate for Matbench experimental gaps.

This phase consumes only the small Phase 123/124 Matbench experimental-gap
artifacts. It tests interpretable linear chemistry and band-gap proxy
descriptors against the Phase 124 ExtraTrees strong-baseline guard. It does not
train a neural model or open A100 training.
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


DEFAULT_PHASE123_DIR = Path("docs/results/phase123_matbench_expt_gap_baseline_gate")
DEFAULT_PHASE124_DIR = Path("docs/results/phase124_matbench_expt_gap_focused_review")
DEFAULT_OUTPUT_DIR = Path("docs/results/phase125_matbench_expt_gap_low_capacity_mechanism_gate")
MIN_SPLIT_ROWS = 200
MIN_RELATIVE_PHASE124_VAL_GAIN = 0.01
MAX_PHASE124_TEST_REVERSAL_RATIO = 1.05
MAX_LOW_CAPACITY_FEATURES = 36


def _load_phase123_module():
    script = Path(__file__).with_name("build_phase123_matbench_expt_gap_baseline_gate.py")
    spec = importlib.util.spec_from_file_location("phase123_helpers", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Phase 123 helper module from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase123 = _load_phase123_module()

ELEMENTS = tuple(phase123.ELEMENTS)
COMMON_ELEMENTS = tuple(phase123.COMMON_ELEMENTS)
ALKALI = set(phase123.ALKALI)
ALKALINE_EARTH = set(phase123.ALKALINE_EARTH)
HALOGENS = set(phase123.HALOGENS)
CHALCOGENS = set(phase123.CHALCOGENS)
PNICTOGENS = set(phase123.PNICTOGENS)
TRANSITION_METALS = set(phase123.TRANSITION_METALS)
POST_TRANSITION = set(phase123.POST_TRANSITION)
METALLOIDS = set(phase123.METALLOIDS)
RARE_EARTHS = set(phase123.RARE_EARTHS)
ELECTRONEGATIVITY = dict(phase123.ELECTRONEGATIVITY)
ATOMIC_NUMBER = dict(phase123.ATOMIC_NUMBER)

NONMETALS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Se", "Br", "I"}
P_BLOCK = {"B", "C", "N", "O", "F", "Al", "Si", "P", "S", "Cl", "Ga", "Ge", "As", "Se", "Br", "In", "Sn", "Sb", "Te", "I", "Tl", "Pb", "Bi"}
D10_ELEMENTS = {"Cu", "Zn", "Ag", "Cd", "Au", "Hg"}
HEAVY_ELEMENTS = {element for element, number in ATOMIC_NUMBER.items() if number >= 49}
LOW_EN_ELEMENTS = {element for element, value in ELECTRONEGATIVITY.items() if value <= 1.30}
HIGH_EN_ELEMENTS = {element for element, value in ELECTRONEGATIVITY.items() if value >= 2.50}

BASE_DESCRIPTOR_FEATURES = (
    "element_count",
    "entropy_fraction",
    "max_fraction",
    "anion_fraction",
    "oxygen_fraction",
    "chalcogen_fraction",
    "halogen_fraction",
    "pnictogen_fraction",
    "alkali_fraction",
    "alkaline_earth_fraction",
    "transition_metal_fraction",
    "post_transition_fraction",
    "metalloid_fraction",
    "rare_earth_fraction",
    "mean_atomic_number",
    "max_atomic_number",
    "mean_electronegativity",
    "electronegativity_range",
)

MECHANISM_PROFILES: dict[str, tuple[str, ...]] = {
    "descriptor_linear": BASE_DESCRIPTOR_FEATURES,
    "ionicity_proxy": (
        "anion_fraction",
        "oxygen_fraction",
        "halogen_fraction",
        "alkali_fraction",
        "alkaline_earth_fraction",
        "transition_metal_fraction",
        "phase125_metal_fraction",
        "phase125_low_en_fraction",
        "phase125_high_en_fraction",
        "mean_electronegativity",
        "electronegativity_range",
        "phase125_en_variance",
        "phase125_ionicity_proxy",
        "phase125_halogen_alkali_interaction",
        "phase125_oxide_ionic_proxy",
        "phase125_cation_anion_balance",
    ),
    "covalency_proxy": (
        "chalcogen_fraction",
        "pnictogen_fraction",
        "metalloid_fraction",
        "post_transition_fraction",
        "mean_electronegativity",
        "electronegativity_range",
        "entropy_fraction",
        "max_fraction",
        "phase125_nonmetal_fraction",
        "phase125_p_block_fraction",
        "phase125_covalency_proxy",
        "phase125_chalcogen_covalent_proxy",
        "phase125_pnictogen_metalloid_interaction",
        "phase125_pblock_anion_interaction",
        "phase125_en_variance",
        "phase125_descriptor_complexity",
    ),
    "heavy_element_proxy": (
        "mean_atomic_number",
        "max_atomic_number",
        "rare_earth_fraction",
        "post_transition_fraction",
        "transition_metal_fraction",
        "phase125_heavy_fraction",
        "phase125_spin_orbit_proxy",
        "phase125_d10_fraction",
        "phase125_heavy_chalcogen_interaction",
        "phase125_post_transition_chalcogen",
        "phase125_transition_chalcogen",
        "phase125_heavy_anion_interaction",
        "phase125_atomic_number_variance",
        "electronegativity_range",
    ),
    "anion_cation_interaction": (
        "anion_fraction",
        "chalcogen_fraction",
        "halogen_fraction",
        "pnictogen_fraction",
        "alkali_fraction",
        "alkaline_earth_fraction",
        "transition_metal_fraction",
        "post_transition_fraction",
        "rare_earth_fraction",
        "phase125_metal_fraction",
        "phase125_nonmetal_fraction",
        "phase125_alkali_alkaline_fraction",
        "phase125_anion_metal_interaction",
        "phase125_chalcogen_metal_interaction",
        "phase125_halogen_alkali_interaction",
        "phase125_pnictogen_transition_interaction",
        "phase125_rare_earth_halogen_interaction",
        "phase125_cation_anion_balance",
        "mean_electronegativity",
        "electronegativity_range",
    ),
    "mechanism_compact": (
        "element_count",
        "entropy_fraction",
        "max_fraction",
        "anion_fraction",
        "oxygen_fraction",
        "chalcogen_fraction",
        "halogen_fraction",
        "pnictogen_fraction",
        "transition_metal_fraction",
        "post_transition_fraction",
        "metalloid_fraction",
        "rare_earth_fraction",
        "mean_atomic_number",
        "max_atomic_number",
        "mean_electronegativity",
        "electronegativity_range",
        "phase125_metal_fraction",
        "phase125_nonmetal_fraction",
        "phase125_p_block_fraction",
        "phase125_heavy_fraction",
        "phase125_spin_orbit_proxy",
        "phase125_en_variance",
        "phase125_ionicity_proxy",
        "phase125_covalency_proxy",
        "phase125_anion_metal_interaction",
        "phase125_chalcogen_covalent_proxy",
        "phase125_heavy_chalcogen_interaction",
        "phase125_descriptor_complexity",
    ),
    "mechanism_full_low_capacity": (
        *BASE_DESCRIPTOR_FEATURES,
        "phase125_metal_fraction",
        "phase125_nonmetal_fraction",
        "phase125_p_block_fraction",
        "phase125_heavy_fraction",
        "phase125_d10_fraction",
        "phase125_low_en_fraction",
        "phase125_high_en_fraction",
        "phase125_spin_orbit_proxy",
        "phase125_en_variance",
        "phase125_atomic_number_variance",
        "phase125_ionicity_proxy",
        "phase125_covalency_proxy",
        "phase125_anion_metal_interaction",
        "phase125_chalcogen_metal_interaction",
        "phase125_post_transition_chalcogen",
        "phase125_heavy_chalcogen_interaction",
        "phase125_descriptor_complexity",
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
    *(ModelSpec("lasso", alpha=value) for value in (1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0)),
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
    "phase124_guard_profile",
    "phase124_guard_method",
    "phase124_guard_val_rmse",
    "phase124_guard_test_rmse",
    "phase124_nearest_neighbor_val_rmse",
    "phase124_nearest_neighbor_test_rmse",
    "phase123_selected_profile",
    "phase123_selected_method",
    "phase123_selected_val_rmse",
    "phase123_selected_test_rmse",
    "val_gain_vs_phase124",
    "relative_val_gain_vs_phase124",
    "test_gain_vs_phase124",
    "test_reversal_ratio_vs_phase124",
    "val_gain_vs_phase123",
    "test_gain_vs_phase123",
    "clears_phase124_validation_guard",
    "clears_phase124_test_guard",
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


def _frac_sum(df: pd.DataFrame, elements: set[str]) -> pd.Series:
    present = [f"frac_{element}" for element in sorted(elements) if f"frac_{element}" in df.columns]
    if not present:
        return pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    return df[present].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)


def _bounded_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.astype(float) / denominator.astype(float).clip(lower=1e-12)


def build_mechanism_feature_table(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    out = df.copy()
    feature_rows: dict[str, dict[str, Any]] = {}

    def register_feature(name: str, *, role: str, expression: str, source: str) -> None:
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

    def add_feature(name: str, values: Any, *, role: str, expression: str, source: str = "phase125") -> None:
        out[name] = pd.Series(values, index=out.index, dtype=float)
        register_feature(name, role=role, expression=expression, source=source)

    for name in BASE_DESCRIPTOR_FEATURES:
        if name not in out.columns:
            raise ValueError(f"Missing Phase 123 descriptor column: {name}")
        register_feature(name, role="phase123_chemistry_descriptor", expression=name, source="phase123")

    metal_fraction = (
        pd.to_numeric(out["alkali_fraction"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["alkaline_earth_fraction"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["transition_metal_fraction"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["post_transition_fraction"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["rare_earth_fraction"], errors="coerce").fillna(0.0)
    )
    nonmetal_fraction = _frac_sum(out, NONMETALS)
    p_block_fraction = _frac_sum(out, P_BLOCK)
    heavy_fraction = _frac_sum(out, HEAVY_ELEMENTS)
    d10_fraction = _frac_sum(out, D10_ELEMENTS)
    low_en_fraction = _frac_sum(out, LOW_EN_ELEMENTS)
    high_en_fraction = _frac_sum(out, HIGH_EN_ELEMENTS)
    alkali_alkaline = pd.to_numeric(out["alkali_fraction"], errors="coerce").fillna(0.0) + pd.to_numeric(
        out["alkaline_earth_fraction"], errors="coerce"
    ).fillna(0.0)

    add_feature("phase125_metal_fraction", metal_fraction, role="cation_fraction_proxy", expression="alkali + alkaline_earth + transition + post_transition + rare_earth fractions")
    add_feature("phase125_nonmetal_fraction", nonmetal_fraction, role="anion_fraction_proxy", expression="sum nonmetal element fractions")
    add_feature("phase125_p_block_fraction", p_block_fraction, role="orbital_family_proxy", expression="sum p-block element fractions")
    add_feature("phase125_heavy_fraction", heavy_fraction, role="heavy_element_proxy", expression="sum atomic-number >= 49 element fractions")
    add_feature("phase125_d10_fraction", d10_fraction, role="d10_closed_shell_proxy", expression="sum Cu/Zn/Ag/Cd/Au/Hg fractions")
    add_feature("phase125_low_en_fraction", low_en_fraction, role="low_electronegativity_proxy", expression="sum element fractions with electronegativity <= 1.30")
    add_feature("phase125_high_en_fraction", high_en_fraction, role="high_electronegativity_proxy", expression="sum element fractions with electronegativity >= 2.50")
    add_feature("phase125_alkali_alkaline_fraction", alkali_alkaline, role="ionic_cation_proxy", expression="alkali_fraction + alkaline_earth_fraction")

    weighted_en = pd.Series(np.zeros(len(out), dtype=float), index=out.index)
    weighted_en2 = pd.Series(np.zeros(len(out), dtype=float), index=out.index)
    weighted_z = pd.Series(np.zeros(len(out), dtype=float), index=out.index)
    weighted_z2 = pd.Series(np.zeros(len(out), dtype=float), index=out.index)
    spin_orbit = pd.Series(np.zeros(len(out), dtype=float), index=out.index)
    for element in ELEMENTS:
        frac = pd.to_numeric(out.get(f"frac_{element}", 0.0), errors="coerce").fillna(0.0)
        en = float(ELECTRONEGATIVITY.get(element, 0.0))
        number = float(ATOMIC_NUMBER.get(element, 0.0))
        weighted_en += frac * en
        weighted_en2 += frac * en * en
        weighted_z += frac * number
        weighted_z2 += frac * number * number
        spin_orbit += frac * (number / max(ATOMIC_NUMBER.values())) ** 2
    add_feature("phase125_en_variance", (weighted_en2 - weighted_en**2).clip(lower=0.0), role="bond_polarity_dispersion", expression="E[EN^2] - E[EN]^2")
    add_feature("phase125_atomic_number_variance", (weighted_z2 - weighted_z**2).clip(lower=0.0), role="atomic_size_dispersion_proxy", expression="E[Z^2] - E[Z]^2")
    add_feature("phase125_spin_orbit_proxy", spin_orbit, role="spin_orbit_heavy_element_proxy", expression="sum(frac_element * (Z / 92)^2)")

    anion_fraction = pd.to_numeric(out["anion_fraction"], errors="coerce").fillna(0.0)
    chalcogen_fraction = pd.to_numeric(out["chalcogen_fraction"], errors="coerce").fillna(0.0)
    halogen_fraction = pd.to_numeric(out["halogen_fraction"], errors="coerce").fillna(0.0)
    pnictogen_fraction = pd.to_numeric(out["pnictogen_fraction"], errors="coerce").fillna(0.0)
    oxygen_fraction = pd.to_numeric(out["oxygen_fraction"], errors="coerce").fillna(0.0)
    transition_fraction = pd.to_numeric(out["transition_metal_fraction"], errors="coerce").fillna(0.0)
    post_transition_fraction = pd.to_numeric(out["post_transition_fraction"], errors="coerce").fillna(0.0)
    metalloid_fraction = pd.to_numeric(out["metalloid_fraction"], errors="coerce").fillna(0.0)
    rare_earth_fraction = pd.to_numeric(out["rare_earth_fraction"], errors="coerce").fillna(0.0)
    entropy_fraction = pd.to_numeric(out["entropy_fraction"], errors="coerce").fillna(0.0)
    en_range = pd.to_numeric(out["electronegativity_range"], errors="coerce").fillna(0.0)
    max_fraction = pd.to_numeric(out["max_fraction"], errors="coerce").fillna(0.0)

    add_feature("phase125_ionicity_proxy", en_range * anion_fraction * metal_fraction, role="ionic_bond_proxy", expression="electronegativity_range * anion_fraction * metal_fraction")
    add_feature("phase125_covalency_proxy", metalloid_fraction * p_block_fraction * (1.0 - max_fraction), role="covalent_network_proxy", expression="metalloid_fraction * p_block_fraction * (1 - max_fraction)")
    add_feature("phase125_anion_metal_interaction", anion_fraction * metal_fraction, role="anion_cation_interaction", expression="anion_fraction * metal_fraction")
    add_feature("phase125_chalcogen_metal_interaction", chalcogen_fraction * metal_fraction, role="chalcogenide_interaction", expression="chalcogen_fraction * metal_fraction")
    add_feature("phase125_halogen_alkali_interaction", halogen_fraction * alkali_alkaline, role="halide_ionic_interaction", expression="halogen_fraction * (alkali_fraction + alkaline_earth_fraction)")
    add_feature("phase125_oxide_ionic_proxy", oxygen_fraction * alkali_alkaline, role="oxide_ionic_proxy", expression="oxygen_fraction * (alkali_fraction + alkaline_earth_fraction)")
    add_feature("phase125_chalcogen_covalent_proxy", chalcogen_fraction * metalloid_fraction, role="covalent_chalcogenide_proxy", expression="chalcogen_fraction * metalloid_fraction")
    add_feature("phase125_pnictogen_metalloid_interaction", pnictogen_fraction * metalloid_fraction, role="pnictide_covalency_proxy", expression="pnictogen_fraction * metalloid_fraction")
    add_feature("phase125_pblock_anion_interaction", p_block_fraction * anion_fraction, role="pblock_anion_proxy", expression="p_block_fraction * anion_fraction")
    add_feature("phase125_pnictogen_transition_interaction", pnictogen_fraction * transition_fraction, role="transition_pnictide_proxy", expression="pnictogen_fraction * transition_metal_fraction")
    add_feature("phase125_rare_earth_halogen_interaction", rare_earth_fraction * halogen_fraction, role="rare_earth_halide_proxy", expression="rare_earth_fraction * halogen_fraction")
    add_feature("phase125_heavy_chalcogen_interaction", heavy_fraction * chalcogen_fraction, role="heavy_chalcogenide_proxy", expression="heavy_fraction * chalcogen_fraction")
    add_feature("phase125_post_transition_chalcogen", post_transition_fraction * chalcogen_fraction, role="post_transition_chalcogenide_proxy", expression="post_transition_fraction * chalcogen_fraction")
    add_feature("phase125_transition_chalcogen", transition_fraction * chalcogen_fraction, role="transition_chalcogenide_proxy", expression="transition_metal_fraction * chalcogen_fraction")
    add_feature("phase125_heavy_anion_interaction", heavy_fraction * anion_fraction, role="heavy_anion_proxy", expression="heavy_fraction * anion_fraction")
    add_feature("phase125_cation_anion_balance", _bounded_ratio(metal_fraction, anion_fraction + nonmetal_fraction), role="bounded_cation_anion_ratio", expression="metal_fraction / max(anion_fraction + nonmetal_fraction, 1e-12)")
    add_feature("phase125_descriptor_complexity", entropy_fraction * (1.0 - max_fraction) * (1.0 + en_range), role="composition_complexity_proxy", expression="entropy_fraction * (1 - max_fraction) * (1 + electronegativity_range)")

    for profile_name, features in MECHANISM_PROFILES.items():
        if len(features) > MAX_LOW_CAPACITY_FEATURES:
            raise ValueError(f"Profile {profile_name} has {len(features)} features, above low-capacity cap")
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
        estimator = Ridge(alpha=float(spec.alpha), random_state=125)
    elif spec.model == "lasso":
        estimator = Lasso(alpha=float(spec.alpha), max_iter=20_000, random_state=125)
    elif spec.model == "elastic_net":
        estimator = ElasticNet(
            alpha=float(spec.alpha),
            l1_ratio=float(spec.l1_ratio),
            max_iter=20_000,
            random_state=125,
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


def _guard_from_phase124(
    *,
    phase123_gate: dict[str, Any],
    phase124_gate: dict[str, Any],
    phase124_split_rows: pd.DataFrame,
) -> dict[str, Any]:
    phase124_val = _safe_float(phase124_gate.get("original_best_admissible_val_rmse"))
    phase124_test = _safe_float(phase124_gate.get("original_best_admissible_test_rmse"))
    if phase124_val is None or phase124_test is None:
        original = phase124_split_rows[phase124_split_rows["split_id"].astype(str) == "phase123_registered_split"]
        if not original.empty:
            row = original.iloc[0].to_dict()
            phase124_val = phase124_val if phase124_val is not None else _safe_float(row.get("best_admissible_val_rmse"))
            phase124_test = phase124_test if phase124_test is not None else _safe_float(row.get("best_admissible_test_rmse"))
    if phase124_val is None or phase124_test is None:
        raise ValueError("Could not determine Phase 124 strong-baseline guard RMSE values")
    return {
        "phase124_guard_profile": phase124_gate.get("original_best_admissible_profile"),
        "phase124_guard_method": phase124_gate.get("original_best_admissible_method"),
        "phase124_guard_val_rmse": phase124_val,
        "phase124_guard_test_rmse": phase124_test,
        "phase124_nearest_neighbor_val_rmse": _safe_float(phase124_gate.get("original_nearest_neighbor_val_rmse")),
        "phase124_nearest_neighbor_test_rmse": _safe_float(phase124_gate.get("original_nearest_neighbor_test_rmse")),
        "phase123_selected_profile": phase123_gate.get("selected_profile"),
        "phase123_selected_method": phase123_gate.get("selected_method"),
        "phase123_selected_val_rmse": _safe_float(phase123_gate.get("selected_validation_rmse")),
        "phase123_selected_test_rmse": _safe_float(phase123_gate.get("selected_test_rmse")),
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

    phase124_val = float(guard["phase124_guard_val_rmse"])
    phase124_test = float(guard["phase124_guard_test_rmse"])
    phase123_val = guard.get("phase123_selected_val_rmse")
    phase123_test = guard.get("phase123_selected_test_rmse")

    for profile_name, feature_names in MECHANISM_PROFILES.items():
        if len(feature_names) > MAX_LOW_CAPACITY_FEATURES:
            raise ValueError(f"Profile {profile_name} has {len(feature_names)} features, above low-capacity cap")
        frame = df[list(feature_names)].apply(pd.to_numeric, errors="coerce")
        medians = frame.iloc[train_idx].median(numeric_only=True).fillna(0.0)
        x_all = frame.fillna(medians).to_numpy(dtype=float)
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
                "val_gain_vs_phase124": phase124_val - val_rmse,
                "relative_val_gain_vs_phase124": (phase124_val - val_rmse) / phase124_val if phase124_val > 0 else None,
                "test_gain_vs_phase124": phase124_test - test_rmse,
                "test_reversal_ratio_vs_phase124": test_rmse / phase124_test if phase124_test > 0 else None,
                "val_gain_vs_phase123": phase123_val - val_rmse if phase123_val is not None else None,
                "test_gain_vs_phase123": phase123_test - test_rmse if phase123_test is not None else None,
                "clears_phase124_validation_guard": (
                    (phase124_val - val_rmse) / phase124_val >= MIN_RELATIVE_PHASE124_VAL_GAIN
                    if phase124_val > 0
                    else False
                ),
                "clears_phase124_test_guard": (
                    test_rmse <= phase124_test * MAX_PHASE124_TEST_REVERSAL_RATIO if phase124_test > 0 else False
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
    phase123_gate: dict[str, Any],
    phase124_gate: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = next(row for row in candidate_rows if _is_true(row["selected_low_capacity"]))
    phase123_ready = phase123_gate.get("status") == "phase123_matbench_expt_gap_gap_ready_focused_review"
    phase124_ready = (
        phase124_gate.get("status")
        == "phase124_matbench_expt_gap_focused_review_ready_low_capacity_mechanism_gate"
    )
    phase124_mechanism_allowed = _is_true(phase124_gate.get("phase124_model_mechanism_allowed"))
    phase124_training_locked = not _is_true(phase124_gate.get("phase124_model_training_allowed"))
    rows = [
        {
            "audit": "phase123_gate_status",
            "status": "pass" if phase123_ready else "block",
            "severity": "blocking" if not phase123_ready else "info",
            "value": phase123_gate.get("status"),
            "threshold": "phase123_matbench_expt_gap_gap_ready_focused_review",
            "reason": "Phase 125 consumes the Phase 123 registered Matbench experimental-gap target table",
        },
        {
            "audit": "phase124_gate_status",
            "status": "pass" if phase124_ready else "block",
            "severity": "blocking" if not phase124_ready else "info",
            "value": phase124_gate.get("status"),
            "threshold": "phase124_matbench_expt_gap_focused_review_ready_low_capacity_mechanism_gate",
            "reason": "low-capacity mechanism design requires the Phase 124 focused-review pass",
        },
        {
            "audit": "phase124_mechanism_allowed",
            "status": "pass" if phase124_mechanism_allowed else "block",
            "severity": "blocking" if not phase124_mechanism_allowed else "info",
            "value": phase124_gate.get("phase124_model_mechanism_allowed"),
            "threshold": True,
            "reason": "Phase 124 must explicitly allow only no-training mechanism design",
        },
        {
            "audit": "phase124_training_lock",
            "status": "pass" if phase124_training_locked else "block",
            "severity": "blocking" if not phase124_training_locked else "info",
            "value": phase124_gate.get("phase124_model_training_allowed"),
            "threshold": False,
            "reason": "Phase 125 must not inherit an open training flag",
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
            "audit": "phase124_validation_guard_gain",
            "status": "pass" if _is_true(selected["clears_phase124_validation_guard"]) else "block",
            "severity": "blocking" if not _is_true(selected["clears_phase124_validation_guard"]) else "info",
            "value": selected["relative_val_gain_vs_phase124"],
            "threshold": MIN_RELATIVE_PHASE124_VAL_GAIN,
            "reason": "validation-only selected low-capacity mechanism must beat the Phase 124 ExtraTrees guard",
        },
        {
            "audit": "phase124_test_reversal_guard",
            "status": "pass" if _is_true(selected["clears_phase124_test_guard"]) else "block",
            "severity": "blocking" if not _is_true(selected["clears_phase124_test_guard"]) else "info",
            "value": selected["test_reversal_ratio_vs_phase124"],
            "threshold": MAX_PHASE124_TEST_REVERSAL_RATIO,
            "reason": "validation gain must not reverse badly on the held-out test split",
        },
        {
            "audit": "phase124_nearest_neighbor_comparison",
            "status": "info",
            "severity": "info",
            "value": selected["phase124_nearest_neighbor_val_rmse"],
            "threshold": "reported only",
            "reason": "Phase 124 nearest-neighbor identity control remains a shortcut comparison, not the Phase 125 selection target",
        },
    ]
    return rows


def build_gate(
    *,
    phase123_gate: dict[str, Any],
    phase124_gate: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = next(row for row in candidate_rows if _is_true(row["selected_low_capacity"]))
    blockers = [row for row in audit_rows if row["status"] == "block"]
    phase124_ready = (
        phase124_gate.get("status")
        == "phase124_matbench_expt_gap_focused_review_ready_low_capacity_mechanism_gate"
    )
    if not phase124_ready:
        status = "phase125_matbench_expt_gap_low_capacity_mechanism_blocked_by_phase124"
        mechanism_positive = False
        next_action = "complete or close Phase 124 before mechanism design"
    elif blockers:
        if any(row["audit"] == "phase124_test_reversal_guard" for row in blockers) and not any(
            row["audit"] == "phase124_validation_guard_gain" for row in blockers
        ):
            status = "phase125_matbench_expt_gap_low_capacity_mechanism_closed_validation_test_reversal"
        else:
            status = "phase125_matbench_expt_gap_low_capacity_mechanism_closed_no_guarded_gain"
        mechanism_positive = False
        next_action = "close the low-capacity mechanism branch as diagnostic; do not train"
    else:
        status = "phase125_matbench_expt_gap_low_capacity_mechanism_ready_focused_validation"
        mechanism_positive = True
        next_action = "run a separate focused validation or symbolic mechanism review before any neural training"
    return {
        "status": status,
        "phase123_status": phase123_gate.get("status"),
        "phase124_status": phase124_gate.get("status"),
        "selected_target": selected["target"],
        "selected_low_capacity_profile": selected["profile"],
        "selected_low_capacity_model": selected["model"],
        "selected_low_capacity_model_label": selected["model_label"],
        "selected_low_capacity_alpha": selected["alpha"],
        "selected_low_capacity_l1_ratio": selected["l1_ratio"],
        "selected_low_capacity_feature_count": selected["feature_count"],
        "selected_low_capacity_val_rmse": selected["val_rmse"],
        "selected_low_capacity_test_rmse": selected["test_rmse"],
        "phase124_guard_profile": selected["phase124_guard_profile"],
        "phase124_guard_method": selected["phase124_guard_method"],
        "phase124_guard_val_rmse": selected["phase124_guard_val_rmse"],
        "phase124_guard_test_rmse": selected["phase124_guard_test_rmse"],
        "phase124_nearest_neighbor_val_rmse": selected["phase124_nearest_neighbor_val_rmse"],
        "phase124_nearest_neighbor_test_rmse": selected["phase124_nearest_neighbor_test_rmse"],
        "phase123_selected_profile": selected["phase123_selected_profile"],
        "phase123_selected_method": selected["phase123_selected_method"],
        "phase123_selected_val_rmse": selected["phase123_selected_val_rmse"],
        "phase123_selected_test_rmse": selected["phase123_selected_test_rmse"],
        "selected_val_gain_vs_phase124": selected["val_gain_vs_phase124"],
        "selected_relative_val_gain_vs_phase124": selected["relative_val_gain_vs_phase124"],
        "selected_test_gain_vs_phase124": selected["test_gain_vs_phase124"],
        "selected_test_reversal_ratio_vs_phase124": selected["test_reversal_ratio_vs_phase124"],
        "blocking_audit_rows": len(blockers),
        "blocking_audits": [row["audit"] for row in blockers],
        "candidate_rows": len(candidate_rows),
        "phase125_low_capacity_mechanism_positive": mechanism_positive,
        "phase125_focused_validation_allowed": mechanism_positive,
        "phase125_model_mechanism_allowed": mechanism_positive,
        "phase125_model_training_allowed": False,
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
        "# Phase 125 Matbench Experimental Gap Low-Capacity Mechanism Gate",
        "",
        f"- Status: `{gate['status']}`",
        f"- Selected low-capacity profile: `{gate['selected_low_capacity_profile']}`",
        f"- Selected model: `{gate['selected_low_capacity_model_label']}`",
        f"- Selected validation RMSE: `{gate['selected_low_capacity_val_rmse']:.6g}`",
        f"- Phase 124 guard validation RMSE: `{gate['phase124_guard_val_rmse']:.6g}`",
        f"- Focused validation allowed: `{gate['phase125_focused_validation_allowed']}`",
        f"- Model training allowed: `{gate['phase125_model_training_allowed']}`",
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
                ("Val gain vs Phase124", "relative_val_gain_vs_phase124"),
                ("Test ratio vs Phase124", "test_reversal_ratio_vs_phase124"),
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


def build_package(*, root: Path, phase123_dir: Path, phase124_dir: Path, output_dir: Path) -> dict[str, Any]:
    field_path = phase123_dir / "phase123_matbench_expt_gap_field_table.csv"
    split_path = phase123_dir / "phase123_matbench_expt_gap_split_manifest.json"
    phase123_gate_path = phase123_dir / "phase123_matbench_expt_gap_gate.json"
    phase124_gate_path = phase124_dir / "phase124_matbench_expt_gap_focused_review_gate.json"
    phase124_split_path = phase124_dir / "phase124_matbench_expt_gap_split_sensitivity_table.csv"

    df = pd.read_csv(field_path)
    split_manifest = _read_json(split_path)
    phase123_gate = _read_json(phase123_gate_path)
    phase124_gate = _read_json(phase124_gate_path)
    phase124_split_rows = pd.read_csv(phase124_split_path)
    target = str(phase124_gate.get("selected_target") or phase123_gate.get("selected_target") or "gap_expt_ev")

    mechanism_df, feature_schema_rows = build_mechanism_feature_table(df)
    guard = _guard_from_phase124(
        phase123_gate=phase123_gate,
        phase124_gate=phase124_gate,
        phase124_split_rows=phase124_split_rows,
    )
    metric_rows, candidate_rows, coefficient_rows = evaluate_low_capacity_models(
        mechanism_df,
        target=target,
        split_manifest=split_manifest,
        guard=guard,
    )
    audit_rows = build_audit_rows(
        phase123_gate=phase123_gate,
        phase124_gate=phase124_gate,
        candidate_rows=candidate_rows,
    )
    gate = build_gate(
        phase123_gate=phase123_gate,
        phase124_gate=phase124_gate,
        candidate_rows=candidate_rows,
        audit_rows=audit_rows,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_table_path = output_dir / "phase125_matbench_expt_gap_mechanism_feature_table.csv"
    feature_schema_path = output_dir / "phase125_matbench_expt_gap_mechanism_schema_table.csv"
    metric_path = output_dir / "phase125_matbench_expt_gap_mechanism_metric_table.csv"
    candidate_path = output_dir / "phase125_matbench_expt_gap_mechanism_candidate_table.csv"
    coefficient_path = output_dir / "phase125_matbench_expt_gap_mechanism_coefficient_table.csv"
    audit_path = output_dir / "phase125_matbench_expt_gap_mechanism_audit_table.csv"
    gate_path = output_dir / "phase125_matbench_expt_gap_low_capacity_mechanism_gate.json"
    markdown_path = output_dir / "phase125_matbench_expt_gap_low_capacity_mechanism.md"
    manifest_path = output_dir / "phase125_matbench_expt_gap_low_capacity_mechanism_manifest.json"

    _write_csv(feature_table_path, mechanism_df.to_dict("records"), tuple(mechanism_df.columns))
    _write_csv(feature_schema_path, feature_schema_rows, FEATURE_SCHEMA_FIELDS)
    _write_csv(metric_path, metric_rows, METRIC_FIELDS)
    _write_csv(candidate_path, candidate_rows, CANDIDATE_FIELDS)
    _write_csv(coefficient_path, coefficient_rows, COEFFICIENT_FIELDS)
    _write_csv(audit_path, audit_rows, AUDIT_FIELDS)
    _write_json(gate_path, gate)
    markdown_path.write_text(build_markdown(gate, audit_rows, candidate_rows, coefficient_rows), encoding="utf-8")

    manifest = {
        "phase": 125,
        "objective": "matbench_expt_gap_low_capacity_mechanism_gate_no_training",
        "inputs": {
            "phase123_dir": _display_path(phase123_dir, root),
            "phase124_dir": _display_path(phase124_dir, root),
            "field_table": _display_path(field_path, root),
            "split_manifest": _display_path(split_path, root),
            "phase123_gate": _display_path(phase123_gate_path, root),
            "phase124_gate": _display_path(phase124_gate_path, root),
            "phase124_split_sensitivity_table": _display_path(phase124_split_path, root),
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
    parser.add_argument("--phase123-dir", type=Path, default=DEFAULT_PHASE123_DIR)
    parser.add_argument("--phase124-dir", type=Path, default=DEFAULT_PHASE124_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    phase123_dir = args.phase123_dir if args.phase123_dir.is_absolute() else root / args.phase123_dir
    phase124_dir = args.phase124_dir if args.phase124_dir.is_absolute() else root / args.phase124_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = build_package(root=root, phase123_dir=phase123_dir, phase124_dir=phase124_dir, output_dir=output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
