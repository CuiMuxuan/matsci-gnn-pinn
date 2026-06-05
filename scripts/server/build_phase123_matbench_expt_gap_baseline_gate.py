#!/usr/bin/env python3
"""Build a Phase 123 baseline-first gate for Matbench experimental band gaps.

This phase opens a fresh small public external data-source intake after the
Matbench steels mechanism branch closed. It downloads Matbench v0.1
``matbench_expt_gap`` if needed, parses formulas including parenthesized groups,
derives chemistry descriptors, and reviews the experimental band-gap target with
strong tabular baselines. It does not train a neural model or open A100 training.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SOURCE_URL = "https://ml.materialsproject.org/projects/matbench_expt_gap.json.gz"
DEFAULT_RAW_PATH = Path("data/raw/external/matbench_expt_gap/matbench_expt_gap.json.gz")
DEFAULT_OUTPUT_DIR = Path("docs/results/phase123_matbench_expt_gap_baseline_gate")
EXPECTED_MIN_BYTES = 20_000
EXPECTED_HEAD_BYTES = 37_200
MIN_ROWS_FOR_REVIEW = 1_000
MIN_SPLIT_ROWS = 200
MIN_RELATIVE_VAL_GAIN = 0.05
SHORTCUT_DOMINANCE_TOLERANCE = 1.02

ELEMENTS = (
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
)
COMMON_ELEMENTS = (
    "O",
    "S",
    "Se",
    "Te",
    "P",
    "Si",
    "Ge",
    "Ga",
    "In",
    "Sn",
    "Sb",
    "Bi",
    "Cu",
    "Zn",
    "Cd",
    "Pb",
    "Li",
    "Na",
    "K",
    "Ba",
    "Ti",
)
ELECTRONEGATIVITY = {
    "H": 2.20,
    "Li": 0.98,
    "Be": 1.57,
    "B": 2.04,
    "C": 2.55,
    "N": 3.04,
    "O": 3.44,
    "F": 3.98,
    "Na": 0.93,
    "Mg": 1.31,
    "Al": 1.61,
    "Si": 1.90,
    "P": 2.19,
    "S": 2.58,
    "Cl": 3.16,
    "K": 0.82,
    "Ca": 1.00,
    "Sc": 1.36,
    "Ti": 1.54,
    "V": 1.63,
    "Cr": 1.66,
    "Mn": 1.55,
    "Fe": 1.83,
    "Co": 1.88,
    "Ni": 1.91,
    "Cu": 1.90,
    "Zn": 1.65,
    "Ga": 1.81,
    "Ge": 2.01,
    "As": 2.18,
    "Se": 2.55,
    "Br": 2.96,
    "Rb": 0.82,
    "Sr": 0.95,
    "Y": 1.22,
    "Zr": 1.33,
    "Nb": 1.60,
    "Mo": 2.16,
    "Tc": 1.90,
    "Ru": 2.20,
    "Rh": 2.28,
    "Pd": 2.20,
    "Ag": 1.93,
    "Cd": 1.69,
    "In": 1.78,
    "Sn": 1.96,
    "Sb": 2.05,
    "Te": 2.10,
    "I": 2.66,
    "Xe": 2.60,
    "Cs": 0.79,
    "Ba": 0.89,
    "La": 1.10,
    "Ce": 1.12,
    "Pr": 1.13,
    "Nd": 1.14,
    "Pm": 1.13,
    "Sm": 1.17,
    "Eu": 1.20,
    "Gd": 1.20,
    "Tb": 1.20,
    "Dy": 1.22,
    "Ho": 1.23,
    "Er": 1.24,
    "Tm": 1.25,
    "Yb": 1.10,
    "Lu": 1.27,
    "Hf": 1.30,
    "Ta": 1.50,
    "W": 2.36,
    "Re": 1.90,
    "Os": 2.20,
    "Ir": 2.20,
    "Pt": 2.28,
    "Au": 2.54,
    "Hg": 2.00,
    "Tl": 1.62,
    "Pb": 2.33,
    "Bi": 2.02,
    "Th": 1.30,
    "U": 1.38,
}
ATOMIC_NUMBER = {element: index + 1 for index, element in enumerate(ELEMENTS)}
ALKALI = {"Li", "Na", "K", "Rb", "Cs"}
ALKALINE_EARTH = {"Be", "Mg", "Ca", "Sr", "Ba"}
HALOGENS = {"F", "Cl", "Br", "I"}
CHALCOGENS = {"O", "S", "Se", "Te"}
PNICTOGENS = {"N", "P", "As", "Sb", "Bi"}
TRANSITION_METALS = {
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
}
POST_TRANSITION = {"Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi"}
METALLOIDS = {"B", "Si", "Ge", "As", "Sb", "Te"}
RARE_EARTHS = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}

PROFILE_COLUMNS = {
    "common_element_fractions": {
        "numeric": tuple(f"frac_{element}" for element in COMMON_ELEMENTS),
        "categorical": (),
    },
    "all_element_fractions": {
        "numeric": tuple(f"frac_{element}" for element in ELEMENTS),
        "categorical": (),
    },
    "chemistry_descriptors": {
        "numeric": (
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
        ),
        "categorical": (),
    },
    "common_plus_descriptors": {
        "numeric": (
            *(f"frac_{element}" for element in COMMON_ELEMENTS),
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
            "mean_electronegativity",
            "electronegativity_range",
        ),
        "categorical": (),
    },
    "chemistry_family_descriptor": {
        "numeric": (
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
            "mean_electronegativity",
            "electronegativity_range",
        ),
        "categorical": ("dominant_element", "chemistry_family_key"),
    },
    "composition_hash_shortcut": {
        "numeric": (),
        "categorical": ("composition_hash16",),
    },
}
MODEL_METHODS = ("knn", "extra_trees", "hist_gradient_boosting")
PROFILE_METHODS = {
    "composition_hash_shortcut": ("knn",),
}

FIELD_FIELDS = (
    "phase123_row_id",
    "composition",
    "gap_expt_ev",
    "dominant_element",
    "chemistry_family_key",
    "composition_hash16",
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
    *(f"frac_{element}" for element in ELEMENTS),
)
METRIC_FIELDS = (
    "target",
    "profile",
    "method",
    "split",
    "n_rows",
    "rmse",
    "mae",
    "r2",
    "nrmse_train_std",
)
REVIEW_FIELDS = (
    "target",
    "row_count",
    "train_rows",
    "val_rows",
    "test_rows",
    "mean_val_rmse",
    "mean_test_rmse",
    "best_profile",
    "best_method",
    "best_val_rmse",
    "best_test_rmse",
    "val_gain_vs_mean",
    "test_gain_vs_mean",
    "shortcut_profile_val_rmse",
    "shortcut_profile_test_rmse",
    "shortcut_blocks",
    "phase123_candidate",
    "status",
    "reason",
)
SCHEMA_FIELDS = (
    "column",
    "non_missing",
    "numeric_non_missing",
    "unique_values",
    "role",
    "min",
    "max",
    "std",
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
    if isinstance(value, (dict, list)):
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_source_file(path: Path, *, source_url: str, force_download: bool = False) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = False
    if force_download or not path.exists() or path.stat().st_size < EXPECTED_MIN_BYTES:
        request = urllib.request.Request(
            source_url,
            headers={"User-Agent": "Mozilla/5.0 Codex Phase123 matbench expt gap gate"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        path.write_bytes(payload)
        downloaded = True
    size = path.stat().st_size
    if size < EXPECTED_MIN_BYTES:
        raise ValueError(f"Downloaded source is too small: {path} has {size} bytes")
    return {
        "path": str(path),
        "source_url": source_url,
        "downloaded": downloaded,
        "byte_size": size,
        "expected_head_byte_size": EXPECTED_HEAD_BYTES,
        "sha256": _sha256(path),
    }


def _parse_number(formula: str, index: int) -> tuple[float, int]:
    start = index
    while index < len(formula) and (formula[index].isdigit() or formula[index] == "."):
        index += 1
    return (float(formula[start:index]) if index > start else 1.0, index)


def _parse_element(formula: str, index: int) -> tuple[str, int]:
    if index >= len(formula) or not formula[index].isupper():
        raise ValueError(f"Expected element symbol at position {index} in {formula!r}")
    start = index
    index += 1
    if index < len(formula) and formula[index].islower():
        index += 1
    return formula[start:index], index


def _parse_group(formula: str, index: int, stop: str | None = None) -> tuple[Counter[str], int]:
    values: Counter[str] = Counter()
    while index < len(formula) and formula[index] != stop:
        if formula[index] == "(":
            inner, index = _parse_group(formula, index + 1, ")")
            if index >= len(formula) or formula[index] != ")":
                raise ValueError(f"Unclosed parenthesis in {formula!r}")
            multiplier, index = _parse_number(formula, index + 1)
            for element, amount in inner.items():
                values[element] += amount * multiplier
        elif formula[index] == ")":
            break
        else:
            element, index = _parse_element(formula, index)
            amount, index = _parse_number(formula, index)
            values[element] += amount
    return values, index


def parse_composition(composition: str) -> dict[str, float]:
    values, index = _parse_group(composition, 0, None)
    if index != len(composition):
        raise ValueError(f"Unexpected trailing formula text in {composition!r} at position {index}")
    total = float(sum(values.values()))
    if total <= 0:
        raise ValueError(f"Composition has non-positive total: {composition}")
    unknown = sorted(element for element in values if element not in ELEMENTS)
    if unknown:
        raise ValueError(f"Unsupported elements in {composition!r}: {unknown}")
    return {element: amount / total for element, amount in values.items()}


def _entropy(fractions: list[float]) -> float:
    positive = [value for value in fractions if value > 0.0]
    return float(-sum(value * math.log(value) for value in positive))


def _weighted_mean(fractions: dict[str, float], values: dict[str, float]) -> float:
    return float(sum(fractions.get(element, 0.0) * values.get(element, 0.0) for element in fractions))


def _weighted_range(fractions: dict[str, float], values: dict[str, float]) -> float:
    present = [values[element] for element, amount in fractions.items() if amount > 0.0 and element in values]
    return float(max(present) - min(present)) if present else 0.0


def _dominant_element(fractions: dict[str, float]) -> str:
    return max(fractions.items(), key=lambda item: (item[1], item[0]))[0]


def _chemistry_family(fractions: dict[str, float]) -> str:
    high = [element for element, value in sorted(fractions.items()) if value >= 0.15]
    if not high:
        high = [_dominant_element(fractions)]
    return "+".join(high[:4])


def load_source_table(path: Path) -> pd.DataFrame:
    payload = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    if payload.get("columns") != ["composition", "gap expt"]:
        raise ValueError(f"Unexpected Matbench expt gap columns: {payload.get('columns')}")
    rows = payload.get("data")
    if not isinstance(rows, list) or len(rows) < MIN_ROWS_FOR_REVIEW:
        raise ValueError(f"Expected at least {MIN_ROWS_FOR_REVIEW} rows, found {len(rows) if isinstance(rows, list) else 'missing'}")
    return pd.DataFrame(rows, columns=["composition", "gap_expt_ev"])


def build_field_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, row in df.reset_index(drop=True).iterrows():
        composition = str(row["composition"])
        fractions = parse_composition(composition)
        element_fractions = {f"frac_{element}": float(fractions.get(element, 0.0)) for element in ELEMENTS}
        anion_fraction = sum(fractions.get(element, 0.0) for element in CHALCOGENS | HALOGENS | PNICTOGENS)
        chalcogen_fraction = sum(fractions.get(element, 0.0) for element in CHALCOGENS)
        halogen_fraction = sum(fractions.get(element, 0.0) for element in HALOGENS)
        pnictogen_fraction = sum(fractions.get(element, 0.0) for element in PNICTOGENS)
        alkali_fraction = sum(fractions.get(element, 0.0) for element in ALKALI)
        alkaline_earth_fraction = sum(fractions.get(element, 0.0) for element in ALKALINE_EARTH)
        transition_fraction = sum(fractions.get(element, 0.0) for element in TRANSITION_METALS)
        post_transition_fraction = sum(fractions.get(element, 0.0) for element in POST_TRANSITION)
        metalloid_fraction = sum(fractions.get(element, 0.0) for element in METALLOIDS)
        rare_earth_fraction = sum(fractions.get(element, 0.0) for element in RARE_EARTHS)
        rows.append(
            {
                "phase123_row_id": f"MBGAP-{index:05d}",
                "composition": composition,
                "gap_expt_ev": float(row["gap_expt_ev"]),
                "dominant_element": _dominant_element(fractions),
                "chemistry_family_key": _chemistry_family(fractions),
                "composition_hash16": hashlib.sha256(composition.encode("utf-8")).hexdigest()[:16],
                "element_count": int(sum(1 for value in fractions.values() if value > 0.0)),
                "entropy_fraction": _entropy(list(fractions.values())),
                "max_fraction": float(max(fractions.values())),
                "anion_fraction": float(anion_fraction),
                "oxygen_fraction": float(fractions.get("O", 0.0)),
                "chalcogen_fraction": float(chalcogen_fraction),
                "halogen_fraction": float(halogen_fraction),
                "pnictogen_fraction": float(pnictogen_fraction),
                "alkali_fraction": float(alkali_fraction),
                "alkaline_earth_fraction": float(alkaline_earth_fraction),
                "transition_metal_fraction": float(transition_fraction),
                "post_transition_fraction": float(post_transition_fraction),
                "metalloid_fraction": float(metalloid_fraction),
                "rare_earth_fraction": float(rare_earth_fraction),
                "mean_atomic_number": _weighted_mean(fractions, ATOMIC_NUMBER),
                "max_atomic_number": float(max(ATOMIC_NUMBER[element] for element in fractions)),
                "mean_electronegativity": _weighted_mean(fractions, ELECTRONEGATIVITY),
                "electronegativity_range": _weighted_range(fractions, ELECTRONEGATIVITY),
                **element_fractions,
            }
        )
    return pd.DataFrame(rows, columns=list(FIELD_FIELDS))


def build_schema_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in df.columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        numeric_count = int(numeric.notna().sum())
        if column == "gap_expt_ev":
            role = "target"
        elif column in {"composition", "composition_hash16"}:
            role = "identity_or_shortcut_audit"
        elif column.startswith("frac_") or column.endswith("_fraction") or column in {
            "element_count",
            "entropy_fraction",
            "max_fraction",
            "mean_atomic_number",
            "max_atomic_number",
            "mean_electronegativity",
            "electronegativity_range",
        }:
            role = "chemistry_feature"
        else:
            role = "chemistry_group_or_descriptor"
        rows.append(
            {
                "column": column,
                "non_missing": int(df[column].notna().sum()),
                "numeric_non_missing": numeric_count,
                "unique_values": int(df[column].astype(str).nunique(dropna=True)),
                "role": role,
                "min": float(numeric.min()) if numeric_count else None,
                "max": float(numeric.max()) if numeric_count else None,
                "std": float(numeric.std()) if numeric_count > 1 else None,
            }
        )
    return rows


def split_by_group(df: pd.DataFrame, group_column: str = "chemistry_family_key") -> dict[str, Any]:
    groups = sorted(str(value) for value in df[group_column].fillna("missing").unique())
    ranked = sorted(groups, key=lambda item: hashlib.sha256(f"phase123::{item}".encode("utf-8")).hexdigest())
    n_groups = len(ranked)
    train_end = max(1, int(round(n_groups * 0.6)))
    val_end = max(train_end + 1, int(round(n_groups * 0.8)))
    val_end = min(val_end, n_groups - 1)
    group_to_split = {}
    for index, group in enumerate(ranked):
        if index < train_end:
            group_to_split[group] = "train"
        elif index < val_end:
            group_to_split[group] = "val"
        else:
            group_to_split[group] = "test"
    assignments = [group_to_split[str(value)] for value in df[group_column].fillna("missing")]
    splits = {
        split: [int(index) for index, label in enumerate(assignments) if label == split]
        for split in ("train", "val", "test")
    }
    group_splits = {
        split: [group for group, label in group_to_split.items() if label == split]
        for split in ("train", "val", "test")
    }
    return {
        "split_strategy": f"group_hash_by_{group_column}",
        "group_column": group_column,
        "n_groups": n_groups,
        "splits": splits,
        "group_splits": group_splits,
        "leakage_safe": sum(len(values) for values in group_splits.values()) == n_groups,
    }


def _one_hot_frame(train: pd.DataFrame, all_rows: pd.DataFrame, profile: dict[str, tuple[str, ...]]) -> tuple[np.ndarray, np.ndarray]:
    frames_train: list[pd.DataFrame] = []
    frames_all: list[pd.DataFrame] = []
    numeric = [column for column in profile["numeric"] if column in all_rows.columns]
    categorical = [column for column in profile["categorical"] if column in all_rows.columns]
    if numeric:
        train_numeric = train[numeric].apply(pd.to_numeric, errors="coerce")
        all_numeric = all_rows[numeric].apply(pd.to_numeric, errors="coerce")
        medians = train_numeric.median(numeric_only=True).fillna(0.0)
        frames_train.append(train_numeric.fillna(medians).reset_index(drop=True))
        frames_all.append(all_numeric.fillna(medians).reset_index(drop=True))
    if categorical:
        train_cat = train[categorical].fillna("missing").astype(str)
        all_cat = all_rows[categorical].fillna("missing").astype(str)
        combined = pd.get_dummies(pd.concat([train_cat, all_cat], axis=0), dummy_na=False)
        frames_train.append(combined.iloc[: len(train_cat)].reset_index(drop=True))
        frames_all.append(combined.iloc[len(train_cat) :].reset_index(drop=True))
    if not frames_train:
        raise ValueError("Profile has no usable feature columns")
    return (
        pd.concat(frames_train, axis=1).to_numpy(dtype=float),
        pd.concat(frames_all, axis=1).to_numpy(dtype=float),
    )


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


def _fit_predict(method: str, x_train: np.ndarray, y_train: np.ndarray, x_all: np.ndarray) -> np.ndarray:
    if method == "knn":
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=max(1, min(12, len(y_train)))),
        )
    elif method == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        model = ExtraTreesRegressor(n_estimators=96, random_state=123, n_jobs=1)
    elif method == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(max_iter=160, random_state=123, early_stopping=False)
    else:
        raise ValueError(f"Unsupported method: {method}")
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_all), dtype=float)


def evaluate_target(df: pd.DataFrame, split_manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target = "gap_expt_ev"
    splits = split_manifest["splits"]
    metric_rows: list[dict[str, Any]] = []
    if any(len(splits[split]) < MIN_SPLIT_ROWS for split in ("train", "val", "test")):
        return metric_rows, {
            "target": target,
            "row_count": int(len(df)),
            "train_rows": len(splits["train"]),
            "val_rows": len(splits["val"]),
            "test_rows": len(splits["test"]),
            "status": "blocked_insufficient_split_rows",
            "reason": "one or more splits is below the minimum row count",
            "phase123_candidate": False,
        }
    y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
    train_idx = splits["train"]
    train_std = float(np.std(y[train_idx])) if train_idx else 0.0
    train_mean = float(np.mean(y[train_idx]))
    mean_pred = np.full_like(y, train_mean, dtype=float)
    for split in ("train", "val", "test"):
        values = _metrics(y[splits[split]], mean_pred[splits[split]], train_std)
        metric_rows.append(
            {
                "target": target,
                "profile": "mean",
                "method": "mean",
                "split": split,
                "n_rows": len(splits[split]),
                **values,
            }
        )
    for profile_name, profile in PROFILE_COLUMNS.items():
        x_train, x_all = _one_hot_frame(df.iloc[train_idx], df, profile)
        y_train = y[train_idx]
        for method in PROFILE_METHODS.get(profile_name, MODEL_METHODS):
            pred = _fit_predict(method, x_train, y_train, x_all)
            for split in ("train", "val", "test"):
                values = _metrics(y[splits[split]], pred[splits[split]], train_std)
                metric_rows.append(
                    {
                        "target": target,
                        "profile": profile_name,
                        "method": method,
                        "split": split,
                        "n_rows": len(splits[split]),
                        **values,
                    }
                )
    mean_val = next(row for row in metric_rows if row["profile"] == "mean" and row["split"] == "val")
    mean_test = next(row for row in metric_rows if row["profile"] == "mean" and row["split"] == "test")
    admissible_val = [
        row
        for row in metric_rows
        if row["split"] == "val" and row["profile"] not in {"mean", "composition_hash_shortcut"}
    ]
    best_val = min(admissible_val, key=lambda row: row["rmse"])
    best_test = next(
        row
        for row in metric_rows
        if row["profile"] == best_val["profile"]
        and row["method"] == best_val["method"]
        and row["split"] == "test"
    )
    shortcut_val = min(
        [row for row in metric_rows if row["split"] == "val" and row["profile"] == "composition_hash_shortcut"],
        key=lambda row: row["rmse"],
    )
    shortcut_test = next(
        row
        for row in metric_rows
        if row["profile"] == shortcut_val["profile"]
        and row["method"] == shortcut_val["method"]
        and row["split"] == "test"
    )
    val_gain = float(mean_val["rmse"]) - float(best_val["rmse"])
    test_gain = float(mean_test["rmse"]) - float(best_test["rmse"])
    min_gain = float(mean_val["rmse"]) * MIN_RELATIVE_VAL_GAIN
    shortcut_blocks = (
        float(shortcut_val["rmse"]) <= float(best_val["rmse"]) * SHORTCUT_DOMINANCE_TOLERANCE
        and float(shortcut_test["rmse"]) <= float(best_test["rmse"]) * SHORTCUT_DOMINANCE_TOLERANCE
    )
    candidate = val_gain > min_gain and test_gain > 0.0 and not shortcut_blocks
    if candidate:
        status = "phase123_candidate_gap_ready_focused_review"
        reason = "chemistry profile beats mean, preserves test gain, and is not shortcut-dominated"
    elif shortcut_blocks:
        status = "blocked_composition_hash_shortcut"
        reason = "composition identity shortcut matches or beats the safe profile"
    elif val_gain <= min_gain:
        status = "blocked_no_validation_gain"
        reason = "best chemistry profile does not clear validation gain threshold"
    else:
        status = "blocked_validation_test_reversal"
        reason = "validation gain does not preserve test gain"
    return metric_rows, {
        "target": target,
        "row_count": int(len(df)),
        "train_rows": len(splits["train"]),
        "val_rows": len(splits["val"]),
        "test_rows": len(splits["test"]),
        "mean_val_rmse": mean_val["rmse"],
        "mean_test_rmse": mean_test["rmse"],
        "best_profile": best_val["profile"],
        "best_method": best_val["method"],
        "best_val_rmse": best_val["rmse"],
        "best_test_rmse": best_test["rmse"],
        "val_gain_vs_mean": val_gain,
        "test_gain_vs_mean": test_gain,
        "shortcut_profile_val_rmse": shortcut_val["rmse"],
        "shortcut_profile_test_rmse": shortcut_test["rmse"],
        "shortcut_blocks": shortcut_blocks,
        "phase123_candidate": candidate,
        "status": status,
        "reason": reason,
    }


def build_gate(review: dict[str, Any], *, source_info: dict[str, Any], split_manifest: dict[str, Any]) -> dict[str, Any]:
    if review["status"] == "phase123_candidate_gap_ready_focused_review":
        status = "phase123_matbench_expt_gap_gap_ready_focused_review"
        focused_allowed = True
        next_action = "enter focused split/shortcut review before any model mechanism"
    elif review["status"] == "blocked_insufficient_split_rows":
        status = "phase123_matbench_expt_gap_incomplete_insufficient_split_rows"
        focused_allowed = False
        next_action = "repair split before focused review"
    else:
        status = "phase123_matbench_expt_gap_closed_no_stable_guarded_gap"
        focused_allowed = False
        next_action = "close as external-data diagnostic or choose a different public target"
    return {
        "status": status,
        "source_name": "Matbench v0.1 matbench_expt_gap",
        "source_url": source_info["source_url"],
        "source_byte_size": source_info["byte_size"],
        "source_sha256": source_info["sha256"],
        "row_count": sum(len(indices) for indices in split_manifest["splits"].values()),
        "group_count": split_manifest["n_groups"],
        "split_counts": {split: len(indices) for split, indices in split_manifest["splits"].items()},
        "leakage_safe_group_split": bool(split_manifest["leakage_safe"]),
        "selected_target": review["target"],
        "selected_profile": review["best_profile"] if focused_allowed else None,
        "selected_method": review["best_method"] if focused_allowed else None,
        "selected_validation_rmse": review["best_val_rmse"] if focused_allowed else None,
        "selected_test_rmse": review["best_test_rmse"] if focused_allowed else None,
        "mean_validation_rmse": review.get("mean_val_rmse"),
        "mean_test_rmse": review.get("mean_test_rmse"),
        "shortcut_profile_val_rmse": review.get("shortcut_profile_val_rmse"),
        "shortcut_profile_test_rmse": review.get("shortcut_profile_test_rmse"),
        "phase123_focused_review_allowed": focused_allowed,
        "phase123_model_mechanism_allowed": False,
        "phase123_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": next_action,
    }


def build_data_card(source_info: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": "Matbench v0.1 matbench_expt_gap",
        "source_url": source_info["source_url"],
        "source_sha256": source_info["sha256"],
        "source_byte_size": source_info["byte_size"],
        "license_note": "Public Matbench benchmark dataset; cite Matbench/matminer in manuscripts.",
        "target": "experimental band gap in eV",
        "row_count": gate["row_count"],
        "training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
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


def build_markdown(gate: dict[str, Any], review: dict[str, Any]) -> str:
    lines = [
        "# Phase 123 Matbench Experimental Band-Gap Baseline Gate",
        "",
        f"- Status: `{gate['status']}`",
        f"- Rows: `{gate['row_count']}`",
        f"- Selected target: `{gate['selected_target']}`",
        f"- Focused review allowed: `{gate['phase123_focused_review_allowed']}`",
        f"- Model training allowed: `{gate['phase123_model_training_allowed']}`",
        f"- A100 training allowed now: `{gate['a100_training_allowed_now']}`",
        "",
        "## Review",
        "",
        _markdown_table(
            [review],
            [
                ("Target", "target"),
                ("Status", "status"),
                ("Best profile", "best_profile"),
                ("Best method", "best_method"),
                ("Val RMSE", "best_val_rmse"),
                ("Test RMSE", "best_test_rmse"),
                ("Shortcut blocks", "shortcut_blocks"),
                ("Reason", "reason"),
            ],
        ),
    ]
    return "\n".join(lines) + "\n"


def build_package(
    *,
    root: Path,
    output_dir: Path,
    raw_path: Path,
    source_url: str = SOURCE_URL,
    force_download: bool = False,
) -> dict[str, Any]:
    source_info = ensure_source_file(raw_path, source_url=source_url, force_download=force_download)
    source_df = load_source_table(raw_path)
    field_table = build_field_table(source_df)
    split_manifest = split_by_group(field_table)
    schema_rows = build_schema_rows(field_table)
    metric_rows, review = evaluate_target(field_table, split_manifest)
    gate = build_gate(review, source_info=source_info, split_manifest=split_manifest)
    data_card = build_data_card(source_info, gate)

    output_dir.mkdir(parents=True, exist_ok=True)
    field_path = output_dir / "phase123_matbench_expt_gap_field_table.csv"
    split_path = output_dir / "phase123_matbench_expt_gap_split_manifest.json"
    schema_path = output_dir / "phase123_matbench_expt_gap_schema_table.csv"
    metric_path = output_dir / "phase123_matbench_expt_gap_metric_table.csv"
    review_path = output_dir / "phase123_matbench_expt_gap_target_review_table.csv"
    gate_path = output_dir / "phase123_matbench_expt_gap_gate.json"
    card_path = output_dir / "phase123_matbench_expt_gap_data_card.json"
    markdown_path = output_dir / "phase123_matbench_expt_gap.md"
    manifest_path = output_dir / "phase123_matbench_expt_gap_manifest.json"

    _write_csv(field_path, field_table.to_dict("records"), FIELD_FIELDS)
    _write_json(split_path, split_manifest)
    _write_csv(schema_path, schema_rows, SCHEMA_FIELDS)
    _write_csv(metric_path, metric_rows, METRIC_FIELDS)
    _write_csv(review_path, [review], REVIEW_FIELDS)
    _write_json(gate_path, gate)
    _write_json(card_path, data_card)
    markdown_path.write_text(build_markdown(gate, review), encoding="utf-8")

    manifest = {
        "phase": 123,
        "objective": "matbench_expt_gap_external_baseline_first_gate",
        "inputs": {
            "source_url": source_url,
            "raw_json_gz": _display_path(raw_path, root),
        },
        "outputs": {
            "field_table": _display_path(field_path, root),
            "split_manifest": _display_path(split_path, root),
            "schema_table": _display_path(schema_path, root),
            "metric_table": _display_path(metric_path, root),
            "target_review_table": _display_path(review_path, root),
            "gate_json": _display_path(gate_path, root),
            "data_card": _display_path(card_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "source_rows": int(len(source_df)),
            "field_rows": int(len(field_table)),
            "schema_rows": len(schema_rows),
            "metric_rows": len(metric_rows),
            "candidate_targets": 1 if gate["phase123_focused_review_allowed"] else 0,
        },
        "gate": gate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--source-url", default=SOURCE_URL)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    raw_path = args.raw_path if args.raw_path.is_absolute() else root / args.raw_path
    manifest = build_package(
        root=root,
        output_dir=output_dir,
        raw_path=raw_path,
        source_url=args.source_url,
        force_download=args.force_download,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
