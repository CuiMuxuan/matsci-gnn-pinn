#!/usr/bin/env python3
"""Build a Phase 138 baseline-first gate for Matbench glass formation.

This phase opens a fresh small public external source after the Phase 137
paper-evidence refresh. It downloads Matbench v0.1 ``matbench_glass`` if
needed, parses formulas through the Phase 123 composition descriptor route,
and reviews the binary ``gfa`` target with strong classification baselines and
shortcut controls. It does not train a neural model or open A100/A800 training.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SOURCE_URL = "https://ml.materialsproject.org/projects/matbench_glass.json.gz"
DEFAULT_RAW_PATH = Path("data/raw/external/matbench_glass/matbench_glass.json.gz")
DEFAULT_OUTPUT_DIR = Path("docs/results/phase138_matbench_glass_baseline_gate")
EXPECTED_MIN_BYTES = 20_000
EXPECTED_HEAD_BYTES = 39_729
MIN_ROWS_FOR_REVIEW = 1_000
MIN_SPLIT_ROWS = 200
MIN_BALANCED_ACCURACY_GAIN = 0.05
SHORTCUT_DOMINANCE_MARGIN = 0.005
TARGET_COLUMN = "gfa"


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
COMMON_ELEMENTS = (
    "B",
    "C",
    "N",
    "O",
    "F",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Zr",
    "Nb",
    "Mo",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Hf",
    "Ta",
    "W",
    "Pt",
    "Au",
)

PROFILE_COLUMNS: dict[str, dict[str, Any]] = {
    "composition_descriptors": {
        "role": "admissible",
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
    "common_element_fractions": {
        "role": "admissible",
        "numeric": tuple(f"frac_{element}" for element in COMMON_ELEMENTS),
        "categorical": (),
    },
    "all_element_fractions": {
        "role": "admissible",
        "numeric": tuple(f"frac_{element}" for element in ELEMENTS),
        "categorical": (),
    },
    "common_plus_descriptors": {
        "role": "admissible",
        "numeric": (
            *(f"frac_{element}" for element in COMMON_ELEMENTS),
            "element_count",
            "entropy_fraction",
            "max_fraction",
            "anion_fraction",
            "oxygen_fraction",
            "transition_metal_fraction",
            "post_transition_fraction",
            "metalloid_fraction",
            "mean_atomic_number",
            "mean_electronegativity",
            "electronegativity_range",
        ),
        "categorical": (),
    },
    "composition_hash_shortcut": {
        "role": "negative_control",
        "numeric": (),
        "categorical": ("composition_hash16",),
    },
    "chemistry_family_shortcut": {
        "role": "negative_control",
        "numeric": (),
        "categorical": ("chemistry_family_key",),
    },
    "dominant_element_shortcut": {
        "role": "negative_control",
        "numeric": (),
        "categorical": ("dominant_element",),
    },
}
MODEL_METHODS = ("knn", "extra_trees", "hist_gradient_boosting")
PROFILE_METHODS = {
    "composition_hash_shortcut": ("knn",),
    "chemistry_family_shortcut": ("knn",),
    "dominant_element_shortcut": ("knn", "extra_trees"),
}

FIELD_FIELDS = (
    "phase138_row_id",
    "composition",
    TARGET_COLUMN,
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
    "balanced_accuracy",
    "accuracy",
    "log_loss",
    "true_positive_rate",
    "true_negative_rate",
    "true_positive_fraction",
    "pred_positive_fraction",
)
REVIEW_FIELDS = (
    "target",
    "row_count",
    "train_rows",
    "val_rows",
    "test_rows",
    "train_positive_fraction",
    "val_positive_fraction",
    "test_positive_fraction",
    "majority_val_balanced_accuracy",
    "majority_test_balanced_accuracy",
    "best_profile",
    "best_method",
    "best_val_balanced_accuracy",
    "best_test_balanced_accuracy",
    "best_val_accuracy",
    "best_test_accuracy",
    "best_val_log_loss",
    "best_test_log_loss",
    "val_balanced_accuracy_gain_vs_majority",
    "test_balanced_accuracy_gain_vs_majority",
    "best_negative_profile",
    "best_negative_method",
    "best_negative_val_balanced_accuracy",
    "best_negative_test_balanced_accuracy",
    "best_negative_val_log_loss",
    "best_negative_test_log_loss",
    "shortcut_blocks",
    "phase138_candidate",
    "status",
    "reason",
)
SCHEMA_FIELDS = tuple(phase123.SCHEMA_FIELDS)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    phase123._write_json(path, payload)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    phase123._write_csv(path, rows, fields)


def _display_path(path: Path, root: Path | None = None) -> str:
    return phase123._display_path(path, root)


def _sha256(path: Path) -> str:
    return phase123._sha256(path)


def ensure_source_file(path: Path, *, source_url: str, force_download: bool = False) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = False
    if force_download or not path.exists() or path.stat().st_size < EXPECTED_MIN_BYTES:
        request = urllib.request.Request(
            source_url,
            headers={"User-Agent": "Mozilla/5.0 Codex Phase138 matbench glass gate"},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
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


def load_source_table(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("columns") != ["composition", TARGET_COLUMN]:
        raise ValueError(f"Unexpected matbench_glass columns: {payload.get('columns')}")
    rows = payload.get("data")
    if not isinstance(rows, list) or len(rows) < MIN_ROWS_FOR_REVIEW:
        found = len(rows) if isinstance(rows, list) else "missing"
        raise ValueError(f"Expected at least {MIN_ROWS_FOR_REVIEW} rows, found {found}")
    return pd.DataFrame(rows, columns=["composition", TARGET_COLUMN])


def _target_to_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)) and value in {0, 1}:
        return int(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return 1
        if lowered in {"false", "0", "no"}:
            return 0
    raise ValueError(f"Unsupported gfa label: {value!r}")


def build_field_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for index, row in df.reset_index(drop=True).iterrows():
        composition = str(row["composition"])
        try:
            fractions = phase123.parse_composition(composition)
            target = _target_to_int(row[TARGET_COLUMN])
        except Exception as exc:  # pragma: no cover - exercised by tests through outcome
            skipped_rows.append({"row_index": int(index), "composition": composition, "reason": str(exc)})
            continue
        element_fractions = {f"frac_{element}": float(fractions.get(element, 0.0)) for element in ELEMENTS}
        anion_fraction = sum(
            fractions.get(element, 0.0)
            for element in phase123.CHALCOGENS | phase123.HALOGENS | phase123.PNICTOGENS
        )
        rows.append(
            {
                "phase138_row_id": f"MBGLASS-{len(rows):05d}",
                "composition": composition,
                TARGET_COLUMN: int(target),
                "dominant_element": phase123._dominant_element(fractions),
                "chemistry_family_key": phase123._chemistry_family(fractions),
                "composition_hash16": hashlib.sha256(composition.encode("utf-8")).hexdigest()[:16],
                "element_count": int(sum(1 for value in fractions.values() if value > 0.0)),
                "entropy_fraction": phase123._entropy(list(fractions.values())),
                "max_fraction": float(max(fractions.values())),
                "anion_fraction": float(anion_fraction),
                "oxygen_fraction": float(fractions.get("O", 0.0)),
                "chalcogen_fraction": float(
                    sum(fractions.get(element, 0.0) for element in phase123.CHALCOGENS)
                ),
                "halogen_fraction": float(
                    sum(fractions.get(element, 0.0) for element in phase123.HALOGENS)
                ),
                "pnictogen_fraction": float(
                    sum(fractions.get(element, 0.0) for element in phase123.PNICTOGENS)
                ),
                "alkali_fraction": float(sum(fractions.get(element, 0.0) for element in phase123.ALKALI)),
                "alkaline_earth_fraction": float(
                    sum(fractions.get(element, 0.0) for element in phase123.ALKALINE_EARTH)
                ),
                "transition_metal_fraction": float(
                    sum(fractions.get(element, 0.0) for element in phase123.TRANSITION_METALS)
                ),
                "post_transition_fraction": float(
                    sum(fractions.get(element, 0.0) for element in phase123.POST_TRANSITION)
                ),
                "metalloid_fraction": float(
                    sum(fractions.get(element, 0.0) for element in phase123.METALLOIDS)
                ),
                "rare_earth_fraction": float(
                    sum(fractions.get(element, 0.0) for element in phase123.RARE_EARTHS)
                ),
                "mean_atomic_number": phase123._weighted_mean(fractions, phase123.ATOMIC_NUMBER),
                "max_atomic_number": float(max(phase123.ATOMIC_NUMBER[element] for element in fractions)),
                "mean_electronegativity": phase123._weighted_mean(fractions, phase123.ELECTRONEGATIVITY),
                "electronegativity_range": phase123._weighted_range(
                    fractions,
                    phase123.ELECTRONEGATIVITY,
                ),
                **element_fractions,
            }
        )
    return pd.DataFrame(rows, columns=list(FIELD_FIELDS)), {
        "raw_rows": int(len(df)),
        "parsed_rows": len(rows),
        "skipped_rows": len(skipped_rows),
        "skipped_row_examples": skipped_rows[:10],
    }


def group_split(
    df: pd.DataFrame,
    *,
    group_column: str = "chemistry_family_key",
    salt: str = "phase138",
) -> dict[str, Any]:
    groups = sorted(str(value) for value in df[group_column].fillna("missing").unique())
    ranked = sorted(groups, key=lambda item: hashlib.sha256(f"{salt}::{item}".encode("utf-8")).hexdigest())
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
        "split_salt": salt,
        "group_column": group_column,
        "n_groups": n_groups,
        "splits": splits,
        "group_splits": group_splits,
        "leakage_safe": sum(len(values) for values in group_splits.values()) == n_groups,
    }


def _positive_proba_from_model(model: Any, x_all: np.ndarray) -> np.ndarray:
    proba = np.asarray(model.predict_proba(x_all), dtype=float)
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        return proba[:, classes.index(1)]
    if len(classes) == 1:
        return np.full(x_all.shape[0], float(classes[0] == 1), dtype=float)
    return proba[:, -1]


def _fit_predict_proba(method: str, x_train: np.ndarray, y_train: np.ndarray, x_all: np.ndarray) -> np.ndarray:
    if len(set(int(value) for value in y_train)) < 2:
        return np.full(x_all.shape[0], float(np.mean(y_train)), dtype=float)
    if method == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=max(1, min(12, len(y_train)))),
        )
    elif method == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        model = ExtraTreesClassifier(n_estimators=128, random_state=138, n_jobs=1, class_weight="balanced")
    elif method == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(max_iter=160, random_state=138, early_stopping=False)
    else:
        raise ValueError(f"Unsupported method: {method}")
    model.fit(x_train, y_train)
    return _positive_proba_from_model(model, x_all)


def _classification_metrics(y_true: np.ndarray, positive_proba: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

    clipped = np.clip(positive_proba.astype(float), 1e-6, 1.0 - 1e-6)
    y_pred = (clipped >= 0.5).astype(int)
    positives = y_true == 1
    negatives = y_true == 0
    true_positive_rate = float(np.mean(y_pred[positives] == 1)) if positives.any() else None
    true_negative_rate = float(np.mean(y_pred[negatives] == 0)) if negatives.any() else None
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "true_positive_rate": true_positive_rate,
        "true_negative_rate": true_negative_rate,
        "true_positive_fraction": float(np.mean(y_true == 1)),
        "pred_positive_fraction": float(np.mean(y_pred == 1)),
    }


def evaluate_target(df: pd.DataFrame, split_manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    splits = split_manifest["splits"]
    metric_rows: list[dict[str, Any]] = []
    if len(df) < MIN_ROWS_FOR_REVIEW or any(len(splits[split]) < MIN_SPLIT_ROWS for split in ("train", "val", "test")):
        return metric_rows, {
            "target": TARGET_COLUMN,
            "row_count": int(len(df)),
            "train_rows": len(splits["train"]),
            "val_rows": len(splits["val"]),
            "test_rows": len(splits["test"]),
            "status": "blocked_insufficient_rows_or_split",
            "reason": "dataset or one split is below the minimum row count",
            "phase138_candidate": False,
        }

    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").to_numpy(dtype=int)
    train_idx = splits["train"]
    train_positive_fraction = float(np.mean(y[train_idx] == 1))
    majority_class = int(train_positive_fraction >= 0.5)
    majority_proba = np.full_like(y, float(majority_class), dtype=float)
    for split in ("train", "val", "test"):
        values = _classification_metrics(y[splits[split]], majority_proba[splits[split]])
        metric_rows.append(
            {
                "target": TARGET_COLUMN,
                "profile": "majority",
                "method": "majority",
                "split": split,
                "n_rows": len(splits[split]),
                **values,
            }
        )

    for profile_name, profile in PROFILE_COLUMNS.items():
        x_train, x_all = phase123._one_hot_frame(df.iloc[train_idx], df, profile)
        y_train = y[train_idx]
        for method in PROFILE_METHODS.get(profile_name, MODEL_METHODS):
            proba = _fit_predict_proba(method, x_train, y_train, x_all)
            for split in ("train", "val", "test"):
                values = _classification_metrics(y[splits[split]], proba[splits[split]])
                metric_rows.append(
                    {
                        "target": TARGET_COLUMN,
                        "profile": profile_name,
                        "method": method,
                        "split": split,
                        "n_rows": len(splits[split]),
                        **values,
                    }
                )

    majority_val = next(row for row in metric_rows if row["profile"] == "majority" and row["split"] == "val")
    majority_test = next(row for row in metric_rows if row["profile"] == "majority" and row["split"] == "test")
    admissible_profiles = {name for name, profile in PROFILE_COLUMNS.items() if profile["role"] == "admissible"}
    negative_profiles = {name for name, profile in PROFILE_COLUMNS.items() if profile["role"] == "negative_control"}
    admissible_val = [row for row in metric_rows if row["split"] == "val" and row["profile"] in admissible_profiles]
    negative_val = [row for row in metric_rows if row["split"] == "val" and row["profile"] in negative_profiles]
    best_val = max(admissible_val, key=lambda row: (row["balanced_accuracy"], -row["log_loss"]))
    best_test = next(
        row
        for row in metric_rows
        if row["profile"] == best_val["profile"]
        and row["method"] == best_val["method"]
        and row["split"] == "test"
    )
    best_negative_val = max(negative_val, key=lambda row: (row["balanced_accuracy"], -row["log_loss"]))
    best_negative_test = next(
        row
        for row in metric_rows
        if row["profile"] == best_negative_val["profile"]
        and row["method"] == best_negative_val["method"]
        and row["split"] == "test"
    )
    val_gain = float(best_val["balanced_accuracy"]) - float(majority_val["balanced_accuracy"])
    test_gain = float(best_test["balanced_accuracy"]) - float(majority_test["balanced_accuracy"])
    shortcut_blocks = (
        float(best_negative_val["balanced_accuracy"])
        >= float(best_val["balanced_accuracy"]) - SHORTCUT_DOMINANCE_MARGIN
        and float(best_negative_test["balanced_accuracy"])
        >= float(majority_test["balanced_accuracy"])
    )
    candidate = val_gain > MIN_BALANCED_ACCURACY_GAIN and test_gain > 0.0 and not shortcut_blocks
    if shortcut_blocks:
        status = "blocked_shortcut_dominance"
        reason = "composition, family, or dominant-element shortcut matches the selected safe classification profile"
    elif val_gain <= MIN_BALANCED_ACCURACY_GAIN:
        status = "blocked_no_validation_gain"
        reason = "best safe profile does not improve validation balanced accuracy over majority by the required margin"
    elif test_gain <= 0.0:
        status = "blocked_validation_test_reversal"
        reason = "validation-selected safe profile does not preserve balanced-accuracy gain on test"
    else:
        status = "ready_focused_review"
        reason = "safe chemistry profile beats majority and shortcut controls"

    review = {
        "target": TARGET_COLUMN,
        "row_count": int(len(df)),
        "train_rows": len(splits["train"]),
        "val_rows": len(splits["val"]),
        "test_rows": len(splits["test"]),
        "train_positive_fraction": train_positive_fraction,
        "val_positive_fraction": float(np.mean(y[splits["val"]] == 1)),
        "test_positive_fraction": float(np.mean(y[splits["test"]] == 1)),
        "majority_val_balanced_accuracy": majority_val["balanced_accuracy"],
        "majority_test_balanced_accuracy": majority_test["balanced_accuracy"],
        "best_profile": best_val["profile"],
        "best_method": best_val["method"],
        "best_val_balanced_accuracy": best_val["balanced_accuracy"],
        "best_test_balanced_accuracy": best_test["balanced_accuracy"],
        "best_val_accuracy": best_val["accuracy"],
        "best_test_accuracy": best_test["accuracy"],
        "best_val_log_loss": best_val["log_loss"],
        "best_test_log_loss": best_test["log_loss"],
        "val_balanced_accuracy_gain_vs_majority": val_gain,
        "test_balanced_accuracy_gain_vs_majority": test_gain,
        "best_negative_profile": best_negative_val["profile"],
        "best_negative_method": best_negative_val["method"],
        "best_negative_val_balanced_accuracy": best_negative_val["balanced_accuracy"],
        "best_negative_test_balanced_accuracy": best_negative_test["balanced_accuracy"],
        "best_negative_val_log_loss": best_negative_val["log_loss"],
        "best_negative_test_log_loss": best_negative_test["log_loss"],
        "shortcut_blocks": shortcut_blocks,
        "phase138_candidate": candidate,
        "status": status,
        "reason": reason,
    }
    return metric_rows, review


def build_schema_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    profile_numeric = set()
    profile_categorical = set()
    for profile in PROFILE_COLUMNS.values():
        profile_numeric.update(profile["numeric"])
        profile_categorical.update(profile["categorical"])
    for column in df.columns:
        series = df[column]
        numeric = pd.to_numeric(series, errors="coerce")
        role = "feature"
        if column == TARGET_COLUMN:
            role = "target"
        elif column in {"phase138_row_id", "composition"}:
            role = "identifier"
        elif column in {"composition_hash16"}:
            role = "identity_or_shortcut_audit"
        elif column in profile_categorical:
            role = "categorical_feature_or_shortcut_audit"
        elif column in profile_numeric or column.startswith("frac_"):
            role = "numeric_chemistry_feature"
        rows.append(
            {
                "column": column,
                "non_missing": int(series.notna().sum()),
                "numeric_non_missing": int(numeric.notna().sum()),
                "unique_values": int(series.nunique(dropna=True)),
                "role": role,
                "min": float(numeric.min()) if numeric.notna().any() else None,
                "max": float(numeric.max()) if numeric.notna().any() else None,
                "std": float(numeric.std()) if numeric.notna().sum() > 1 else None,
            }
        )
    return rows


def build_gate(
    *,
    source_info: dict[str, Any],
    split_manifest: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, Any]:
    focused_allowed = bool(review.get("phase138_candidate"))
    if focused_allowed:
        status = "phase138_matbench_glass_ready_focused_review"
        next_action = "enter focused split/shortcut review before any model mechanism"
    elif review.get("status") == "blocked_insufficient_rows_or_split":
        status = "phase138_matbench_glass_incomplete_insufficient_split_rows"
        next_action = "repair split or choose another public target"
    else:
        status = "phase138_matbench_glass_closed_no_stable_guarded_gap"
        next_action = "close as external-data diagnostic or choose another public target"
    return {
        "status": status,
        "source_name": "Matbench v0.1 matbench_glass",
        "source_url": source_info["source_url"],
        "source_byte_size": source_info["byte_size"],
        "source_sha256": source_info["sha256"],
        "row_count": review["row_count"],
        "group_count": split_manifest["n_groups"],
        "split_counts": {split: len(indices) for split, indices in split_manifest["splits"].items()},
        "leakage_safe_group_split": bool(split_manifest["leakage_safe"]),
        "selected_target": review["target"],
        "selected_profile": review.get("best_profile") if focused_allowed else None,
        "selected_method": review.get("best_method") if focused_allowed else None,
        "selected_validation_balanced_accuracy": (
            review.get("best_val_balanced_accuracy") if focused_allowed else None
        ),
        "selected_test_balanced_accuracy": review.get("best_test_balanced_accuracy") if focused_allowed else None,
        "majority_validation_balanced_accuracy": review.get("majority_val_balanced_accuracy"),
        "majority_test_balanced_accuracy": review.get("majority_test_balanced_accuracy"),
        "best_negative_profile": review.get("best_negative_profile"),
        "best_negative_method": review.get("best_negative_method"),
        "best_negative_validation_balanced_accuracy": review.get("best_negative_val_balanced_accuracy"),
        "best_negative_test_balanced_accuracy": review.get("best_negative_test_balanced_accuracy"),
        "train_positive_fraction": review.get("train_positive_fraction"),
        "val_positive_fraction": review.get("val_positive_fraction"),
        "test_positive_fraction": review.get("test_positive_fraction"),
        "phase138_focused_review_allowed": focused_allowed,
        "phase138_model_mechanism_allowed": False,
        "phase138_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": next_action,
        "reason": review.get("reason"),
    }


def build_data_card(source_info: dict[str, Any], gate: dict[str, Any], parse_audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": "Matbench v0.1 matbench_glass",
        "source_url": source_info["source_url"],
        "source_sha256": source_info["sha256"],
        "source_byte_size": source_info["byte_size"],
        "license_note": "Public Matbench benchmark dataset; cite Matbench/matminer in manuscripts.",
        "target": "gfa binary glass-forming ability label",
        "input": "composition formula",
        "row_count": gate["row_count"],
        "raw_row_count": parse_audit["raw_rows"],
        "skipped_row_count": parse_audit["skipped_rows"],
        "training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(_fmt(row.get(key)) for _, key in columns) + " |")
    return "\n".join([header, divider, *body])


def build_markdown(gate: dict[str, Any], review: dict[str, Any]) -> str:
    lines = [
        "# Phase 138 Matbench Glass Baseline Gate",
        "",
        f"- Status: `{gate['status']}`",
        f"- Rows: `{gate['row_count']}`",
        f"- Selected target: `{gate['selected_target']}`",
        f"- Focused review allowed: `{gate['phase138_focused_review_allowed']}`",
        f"- Model training allowed: `{gate['phase138_model_training_allowed']}`",
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
                ("Val BA", "best_val_balanced_accuracy"),
                ("Test BA", "best_test_balanced_accuracy"),
                ("Negative profile", "best_negative_profile"),
                ("Shortcut blocks", "shortcut_blocks"),
                ("Reason", "reason"),
            ],
        ),
    ]
    return "\n".join(lines) + "\n"


def build_package(
    *,
    root: Path,
    raw_path: Path,
    output_dir: Path,
    source_url: str = SOURCE_URL,
    force_download: bool = False,
) -> dict[str, Any]:
    source_info = ensure_source_file(raw_path, source_url=source_url, force_download=force_download)
    source_df = load_source_table(raw_path)
    field_df, parse_audit = build_field_table(source_df)
    split_manifest = group_split(field_df)
    schema_rows = build_schema_rows(field_df)
    metric_rows, review = evaluate_target(field_df, split_manifest)
    gate = build_gate(source_info=source_info, split_manifest=split_manifest, review=review)
    gate.update(
        {
            "raw_row_count": parse_audit["raw_rows"],
            "parsed_row_count": parse_audit["parsed_rows"],
            "skipped_row_count": parse_audit["skipped_rows"],
        }
    )
    data_card = build_data_card(source_info, gate, parse_audit)

    output_dir.mkdir(parents=True, exist_ok=True)
    field_path = output_dir / "phase138_matbench_glass_field_table.csv"
    split_path = output_dir / "phase138_matbench_glass_split_manifest.json"
    schema_path = output_dir / "phase138_matbench_glass_schema_table.csv"
    metric_path = output_dir / "phase138_matbench_glass_metric_table.csv"
    review_path = output_dir / "phase138_matbench_glass_target_review_table.csv"
    gate_path = output_dir / "phase138_matbench_glass_gate.json"
    card_path = output_dir / "phase138_matbench_glass_data_card.json"
    markdown_path = output_dir / "phase138_matbench_glass.md"
    manifest_path = output_dir / "phase138_matbench_glass_manifest.json"

    _write_csv(field_path, field_df.to_dict("records"), FIELD_FIELDS)
    _write_json(split_path, split_manifest)
    _write_csv(schema_path, schema_rows, SCHEMA_FIELDS)
    _write_csv(metric_path, metric_rows, METRIC_FIELDS)
    _write_csv(review_path, [review], REVIEW_FIELDS)
    _write_json(gate_path, gate)
    _write_json(card_path, data_card)
    markdown_path.write_text(build_markdown(gate, review), encoding="utf-8")

    manifest = {
        "phase": 138,
        "objective": "matbench_glass_classification_baseline_first_gate_no_training",
        "source": source_info,
        "inputs": {"raw_path": _display_path(raw_path, root), "source_url": source_url},
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
            "raw_rows": parse_audit["raw_rows"],
            "field_rows": int(len(field_df)),
            "skipped_rows": parse_audit["skipped_rows"],
            "schema_rows": len(schema_rows),
            "metric_rows": len(metric_rows),
            "candidate_targets": 1 if gate["phase138_focused_review_allowed"] else 0,
        },
        "parse_audit": parse_audit,
        "gate": gate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--source-url", default=SOURCE_URL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    raw_path = args.raw_path if args.raw_path.is_absolute() else root / args.raw_path
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = build_package(
        root=root,
        raw_path=raw_path,
        output_dir=output_dir,
        source_url=args.source_url,
        force_download=args.force_download,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
