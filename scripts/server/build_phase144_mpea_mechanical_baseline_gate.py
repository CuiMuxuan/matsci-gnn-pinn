#!/usr/bin/env python3
"""Build a Phase 144 baseline-first gate for the MPEA mechanical dataset.

This phase opens a fresh small public external source after the Phase 143 paper
evidence refresh. It downloads the Citrine Informatics MPEA CSV if needed,
parses multi-principal-element alloy formulas through the existing Phase 123
chemistry route, and reviews mechanical-property targets with strong tabular
baselines plus shortcut controls. It does not train a neural model or open
A100/A800 training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SOURCE_URL = "https://raw.githubusercontent.com/CitrineInformatics/MPEA_dataset/master/MPEA_dataset.csv"
DEFAULT_RAW_PATH = Path("data/raw/external/mpea_dataset/MPEA_dataset.csv")
DEFAULT_OUTPUT_DIR = Path("docs/results/phase144_mpea_mechanical_baseline_gate")
EXPECTED_MIN_BYTES = 100_000
EXPECTED_HEAD_BYTES = 332_126
MIN_ROWS_FOR_REVIEW = 80
MIN_SPLIT_ROWS = 20
MIN_RELATIVE_VAL_GAIN = 0.08
SHORTCUT_DOMINANCE_TOLERANCE = 1.02

SOURCE_COLUMNS = (
    "IDENTIFIER: Reference ID",
    "FORMULA",
    "PROPERTY: Microstructure",
    "PROPERTY: Processing method",
    "PROPERTY: BCC/FCC/other",
    "PROPERTY: grain size ($\\mu$m)",
    "PROPERTY: Exp. Density (g/cm$^3$)",
    "PROPERTY: Calculated Density (g/cm$^3$)",
    "PROPERTY: HV",
    "PROPERTY: Type of test",
    "PROPERTY: Test temperature ($^\\circ$C)",
    "PROPERTY: YS (MPa)",
    "PROPERTY: UTS (MPa)",
    "PROPERTY: Elongation (%)",
    "PROPERTY: Elongation plastic (%)",
    "PROPERTY: Exp. Young modulus (GPa)",
    "PROPERTY: Calculated Young modulus (GPa)",
    "PROPERTY: O content (wppm)",
    "PROPERTY: N content (wppm)",
    "PROPERTY: C content (wppm)",
    "REFERENCE: doi",
    "REFERENCE: year",
    "REFERENCE: title",
)

TARGET_COLUMNS = {
    "hardness_hv": "PROPERTY: HV",
    "yield_strength_mpa": "PROPERTY: YS (MPa)",
    "ultimate_tensile_strength_mpa": "PROPERTY: UTS (MPa)",
    "elongation_pct": "PROPERTY: Elongation (%)",
    "young_modulus_gpa": "PROPERTY: Exp. Young modulus (GPa)",
}

TARGET_LABELS = {
    "hardness_hv": "Vickers hardness",
    "yield_strength_mpa": "yield strength in MPa",
    "ultimate_tensile_strength_mpa": "ultimate tensile strength in MPa",
    "elongation_pct": "elongation in percent",
    "young_modulus_gpa": "experimental Young modulus in GPa",
}


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
    "Al",
    "Co",
    "Cr",
    "Cu",
    "Fe",
    "Hf",
    "Mn",
    "Mo",
    "Nb",
    "Ni",
    "Ta",
    "Ti",
    "V",
    "W",
    "Zr",
)

COMPOSITION_DESCRIPTOR_COLUMNS = (
    "element_count",
    "entropy_fraction",
    "max_fraction",
    "transition_metal_fraction",
    "post_transition_fraction",
    "rare_earth_fraction",
    "mean_atomic_number",
    "max_atomic_number",
    "mean_electronegativity",
    "electronegativity_range",
)
PROCESS_NUMERIC_COLUMNS = (
    "grain_size_um",
    "exp_density_g_cm3",
    "calc_density_g_cm3",
    "test_temperature_c",
    "oxygen_wppm",
    "nitrogen_wppm",
    "carbon_wppm",
    "reference_year",
)
PROCESS_CATEGORICAL_COLUMNS = (
    "microstructure",
    "processing_method",
    "phase_family",
    "test_type",
)

PROFILE_COLUMNS: dict[str, dict[str, Any]] = {
    "composition_descriptors": {
        "role": "admissible",
        "numeric": COMPOSITION_DESCRIPTOR_COLUMNS,
        "categorical": (),
    },
    "common_element_fractions": {
        "role": "admissible",
        "numeric": tuple(f"frac_{element}" for element in COMMON_ELEMENTS),
        "categorical": (),
    },
    "composition_process_context": {
        "role": "admissible",
        "numeric": (
            *COMPOSITION_DESCRIPTOR_COLUMNS,
            *(f"frac_{element}" for element in COMMON_ELEMENTS),
            *PROCESS_NUMERIC_COLUMNS,
        ),
        "categorical": PROCESS_CATEGORICAL_COLUMNS,
    },
    "process_only_control": {
        "role": "negative_control",
        "numeric": PROCESS_NUMERIC_COLUMNS,
        "categorical": PROCESS_CATEGORICAL_COLUMNS,
    },
    "formula_hash_shortcut": {
        "role": "negative_control",
        "numeric": (),
        "categorical": ("formula_hash16",),
    },
    "reference_shortcut": {
        "role": "negative_control",
        "numeric": (),
        "categorical": ("reference_hash16",),
    },
    "dominant_element_shortcut": {
        "role": "negative_control",
        "numeric": (),
        "categorical": ("dominant_element",),
    },
}
MODEL_METHODS = ("knn", "extra_trees", "hist_gradient_boosting")
PROFILE_METHODS = {
    "formula_hash_shortcut": ("knn",),
    "reference_shortcut": ("knn",),
    "dominant_element_shortcut": ("knn", "extra_trees"),
    "process_only_control": ("knn", "extra_trees", "hist_gradient_boosting"),
}

FIELD_FIELDS = (
    "phase144_row_id",
    "reference_id",
    "formula",
    "normalized_formula",
    "microstructure",
    "processing_method",
    "phase_family",
    "grain_size_um",
    "exp_density_g_cm3",
    "calc_density_g_cm3",
    "hardness_hv",
    "test_type",
    "test_temperature_c",
    "yield_strength_mpa",
    "ultimate_tensile_strength_mpa",
    "elongation_pct",
    "elongation_plastic_pct",
    "young_modulus_gpa",
    "calculated_young_modulus_gpa",
    "oxygen_wppm",
    "nitrogen_wppm",
    "carbon_wppm",
    "reference_doi",
    "reference_year",
    "reference_title",
    "dominant_element",
    "chemistry_family_key",
    "formula_hash16",
    "reference_hash16",
    "formula_reference_hash16",
    *COMPOSITION_DESCRIPTOR_COLUMNS,
    *(f"frac_{element}" for element in ELEMENTS),
)

METRIC_FIELDS = (
    "target",
    "target_label",
    "profile",
    "profile_role",
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
    "target_label",
    "row_count",
    "train_rows",
    "val_rows",
    "test_rows",
    "train_mean",
    "train_std",
    "mean_val_rmse",
    "mean_test_rmse",
    "best_profile",
    "best_method",
    "best_val_rmse",
    "best_test_rmse",
    "best_val_nrmse",
    "best_test_nrmse",
    "val_gain_vs_mean",
    "test_gain_vs_mean",
    "relative_val_gain_vs_mean",
    "best_negative_profile",
    "best_negative_method",
    "best_negative_val_rmse",
    "best_negative_test_rmse",
    "shortcut_blocks",
    "phase144_candidate",
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


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, train_std: float) -> dict[str, Any]:
    return phase123._metrics(y_true, y_pred, train_std)


def _fit_predict(method: str, x_train: np.ndarray, y_train: np.ndarray, x_all: np.ndarray) -> np.ndarray:
    return phase123._fit_predict(method, x_train, y_train, x_all)


def ensure_source_file(path: Path, *, source_url: str, force_download: bool = False) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = False
    if force_download or not path.exists() or path.stat().st_size < EXPECTED_MIN_BYTES:
        request = urllib.request.Request(
            source_url,
            headers={"User-Agent": "Mozilla/5.0 Codex Phase144 MPEA gate"},
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


def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "missing"
    text = str(value).strip()
    return text if text else "missing"


def _to_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _normalize_formula(formula: str) -> str:
    return re.sub(r"\s+", "", formula.strip())


def load_source_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [column for column in SOURCE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing MPEA source columns: {missing}")
    if len(df) < MIN_ROWS_FOR_REVIEW:
        raise ValueError(f"Expected at least {MIN_ROWS_FOR_REVIEW} rows, found {len(df)}")
    return df[list(SOURCE_COLUMNS)].copy()


def build_field_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for index, row in df.reset_index(drop=True).iterrows():
        formula = _clean_text(row["FORMULA"])
        normalized_formula = _normalize_formula(formula)
        try:
            fractions = phase123.parse_composition(normalized_formula)
        except Exception as exc:  # pragma: no cover - covered by tests through outcome
            skipped_rows.append({"row_index": int(index), "formula": formula, "reason": str(exc)})
            continue
        reference_id = _clean_text(row["IDENTIFIER: Reference ID"])
        reference_doi = _clean_text(row["REFERENCE: doi"])
        reference_title = _clean_text(row["REFERENCE: title"])
        formula_hash = hashlib.sha256(normalized_formula.encode("utf-8")).hexdigest()[:16]
        reference_hash = hashlib.sha256(f"{reference_id}::{reference_doi}".encode("utf-8")).hexdigest()[:16]
        element_fractions = {f"frac_{element}": float(fractions.get(element, 0.0)) for element in ELEMENTS}
        rows.append(
            {
                "phase144_row_id": f"MPEA-{len(rows):05d}",
                "reference_id": reference_id,
                "formula": formula,
                "normalized_formula": normalized_formula,
                "microstructure": _clean_text(row["PROPERTY: Microstructure"]),
                "processing_method": _clean_text(row["PROPERTY: Processing method"]),
                "phase_family": _clean_text(row["PROPERTY: BCC/FCC/other"]),
                "grain_size_um": _to_float(row["PROPERTY: grain size ($\\mu$m)"]),
                "exp_density_g_cm3": _to_float(row["PROPERTY: Exp. Density (g/cm$^3$)"]),
                "calc_density_g_cm3": _to_float(row["PROPERTY: Calculated Density (g/cm$^3$)"]),
                "hardness_hv": _to_float(row["PROPERTY: HV"]),
                "test_type": _clean_text(row["PROPERTY: Type of test"]),
                "test_temperature_c": _to_float(row["PROPERTY: Test temperature ($^\\circ$C)"]),
                "yield_strength_mpa": _to_float(row["PROPERTY: YS (MPa)"]),
                "ultimate_tensile_strength_mpa": _to_float(row["PROPERTY: UTS (MPa)"]),
                "elongation_pct": _to_float(row["PROPERTY: Elongation (%)"]),
                "elongation_plastic_pct": _to_float(row["PROPERTY: Elongation plastic (%)"]),
                "young_modulus_gpa": _to_float(row["PROPERTY: Exp. Young modulus (GPa)"]),
                "calculated_young_modulus_gpa": _to_float(row["PROPERTY: Calculated Young modulus (GPa)"]),
                "oxygen_wppm": _to_float(row["PROPERTY: O content (wppm)"]),
                "nitrogen_wppm": _to_float(row["PROPERTY: N content (wppm)"]),
                "carbon_wppm": _to_float(row["PROPERTY: C content (wppm)"]),
                "reference_doi": reference_doi,
                "reference_year": _to_float(row["REFERENCE: year"]),
                "reference_title": reference_title,
                "dominant_element": phase123._dominant_element(fractions),
                "chemistry_family_key": phase123._chemistry_family(fractions),
                "formula_hash16": formula_hash,
                "reference_hash16": reference_hash,
                "formula_reference_hash16": hashlib.sha256(
                    f"{normalized_formula}::{reference_hash}".encode("utf-8")
                ).hexdigest()[:16],
                "element_count": int(sum(1 for value in fractions.values() if value > 0.0)),
                "entropy_fraction": phase123._entropy(list(fractions.values())),
                "max_fraction": float(max(fractions.values())),
                "transition_metal_fraction": float(
                    sum(fractions.get(element, 0.0) for element in phase123.TRANSITION_METALS)
                ),
                "post_transition_fraction": float(
                    sum(fractions.get(element, 0.0) for element in phase123.POST_TRANSITION)
                ),
                "rare_earth_fraction": float(
                    sum(fractions.get(element, 0.0) for element in phase123.RARE_EARTHS)
                ),
                "mean_atomic_number": phase123._weighted_mean(fractions, phase123.ATOMIC_NUMBER),
                "max_atomic_number": float(max(phase123.ATOMIC_NUMBER[element] for element in fractions)),
                "mean_electronegativity": phase123._weighted_mean(fractions, phase123.ELECTRONEGATIVITY),
                "electronegativity_range": phase123._weighted_range(fractions, phase123.ELECTRONEGATIVITY),
                **element_fractions,
            }
        )
    return pd.DataFrame(rows, columns=list(FIELD_FIELDS)), {
        "raw_rows": int(len(df)),
        "parsed_rows": len(rows),
        "skipped_rows": len(skipped_rows),
        "skipped_row_examples": skipped_rows[:10],
    }


def split_by_group(
    df: pd.DataFrame,
    *,
    group_column: str = "formula_hash16",
    salt: str = "phase144_mpea",
) -> dict[str, Any]:
    groups = sorted(str(value) for value in df[group_column].fillna("missing").unique())
    ranked = sorted(groups, key=lambda item: hashlib.sha256(f"{salt}::{item}".encode("utf-8")).hexdigest())
    n_groups = len(ranked)
    train_end = max(1, int(round(n_groups * 0.6)))
    val_end = max(train_end + 1, int(round(n_groups * 0.8)))
    val_end = min(val_end, n_groups - 1)
    group_to_split: dict[str, str] = {}
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


def _target_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    target_df = df[pd.to_numeric(df[target], errors="coerce").notna()].copy()
    target_df[target] = pd.to_numeric(target_df[target], errors="coerce")
    return target_df.reset_index(drop=True)


def _evaluate_one_target(df: pd.DataFrame, target: str, split_manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    splits = split_manifest["splits"]
    metric_rows: list[dict[str, Any]] = []
    label = TARGET_LABELS[target]
    if len(df) < MIN_ROWS_FOR_REVIEW or any(len(splits[split]) < MIN_SPLIT_ROWS for split in ("train", "val", "test")):
        return metric_rows, {
            "target": target,
            "target_label": label,
            "row_count": int(len(df)),
            "train_rows": len(splits["train"]),
            "val_rows": len(splits["val"]),
            "test_rows": len(splits["test"]),
            "status": "blocked_insufficient_rows_or_split",
            "reason": "target or one split is below the minimum row count",
            "phase144_candidate": False,
        }
    y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
    train_idx = splits["train"]
    train_mean = float(np.mean(y[train_idx]))
    train_std = float(np.std(y[train_idx]))
    mean_pred = np.full_like(y, train_mean, dtype=float)
    for split in ("train", "val", "test"):
        values = _metrics(y[splits[split]], mean_pred[splits[split]], train_std)
        metric_rows.append(
            {
                "target": target,
                "target_label": label,
                "profile": "mean",
                "profile_role": "baseline",
                "method": "mean",
                "split": split,
                "n_rows": len(splits[split]),
                **values,
            }
        )
    for profile_name, profile in PROFILE_COLUMNS.items():
        x_train, x_all = phase123._one_hot_frame(df.iloc[train_idx], df, profile)
        y_train = y[train_idx]
        for method in PROFILE_METHODS.get(profile_name, MODEL_METHODS):
            pred = _fit_predict(method, x_train, y_train, x_all)
            for split in ("train", "val", "test"):
                values = _metrics(y[splits[split]], pred[splits[split]], train_std)
                metric_rows.append(
                    {
                        "target": target,
                        "target_label": label,
                        "profile": profile_name,
                        "profile_role": profile["role"],
                        "method": method,
                        "split": split,
                        "n_rows": len(splits[split]),
                        **values,
                    }
                )
    mean_val = next(row for row in metric_rows if row["profile"] == "mean" and row["split"] == "val")
    mean_test = next(row for row in metric_rows if row["profile"] == "mean" and row["split"] == "test")
    admissible_val = [
        row for row in metric_rows if row["split"] == "val" and row["profile_role"] == "admissible"
    ]
    negative_val = [
        row for row in metric_rows if row["split"] == "val" and row["profile_role"] == "negative_control"
    ]
    best_val = min(admissible_val, key=lambda row: row["rmse"])
    best_test = next(
        row
        for row in metric_rows
        if row["profile"] == best_val["profile"]
        and row["method"] == best_val["method"]
        and row["split"] == "test"
    )
    best_negative_val = min(negative_val, key=lambda row: row["rmse"])
    best_negative_test = next(
        row
        for row in metric_rows
        if row["profile"] == best_negative_val["profile"]
        and row["method"] == best_negative_val["method"]
        and row["split"] == "test"
    )
    val_gain = float(mean_val["rmse"]) - float(best_val["rmse"])
    test_gain = float(mean_test["rmse"]) - float(best_test["rmse"])
    rel_gain = val_gain / float(mean_val["rmse"]) if float(mean_val["rmse"]) > 0.0 else 0.0
    shortcut_blocks = (
        float(best_negative_val["rmse"]) <= float(best_val["rmse"]) * SHORTCUT_DOMINANCE_TOLERANCE
    )
    candidate = bool(rel_gain >= MIN_RELATIVE_VAL_GAIN and test_gain > 0.0 and not shortcut_blocks)
    if shortcut_blocks:
        status = "blocked_shortcut_or_process_control_dominance"
        reason = "best negative-control profile matches or beats the admissible validation RMSE"
    elif rel_gain < MIN_RELATIVE_VAL_GAIN:
        status = "closed_no_validation_gain"
        reason = "best admissible profile does not clear the relative validation gain guard"
    elif test_gain <= 0.0 or float(best_test["rmse"]) > float(mean_test["rmse"]):
        status = "closed_validation_test_reversal"
        reason = "validation-selected profile does not preserve a test RMSE gain over mean"
    else:
        status = "ready_focused_review"
        reason = "admissible profile beats mean, preserves test gain, and is not control-dominated"
    return metric_rows, {
        "target": target,
        "target_label": label,
        "row_count": int(len(df)),
        "train_rows": len(splits["train"]),
        "val_rows": len(splits["val"]),
        "test_rows": len(splits["test"]),
        "train_mean": train_mean,
        "train_std": train_std,
        "mean_val_rmse": mean_val["rmse"],
        "mean_test_rmse": mean_test["rmse"],
        "best_profile": best_val["profile"],
        "best_method": best_val["method"],
        "best_val_rmse": best_val["rmse"],
        "best_test_rmse": best_test["rmse"],
        "best_val_nrmse": best_val["nrmse_train_std"],
        "best_test_nrmse": best_test["nrmse_train_std"],
        "val_gain_vs_mean": val_gain,
        "test_gain_vs_mean": test_gain,
        "relative_val_gain_vs_mean": rel_gain,
        "best_negative_profile": best_negative_val["profile"],
        "best_negative_method": best_negative_val["method"],
        "best_negative_val_rmse": best_negative_val["rmse"],
        "best_negative_test_rmse": best_negative_test["rmse"],
        "shortcut_blocks": shortcut_blocks,
        "phase144_candidate": candidate,
        "status": status,
        "reason": reason,
    }


def evaluate_targets(df: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    metric_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    split_reviews: dict[str, Any] = {}
    for target in TARGET_COLUMNS:
        target_df = _target_frame(df, target)
        split_manifest = split_by_group(target_df, salt=f"phase144_mpea::{target}")
        target_metrics, review = _evaluate_one_target(target_df, target, split_manifest)
        metric_rows.extend(target_metrics)
        review_rows.append(review)
        split_reviews[target] = split_manifest
    candidates = [row for row in review_rows if row.get("phase144_candidate")]
    if candidates:
        selected = max(
            candidates,
            key=lambda row: (
                float(row.get("relative_val_gain_vs_mean") or 0.0),
                -float(row.get("best_val_rmse") or float("inf")),
            ),
        )
    else:
        sufficient = [row for row in review_rows if row.get("status") != "blocked_insufficient_rows_or_split"]
        selected = min(
            sufficient or review_rows,
            key=lambda row: float(row.get("best_val_rmse") or row.get("mean_val_rmse") or float("inf")),
        )
    return metric_rows, review_rows, {"selected": selected, "split_reviews": split_reviews}


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
        if column in TARGET_COLUMNS:
            role = "candidate_target"
        elif column in {"phase144_row_id", "formula", "normalized_formula", "reference_title", "reference_doi"}:
            role = "identifier_or_provenance"
        elif column in {"formula_hash16", "reference_hash16", "formula_reference_hash16"}:
            role = "identity_or_shortcut_audit"
        elif column in profile_categorical:
            role = "categorical_feature_or_shortcut_audit"
        elif column in profile_numeric or column.startswith("frac_"):
            role = "numeric_feature"
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
    parse_audit: dict[str, Any],
    review_rows: list[dict[str, Any]],
    selected: dict[str, Any],
) -> dict[str, Any]:
    focused_allowed = bool(selected.get("phase144_candidate"))
    if focused_allowed:
        status = "phase144_mpea_mechanical_ready_focused_review"
        next_action = "enter focused split/shortcut review before any model mechanism"
    elif selected.get("status") == "blocked_insufficient_rows_or_split":
        status = "phase144_mpea_mechanical_incomplete_insufficient_split_rows"
        next_action = "repair target filtering/split or choose another public source"
    else:
        status = "phase144_mpea_mechanical_closed_no_stable_guarded_gap"
        next_action = "close as external-data diagnostic or choose another public source"
    return {
        "status": status,
        "source_name": "Citrine Informatics MPEA dataset",
        "source_url": source_info["source_url"],
        "source_byte_size": source_info["byte_size"],
        "source_sha256": source_info["sha256"],
        "raw_row_count": parse_audit["raw_rows"],
        "parsed_row_count": parse_audit["parsed_rows"],
        "skipped_row_count": parse_audit["skipped_rows"],
        "candidate_targets_reviewed": len(review_rows),
        "candidate_targets_ready": sum(1 for row in review_rows if row.get("phase144_candidate")),
        "selected_target": selected["target"],
        "selected_target_label": selected["target_label"],
        "selected_row_count": selected["row_count"],
        "selected_profile": selected.get("best_profile") if focused_allowed else None,
        "selected_method": selected.get("best_method") if focused_allowed else None,
        "selected_validation_rmse": selected.get("best_val_rmse") if focused_allowed else None,
        "selected_test_rmse": selected.get("best_test_rmse") if focused_allowed else None,
        "mean_validation_rmse": selected.get("mean_val_rmse"),
        "mean_test_rmse": selected.get("mean_test_rmse"),
        "best_negative_profile": selected.get("best_negative_profile"),
        "best_negative_method": selected.get("best_negative_method"),
        "best_negative_validation_rmse": selected.get("best_negative_val_rmse"),
        "best_negative_test_rmse": selected.get("best_negative_test_rmse"),
        "phase144_focused_review_allowed": focused_allowed,
        "phase144_model_mechanism_allowed": False,
        "phase144_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": next_action,
        "reason": selected.get("reason"),
    }


def build_data_card(source_info: dict[str, Any], gate: dict[str, Any], parse_audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": "Citrine Informatics MPEA dataset",
        "source_url": source_info["source_url"],
        "source_sha256": source_info["sha256"],
        "source_byte_size": source_info["byte_size"],
        "license_note": "Public GitHub dataset with Apache-2.0 license in the source repository.",
        "input": "multi-principal-element alloy formula plus process/test context",
        "candidate_targets": TARGET_LABELS,
        "raw_row_count": parse_audit["raw_rows"],
        "parsed_row_count": parse_audit["parsed_rows"],
        "skipped_row_count": parse_audit["skipped_rows"],
        "training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(_fmt(row.get(key)) for _, key in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def build_markdown(gate: dict[str, Any], review_rows: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "# Phase 144 MPEA Mechanical Baseline Gate",
            "",
            f"- Status: `{gate['status']}`",
            f"- Source rows: `{gate['parsed_row_count']}`",
            f"- Selected target: `{gate['selected_target']}`",
            f"- Focused review allowed: `{gate['phase144_focused_review_allowed']}`",
            f"- Model training allowed: `{gate['phase144_model_training_allowed']}`",
            f"- A100 training allowed now: `{gate['a100_training_allowed_now']}`",
            f"- A100 80GB request now: `{gate['a100_80gb_request_now']}`",
            "",
            "## Target Review",
            "",
            _markdown_table(
                review_rows,
                [
                    ("Target", "target"),
                    ("Status", "status"),
                    ("Rows", "row_count"),
                    ("Best profile", "best_profile"),
                    ("Best method", "best_method"),
                    ("Val RMSE", "best_val_rmse"),
                    ("Test RMSE", "best_test_rmse"),
                    ("Negative profile", "best_negative_profile"),
                    ("Shortcut blocks", "shortcut_blocks"),
                    ("Reason", "reason"),
                ],
            ),
        ]
    ) + "\n"


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
    schema_rows = build_schema_rows(field_df)
    metric_rows, review_rows, review_payload = evaluate_targets(field_df)
    selected = review_payload["selected"]
    gate = build_gate(
        source_info=source_info,
        parse_audit=parse_audit,
        review_rows=review_rows,
        selected=selected,
    )
    data_card = build_data_card(source_info, gate, parse_audit)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "phase144_mpea_mechanical"
    field_path = output_dir / f"{prefix}_field_table.csv"
    split_path = output_dir / f"{prefix}_split_manifest.json"
    schema_path = output_dir / f"{prefix}_schema_table.csv"
    metric_path = output_dir / f"{prefix}_metric_table.csv"
    review_path = output_dir / f"{prefix}_target_review_table.csv"
    gate_path = output_dir / f"{prefix}_gate.json"
    card_path = output_dir / f"{prefix}_data_card.json"
    markdown_path = output_dir / f"{prefix}.md"
    manifest_path = output_dir / f"{prefix}_manifest.json"

    _write_csv(field_path, field_df.to_dict("records"), FIELD_FIELDS)
    _write_json(split_path, review_payload["split_reviews"])
    _write_csv(schema_path, schema_rows, SCHEMA_FIELDS)
    _write_csv(metric_path, metric_rows, METRIC_FIELDS)
    _write_csv(review_path, review_rows, REVIEW_FIELDS)
    _write_json(gate_path, gate)
    _write_json(card_path, data_card)
    markdown_path.write_text(build_markdown(gate, review_rows), encoding="utf-8")

    manifest = {
        "phase": 144,
        "objective": "mpea_mechanical_baseline_first_gate_no_training",
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
            "candidate_targets": len(review_rows),
            "candidate_targets_ready": gate["candidate_targets_ready"],
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
