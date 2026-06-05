#!/usr/bin/env python3
"""Build a Phase 128 baseline-first gate for Matbench dielectric.

This phase opens a fresh small public external data-source intake after the
Matbench phonons branch closed. It downloads Matbench v0.1
``matbench_dielectric`` if needed, parses structure JSON dictionaries into
composition and lattice descriptors, and reviews the refractive-index target
``n`` with strong tabular baselines and shortcut controls. It does not train a
neural model or open A100 training.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SOURCE_URL = "https://ml.materialsproject.org/projects/matbench_dielectric.json.gz"
DEFAULT_RAW_PATH = Path("data/raw/external/matbench_dielectric/matbench_dielectric.json.gz")
DEFAULT_OUTPUT_DIR = Path("docs/results/phase128_matbench_dielectric_baseline_gate")
EXPECTED_MIN_BYTES = 1_000_000
EXPECTED_HEAD_BYTES = 3_608_015
MIN_ROWS_FOR_REVIEW = 500
MIN_SPLIT_ROWS = 100
MIN_RELATIVE_VAL_GAIN = 0.05
SHORTCUT_DOMINANCE_TOLERANCE = 1.02


def _load_phase126_module():
    script = Path(__file__).with_name("build_phase126_matbench_phonons_baseline_gate.py")
    spec = importlib.util.spec_from_file_location("phase126_helpers", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Phase 126 helper module from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase126 = _load_phase126_module()

ELEMENTS = tuple(phase126.ELEMENTS)
COMMON_ELEMENTS = tuple(phase126.COMMON_ELEMENTS)
ELECTRONEGATIVITY = dict(phase126.ELECTRONEGATIVITY)
ATOMIC_NUMBER = dict(phase126.ATOMIC_NUMBER)
HALOGENS = set(phase126.HALOGENS)
CHALCOGENS = set(phase126.CHALCOGENS)
PNICTOGENS = set(phase126.PNICTOGENS)
ALKALI = set(phase126.ALKALI)
ALKALINE_EARTH = set(phase126.ALKALINE_EARTH)
TRANSITION_METALS = set(phase126.TRANSITION_METALS)
POST_TRANSITION = set(phase126.POST_TRANSITION)
METALLOIDS = set(phase126.METALLOIDS)
RARE_EARTHS = set(phase126.RARE_EARTHS)

PROFILE_COLUMNS: dict[str, dict[str, Any]] = {
    "composition_descriptors": {
        **phase126.PROFILE_COLUMNS["composition_descriptors"],
        "role": "admissible",
    },
    "common_element_fractions": {
        **phase126.PROFILE_COLUMNS["common_element_fractions"],
        "role": "admissible",
    },
    "lattice_descriptors": {
        **phase126.PROFILE_COLUMNS["lattice_descriptors"],
        "role": "admissible",
    },
    "composition_lattice_descriptors": {
        **phase126.PROFILE_COLUMNS["composition_lattice_descriptors"],
        "role": "admissible",
    },
    "composition_hash_shortcut": {
        **phase126.PROFILE_COLUMNS["composition_hash_shortcut"],
        "role": "negative_control",
    },
    "chemistry_family_shortcut": {
        **phase126.PROFILE_COLUMNS["chemistry_family_shortcut"],
        "role": "negative_control",
    },
    "dominant_element_shortcut": {
        "role": "negative_control",
        "numeric": (),
        "categorical": ("dominant_element",),
    },
}
MODEL_METHODS = phase126.MODEL_METHODS
PROFILE_METHODS = {
    "composition_hash_shortcut": ("knn",),
    "chemistry_family_shortcut": ("knn",),
    "dominant_element_shortcut": ("knn", "extra_trees"),
}

FIELD_FIELDS = (
    "phase128_row_id",
    "composition",
    "refractive_index_n",
    "dominant_element",
    "chemistry_family_key",
    "composition_hash16",
    "n_sites",
    "element_count",
    "entropy_fraction",
    "max_fraction",
    "anion_fraction",
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
    "lattice_a",
    "lattice_b",
    "lattice_c",
    "lattice_alpha",
    "lattice_beta",
    "lattice_gamma",
    "lattice_volume",
    "volume_per_site",
    "abc_anisotropy",
    "angle_deviation",
    "density_z_proxy",
    "site_z_mean",
    "site_z_std",
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
    "best_negative_profile",
    "best_negative_method",
    "best_negative_val_rmse",
    "best_negative_test_rmse",
    "shortcut_blocks",
    "phase128_candidate",
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
            headers={"User-Agent": "Mozilla/5.0 Codex Phase128 matbench dielectric gate"},
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


def load_matbench_payload(path: Path) -> tuple[list[str], list[list[Any]]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    columns = payload.get("columns")
    data = payload.get("data")
    if columns != ["structure", "n"]:
        raise ValueError(f"Unexpected matbench_dielectric columns: {columns}")
    if not isinstance(data, list):
        raise ValueError("Expected split-orient data list")
    return list(columns), data


def _family_label(element: str) -> str:
    return phase126._family_label(element)


def _fraction_sum(fractions: dict[str, float], elements: set[str]) -> float:
    return float(sum(fractions.get(element, 0.0) for element in elements))


def _weighted_mean(fractions: dict[str, float], values: dict[str, float]) -> float:
    return float(sum(fractions.get(element, 0.0) * float(values.get(element, 0.0)) for element in ELEMENTS))


def _weighted_variance(fractions: dict[str, float], values: dict[str, float]) -> float:
    mean = _weighted_mean(fractions, values)
    return float(sum(fractions.get(element, 0.0) * (float(values.get(element, 0.0)) - mean) ** 2 for element in ELEMENTS))


def build_field_table(rows: list[list[Any]]) -> pd.DataFrame:
    out_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(f"Unexpected row shape at {index}: {type(row).__name__}")
        structure, target = row
        if not isinstance(structure, dict):
            raise ValueError(f"Expected structure dict at row {index}")
        fractions = phase126._extract_composition(structure)
        positive = {element: value for element, value in fractions.items() if value > 1e-12}
        dominant = max(positive, key=positive.get)
        families = Counter(_family_label(element) for element in positive)
        family_signature = "+".join(f"{family}{count}" for family, count in sorted(families.items()))
        chemistry_family_key = f"{dominant}|{family_signature}|n{min(len(positive), 6)}"
        en_values = [ELECTRONEGATIVITY[element] for element in positive if element in ELECTRONEGATIVITY]
        z_values = [ATOMIC_NUMBER[element] for element in positive if element in ATOMIC_NUMBER]
        lattice = structure.get("lattice", {}) if isinstance(structure.get("lattice"), dict) else {}
        sites = structure.get("sites", []) if isinstance(structure.get("sites"), list) else []
        n_sites = int(len(sites))
        a = phase126._lattice_float(lattice, "a")
        b = phase126._lattice_float(lattice, "b")
        c = phase126._lattice_float(lattice, "c")
        alpha = phase126._lattice_float(lattice, "alpha", 90.0)
        beta = phase126._lattice_float(lattice, "beta", 90.0)
        gamma = phase126._lattice_float(lattice, "gamma", 90.0)
        volume = phase126._lattice_float(lattice, "volume")
        lengths = [value for value in (a, b, c) if value > 0.0]
        abc_anisotropy = max(lengths) / min(lengths) if lengths else 0.0
        angle_deviation = (abs(alpha - 90.0) + abs(beta - 90.0) + abs(gamma - 90.0)) / 270.0
        mean_z = _weighted_mean(fractions, {element: float(ATOMIC_NUMBER.get(element, 0.0)) for element in ELEMENTS})
        var_z = _weighted_variance(fractions, {element: float(ATOMIC_NUMBER.get(element, 0.0)) for element in ELEMENTS})
        composition = phase126._composition_label(fractions)
        record = {
            "phase128_row_id": f"P128-{index:05d}",
            "composition": composition,
            "refractive_index_n": float(target),
            "dominant_element": dominant,
            "chemistry_family_key": chemistry_family_key,
            "composition_hash16": phase126._stable_hash(composition)[:16],
            "n_sites": n_sites,
            "element_count": len(positive),
            "entropy_fraction": phase126._entropy(list(positive.values())),
            "max_fraction": float(max(positive.values())),
            "anion_fraction": _fraction_sum(fractions, HALOGENS | CHALCOGENS | PNICTOGENS),
            "chalcogen_fraction": _fraction_sum(fractions, CHALCOGENS),
            "halogen_fraction": _fraction_sum(fractions, HALOGENS),
            "pnictogen_fraction": _fraction_sum(fractions, PNICTOGENS),
            "alkali_fraction": _fraction_sum(fractions, ALKALI),
            "alkaline_earth_fraction": _fraction_sum(fractions, ALKALINE_EARTH),
            "transition_metal_fraction": _fraction_sum(fractions, TRANSITION_METALS),
            "post_transition_fraction": _fraction_sum(fractions, POST_TRANSITION),
            "metalloid_fraction": _fraction_sum(fractions, METALLOIDS),
            "rare_earth_fraction": _fraction_sum(fractions, RARE_EARTHS),
            "mean_atomic_number": mean_z,
            "max_atomic_number": float(max(z_values)) if z_values else 0.0,
            "mean_electronegativity": _weighted_mean(fractions, ELECTRONEGATIVITY),
            "electronegativity_range": float(max(en_values) - min(en_values)) if en_values else 0.0,
            "lattice_a": a,
            "lattice_b": b,
            "lattice_c": c,
            "lattice_alpha": alpha,
            "lattice_beta": beta,
            "lattice_gamma": gamma,
            "lattice_volume": volume,
            "volume_per_site": volume / max(1, n_sites),
            "abc_anisotropy": abc_anisotropy,
            "angle_deviation": angle_deviation,
            "density_z_proxy": mean_z * max(1, n_sites) / volume if volume > 0.0 else 0.0,
            "site_z_mean": mean_z,
            "site_z_std": math.sqrt(max(0.0, var_z)),
        }
        for element in ELEMENTS:
            record[f"frac_{element}"] = fractions.get(element, 0.0)
        out_rows.append(record)
    return pd.DataFrame(out_rows, columns=list(FIELD_FIELDS))


def group_split(df: pd.DataFrame, *, group_column: str = "chemistry_family_key", salt: str = "phase128") -> dict[str, Any]:
    return phase126.group_split(df, group_column=group_column, salt=salt)


def evaluate_target(df: pd.DataFrame, split_manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target = "refractive_index_n"
    splits = split_manifest["splits"]
    metric_rows: list[dict[str, Any]] = []
    if len(df) < MIN_ROWS_FOR_REVIEW or any(len(splits[split]) < MIN_SPLIT_ROWS for split in ("train", "val", "test")):
        return metric_rows, {
            "target": target,
            "row_count": int(len(df)),
            "train_rows": len(splits["train"]),
            "val_rows": len(splits["val"]),
            "test_rows": len(splits["test"]),
            "status": "blocked_insufficient_rows_or_split",
            "reason": "dataset or one split is below the minimum row count",
            "phase128_candidate": False,
        }
    y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
    train_idx = splits["train"]
    train_std = float(np.std(y[train_idx])) if train_idx else 0.0
    train_mean = float(np.mean(y[train_idx]))
    mean_pred = np.full_like(y, train_mean, dtype=float)
    for split in ("train", "val", "test"):
        values = phase126._metrics(y[splits[split]], mean_pred[splits[split]], train_std)
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
        x_train, x_all = phase126._one_hot_frame(df.iloc[train_idx], df, profile)
        y_train = y[train_idx]
        for method in PROFILE_METHODS.get(profile_name, MODEL_METHODS):
            pred = phase126.phase123._fit_predict(method, x_train, y_train, x_all)
            for split in ("train", "val", "test"):
                values = phase126._metrics(y[splits[split]], pred[splits[split]], train_std)
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
    admissible_profiles = {name for name, profile in PROFILE_COLUMNS.items() if profile["role"] == "admissible"}
    negative_profiles = {name for name, profile in PROFILE_COLUMNS.items() if profile["role"] == "negative_control"}
    admissible_val = [row for row in metric_rows if row["split"] == "val" and row["profile"] in admissible_profiles]
    negative_val = [row for row in metric_rows if row["split"] == "val" and row["profile"] in negative_profiles]
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
    min_gain = float(mean_val["rmse"]) * MIN_RELATIVE_VAL_GAIN
    shortcut_blocks = (
        float(best_negative_val["rmse"]) <= float(best_val["rmse"]) * SHORTCUT_DOMINANCE_TOLERANCE
        and float(mean_test["rmse"]) - float(best_negative_test["rmse"]) > 0.0
    )
    candidate = val_gain > min_gain and test_gain > 0.0 and not shortcut_blocks
    if shortcut_blocks:
        status = "blocked_shortcut_dominance"
        reason = "composition, family, or dominant-element shortcut dominates the selected safe profile"
    elif val_gain <= min_gain:
        status = "blocked_no_validation_gain"
        reason = "best safe profile does not improve validation RMSE over train mean by the required margin"
    elif test_gain <= 0.0:
        status = "blocked_validation_test_reversal"
        reason = "validation-selected safe profile does not preserve gain on test"
    else:
        status = "ready_focused_review"
        reason = "safe structure/composition profile beats mean and shortcut controls"
    review = {
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
        "best_negative_profile": best_negative_val["profile"],
        "best_negative_method": best_negative_val["method"],
        "best_negative_val_rmse": best_negative_val["rmse"],
        "best_negative_test_rmse": best_negative_test["rmse"],
        "shortcut_blocks": shortcut_blocks,
        "phase128_candidate": candidate,
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
        if column == "refractive_index_n":
            role = "target"
        elif column in {"phase128_row_id", "composition"}:
            role = "identifier"
        elif column in profile_categorical:
            role = "categorical_feature"
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


def build_gate(*, source_info: dict[str, Any], split_manifest: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    status = (
        "phase128_matbench_dielectric_n_ready_focused_review"
        if review.get("phase128_candidate")
        else "phase128_matbench_dielectric_n_closed_no_stable_guarded_gap"
    )
    return {
        "status": status,
        "source_url": source_info["source_url"],
        "source_byte_size": source_info["byte_size"],
        "source_sha256": source_info["sha256"],
        "selected_target": review["target"],
        "selected_profile": review.get("best_profile"),
        "selected_method": review.get("best_method"),
        "selected_validation_rmse": review.get("best_val_rmse"),
        "selected_test_rmse": review.get("best_test_rmse"),
        "mean_validation_rmse": review.get("mean_val_rmse"),
        "mean_test_rmse": review.get("mean_test_rmse"),
        "best_negative_profile": review.get("best_negative_profile"),
        "best_negative_method": review.get("best_negative_method"),
        "best_negative_val_rmse": review.get("best_negative_val_rmse"),
        "best_negative_test_rmse": review.get("best_negative_test_rmse"),
        "row_count": review["row_count"],
        "group_split": split_manifest["group_column"],
        "group_count": split_manifest["n_groups"],
        "split_counts": {split: len(split_manifest["splits"][split]) for split in ("train", "val", "test")},
        "leakage_safe": bool(split_manifest["leakage_safe"]),
        "phase128_focused_review_allowed": bool(review.get("phase128_candidate")),
        "phase128_model_mechanism_allowed": False,
        "phase128_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "reason": review.get("reason"),
    }


def build_markdown(gate: dict[str, Any], review: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Phase 128 Matbench Dielectric Baseline Gate",
            "",
            f"- Status: `{gate['status']}`",
            f"- Target: `{gate['selected_target']}`",
            f"- Rows: `{gate['row_count']}`",
            f"- Group split: `{gate['group_split']}` with `{gate['group_count']}` groups",
            f"- Selected profile/method: `{gate['selected_profile']}` / `{gate['selected_method']}`",
            f"- Selected validation/test RMSE: `{gate['selected_validation_rmse']:.6g}` / `{gate['selected_test_rmse']:.6g}`",
            f"- Mean validation/test RMSE: `{gate['mean_validation_rmse']:.6g}` / `{gate['mean_test_rmse']:.6g}`",
            f"- Best negative control: `{gate['best_negative_profile']}` / `{gate['best_negative_method']}`",
            f"- Focused review allowed: `{gate['phase128_focused_review_allowed']}`",
            f"- Model training allowed: `{gate['phase128_model_training_allowed']}`",
            f"- A100 training allowed now: `{gate['a100_training_allowed_now']}`",
            "",
            "## Review Reason",
            "",
            str(review.get("reason")),
            "",
        ]
    )


def build_package(
    *,
    root: Path,
    raw_path: Path,
    output_dir: Path,
    source_url: str = SOURCE_URL,
    force_download: bool = False,
) -> dict[str, Any]:
    source_info = ensure_source_file(raw_path, source_url=source_url, force_download=force_download)
    _, raw_rows = load_matbench_payload(raw_path)
    field_df = build_field_table(raw_rows)
    split_manifest = group_split(field_df)
    metric_rows, review = evaluate_target(field_df, split_manifest)
    schema_rows = build_schema_rows(field_df)
    gate = build_gate(source_info=source_info, split_manifest=split_manifest, review=review)

    output_dir.mkdir(parents=True, exist_ok=True)
    field_path = output_dir / "phase128_matbench_dielectric_field_table.csv"
    split_path = output_dir / "phase128_matbench_dielectric_split_manifest.json"
    metric_path = output_dir / "phase128_matbench_dielectric_metric_table.csv"
    review_path = output_dir / "phase128_matbench_dielectric_target_review_table.csv"
    schema_path = output_dir / "phase128_matbench_dielectric_schema_table.csv"
    data_card_path = output_dir / "phase128_matbench_dielectric_data_card.json"
    gate_path = output_dir / "phase128_matbench_dielectric_gate.json"
    markdown_path = output_dir / "phase128_matbench_dielectric.md"
    manifest_path = output_dir / "phase128_matbench_dielectric_manifest.json"

    _write_csv(field_path, field_df.to_dict("records"), FIELD_FIELDS)
    _write_json(split_path, split_manifest)
    _write_csv(metric_path, metric_rows, METRIC_FIELDS)
    _write_csv(review_path, [review], REVIEW_FIELDS)
    _write_csv(schema_path, schema_rows, SCHEMA_FIELDS)
    _write_json(
        data_card_path,
        {
            "dataset": "matbench_dielectric",
            "source_url": source_url,
            "raw_path": _display_path(raw_path, root),
            "byte_size": source_info["byte_size"],
            "sha256": source_info["sha256"],
            "target": "refractive_index_n",
            "input": "pymatgen Structure JSON",
            "row_count": int(len(field_df)),
            "phase128_no_training": True,
        },
    )
    _write_json(gate_path, gate)
    markdown_path.write_text(build_markdown(gate, review), encoding="utf-8")

    manifest = {
        "phase": 128,
        "objective": "matbench_dielectric_baseline_first_gate_no_training",
        "source": source_info,
        "inputs": {"raw_path": _display_path(raw_path, root), "source_url": source_url},
        "outputs": {
            "field_table": _display_path(field_path, root),
            "split_manifest": _display_path(split_path, root),
            "metric_table": _display_path(metric_path, root),
            "target_review_table": _display_path(review_path, root),
            "schema_table": _display_path(schema_path, root),
            "data_card": _display_path(data_card_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "field_rows": int(len(field_df)),
            "metric_rows": len(metric_rows),
            "schema_rows": len(schema_rows),
        },
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
