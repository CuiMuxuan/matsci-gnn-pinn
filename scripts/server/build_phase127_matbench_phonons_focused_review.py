#!/usr/bin/env python3
"""Build Phase 127 focused review for the Matbench phonons gate.

This phase consumes only small Phase 126 artifacts. It checks whether the
Phase 126 phonon peak gap is stable under alternate grouped/binned splits and
whether shortcut profiles, nearest-neighbor composition/lattice identity, or
target-distribution imbalance explain the signal. It does not train a neural
model or open A100 training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PHASE126_DIR = Path("docs/results/phase126_matbench_phonons_baseline_gate")
DEFAULT_OUTPUT_DIR = Path("docs/results/phase127_matbench_phonons_focused_review")
MIN_SPLIT_ROWS = 100
MIN_RELATIVE_VAL_GAIN = 0.05
MIN_STABLE_SPLIT_PASS_RATE = 0.75
NEGATIVE_DOMINANCE_TOLERANCE = 1.02
MAX_ORIGINAL_TARGET_SHIFT_Z = 0.75
MAX_NEAR_DUPLICATE_FRACTION = 0.60
NEAR_DUPLICATE_DISTANCE = 0.05


def _load_phase126_module():
    script = Path(__file__).with_name("build_phase126_matbench_phonons_baseline_gate.py")
    spec = importlib.util.spec_from_file_location("phase126_helpers", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Phase 126 helper module from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase126 = _load_phase126_module()

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
NN_FEATURE_COLUMNS = tuple(
    column
    for column in (
        *(f"frac_{element}" for element in phase126.ELEMENTS),
        "n_sites",
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
    )
)
SPLIT_PLAN = (
    ("phase126_registered_split", "phase126_manifest", "phase126"),
    ("chemistry_family_hash_0", "group:chemistry_family_key", "phase127_family_0"),
    ("chemistry_family_hash_1", "group:chemistry_family_key", "phase127_family_1"),
    ("chemistry_family_hash_2", "group:chemistry_family_key", "phase127_family_2"),
    ("dominant_element_hash", "group:dominant_element", "phase127_dominant"),
    ("lattice_volume_bins", "bins:lattice_volume", "phase127_lattice_volume"),
    ("volume_per_site_bins", "bins:volume_per_site", "phase127_volume_per_site"),
    ("element_count_bins", "bins:element_count", "phase127_element_count"),
    ("density_anisotropy_bins", "bins:density_z_proxy,abc_anisotropy", "phase127_density_aniso"),
)

PROFILE_FIELDS = (
    "target",
    "split_id",
    "split_strategy",
    "profile",
    "profile_role",
    "method",
    "train_rows",
    "val_rows",
    "test_rows",
    "train_rmse",
    "val_rmse",
    "test_rmse",
    "mean_val_rmse",
    "mean_test_rmse",
    "val_gain_vs_mean",
    "test_gain_vs_mean",
    "relative_val_gain",
    "selected_admissible",
    "selected_negative_control",
)
SPLIT_FIELDS = (
    "target",
    "split_id",
    "split_strategy",
    "split_viable",
    "split_reason",
    "train_rows",
    "val_rows",
    "test_rows",
    "train_target_mean",
    "val_target_mean",
    "test_target_mean",
    "train_target_median",
    "val_target_median",
    "test_target_median",
    "train_target_q90",
    "val_target_q90",
    "test_target_q90",
    "target_mean_shift_z",
    "target_median_shift_z",
    "target_q90_shift_z",
    "target_distribution_shift_z",
    "mean_val_rmse",
    "mean_test_rmse",
    "best_admissible_profile",
    "best_admissible_method",
    "best_admissible_val_rmse",
    "best_admissible_test_rmse",
    "best_admissible_val_gain_vs_mean",
    "best_admissible_test_gain_vs_mean",
    "best_admissible_relative_val_gain",
    "split_pass",
    "best_negative_profile",
    "best_negative_method",
    "best_negative_val_rmse",
    "best_negative_test_rmse",
    "negative_control_dominates",
    "nearest_neighbor_val_rmse",
    "nearest_neighbor_test_rmse",
    "nearest_neighbor_val_gain_vs_mean",
    "nearest_neighbor_test_gain_vs_mean",
    "nearest_neighbor_dominates",
    "val_near_duplicate_fraction",
    "test_near_duplicate_fraction",
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


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _blocked_split(strategy: str, reason: str) -> dict[str, Any]:
    return {
        "split_strategy": strategy,
        "split_viable": False,
        "reason": reason,
        "splits": {"train": [], "val": [], "test": []},
        "group_splits": {"train": [], "val": [], "test": []},
        "n_groups": 0,
    }


def _split_from_keys(keys: list[str], groups: list[str], strategy: str, salt: str) -> dict[str, Any]:
    if len(groups) < 3:
        return _blocked_split(strategy, "fewer than three groups")
    ranked = sorted(groups, key=lambda item: _stable_hash(f"{salt}::{item}"))
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
    assignments = [group_to_split[key] for key in keys]
    splits = {
        split: [int(index) for index, label in enumerate(assignments) if label == split]
        for split in ("train", "val", "test")
    }
    group_splits = {
        split: [group for group, label in group_to_split.items() if label == split]
        for split in ("train", "val", "test")
    }
    return {
        "split_strategy": strategy,
        "split_salt": salt,
        "split_viable": True,
        "reason": "ok",
        "splits": splits,
        "group_splits": group_splits,
        "n_groups": n_groups,
        "leakage_safe": sum(len(values) for values in group_splits.values()) == n_groups,
    }


def _group_split(df: pd.DataFrame, column: str, salt: str) -> dict[str, Any]:
    if column not in df.columns:
        return _blocked_split(f"group_hash_by_{column}", "group column missing")
    keys = df[column].fillna("missing").astype(str).tolist()
    groups = sorted(set(keys))
    return _split_from_keys(keys, groups, f"group_hash_by_{column}", salt)


def _bin_split(df: pd.DataFrame, columns: tuple[str, ...], salt: str) -> dict[str, Any]:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        return _blocked_split(f"bins_{'_'.join(columns)}", f"missing columns: {missing}")
    keys: list[str] = []
    ranks = {
        column: pd.to_numeric(df[column], errors="coerce").rank(method="first").to_numpy(dtype=float)
        for column in columns
    }
    n_rows = len(df)
    for row_index in range(n_rows):
        parts = []
        for column in columns:
            quartile = int(min(3, max(0, math.floor((ranks[column][row_index] - 1.0) / n_rows * 4.0))))
            parts.append(f"{column}:{quartile}")
        keys.append("|".join(parts))
    groups = sorted(set(keys))
    return _split_from_keys(keys, groups, f"quartile_bins_by_{'_'.join(columns)}", salt)


def _phase126_split(split_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "split_strategy": split_manifest.get("split_strategy", "phase126_registered_split"),
        "split_salt": "phase126",
        "split_viable": True,
        "reason": "ok",
        "splits": split_manifest["splits"],
        "group_splits": split_manifest.get("group_splits", {}),
        "n_groups": split_manifest.get("n_groups"),
        "leakage_safe": split_manifest.get("leakage_safe"),
    }


def build_split_reviews(df: pd.DataFrame, split_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    reviews: dict[str, dict[str, Any]] = {}
    for split_id, split_spec, salt in SPLIT_PLAN:
        if split_spec == "phase126_manifest":
            reviews[split_id] = _phase126_split(split_manifest)
        elif split_spec.startswith("group:"):
            reviews[split_id] = _group_split(df, split_spec.split(":", 1)[1], salt)
        elif split_spec.startswith("bins:"):
            columns = tuple(split_spec.split(":", 1)[1].split(","))
            reviews[split_id] = _bin_split(df, columns, salt)
        else:
            raise ValueError(f"Unsupported split spec: {split_spec}")
    return reviews


def _standardized_feature_matrix(df: pd.DataFrame, train_idx: list[int]) -> np.ndarray:
    columns = [column for column in NN_FEATURE_COLUMNS if column in df.columns]
    numeric = df[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    train = numeric.iloc[train_idx]
    means = train.mean(axis=0)
    stds = train.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    return ((numeric - means) / stds).to_numpy(dtype=float)


def _nearest_neighbor_control(
    df: pd.DataFrame,
    *,
    target: str,
    splits: dict[str, list[int]],
    mean_val_rmse: float,
    mean_test_rmse: float,
    train_std: float,
) -> dict[str, Any]:
    from sklearn.neighbors import NearestNeighbors

    y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
    train_idx = splits["train"]
    x_all = _standardized_feature_matrix(df, train_idx)
    model = NearestNeighbors(n_neighbors=1, metric="manhattan")
    model.fit(x_all[train_idx])
    distances, indices = model.kneighbors(x_all)
    pred = y[np.asarray(train_idx, dtype=int)[indices[:, 0]]]
    val_metrics = phase126._metrics(y[splits["val"]], pred[splits["val"]], train_std)
    test_metrics = phase126._metrics(y[splits["test"]], pred[splits["test"]], train_std)
    val_dist = distances[splits["val"], 0] if splits["val"] else np.array([], dtype=float)
    test_dist = distances[splits["test"], 0] if splits["test"] else np.array([], dtype=float)
    return {
        "nearest_neighbor_val_rmse": val_metrics["rmse"],
        "nearest_neighbor_test_rmse": test_metrics["rmse"],
        "nearest_neighbor_val_gain_vs_mean": mean_val_rmse - float(val_metrics["rmse"]),
        "nearest_neighbor_test_gain_vs_mean": mean_test_rmse - float(test_metrics["rmse"]),
        "val_near_duplicate_fraction": float(np.mean(val_dist <= NEAR_DUPLICATE_DISTANCE)) if len(val_dist) else 0.0,
        "test_near_duplicate_fraction": float(np.mean(test_dist <= NEAR_DUPLICATE_DISTANCE)) if len(test_dist) else 0.0,
    }


def _target_distribution(
    y: np.ndarray,
    splits: dict[str, list[int]],
    train_std: float,
) -> dict[str, Any]:
    stats: dict[str, dict[str, float]] = {}
    for split in ("train", "val", "test"):
        values = y[splits[split]]
        stats[split] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "q90": float(np.quantile(values, 0.90)),
        }
    denom = train_std if train_std > 0.0 else 1.0
    mean_shift = max(
        abs(stats["val"]["mean"] - stats["train"]["mean"]) / denom,
        abs(stats["test"]["mean"] - stats["train"]["mean"]) / denom,
    )
    median_shift = max(
        abs(stats["val"]["median"] - stats["train"]["median"]) / denom,
        abs(stats["test"]["median"] - stats["train"]["median"]) / denom,
    )
    q90_shift = max(
        abs(stats["val"]["q90"] - stats["train"]["q90"]) / denom,
        abs(stats["test"]["q90"] - stats["train"]["q90"]) / denom,
    )
    return {
        "train_target_mean": stats["train"]["mean"],
        "val_target_mean": stats["val"]["mean"],
        "test_target_mean": stats["test"]["mean"],
        "train_target_median": stats["train"]["median"],
        "val_target_median": stats["val"]["median"],
        "test_target_median": stats["test"]["median"],
        "train_target_q90": stats["train"]["q90"],
        "val_target_q90": stats["val"]["q90"],
        "test_target_q90": stats["test"]["q90"],
        "target_mean_shift_z": mean_shift,
        "target_median_shift_z": median_shift,
        "target_q90_shift_z": q90_shift,
        "target_distribution_shift_z": max(mean_shift, median_shift, q90_shift),
    }


def evaluate_split(
    df: pd.DataFrame,
    *,
    target: str,
    split_id: str,
    split_info: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    splits = split_info["splits"]
    counts = {split: len(splits[split]) for split in ("train", "val", "test")}
    base = {
        "target": target,
        "split_id": split_id,
        "split_strategy": split_info.get("split_strategy"),
        "train_rows": counts["train"],
        "val_rows": counts["val"],
        "test_rows": counts["test"],
    }
    if not split_info.get("split_viable", True):
        return [], {**base, "split_viable": False, "split_reason": split_info.get("reason"), "split_pass": False}
    if min(counts.values()) < MIN_SPLIT_ROWS:
        return [], {
            **base,
            "split_viable": False,
            "split_reason": "one or more splits is below minimum row count",
            "split_pass": False,
            "negative_control_dominates": False,
            "nearest_neighbor_dominates": False,
        }
    y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
    train_idx = splits["train"]
    train_std = float(np.std(y[train_idx])) if train_idx else 0.0
    train_mean = float(np.mean(y[train_idx]))
    mean_pred = np.full_like(y, train_mean, dtype=float)
    mean_val = phase126._metrics(y[splits["val"]], mean_pred[splits["val"]], train_std)
    mean_test = phase126._metrics(y[splits["test"]], mean_pred[splits["test"]], train_std)

    rows: list[dict[str, Any]] = []
    for profile_name, profile in PROFILE_COLUMNS.items():
        x_train, x_all = phase126._one_hot_frame(df.iloc[train_idx], df, profile)
        y_train = y[train_idx]
        for method in PROFILE_METHODS.get(profile_name, MODEL_METHODS):
            pred = phase126.phase123._fit_predict(method, x_train, y_train, x_all)
            split_metrics = {
                split: phase126._metrics(y[splits[split]], pred[splits[split]], train_std)
                for split in ("train", "val", "test")
            }
            val_rmse = float(split_metrics["val"]["rmse"])
            test_rmse = float(split_metrics["test"]["rmse"])
            val_gain = float(mean_val["rmse"]) - val_rmse
            test_gain = float(mean_test["rmse"]) - test_rmse
            rows.append(
                {
                    "target": target,
                    "split_id": split_id,
                    "split_strategy": split_info.get("split_strategy"),
                    "profile": profile_name,
                    "profile_role": profile["role"],
                    "method": method,
                    "train_rows": counts["train"],
                    "val_rows": counts["val"],
                    "test_rows": counts["test"],
                    "train_rmse": split_metrics["train"]["rmse"],
                    "val_rmse": val_rmse,
                    "test_rmse": test_rmse,
                    "mean_val_rmse": mean_val["rmse"],
                    "mean_test_rmse": mean_test["rmse"],
                    "val_gain_vs_mean": val_gain,
                    "test_gain_vs_mean": test_gain,
                    "relative_val_gain": val_gain / float(mean_val["rmse"]) if float(mean_val["rmse"]) > 0 else None,
                    "selected_admissible": False,
                    "selected_negative_control": False,
                }
            )
    admissible_rows = [row for row in rows if row["profile_role"] == "admissible"]
    negative_rows = [row for row in rows if row["profile_role"] == "negative_control"]
    best_admissible = min(admissible_rows, key=lambda row: float(row["val_rmse"]))
    best_negative = min(negative_rows, key=lambda row: float(row["val_rmse"]))
    best_admissible["selected_admissible"] = True
    best_negative["selected_negative_control"] = True
    min_gain = float(mean_val["rmse"]) * MIN_RELATIVE_VAL_GAIN
    split_pass = (
        float(best_admissible["val_gain_vs_mean"]) > min_gain
        and float(best_admissible["test_gain_vs_mean"]) > 0.0
    )
    negative_dominates = (
        float(best_negative["val_rmse"]) <= float(best_admissible["val_rmse"]) * NEGATIVE_DOMINANCE_TOLERANCE
        and float(best_negative["test_gain_vs_mean"]) > 0.0
    )
    nearest = _nearest_neighbor_control(
        df,
        target=target,
        splits=splits,
        mean_val_rmse=float(mean_val["rmse"]),
        mean_test_rmse=float(mean_test["rmse"]),
        train_std=train_std,
    )
    nearest_dominates = (
        float(nearest["nearest_neighbor_val_rmse"]) <= float(best_admissible["val_rmse"]) * NEGATIVE_DOMINANCE_TOLERANCE
        and float(nearest["nearest_neighbor_test_gain_vs_mean"]) > 0.0
    )
    distribution = _target_distribution(y, splits, train_std)
    summary = {
        **base,
        "split_viable": True,
        "split_reason": "ok",
        **distribution,
        "mean_val_rmse": mean_val["rmse"],
        "mean_test_rmse": mean_test["rmse"],
        "best_admissible_profile": best_admissible["profile"],
        "best_admissible_method": best_admissible["method"],
        "best_admissible_val_rmse": best_admissible["val_rmse"],
        "best_admissible_test_rmse": best_admissible["test_rmse"],
        "best_admissible_val_gain_vs_mean": best_admissible["val_gain_vs_mean"],
        "best_admissible_test_gain_vs_mean": best_admissible["test_gain_vs_mean"],
        "best_admissible_relative_val_gain": best_admissible["relative_val_gain"],
        "split_pass": split_pass,
        "best_negative_profile": best_negative["profile"],
        "best_negative_method": best_negative["method"],
        "best_negative_val_rmse": best_negative["val_rmse"],
        "best_negative_test_rmse": best_negative["test_rmse"],
        "negative_control_dominates": negative_dominates,
        **nearest,
        "nearest_neighbor_dominates": nearest_dominates,
    }
    return rows, summary


def build_audit_rows(*, phase126_gate: dict[str, Any], split_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    phase126_ready = phase126_gate.get("status") == "phase126_matbench_phonons_peak_ready_focused_review"
    rows.append(
        {
            "audit": "phase126_gate_status",
            "status": "pass" if phase126_ready else "block",
            "severity": "blocking" if not phase126_ready else "info",
            "value": phase126_gate.get("status"),
            "threshold": "phase126_matbench_phonons_peak_ready_focused_review",
            "reason": "focused review requires a Phase 126 focused-review gate",
        }
    )
    original = next((row for row in split_rows if row.get("split_id") == "phase126_registered_split"), None)
    if original and original.get("split_viable"):
        rows.append(
            {
                "audit": "original_split_admissible_gain",
                "status": "pass" if _is_true(original.get("split_pass")) else "block",
                "severity": "blocking" if not _is_true(original.get("split_pass")) else "info",
                "value": original.get("best_admissible_relative_val_gain"),
                "threshold": MIN_RELATIVE_VAL_GAIN,
                "reason": "Phase 126 selected phonon target must preserve validation and test gain",
            }
        )
        rows.append(
            {
                "audit": "original_split_shortcut_dominance",
                "status": "block" if _is_true(original.get("negative_control_dominates")) else "pass",
                "severity": "blocking" if _is_true(original.get("negative_control_dominates")) else "info",
                "value": original.get("best_negative_profile"),
                "threshold": f"negative val RMSE > admissible * {NEGATIVE_DOMINANCE_TOLERANCE}",
                "reason": "composition, chemistry-family, or dominant-element shortcuts must not dominate",
            }
        )
        rows.append(
            {
                "audit": "original_split_nearest_neighbor_dominance",
                "status": "block" if _is_true(original.get("nearest_neighbor_dominates")) else "pass",
                "severity": "blocking" if _is_true(original.get("nearest_neighbor_dominates")) else "info",
                "value": original.get("nearest_neighbor_val_rmse"),
                "threshold": f"nearest val RMSE > admissible * {NEGATIVE_DOMINANCE_TOLERANCE}",
                "reason": "nearest-neighbor composition/lattice identity control must not dominate",
            }
        )
        rows.append(
            {
                "audit": "original_split_target_distribution_balance",
                "status": "block"
                if float(original.get("target_distribution_shift_z") or 0.0) > MAX_ORIGINAL_TARGET_SHIFT_Z
                else "pass",
                "severity": "blocking"
                if float(original.get("target_distribution_shift_z") or 0.0) > MAX_ORIGINAL_TARGET_SHIFT_Z
                else "info",
                "value": original.get("target_distribution_shift_z"),
                "threshold": MAX_ORIGINAL_TARGET_SHIFT_Z,
                "reason": "registered split target mean/median/q90 shift must not dominate the interpretation",
            }
        )
        rows.append(
            {
                "audit": "original_split_near_duplicate_fraction",
                "status": "block"
                if max(float(original.get("val_near_duplicate_fraction") or 0.0), float(original.get("test_near_duplicate_fraction") or 0.0))
                > MAX_NEAR_DUPLICATE_FRACTION
                else "pass",
                "severity": "blocking"
                if max(float(original.get("val_near_duplicate_fraction") or 0.0), float(original.get("test_near_duplicate_fraction") or 0.0))
                > MAX_NEAR_DUPLICATE_FRACTION
                else "info",
                "value": max(float(original.get("val_near_duplicate_fraction") or 0.0), float(original.get("test_near_duplicate_fraction") or 0.0)),
                "threshold": MAX_NEAR_DUPLICATE_FRACTION,
                "reason": "registered split must not contain too many near-duplicate composition/lattice rows relative to train",
            }
        )
    else:
        rows.append(
            {
                "audit": "original_split_viability",
                "status": "block",
                "severity": "blocking",
                "value": original.get("split_reason") if original else "missing",
                "threshold": f"all splits >= {MIN_SPLIT_ROWS} rows",
                "reason": "Phase 126 registered split must be reviewable",
            }
        )
    viable = [row for row in split_rows if _is_true(row.get("split_viable"))]
    passed = [row for row in viable if _is_true(row.get("split_pass"))]
    stable_rate = len(passed) / len(viable) if viable else 0.0
    rows.append(
        {
            "audit": "split_sensitivity_pass_rate",
            "status": "pass" if stable_rate >= MIN_STABLE_SPLIT_PASS_RATE else "block",
            "severity": "blocking" if stable_rate < MIN_STABLE_SPLIT_PASS_RATE else "info",
            "value": stable_rate,
            "threshold": MIN_STABLE_SPLIT_PASS_RATE,
            "reason": "phonon target gain must survive deterministic chemistry, dominant-element, and lattice split perturbations",
        }
    )
    shortcut_dominant = [row for row in viable if _is_true(row.get("negative_control_dominates"))]
    rows.append(
        {
            "audit": "shortcut_dominant_split_count",
            "status": "block" if shortcut_dominant else "pass",
            "severity": "blocking" if shortcut_dominant else "info",
            "value": len(shortcut_dominant),
            "threshold": 0,
            "reason": "no viable split may be dominated by composition, chemistry-family, or dominant-element shortcuts",
        }
    )
    nearest_dominant = [row for row in viable if _is_true(row.get("nearest_neighbor_dominates"))]
    rows.append(
        {
            "audit": "nearest_neighbor_dominant_split_count",
            "status": "block" if nearest_dominant else "pass",
            "severity": "blocking" if nearest_dominant else "info",
            "value": len(nearest_dominant),
            "threshold": 0,
            "reason": "no viable split may be dominated by nearest-neighbor composition/lattice identity control",
        }
    )
    target_imbalanced = [
        row
        for row in viable
        if float(row.get("target_distribution_shift_z") or 0.0) > MAX_ORIGINAL_TARGET_SHIFT_Z
    ]
    rows.append(
        {
            "audit": "target_distribution_imbalanced_split_count",
            "status": "block" if target_imbalanced else "pass",
            "severity": "blocking" if target_imbalanced else "info",
            "value": len(target_imbalanced),
            "threshold": 0,
            "reason": "no viable split may have severe train/validation/test phonon target mean, median, or q90 imbalance",
        }
    )
    return rows


def build_gate(*, phase126_gate: dict[str, Any], split_rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    blockers = [row for row in audit_rows if row["status"] == "block"]
    viable = [row for row in split_rows if _is_true(row.get("split_viable"))]
    passed = [row for row in viable if _is_true(row.get("split_pass"))]
    shortcut_dominant = [row for row in viable if _is_true(row.get("negative_control_dominates"))]
    nearest_dominant = [row for row in viable if _is_true(row.get("nearest_neighbor_dominates"))]
    target_imbalanced = [
        row
        for row in viable
        if float(row.get("target_distribution_shift_z") or 0.0) > MAX_ORIGINAL_TARGET_SHIFT_Z
    ]
    original = next((row for row in split_rows if row.get("split_id") == "phase126_registered_split"), {})
    if phase126_gate.get("status") != "phase126_matbench_phonons_peak_ready_focused_review":
        status = "phase127_matbench_phonons_review_blocked_by_phase126"
        mechanism_allowed = False
        next_action = "complete or close Phase 126 before focused review"
    elif blockers:
        status = "phase127_matbench_phonons_focused_review_closed_split_sensitivity_or_shortcut"
        mechanism_allowed = False
        next_action = "close the Phase 126 phonons target as diagnostic; do not train"
    else:
        status = "phase127_matbench_phonons_focused_review_ready_low_capacity_mechanism_gate"
        mechanism_allowed = True
        next_action = "design a separate no-training low-capacity phonon mechanism gate; keep model training closed"
    return {
        "status": status,
        "phase126_status": phase126_gate.get("status"),
        "selected_target": phase126_gate.get("selected_target", "last_phdos_peak"),
        "phase126_selected_profile": phase126_gate.get("selected_profile"),
        "phase126_selected_method": phase126_gate.get("selected_method"),
        "viable_split_reviews": len(viable),
        "passed_split_reviews": len(passed),
        "split_pass_rate": len(passed) / len(viable) if viable else 0.0,
        "shortcut_dominant_splits": len(shortcut_dominant),
        "nearest_neighbor_dominant_splits": len(nearest_dominant),
        "target_distribution_imbalanced_splits": len(target_imbalanced),
        "blocking_audit_rows": len(blockers),
        "blocking_audits": [row["audit"] for row in blockers],
        "original_best_admissible_profile": original.get("best_admissible_profile"),
        "original_best_admissible_method": original.get("best_admissible_method"),
        "original_best_admissible_val_rmse": original.get("best_admissible_val_rmse"),
        "original_best_admissible_test_rmse": original.get("best_admissible_test_rmse"),
        "original_best_negative_profile": original.get("best_negative_profile"),
        "original_best_negative_method": original.get("best_negative_method"),
        "original_best_negative_val_rmse": original.get("best_negative_val_rmse"),
        "original_best_negative_test_rmse": original.get("best_negative_test_rmse"),
        "original_nearest_neighbor_val_rmse": original.get("nearest_neighbor_val_rmse"),
        "original_nearest_neighbor_test_rmse": original.get("nearest_neighbor_test_rmse"),
        "original_target_distribution_shift_z": original.get("target_distribution_shift_z"),
        "phase127_model_mechanism_allowed": mechanism_allowed,
        "phase127_low_capacity_mechanism_design_allowed": mechanism_allowed,
        "phase127_model_training_allowed": False,
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


def build_markdown(gate: dict[str, Any], audit_rows: list[dict[str, Any]], split_rows: list[dict[str, Any]]) -> str:
    blocking = [row for row in audit_rows if row["status"] == "block"]
    viable = [row for row in split_rows if _is_true(row.get("split_viable"))]
    lines = [
        "# Phase 127 Matbench Phonons Focused Review",
        "",
        f"- Status: `{gate['status']}`",
        f"- Split pass rate: `{gate['split_pass_rate']:.6g}`",
        f"- Blocking audits: `{', '.join(gate['blocking_audits']) or 'none'}`",
        f"- Low-capacity mechanism design allowed: `{gate['phase127_low_capacity_mechanism_design_allowed']}`",
        f"- Model training allowed: `{gate['phase127_model_training_allowed']}`",
        "",
        "## Blocking Audits",
        "",
        _markdown_table(blocking, [("Audit", "audit"), ("Value", "value"), ("Threshold", "threshold"), ("Reason", "reason")]),
        "",
        "## Split Reviews",
        "",
        _markdown_table(
            viable,
            [
                ("Split", "split_id"),
                ("Pass", "split_pass"),
                ("Best profile", "best_admissible_profile"),
                ("Val RMSE", "best_admissible_val_rmse"),
                ("Test RMSE", "best_admissible_test_rmse"),
                ("Shortcut", "negative_control_dominates"),
                ("NN", "nearest_neighbor_dominates"),
                ("Target shift", "target_distribution_shift_z"),
            ],
        ),
    ]
    return "\n".join(lines) + "\n"


def build_package(*, root: Path, phase126_dir: Path, output_dir: Path) -> dict[str, Any]:
    field_path = phase126_dir / "phase126_matbench_phonons_field_table.csv"
    split_path = phase126_dir / "phase126_matbench_phonons_split_manifest.json"
    gate_path = phase126_dir / "phase126_matbench_phonons_gate.json"
    df = pd.read_csv(field_path)
    split_manifest = _read_json(split_path)
    phase126_gate = _read_json(gate_path)
    target = str(phase126_gate.get("selected_target") or "last_phdos_peak")

    split_infos = build_split_reviews(df, split_manifest)
    profile_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for split_id, split_info in split_infos.items():
        rows, summary = evaluate_split(df, target=target, split_id=split_id, split_info=split_info)
        profile_rows.extend(rows)
        split_rows.append(summary)
    audit_rows = build_audit_rows(phase126_gate=phase126_gate, split_rows=split_rows)
    gate = build_gate(phase126_gate=phase126_gate, split_rows=split_rows, audit_rows=audit_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "phase127_matbench_phonons_focused_profile_table.csv"
    split_review_path = output_dir / "phase127_matbench_phonons_split_sensitivity_table.csv"
    audit_path = output_dir / "phase127_matbench_phonons_shortcut_audit_table.csv"
    split_manifest_path = output_dir / "phase127_matbench_phonons_split_review_manifest.json"
    gate_out = output_dir / "phase127_matbench_phonons_focused_review_gate.json"
    markdown_path = output_dir / "phase127_matbench_phonons_focused_review.md"
    manifest_path = output_dir / "phase127_matbench_phonons_focused_review_manifest.json"

    _write_csv(profile_path, profile_rows, PROFILE_FIELDS)
    _write_csv(split_review_path, split_rows, SPLIT_FIELDS)
    _write_csv(audit_path, audit_rows, AUDIT_FIELDS)
    _write_json(split_manifest_path, split_infos)
    _write_json(gate_out, gate)
    markdown_path.write_text(build_markdown(gate, audit_rows, split_rows), encoding="utf-8")

    manifest = {
        "phase": 127,
        "objective": "matbench_phonons_focused_split_shortcut_review_no_training",
        "inputs": {
            "phase126_dir": _display_path(phase126_dir, root),
            "field_table": _display_path(field_path, root),
            "split_manifest": _display_path(split_path, root),
            "phase126_gate": _display_path(gate_path, root),
        },
        "outputs": {
            "profile_table": _display_path(profile_path, root),
            "split_sensitivity_table": _display_path(split_review_path, root),
            "shortcut_audit_table": _display_path(audit_path, root),
            "split_review_manifest": _display_path(split_manifest_path, root),
            "gate_json": _display_path(gate_out, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "profile_rows": len(profile_rows),
            "split_reviews": len(split_rows),
            "viable_split_reviews": gate["viable_split_reviews"],
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
    parser.add_argument("--phase126-dir", type=Path, default=DEFAULT_PHASE126_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    phase126_dir = args.phase126_dir if args.phase126_dir.is_absolute() else root / args.phase126_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = build_package(root=root, phase126_dir=phase126_dir, output_dir=output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
