#!/usr/bin/env python3
"""Build Phase 139 focused review for the Matbench glass gate.

This phase consumes only small Phase 138 artifacts. It checks whether the
Phase 138 glass-forming-ability classification gap is stable under alternate
grouped/binned splits and whether shortcut profiles, nearest-neighbor
composition identity, or class-balance shifts explain the signal. It does not
train a neural model or open A100/A800 training.
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


DEFAULT_PHASE138_DIR = Path("docs/results/phase138_matbench_glass_baseline_gate")
DEFAULT_OUTPUT_DIR = Path("docs/results/phase139_matbench_glass_focused_review")
MIN_SPLIT_ROWS = 100
MIN_BALANCED_ACCURACY_GAIN = 0.05
MIN_STABLE_SPLIT_PASS_RATE = 0.75
SHORTCUT_DOMINANCE_MARGIN = 0.005
MAX_CLASS_BALANCE_SHIFT = 0.20
MAX_NEAR_DUPLICATE_FRACTION = 0.60
NEAR_DUPLICATE_DISTANCE = 0.05


def _load_phase138_module():
    script = Path(__file__).with_name("build_phase138_matbench_glass_baseline_gate.py")
    spec = importlib.util.spec_from_file_location("phase138_helpers", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Phase 138 helper module from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase138 = _load_phase138_module()

TARGET = "gfa"
PROFILE_COLUMNS: dict[str, dict[str, Any]] = dict(phase138.PROFILE_COLUMNS)
MODEL_METHODS = phase138.MODEL_METHODS
PROFILE_METHODS = dict(phase138.PROFILE_METHODS)
NN_FEATURE_COLUMNS = tuple(
    column
    for column in (
        *(f"frac_{element}" for element in phase138.ELEMENTS),
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
)
SPLIT_PLAN = (
    ("phase138_registered_split", "phase138_manifest", "phase138"),
    ("chemistry_family_hash_0", "group:chemistry_family_key", "phase139_family_0"),
    ("chemistry_family_hash_1", "group:chemistry_family_key", "phase139_family_1"),
    ("chemistry_family_hash_2", "group:chemistry_family_key", "phase139_family_2"),
    ("dominant_element_hash", "group:dominant_element", "phase139_dominant"),
    ("element_count_bins", "bins:element_count", "phase139_element_count"),
    ("entropy_bins", "bins:entropy_fraction", "phase139_entropy"),
    ("transition_metal_bins", "bins:transition_metal_fraction", "phase139_transition"),
    ("metalloid_bins", "bins:metalloid_fraction", "phase139_metalloid"),
    ("max_fraction_bins", "bins:max_fraction", "phase139_max_fraction"),
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
    "train_balanced_accuracy",
    "val_balanced_accuracy",
    "test_balanced_accuracy",
    "val_accuracy",
    "test_accuracy",
    "val_log_loss",
    "test_log_loss",
    "majority_val_balanced_accuracy",
    "majority_test_balanced_accuracy",
    "val_balanced_accuracy_gain_vs_majority",
    "test_balanced_accuracy_gain_vs_majority",
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
    "train_positive_fraction",
    "val_positive_fraction",
    "test_positive_fraction",
    "class_balance_shift",
    "majority_val_balanced_accuracy",
    "majority_test_balanced_accuracy",
    "best_admissible_profile",
    "best_admissible_method",
    "best_admissible_val_balanced_accuracy",
    "best_admissible_test_balanced_accuracy",
    "best_admissible_val_accuracy",
    "best_admissible_test_accuracy",
    "best_admissible_val_log_loss",
    "best_admissible_test_log_loss",
    "best_admissible_val_gain_vs_majority",
    "best_admissible_test_gain_vs_majority",
    "split_pass",
    "best_negative_profile",
    "best_negative_method",
    "best_negative_val_balanced_accuracy",
    "best_negative_test_balanced_accuracy",
    "best_negative_val_log_loss",
    "best_negative_test_log_loss",
    "negative_control_dominates",
    "nearest_neighbor_val_balanced_accuracy",
    "nearest_neighbor_test_balanced_accuracy",
    "nearest_neighbor_val_gain_vs_majority",
    "nearest_neighbor_test_gain_vs_majority",
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


def _registered_split(split_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "split_strategy": split_manifest.get("split_strategy", "phase138_registered_split"),
        "split_salt": "phase138",
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
        if split_spec == "phase138_manifest":
            reviews[split_id] = _registered_split(split_manifest)
        elif split_spec.startswith("group:"):
            reviews[split_id] = _group_split(df, split_spec.split(":", 1)[1], salt)
        elif split_spec.startswith("bins:"):
            columns = tuple(split_spec.split(":", 1)[1].split(","))
            reviews[split_id] = _bin_split(df, columns, salt)
        else:
            raise ValueError(f"Unsupported split spec: {split_spec}")
    return reviews


def _binary_metrics(y_true: np.ndarray, positive_proba: np.ndarray) -> dict[str, Any]:
    clipped = np.clip(positive_proba.astype(float), 1e-6, 1.0 - 1e-6)
    pred = (clipped >= 0.5).astype(int)
    positives = y_true == 1
    negatives = y_true == 0
    true_positive_rate = float(np.mean(pred[positives] == 1)) if positives.any() else None
    true_negative_rate = float(np.mean(pred[negatives] == 0)) if negatives.any() else None
    if true_positive_rate is None or true_negative_rate is None:
        balanced_accuracy = 0.5
    else:
        balanced_accuracy = float(0.5 * (true_positive_rate + true_negative_rate))
    accuracy = float(np.mean(pred == y_true)) if len(y_true) else 0.0
    loss = -np.mean(y_true * np.log(clipped) + (1 - y_true) * np.log(1.0 - clipped)) if len(y_true) else 0.0
    return {
        "balanced_accuracy": balanced_accuracy,
        "accuracy": accuracy,
        "log_loss": float(loss),
        "true_positive_rate": true_positive_rate,
        "true_negative_rate": true_negative_rate,
        "true_positive_fraction": float(np.mean(y_true == 1)) if len(y_true) else 0.0,
        "pred_positive_fraction": float(np.mean(pred == 1)) if len(pred) else 0.0,
    }


def _majority_proba(y: np.ndarray, train_idx: list[int]) -> np.ndarray:
    train_positive = float(np.mean(y[train_idx] == 1)) if train_idx else 0.0
    majority_class = int(train_positive >= 0.5)
    return np.full(y.shape[0], float(majority_class), dtype=float)


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
    majority_val_ba: float,
    majority_test_ba: float,
) -> dict[str, Any]:
    from sklearn.neighbors import NearestNeighbors

    y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=int)
    train_idx = splits["train"]
    x_all = _standardized_feature_matrix(df, train_idx)
    model = NearestNeighbors(n_neighbors=1, metric="manhattan")
    model.fit(x_all[train_idx])
    distances, indices = model.kneighbors(x_all)
    train_labels = y[np.asarray(train_idx, dtype=int)]
    pred_labels = train_labels[indices[:, 0]]
    proba = pred_labels.astype(float)
    val_metrics = _binary_metrics(y[splits["val"]], proba[splits["val"]])
    test_metrics = _binary_metrics(y[splits["test"]], proba[splits["test"]])
    val_dist = distances[splits["val"], 0] if splits["val"] else np.array([], dtype=float)
    test_dist = distances[splits["test"], 0] if splits["test"] else np.array([], dtype=float)
    return {
        "nearest_neighbor_val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "nearest_neighbor_test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "nearest_neighbor_val_gain_vs_majority": float(val_metrics["balanced_accuracy"]) - majority_val_ba,
        "nearest_neighbor_test_gain_vs_majority": float(test_metrics["balanced_accuracy"]) - majority_test_ba,
        "val_near_duplicate_fraction": float(np.mean(val_dist <= NEAR_DUPLICATE_DISTANCE)) if len(val_dist) else 0.0,
        "test_near_duplicate_fraction": float(np.mean(test_dist <= NEAR_DUPLICATE_DISTANCE)) if len(test_dist) else 0.0,
    }


def _class_distribution(y: np.ndarray, splits: dict[str, list[int]]) -> dict[str, Any]:
    fractions = {
        split: float(np.mean(y[splits[split]] == 1)) if splits[split] else 0.0
        for split in ("train", "val", "test")
    }
    return {
        "train_positive_fraction": fractions["train"],
        "val_positive_fraction": fractions["val"],
        "test_positive_fraction": fractions["test"],
        "class_balance_shift": max(
            abs(fractions["val"] - fractions["train"]),
            abs(fractions["test"] - fractions["train"]),
        ),
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
            "class_balance_shift": None,
        }

    y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=int)
    train_idx = splits["train"]
    majority = _majority_proba(y, train_idx)
    majority_val = _binary_metrics(y[splits["val"]], majority[splits["val"]])
    majority_test = _binary_metrics(y[splits["test"]], majority[splits["test"]])

    rows: list[dict[str, Any]] = []
    for profile_name, profile in PROFILE_COLUMNS.items():
        x_train, x_all = phase138.phase123._one_hot_frame(df.iloc[train_idx], df, profile)
        y_train = y[train_idx]
        for method in PROFILE_METHODS.get(profile_name, MODEL_METHODS):
            proba = phase138._fit_predict_proba(method, x_train, y_train, x_all)
            split_metrics = {
                split: _binary_metrics(y[splits[split]], proba[splits[split]])
                for split in ("train", "val", "test")
            }
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
                    "train_balanced_accuracy": split_metrics["train"]["balanced_accuracy"],
                    "val_balanced_accuracy": split_metrics["val"]["balanced_accuracy"],
                    "test_balanced_accuracy": split_metrics["test"]["balanced_accuracy"],
                    "val_accuracy": split_metrics["val"]["accuracy"],
                    "test_accuracy": split_metrics["test"]["accuracy"],
                    "val_log_loss": split_metrics["val"]["log_loss"],
                    "test_log_loss": split_metrics["test"]["log_loss"],
                    "majority_val_balanced_accuracy": majority_val["balanced_accuracy"],
                    "majority_test_balanced_accuracy": majority_test["balanced_accuracy"],
                    "val_balanced_accuracy_gain_vs_majority": float(split_metrics["val"]["balanced_accuracy"])
                    - float(majority_val["balanced_accuracy"]),
                    "test_balanced_accuracy_gain_vs_majority": float(split_metrics["test"]["balanced_accuracy"])
                    - float(majority_test["balanced_accuracy"]),
                    "selected_admissible": False,
                    "selected_negative_control": False,
                }
            )

    admissible_rows = [row for row in rows if row["profile_role"] == "admissible"]
    negative_rows = [row for row in rows if row["profile_role"] == "negative_control"]
    best_admissible = max(
        admissible_rows,
        key=lambda row: (float(row["val_balanced_accuracy"]), -float(row["val_log_loss"])),
    )
    best_negative = max(
        negative_rows,
        key=lambda row: (float(row["val_balanced_accuracy"]), -float(row["val_log_loss"])),
    )
    best_admissible["selected_admissible"] = True
    best_negative["selected_negative_control"] = True
    split_pass = (
        float(best_admissible["val_balanced_accuracy_gain_vs_majority"]) > MIN_BALANCED_ACCURACY_GAIN
        and float(best_admissible["test_balanced_accuracy_gain_vs_majority"]) > 0.0
    )
    negative_dominates = (
        float(best_negative["val_balanced_accuracy"])
        >= float(best_admissible["val_balanced_accuracy"]) - SHORTCUT_DOMINANCE_MARGIN
        and float(best_negative["test_balanced_accuracy"]) >= float(majority_test["balanced_accuracy"])
    )
    nearest = _nearest_neighbor_control(
        df,
        target=target,
        splits=splits,
        majority_val_ba=float(majority_val["balanced_accuracy"]),
        majority_test_ba=float(majority_test["balanced_accuracy"]),
    )
    nearest_dominates = (
        float(nearest["nearest_neighbor_val_balanced_accuracy"])
        >= float(best_admissible["val_balanced_accuracy"]) - SHORTCUT_DOMINANCE_MARGIN
        and float(nearest["nearest_neighbor_test_balanced_accuracy"]) >= float(majority_test["balanced_accuracy"])
    )
    distribution = _class_distribution(y, splits)
    summary = {
        **base,
        "split_viable": True,
        "split_reason": "ok",
        **distribution,
        "majority_val_balanced_accuracy": majority_val["balanced_accuracy"],
        "majority_test_balanced_accuracy": majority_test["balanced_accuracy"],
        "best_admissible_profile": best_admissible["profile"],
        "best_admissible_method": best_admissible["method"],
        "best_admissible_val_balanced_accuracy": best_admissible["val_balanced_accuracy"],
        "best_admissible_test_balanced_accuracy": best_admissible["test_balanced_accuracy"],
        "best_admissible_val_accuracy": best_admissible["val_accuracy"],
        "best_admissible_test_accuracy": best_admissible["test_accuracy"],
        "best_admissible_val_log_loss": best_admissible["val_log_loss"],
        "best_admissible_test_log_loss": best_admissible["test_log_loss"],
        "best_admissible_val_gain_vs_majority": best_admissible["val_balanced_accuracy_gain_vs_majority"],
        "best_admissible_test_gain_vs_majority": best_admissible["test_balanced_accuracy_gain_vs_majority"],
        "split_pass": split_pass,
        "best_negative_profile": best_negative["profile"],
        "best_negative_method": best_negative["method"],
        "best_negative_val_balanced_accuracy": best_negative["val_balanced_accuracy"],
        "best_negative_test_balanced_accuracy": best_negative["test_balanced_accuracy"],
        "best_negative_val_log_loss": best_negative["val_log_loss"],
        "best_negative_test_log_loss": best_negative["test_log_loss"],
        "negative_control_dominates": negative_dominates,
        **nearest,
        "nearest_neighbor_dominates": nearest_dominates,
    }
    return rows, summary


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


def build_audit_rows(*, phase138_gate: dict[str, Any], split_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    phase138_ready = phase138_gate.get("status") == "phase138_matbench_glass_ready_focused_review"
    rows.append(
        _audit_row(
            audit="phase138_gate_status",
            status="pass" if phase138_ready else "block",
            severity="blocking" if not phase138_ready else "info",
            value=phase138_gate.get("status"),
            threshold="phase138_matbench_glass_ready_focused_review",
            reason="focused review requires a Phase 138 focused-review gate",
        )
    )
    original = next((row for row in split_rows if row.get("split_id") == "phase138_registered_split"), None)
    if original and original.get("split_viable"):
        rows.append(
            _audit_row(
                audit="original_split_admissible_gain",
                status="pass" if _is_true(original.get("split_pass")) else "block",
                severity="blocking" if not _is_true(original.get("split_pass")) else "info",
                value=original.get("best_admissible_val_gain_vs_majority"),
                threshold=MIN_BALANCED_ACCURACY_GAIN,
                reason="Phase 138 selected glass target must preserve validation and test balanced-accuracy gain",
            )
        )
        rows.append(
            _audit_row(
                audit="original_split_shortcut_dominance",
                status="block" if _is_true(original.get("negative_control_dominates")) else "pass",
                severity="blocking" if _is_true(original.get("negative_control_dominates")) else "info",
                value=original.get("best_negative_profile"),
                threshold=f"negative val BA < admissible - {SHORTCUT_DOMINANCE_MARGIN}",
                reason="composition, chemistry-family, or dominant-element shortcuts must not dominate",
            )
        )
        rows.append(
            _audit_row(
                audit="original_split_nearest_neighbor_dominance",
                status="block" if _is_true(original.get("nearest_neighbor_dominates")) else "pass",
                severity="blocking" if _is_true(original.get("nearest_neighbor_dominates")) else "info",
                value=original.get("nearest_neighbor_val_balanced_accuracy"),
                threshold=f"nearest val BA < admissible - {SHORTCUT_DOMINANCE_MARGIN}",
                reason="nearest-neighbor composition identity control must not dominate",
            )
        )
        balance_shift = float(original.get("class_balance_shift") or 0.0)
        rows.append(
            _audit_row(
                audit="original_split_class_balance",
                status="block" if balance_shift > MAX_CLASS_BALANCE_SHIFT else "pass",
                severity="blocking" if balance_shift > MAX_CLASS_BALANCE_SHIFT else "info",
                value=balance_shift,
                threshold=MAX_CLASS_BALANCE_SHIFT,
                reason="registered split class balance must not dominate interpretation",
            )
        )
        near_duplicate_fraction = max(
            float(original.get("val_near_duplicate_fraction") or 0.0),
            float(original.get("test_near_duplicate_fraction") or 0.0),
        )
        rows.append(
            _audit_row(
                audit="original_split_near_duplicate_fraction",
                status="block" if near_duplicate_fraction > MAX_NEAR_DUPLICATE_FRACTION else "pass",
                severity="blocking" if near_duplicate_fraction > MAX_NEAR_DUPLICATE_FRACTION else "info",
                value=near_duplicate_fraction,
                threshold=MAX_NEAR_DUPLICATE_FRACTION,
                reason="registered split must not contain too many near-duplicate composition rows relative to train",
            )
        )
    else:
        rows.append(
            _audit_row(
                audit="original_split_viability",
                status="block",
                severity="blocking",
                value=original.get("split_reason") if original else "missing",
                threshold=f"all splits >= {MIN_SPLIT_ROWS} rows",
                reason="Phase 138 registered split must be reviewable",
            )
        )
    viable = [row for row in split_rows if _is_true(row.get("split_viable"))]
    passed = [row for row in viable if _is_true(row.get("split_pass"))]
    stable_rate = len(passed) / len(viable) if viable else 0.0
    rows.append(
        _audit_row(
            audit="split_sensitivity_pass_rate",
            status="pass" if stable_rate >= MIN_STABLE_SPLIT_PASS_RATE else "block",
            severity="blocking" if stable_rate < MIN_STABLE_SPLIT_PASS_RATE else "info",
            value=stable_rate,
            threshold=MIN_STABLE_SPLIT_PASS_RATE,
            reason="glass target gain must survive deterministic chemistry, dominant-element, and descriptor split perturbations",
        )
    )
    shortcut_dominant = [row for row in viable if _is_true(row.get("negative_control_dominates"))]
    rows.append(
        _audit_row(
            audit="shortcut_dominant_split_count",
            status="block" if shortcut_dominant else "pass",
            severity="blocking" if shortcut_dominant else "info",
            value=len(shortcut_dominant),
            threshold=0,
            reason="no viable split may be dominated by composition, chemistry-family, or dominant-element shortcuts",
        )
    )
    nearest_dominant = [row for row in viable if _is_true(row.get("nearest_neighbor_dominates"))]
    rows.append(
        _audit_row(
            audit="nearest_neighbor_dominant_split_count",
            status="block" if nearest_dominant else "pass",
            severity="blocking" if nearest_dominant else "info",
            value=len(nearest_dominant),
            threshold=0,
            reason="no viable split may be dominated by nearest-neighbor composition identity control",
        )
    )
    class_imbalanced = [
        row
        for row in viable
        if float(row.get("class_balance_shift") or 0.0) > MAX_CLASS_BALANCE_SHIFT
    ]
    rows.append(
        _audit_row(
            audit="class_balance_imbalanced_split_count",
            status="block" if class_imbalanced else "pass",
            severity="blocking" if class_imbalanced else "info",
            value=len(class_imbalanced),
            threshold=0,
            reason="no viable split may have severe train/validation/test glass class-balance imbalance",
        )
    )
    return rows


def build_gate(*, phase138_gate: dict[str, Any], split_rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    blockers = [row for row in audit_rows if row["status"] == "block"]
    viable = [row for row in split_rows if _is_true(row.get("split_viable"))]
    passed = [row for row in viable if _is_true(row.get("split_pass"))]
    shortcut_dominant = [row for row in viable if _is_true(row.get("negative_control_dominates"))]
    nearest_dominant = [row for row in viable if _is_true(row.get("nearest_neighbor_dominates"))]
    class_imbalanced = [
        row
        for row in viable
        if float(row.get("class_balance_shift") or 0.0) > MAX_CLASS_BALANCE_SHIFT
    ]
    original = next((row for row in split_rows if row.get("split_id") == "phase138_registered_split"), {})
    if phase138_gate.get("status") != "phase138_matbench_glass_ready_focused_review":
        status = "phase139_matbench_glass_review_blocked_by_phase138"
        mechanism_allowed = False
        next_action = "complete or close Phase 138 before focused review"
    elif blockers:
        status = "phase139_matbench_glass_focused_review_closed_split_sensitivity_or_shortcut"
        mechanism_allowed = False
        next_action = "close the Phase 138 glass target as diagnostic; do not train"
    else:
        status = "phase139_matbench_glass_focused_review_ready_low_capacity_mechanism_gate"
        mechanism_allowed = True
        next_action = "design a separate no-training low-capacity glass mechanism gate; keep model training closed"
    return {
        "status": status,
        "phase138_status": phase138_gate.get("status"),
        "selected_target": phase138_gate.get("selected_target", TARGET),
        "phase138_selected_profile": phase138_gate.get("selected_profile"),
        "phase138_selected_method": phase138_gate.get("selected_method"),
        "viable_split_reviews": len(viable),
        "passed_split_reviews": len(passed),
        "split_pass_rate": len(passed) / len(viable) if viable else 0.0,
        "shortcut_dominant_splits": len(shortcut_dominant),
        "nearest_neighbor_dominant_splits": len(nearest_dominant),
        "class_balance_imbalanced_splits": len(class_imbalanced),
        "blocking_audit_rows": len(blockers),
        "blocking_audits": [row["audit"] for row in blockers],
        "original_best_admissible_profile": original.get("best_admissible_profile"),
        "original_best_admissible_method": original.get("best_admissible_method"),
        "original_best_admissible_val_balanced_accuracy": original.get("best_admissible_val_balanced_accuracy"),
        "original_best_admissible_test_balanced_accuracy": original.get("best_admissible_test_balanced_accuracy"),
        "original_best_negative_profile": original.get("best_negative_profile"),
        "original_best_negative_method": original.get("best_negative_method"),
        "original_best_negative_val_balanced_accuracy": original.get("best_negative_val_balanced_accuracy"),
        "original_best_negative_test_balanced_accuracy": original.get("best_negative_test_balanced_accuracy"),
        "original_nearest_neighbor_val_balanced_accuracy": original.get("nearest_neighbor_val_balanced_accuracy"),
        "original_nearest_neighbor_test_balanced_accuracy": original.get("nearest_neighbor_test_balanced_accuracy"),
        "original_class_balance_shift": original.get("class_balance_shift"),
        "phase139_model_mechanism_allowed": mechanism_allowed,
        "phase139_low_capacity_mechanism_design_allowed": mechanism_allowed,
        "phase139_model_training_allowed": False,
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
        "# Phase 139 Matbench Glass Focused Review",
        "",
        f"- Status: `{gate['status']}`",
        f"- Split pass rate: `{gate['split_pass_rate']:.6g}`",
        f"- Blocking audits: `{', '.join(gate['blocking_audits']) or 'none'}`",
        f"- Low-capacity mechanism design allowed: `{gate['phase139_low_capacity_mechanism_design_allowed']}`",
        f"- Model training allowed: `{gate['phase139_model_training_allowed']}`",
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
                ("Val BA", "best_admissible_val_balanced_accuracy"),
                ("Test BA", "best_admissible_test_balanced_accuracy"),
                ("Shortcut", "negative_control_dominates"),
                ("NN", "nearest_neighbor_dominates"),
                ("Class shift", "class_balance_shift"),
            ],
        ),
    ]
    return "\n".join(lines) + "\n"


def build_package(*, root: Path, phase138_dir: Path, output_dir: Path) -> dict[str, Any]:
    field_path = phase138_dir / "phase138_matbench_glass_field_table.csv"
    split_path = phase138_dir / "phase138_matbench_glass_split_manifest.json"
    gate_path = phase138_dir / "phase138_matbench_glass_gate.json"
    df = pd.read_csv(field_path)
    split_manifest = _read_json(split_path)
    phase138_gate = _read_json(gate_path)
    target = str(phase138_gate.get("selected_target") or TARGET)

    split_infos = build_split_reviews(df, split_manifest)
    profile_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for split_id, split_info in split_infos.items():
        rows, summary = evaluate_split(df, target=target, split_id=split_id, split_info=split_info)
        profile_rows.extend(rows)
        split_rows.append(summary)
    audit_rows = build_audit_rows(phase138_gate=phase138_gate, split_rows=split_rows)
    gate = build_gate(phase138_gate=phase138_gate, split_rows=split_rows, audit_rows=audit_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "phase139_matbench_glass_focused_profile_table.csv"
    split_review_path = output_dir / "phase139_matbench_glass_split_sensitivity_table.csv"
    audit_path = output_dir / "phase139_matbench_glass_shortcut_audit_table.csv"
    split_manifest_path = output_dir / "phase139_matbench_glass_split_review_manifest.json"
    gate_out = output_dir / "phase139_matbench_glass_focused_review_gate.json"
    markdown_path = output_dir / "phase139_matbench_glass_focused_review.md"
    manifest_path = output_dir / "phase139_matbench_glass_focused_review_manifest.json"

    _write_csv(profile_path, profile_rows, PROFILE_FIELDS)
    _write_csv(split_review_path, split_rows, SPLIT_FIELDS)
    _write_csv(audit_path, audit_rows, AUDIT_FIELDS)
    _write_json(split_manifest_path, split_infos)
    _write_json(gate_out, gate)
    markdown_path.write_text(build_markdown(gate, audit_rows, split_rows), encoding="utf-8")

    manifest = {
        "phase": 139,
        "objective": "matbench_glass_focused_split_shortcut_review_no_training",
        "inputs": {
            "phase138_dir": _display_path(phase138_dir, root),
            "field_table": _display_path(field_path, root),
            "split_manifest": _display_path(split_path, root),
            "phase138_gate": _display_path(gate_path, root),
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
    parser.add_argument("--phase138-dir", type=Path, default=DEFAULT_PHASE138_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    phase138_dir = args.phase138_dir if args.phase138_dir.is_absolute() else root / args.phase138_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = build_package(root=root, phase138_dir=phase138_dir, output_dir=output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
