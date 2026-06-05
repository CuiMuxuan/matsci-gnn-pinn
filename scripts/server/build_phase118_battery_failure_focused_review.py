#!/usr/bin/env python3
"""Build Phase 118 focused review for the Battery Failure Databank gate.

This phase consumes only the small Phase 117 artifacts. It audits whether the
selected Phase 117 target is stable under alternative leakage-safe splits and
whether the apparent signal is dominated by target-family, post-test, cell, or
series shortcuts. It does not train a neural model or open A100 training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PHASE117_DIR = Path("docs/results/phase117_battery_failure_databank_gate")
DEFAULT_OUTPUT_DIR = Path("docs/results/phase118_battery_failure_focused_review")

MIN_SPLIT_ROWS = 20
MIN_RELATIVE_VAL_GAIN = 0.05
MIN_STABLE_SPLIT_PASS_RATE = 0.80
NEGATIVE_DOMINANCE_TOLERANCE = 1.02
DEPENDENCY_CORR_BLOCK_THRESHOLD = 0.70

SELECTED_TARGET_FALLBACK = "Post-Test-Mass-Unrecovered-g"

CELL_NUMERIC_FEATURES = (
    "Cell-Capacity-Ah",
    "Cell-Nominal-Voltage-V",
    "Cell-Energy-Wh",
    "Pre-Test-Cell-Open-Circuit-Voltage-V",
    "Pre-Test-Cell-Mass-g",
)
TRIGGER_NUMERIC_FEATURES = (
    "Heater-Power-W",
    "Heater-Time-On-s",
    "Energy-Applied-to-Trigger-kJ",
    "Avg-Cell-Temp-At-Trigger-degC",
)
CELL_CATEGORICAL_FEATURES = ("Cell-Format", "S-FTRC-Generation")
TRIGGER_CATEGORICAL_FEATURES = (
    "Trigger-Mechanism",
    "Pressure-Assisted-Seal-Configuration-Positive",
    "Pressure-Assisted-Seal-Configuration-Negative",
)
TARGET_FAMILY_COLUMNS = (
    "Corrected-Total-Energy-Yield-kJ",
    "Baseline-Plus-Heat-Loss-Total-Energy-Yield-kJ",
    "Baseline-Total-Energy-Yield-kJ",
    "Energy-Percent-Positive-Ejecta-%",
    "Energy-Percent-Negative-Ejecta-%",
)

PROFILE_COLUMNS: dict[str, dict[str, Any]] = {
    "cell_pretest": {
        "role": "admissible",
        "numeric": CELL_NUMERIC_FEATURES,
        "categorical": CELL_CATEGORICAL_FEATURES,
    },
    "trigger_only": {
        "role": "admissible",
        "numeric": TRIGGER_NUMERIC_FEATURES,
        "categorical": ("Trigger-Mechanism",),
    },
    "cell_trigger_safe": {
        "role": "admissible",
        "numeric": CELL_NUMERIC_FEATURES + TRIGGER_NUMERIC_FEATURES,
        "categorical": CELL_CATEGORICAL_FEATURES + TRIGGER_CATEGORICAL_FEATURES,
    },
    "cell_trigger_no_pretest_mass": {
        "role": "admissible",
        "numeric": (
            "Cell-Capacity-Ah",
            "Cell-Nominal-Voltage-V",
            "Cell-Energy-Wh",
            "Pre-Test-Cell-Open-Circuit-Voltage-V",
            *TRIGGER_NUMERIC_FEATURES,
        ),
        "categorical": CELL_CATEGORICAL_FEATURES + TRIGGER_CATEGORICAL_FEATURES,
    },
    "cell_mass_only": {
        "role": "scaling_control",
        "numeric": ("Pre-Test-Cell-Mass-g",),
        "categorical": (),
    },
    "series_shortcut": {
        "role": "negative_control",
        "numeric": (),
        "categorical": ("Test-Series",),
    },
    "cell_description_shortcut": {
        "role": "negative_control",
        "numeric": (),
        "categorical": ("Cell-Description",),
    },
    "target_family_leakage": {
        "role": "negative_control",
        "numeric": TARGET_FAMILY_COLUMNS,
        "categorical": (),
    },
    "target_family_plus_shortcuts": {
        "role": "negative_control",
        "numeric": TARGET_FAMILY_COLUMNS,
        "categorical": ("Test-Series", "Cell-Description"),
    },
}

MODEL_METHODS = ("knn", "extra_trees", "hist_gradient_boosting")
SPLIT_PLAN = (
    ("phase117_registered_split", "phase117_manifest", "phase117"),
    ("cell_description_hash_0", "Cell-Description", "phase118_cell_0"),
    ("cell_description_hash_1", "Cell-Description", "phase118_cell_1"),
    ("cell_description_hash_2", "Cell-Description", "phase118_cell_2"),
    ("cell_description_hash_3", "Cell-Description", "phase118_cell_3"),
    ("cell_description_hash_4", "Cell-Description", "phase118_cell_4"),
    ("test_series_hash", "Test-Series", "phase118_series"),
    ("s_ftrc_generation_hash", "S-FTRC-Generation", "phase118_generation"),
    ("trigger_mechanism_hash", "Trigger-Mechanism", "phase118_trigger"),
    ("cell_format_hash", "Cell-Format", "phase118_format"),
)

PROFILE_FIELDS = (
    "target",
    "split_id",
    "group_column",
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
    "group_column",
    "split_salt",
    "split_viable",
    "split_reason",
    "train_rows",
    "val_rows",
    "test_rows",
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
)
DEPENDENCY_FIELDS = (
    "target",
    "dependency_column",
    "dependency_family",
    "non_missing_pairs",
    "pearson_corr",
    "abs_pearson_corr",
    "status",
    "reason",
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


def _profile_role(profile: str) -> str:
    return str(PROFILE_COLUMNS[profile]["role"])


def _group_key(value: Any) -> str:
    text = str(value).strip()
    return text if text and text.lower() != "nan" else "missing_group"


def _hash_group_split(df: pd.DataFrame, *, group_column: str, salt: str) -> dict[str, Any]:
    if group_column not in df.columns:
        return {
            "split_strategy": f"missing_group_column_{group_column}",
            "group_column": group_column,
            "split_viable": False,
            "reason": "group column missing",
            "splits": {"train": [], "val": [], "test": []},
            "group_splits": {"train": [], "val": [], "test": []},
            "n_groups": 0,
        }
    groups = sorted({_group_key(value) for value in df[group_column]})
    if len(groups) < 3:
        return {
            "split_strategy": f"group_hash_by_{group_column}",
            "group_column": group_column,
            "split_viable": False,
            "reason": "fewer than three groups",
            "splits": {"train": [], "val": [], "test": []},
            "group_splits": {"train": [], "val": [], "test": []},
            "n_groups": len(groups),
        }
    ranked = sorted(groups, key=lambda item: hashlib.sha256(f"{salt}::{item}".encode()).hexdigest())
    n_groups = len(ranked)
    train_count = max(1, int(round(n_groups * 0.6)))
    val_count = max(1, int(round(n_groups * 0.2)))
    if train_count + val_count >= n_groups:
        train_count = max(1, n_groups - 2)
        val_count = 1
    group_to_split: dict[str, str] = {}
    for index, group in enumerate(ranked):
        if index < train_count:
            group_to_split[group] = "train"
        elif index < train_count + val_count:
            group_to_split[group] = "val"
        else:
            group_to_split[group] = "test"
    assignments = [group_to_split[_group_key(value)] for value in df[group_column]]
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
        "split_salt": salt,
        "split_viable": True,
        "reason": "ok",
        "splits": splits,
        "group_splits": group_splits,
        "n_groups": n_groups,
        "leakage_safe": sum(len(values) for values in group_splits.values()) == n_groups,
    }


def _remap_phase117_split(target_df: pd.DataFrame, split_manifest: dict[str, Any]) -> dict[str, Any]:
    old_to_new = {
        int(old_index): int(new_index)
        for new_index, old_index in enumerate(target_df["source_index"].tolist())
    }
    splits = {
        split: [old_to_new[int(index)] for index in indices if int(index) in old_to_new]
        for split, indices in split_manifest["splits"].items()
    }
    return {
        "split_strategy": split_manifest.get("split_strategy", "phase117_registered_split"),
        "group_column": split_manifest.get("group_column", "Cell-Description"),
        "split_salt": "phase117",
        "split_viable": True,
        "reason": "ok",
        "splits": splits,
        "group_splits": split_manifest.get("group_splits", {}),
        "n_groups": split_manifest.get("n_groups"),
        "leakage_safe": split_manifest.get("leakage_safe"),
    }


def _target_subset(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in df.columns:
        raise ValueError(f"Selected target is missing from field table: {target}")
    mask = pd.to_numeric(df[target], errors="coerce").notna()
    return df.loc[mask].copy().reset_index(drop=False).rename(columns={"index": "source_index"})


def _one_hot_frame(
    train: pd.DataFrame,
    all_rows: pd.DataFrame,
    profile: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
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
    x_train = pd.concat(frames_train, axis=1).to_numpy(dtype=float)
    x_all = pd.concat(frames_all, axis=1).to_numpy(dtype=float)
    return x_train, x_all


def _fit_predict(method: str, x_train: np.ndarray, y_train: np.ndarray, x_all: np.ndarray) -> np.ndarray:
    if method == "knn":
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=max(1, min(8, len(y_train)))),
        )
    elif method == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        model = ExtraTreesRegressor(n_estimators=200, random_state=118, n_jobs=-1)
    elif method == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(
            max_iter=200,
            random_state=118,
            early_stopping=False,
        )
    else:
        raise ValueError(f"Unsupported method: {method}")
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_all), dtype=float)


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


def _evaluate_split(
    target_df: pd.DataFrame,
    *,
    target: str,
    split_id: str,
    split_info: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    splits = split_info["splits"]
    counts = {split: len(splits[split]) for split in ("train", "val", "test")}
    base_summary: dict[str, Any] = {
        "target": target,
        "split_id": split_id,
        "group_column": split_info.get("group_column"),
        "split_salt": split_info.get("split_salt"),
        "train_rows": counts["train"],
        "val_rows": counts["val"],
        "test_rows": counts["test"],
    }
    if not split_info.get("split_viable", True):
        return [], {
            **base_summary,
            "split_viable": False,
            "split_reason": split_info.get("reason", "split is not viable"),
            "split_pass": False,
            "negative_control_dominates": False,
        }
    if min(counts.values()) < MIN_SPLIT_ROWS:
        return [], {
            **base_summary,
            "split_viable": False,
            "split_reason": "one or more splits is below minimum row count",
            "split_pass": False,
            "negative_control_dominates": False,
        }

    y = pd.to_numeric(target_df[target], errors="coerce").to_numpy(dtype=float)
    train_idx = splits["train"]
    train_std = float(np.std(y[train_idx])) if train_idx else 0.0
    train_mean = float(np.mean(y[train_idx]))
    mean_pred = np.full_like(y, train_mean, dtype=float)
    mean_val = _metrics(y[splits["val"]], mean_pred[splits["val"]], train_std)
    mean_test = _metrics(y[splits["test"]], mean_pred[splits["test"]], train_std)

    rows: list[dict[str, Any]] = []
    for profile_name, profile in PROFILE_COLUMNS.items():
        x_train, x_all = _one_hot_frame(target_df.iloc[train_idx], target_df, profile)
        y_train = y[train_idx]
        for method in MODEL_METHODS:
            pred = _fit_predict(method, x_train, y_train, x_all)
            split_metrics = {
                split: _metrics(y[splits[split]], pred[splits[split]], train_std)
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
                    "group_column": split_info.get("group_column"),
                    "profile": profile_name,
                    "profile_role": _profile_role(profile_name),
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
                    "relative_val_gain": val_gain / float(mean_val["rmse"])
                    if float(mean_val["rmse"]) > 0
                    else None,
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
        float(best_negative["val_rmse"])
        <= float(best_admissible["val_rmse"]) * NEGATIVE_DOMINANCE_TOLERANCE
        and float(best_negative["test_gain_vs_mean"]) > 0.0
    )
    summary = {
        **base_summary,
        "split_viable": True,
        "split_reason": "ok",
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
    }
    return rows, summary


def _dependency_rows(df: pd.DataFrame, *, target: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_values = pd.to_numeric(df[target], errors="coerce")
    dependency_columns = [
        *TARGET_FAMILY_COLUMNS,
        "Pre-Test-Cell-Mass-g",
        "Cell-Energy-Wh",
        "Energy-Applied-to-Trigger-kJ",
    ]
    for column in dependency_columns:
        if column == target or column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        mask = target_values.notna() & values.notna()
        count = int(mask.sum())
        corr = float(values[mask].corr(target_values[mask])) if count > 2 else None
        family = "target_family" if column in TARGET_FAMILY_COLUMNS else "safe_feature_scaling"
        blocked = corr is not None and abs(corr) >= DEPENDENCY_CORR_BLOCK_THRESHOLD
        rows.append(
            {
                "target": target,
                "dependency_column": column,
                "dependency_family": family,
                "non_missing_pairs": count,
                "pearson_corr": corr,
                "abs_pearson_corr": abs(corr) if corr is not None else None,
                "status": "high_dependency_risk" if blocked else "reviewed",
                "reason": "absolute correlation exceeds dependency risk threshold"
                if blocked
                else "below dependency risk threshold",
            }
        )
    return rows


def _audit_rows(
    *,
    phase117_gate: dict[str, Any],
    target: str,
    profile_rows: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
    dependency_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    phase117_ready = (
        phase117_gate.get("status") == "phase117_battery_failure_databank_gap_ready_focused_review"
    )
    rows.append(
        {
            "audit": "phase117_gate_status",
            "status": "pass" if phase117_ready else "block",
            "severity": "blocking" if not phase117_ready else "info",
            "value": phase117_gate.get("status"),
            "threshold": "phase117_battery_failure_databank_gap_ready_focused_review",
            "reason": "focused review requires a Phase 117 focused-review gate",
        }
    )
    is_post_test = target.startswith("Post-Test")
    rows.append(
        {
            "audit": "selected_target_post_test_family",
            "status": "risk" if is_post_test else "pass",
            "severity": "high" if is_post_test else "info",
            "value": target,
            "threshold": "not a post-test field",
            "reason": "post-test targets require leakage and scaling review before mechanism design",
        }
    )
    original = next(
        (row for row in split_rows if row.get("split_id") == "phase117_registered_split"),
        None,
    )
    if original and original.get("split_viable"):
        rows.append(
            {
                "audit": "original_split_admissible_gain",
                "status": "pass" if _is_true(original.get("split_pass")) else "block",
                "severity": "blocking" if not _is_true(original.get("split_pass")) else "info",
                "value": original.get("best_admissible_relative_val_gain"),
                "threshold": MIN_RELATIVE_VAL_GAIN,
                "reason": "selected target must preserve admissible validation and test gain",
            }
        )
        rows.append(
            {
                "audit": "original_split_negative_control_dominance",
                "status": "block" if _is_true(original.get("negative_control_dominates")) else "pass",
                "severity": "blocking" if _is_true(original.get("negative_control_dominates")) else "info",
                "value": original.get("best_negative_profile"),
                "threshold": f"negative val RMSE > admissible * {NEGATIVE_DOMINANCE_TOLERANCE}",
                "reason": "target-family or shortcut negative controls must not dominate the selected safe profile",
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
                "reason": "Phase 117 registered split must be reviewable",
            }
        )

    viable = [row for row in split_rows if _is_true(row.get("split_viable"))]
    if viable:
        pass_count = sum(1 for row in viable if _is_true(row.get("split_pass")))
        stable_rate = pass_count / len(viable)
    else:
        stable_rate = 0.0
    rows.append(
        {
            "audit": "split_sensitivity_pass_rate",
            "status": "pass" if stable_rate >= MIN_STABLE_SPLIT_PASS_RATE else "block",
            "severity": "blocking" if stable_rate < MIN_STABLE_SPLIT_PASS_RATE else "info",
            "value": stable_rate,
            "threshold": MIN_STABLE_SPLIT_PASS_RATE,
            "reason": "admissible validation/test gain must survive deterministic group split perturbations",
        }
    )
    high_dependencies = [
        row
        for row in dependency_rows
        if row["dependency_family"] == "target_family"
        and row["abs_pearson_corr"] is not None
        and float(row["abs_pearson_corr"]) >= DEPENDENCY_CORR_BLOCK_THRESHOLD
    ]
    max_target_dependency = max(
        [float(row["abs_pearson_corr"]) for row in dependency_rows if row["dependency_family"] == "target_family"],
        default=0.0,
    )
    rows.append(
        {
            "audit": "target_family_dependency",
            "status": "block" if high_dependencies else "pass",
            "severity": "blocking" if high_dependencies else "info",
            "value": max_target_dependency,
            "threshold": DEPENDENCY_CORR_BLOCK_THRESHOLD,
            "reason": "selected target must not be tightly coupled to other derived target-family columns",
        }
    )
    mass_row = next(
        (row for row in dependency_rows if row["dependency_column"] == "Pre-Test-Cell-Mass-g"),
        None,
    )
    mass_corr = float(mass_row["abs_pearson_corr"]) if mass_row and mass_row["abs_pearson_corr"] is not None else 0.0
    rows.append(
        {
            "audit": "pretest_mass_scaling_dependency",
            "status": "risk" if mass_corr >= DEPENDENCY_CORR_BLOCK_THRESHOLD else "pass",
            "severity": "medium" if mass_corr >= DEPENDENCY_CORR_BLOCK_THRESHOLD else "info",
            "value": mass_corr,
            "threshold": DEPENDENCY_CORR_BLOCK_THRESHOLD,
            "reason": "mass-derived targets can be dominated by pre-test cell inventory scaling",
        }
    )
    mass_controls = [
        row
        for row in profile_rows
        if row["split_id"] == "phase117_registered_split"
        and row["profile"] == "cell_mass_only"
        and row["method"] in MODEL_METHODS
    ]
    selected_admissible = [
        row
        for row in profile_rows
        if row["split_id"] == "phase117_registered_split" and _is_true(row["selected_admissible"])
    ]
    if mass_controls and selected_admissible:
        best_mass = min(mass_controls, key=lambda row: float(row["val_rmse"]))
        selected = selected_admissible[0]
        mass_dominates = (
            float(best_mass["val_rmse"]) <= float(selected["val_rmse"]) * NEGATIVE_DOMINANCE_TOLERANCE
            and float(best_mass["test_gain_vs_mean"]) > 0.0
        )
        rows.append(
            {
                "audit": "cell_mass_only_scaling_control",
                "status": "risk" if mass_dominates else "pass",
                "severity": "medium" if mass_dominates else "info",
                "value": best_mass["val_rmse"],
                "threshold": f"mass-only val RMSE > admissible * {NEGATIVE_DOMINANCE_TOLERANCE}",
                "reason": "mass-only scaling should not explain nearly all selected-target signal",
            }
        )
    return rows


def _build_gate(
    *,
    target: str,
    phase117_gate: dict[str, Any],
    profile_rows: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
    dependency_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    blockers = [row for row in audit_rows if row["status"] == "block"]
    viable_splits = [row for row in split_rows if _is_true(row.get("split_viable"))]
    passed_splits = [row for row in viable_splits if _is_true(row.get("split_pass"))]
    negative_dominant_splits = [
        row for row in viable_splits if _is_true(row.get("negative_control_dominates"))
    ]
    original = next(
        (row for row in split_rows if row.get("split_id") == "phase117_registered_split"),
        None,
    )
    if phase117_gate.get("status") != "phase117_battery_failure_databank_gap_ready_focused_review":
        status = "phase118_battery_failure_review_blocked_by_phase117"
        next_action = "complete or close Phase 117 before focused target review"
        mechanism_allowed = False
    elif blockers:
        status = "phase118_battery_failure_focused_review_closed_leakage_or_split_sensitivity"
        next_action = "close the Phase 117 selected target as diagnostic; do not train"
        mechanism_allowed = False
    else:
        status = "phase118_battery_failure_focused_review_ready_low_capacity_mechanism_gate"
        next_action = "design a separate low-capacity no-training mechanism gate; keep model training closed"
        mechanism_allowed = True
    return {
        "status": status,
        "phase117_status": phase117_gate.get("status"),
        "selected_target": target,
        "phase117_selected_profile": phase117_gate.get("selected_profile"),
        "phase117_selected_method": phase117_gate.get("selected_method"),
        "row_count": int(max((row.get("train_rows", 0) + row.get("val_rows", 0) + row.get("test_rows", 0)) for row in split_rows)),
        "viable_split_reviews": len(viable_splits),
        "passed_split_reviews": len(passed_splits),
        "split_pass_rate": len(passed_splits) / len(viable_splits) if viable_splits else 0.0,
        "negative_control_dominant_splits": len(negative_dominant_splits),
        "dependency_risk_rows": sum(1 for row in dependency_rows if row["status"] == "high_dependency_risk"),
        "blocking_audit_rows": len(blockers),
        "blocking_audits": [row["audit"] for row in blockers],
        "original_best_admissible_profile": original.get("best_admissible_profile") if original else None,
        "original_best_admissible_method": original.get("best_admissible_method") if original else None,
        "original_best_admissible_val_rmse": original.get("best_admissible_val_rmse") if original else None,
        "original_best_admissible_test_rmse": original.get("best_admissible_test_rmse") if original else None,
        "original_best_negative_profile": original.get("best_negative_profile") if original else None,
        "original_best_negative_method": original.get("best_negative_method") if original else None,
        "phase118_model_mechanism_allowed": mechanism_allowed,
        "phase118_low_capacity_mechanism_design_allowed": mechanism_allowed,
        "phase118_model_training_allowed": False,
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
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def _build_markdown(gate: dict[str, Any], audit_rows: list[dict[str, Any]], split_rows: list[dict[str, Any]]) -> str:
    blocking = [row for row in audit_rows if row["status"] == "block"]
    viable = [row for row in split_rows if _is_true(row.get("split_viable"))]
    lines = [
        "# Phase 118 Battery Failure Focused Review",
        "",
        f"- Status: `{gate['status']}`",
        f"- Selected target: `{gate['selected_target']}`",
        f"- Viable split reviews: `{gate['viable_split_reviews']}`",
        f"- Split pass rate: `{gate['split_pass_rate']:.6g}`",
        f"- Blocking audits: `{', '.join(gate['blocking_audits']) or 'none'}`",
        f"- Low-capacity mechanism design allowed: `{gate['phase118_low_capacity_mechanism_design_allowed']}`",
        f"- Model training allowed: `{gate['phase118_model_training_allowed']}`",
        f"- A100 training allowed now: `{gate['a100_training_allowed_now']}`",
        "",
        "## Blocking Audits",
        "",
        _markdown_table(
            blocking,
            [
                ("Audit", "audit"),
                ("Status", "status"),
                ("Value", "value"),
                ("Threshold", "threshold"),
                ("Reason", "reason"),
            ],
        ),
        "",
        "## Split Review Summary",
        "",
        _markdown_table(
            viable,
            [
                ("Split", "split_id"),
                ("Group", "group_column"),
                ("Pass", "split_pass"),
                ("Best profile", "best_admissible_profile"),
                ("Best val RMSE", "best_admissible_val_rmse"),
                ("Best test RMSE", "best_admissible_test_rmse"),
                ("Negative dominates", "negative_control_dominates"),
            ],
        ),
    ]
    return "\n".join(lines) + "\n"


def build_package(
    *,
    root: Path,
    phase117_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    field_table_path = phase117_dir / "phase117_battery_failure_databank_field_table.csv"
    split_manifest_path = phase117_dir / "phase117_battery_failure_databank_split_manifest.json"
    gate_path = phase117_dir / "phase117_battery_failure_databank_gate.json"
    review_path = phase117_dir / "phase117_battery_failure_databank_target_review_table.csv"

    field_table = pd.read_csv(field_table_path)
    phase117_split = _read_json(split_manifest_path)
    phase117_gate = _read_json(gate_path)
    target = str(phase117_gate.get("selected_target") or SELECTED_TARGET_FALLBACK)
    target_df = _target_subset(field_table, target)

    profile_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    split_manifests: dict[str, Any] = {}
    for split_id, group_column, salt in SPLIT_PLAN:
        if group_column == "phase117_manifest":
            split_info = _remap_phase117_split(target_df, phase117_split)
        else:
            split_info = _hash_group_split(target_df, group_column=group_column, salt=salt)
        split_manifests[split_id] = split_info
        rows, summary = _evaluate_split(
            target_df,
            target=target,
            split_id=split_id,
            split_info=split_info,
        )
        profile_rows.extend(rows)
        split_rows.append(summary)

    dependency_rows = _dependency_rows(target_df, target=target)
    audit_rows = _audit_rows(
        phase117_gate=phase117_gate,
        target=target,
        profile_rows=profile_rows,
        split_rows=split_rows,
        dependency_rows=dependency_rows,
    )
    gate = _build_gate(
        target=target,
        phase117_gate=phase117_gate,
        profile_rows=profile_rows,
        split_rows=split_rows,
        dependency_rows=dependency_rows,
        audit_rows=audit_rows,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "phase118_battery_failure_focused_profile_table.csv"
    split_path = output_dir / "phase118_battery_failure_split_sensitivity_table.csv"
    dependency_path = output_dir / "phase118_battery_failure_target_family_dependency_table.csv"
    audit_path = output_dir / "phase118_battery_failure_leakage_shortcut_audit_table.csv"
    split_manifest_out = output_dir / "phase118_battery_failure_split_review_manifest.json"
    gate_out = output_dir / "phase118_battery_failure_focused_review_gate.json"
    markdown_path = output_dir / "phase118_battery_failure_focused_review.md"
    manifest_path = output_dir / "phase118_battery_failure_focused_review_manifest.json"

    _write_csv(profile_path, profile_rows, PROFILE_FIELDS)
    _write_csv(split_path, split_rows, SPLIT_FIELDS)
    _write_csv(dependency_path, dependency_rows, DEPENDENCY_FIELDS)
    _write_csv(audit_path, audit_rows, AUDIT_FIELDS)
    _write_json(split_manifest_out, split_manifests)
    _write_json(gate_out, gate)
    markdown_path.write_text(_build_markdown(gate, audit_rows, split_rows), encoding="utf-8")

    manifest = {
        "phase": 118,
        "objective": "battery_failure_focused_leakage_target_review_no_training",
        "inputs": {
            "phase117_dir": _display_path(phase117_dir, root),
            "field_table": _display_path(field_table_path, root),
            "split_manifest": _display_path(split_manifest_path, root),
            "phase117_gate": _display_path(gate_path, root),
            "phase117_target_review_table": _display_path(review_path, root),
        },
        "outputs": {
            "profile_table": _display_path(profile_path, root),
            "split_sensitivity_table": _display_path(split_path, root),
            "target_family_dependency_table": _display_path(dependency_path, root),
            "leakage_shortcut_audit_table": _display_path(audit_path, root),
            "split_review_manifest": _display_path(split_manifest_out, root),
            "gate_json": _display_path(gate_out, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "target_rows": int(len(target_df)),
            "profile_rows": len(profile_rows),
            "split_reviews": len(split_rows),
            "viable_split_reviews": gate["viable_split_reviews"],
            "dependency_rows": len(dependency_rows),
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
    parser.add_argument("--phase117-dir", type=Path, default=DEFAULT_PHASE117_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    phase117_dir = args.phase117_dir if args.phase117_dir.is_absolute() else root / args.phase117_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = build_package(root=root, phase117_dir=phase117_dir, output_dir=output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
