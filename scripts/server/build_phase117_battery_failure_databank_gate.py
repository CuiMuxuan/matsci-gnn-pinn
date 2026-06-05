#!/usr/bin/env python3
"""Build a Phase 117 baseline-first gate for the Battery Failure Databank.

This phase is a fresh external-data intake and strong-baseline review. It
downloads a small public spreadsheet if needed, builds a structured table, and
checks whether any target has validation-visible signal that survives test.
It does not train a neural model or open A100 training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SOURCE_URL = (
    "https://www.nlr.gov/media/docs/libraries/tsdc/"
    "battery-failure-databank-revision2-feb24.xlsx?sfvrsn=7b7f89_1"
)
DEFAULT_RAW_PATH = Path(
    "data/raw/external/battery_failure_databank/"
    "battery-failure-databank-revision2-feb24.xlsx"
)
SOURCE_SHEET = "Battery Failure Databank"
EXPECTED_MIN_BYTES = 100_000
EXPECTED_HEAD_BYTES = 175_052
MIN_ROWS_FOR_REVIEW = 150
MIN_SPLIT_ROWS = 20
MIN_RELATIVE_VAL_GAIN = 0.05

TARGET_COLUMNS = (
    "Corrected-Total-Energy-Yield-kJ",
    "Baseline-Plus-Heat-Loss-Total-Energy-Yield-kJ",
    "Baseline-Total-Energy-Yield-kJ",
    "Energy-Percent-Positive-Ejecta-%",
    "Energy-Percent-Negative-Ejecta-%",
    "Post-Test-Mass-Unrecovered-g",
)
NUMERIC_FEATURES = (
    "Cell-Capacity-Ah",
    "Cell-Nominal-Voltage-V",
    "Cell-Energy-Wh",
    "Pre-Test-Cell-Open-Circuit-Voltage-V",
    "Pre-Test-Cell-Mass-g",
    "Heater-Power-W",
    "Heater-Time-On-s",
    "Energy-Applied-to-Trigger-kJ",
    "Avg-Cell-Temp-At-Trigger-degC",
)
CATEGORICAL_FEATURES = (
    "Cell-Format",
    "Trigger-Mechanism",
    "Pressure-Assisted-Seal-Configuration-Positive",
    "Pressure-Assisted-Seal-Configuration-Negative",
    "S-FTRC-Generation",
)
PROFILE_COLUMNS = {
    "cell_pretest": {
        "numeric": (
            "Cell-Capacity-Ah",
            "Cell-Nominal-Voltage-V",
            "Cell-Energy-Wh",
            "Pre-Test-Cell-Open-Circuit-Voltage-V",
            "Pre-Test-Cell-Mass-g",
        ),
        "categorical": ("Cell-Format", "S-FTRC-Generation"),
    },
    "trigger_numeric": {
        "numeric": (
            "Cell-Capacity-Ah",
            "Cell-Energy-Wh",
            "Heater-Power-W",
            "Heater-Time-On-s",
            "Energy-Applied-to-Trigger-kJ",
            "Avg-Cell-Temp-At-Trigger-degC",
        ),
        "categorical": ("Trigger-Mechanism",),
    },
    "cell_trigger_safe": {
        "numeric": NUMERIC_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
    },
    "series_shortcut": {
        "numeric": (),
        "categorical": ("Test-Series",),
    },
}
MODEL_METHODS = ("knn", "extra_trees", "hist_gradient_boosting")

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
    "phase117_candidate",
    "status",
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


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...] | list[str]) -> None:
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
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
                )
            },
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


def _role_for_column(column: str) -> str:
    if column in TARGET_COLUMNS:
        return "candidate_target"
    if column in NUMERIC_FEATURES or column in CATEGORICAL_FEATURES:
        return "safe_input_feature"
    if column in {"Cell-Description", "Test-Series", "Test-ID"}:
        return "split_or_shortcut_audit"
    if column.startswith("Post-Test"):
        return "post_test_not_input"
    if column.startswith("Energy-Fraction") or column.startswith("Energy-Percent"):
        return "derived_target_family"
    return "metadata_or_unused"


def build_schema_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in df.columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        non_missing = int(df[column].notna().sum())
        numeric_count = int(numeric.notna().sum())
        rows.append(
            {
                "column": str(column),
                "non_missing": non_missing,
                "numeric_non_missing": numeric_count,
                "unique_values": int(df[column].astype(str).nunique(dropna=True)),
                "role": _role_for_column(str(column)),
                "min": float(numeric.min()) if numeric_count else None,
                "max": float(numeric.max()) if numeric_count else None,
                "std": float(numeric.std()) if numeric_count > 1 else None,
            }
        )
    return rows


def load_source_table(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=SOURCE_SHEET)
    if len(df) < MIN_ROWS_FOR_REVIEW:
        raise ValueError(f"Expected at least {MIN_ROWS_FOR_REVIEW} rows, found {len(df)}")
    required = set(TARGET_COLUMNS) | set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES) | {
        "Cell-Description",
        "Test-Series",
    }
    missing = [column for column in sorted(required) if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required Battery Failure columns: {missing}")
    return df


def build_field_table(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Cell-Description",
        "Test-ID",
        "Test-Series",
        *CATEGORICAL_FEATURES,
        *NUMERIC_FEATURES,
        *TARGET_COLUMNS,
    ]
    present = [column for column in columns if column in df.columns]
    table = df[present].copy()
    for column in NUMERIC_FEATURES + TARGET_COLUMNS:
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce")
    for column in CATEGORICAL_FEATURES + ("Cell-Description", "Test-ID", "Test-Series"):
        if column in table.columns:
            table[column] = table[column].fillna("missing").astype(str)
    table.insert(0, "phase117_row_id", [f"BFDB-{index:04d}" for index in range(len(table))])
    return table


def _group_key(value: Any) -> str:
    text = str(value).strip()
    return text if text and text.lower() != "nan" else "missing_group"


def split_by_group(df: pd.DataFrame, group_column: str = "Cell-Description") -> dict[str, Any]:
    groups = sorted({_group_key(value) for value in df[group_column]})
    ranked = sorted(
        groups,
        key=lambda item: hashlib.sha256(f"phase117::{item}".encode("utf-8")).hexdigest(),
    )
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
        "n_groups": n_groups,
        "splits": splits,
        "group_splits": group_splits,
        "leakage_safe": sum(len(groups) for groups in group_splits.values()) == n_groups,
    }


def _target_subset(df: pd.DataFrame, target: str, split_manifest: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    mask = pd.to_numeric(df[target], errors="coerce").notna()
    subset = df.loc[mask].copy().reset_index(drop=False).rename(columns={"index": "source_index"})
    old_to_new = {int(old): int(new) for new, old in enumerate(subset["source_index"].tolist())}
    splits = {
        split: [old_to_new[index] for index in indices if index in old_to_new]
        for split, indices in split_manifest["splits"].items()
    }
    return subset, splits


def _one_hot_frame(train: pd.DataFrame, all_rows: pd.DataFrame, profile: dict[str, tuple[str, ...]]) -> tuple[np.ndarray, np.ndarray]:
    frames_train: list[pd.DataFrame] = []
    frames_all: list[pd.DataFrame] = []
    numeric = [column for column in profile["numeric"] if column in all_rows.columns]
    categorical = [column for column in profile["categorical"] if column in all_rows.columns]
    if numeric:
        train_numeric = train[numeric].apply(pd.to_numeric, errors="coerce")
        all_numeric = all_rows[numeric].apply(pd.to_numeric, errors="coerce")
        medians = train_numeric.median(numeric_only=True).fillna(0.0)
        frames_train.append(train_numeric.fillna(medians))
        frames_all.append(all_numeric.fillna(medians))
    if categorical:
        train_cat = train[categorical].fillna("missing").astype(str)
        all_cat = all_rows[categorical].fillna("missing").astype(str)
        combined = pd.get_dummies(pd.concat([train_cat, all_cat], axis=0), dummy_na=False)
        frames_train.append(combined.iloc[: len(train_cat)].reset_index(drop=True))
        frames_all.append(combined.iloc[len(train_cat) :].reset_index(drop=True))
    if not frames_train:
        raise ValueError("Profile has no usable feature columns")
    x_train = pd.concat([frame.reset_index(drop=True) for frame in frames_train], axis=1).to_numpy(dtype=float)
    x_all = pd.concat([frame.reset_index(drop=True) for frame in frames_all], axis=1).to_numpy(dtype=float)
    return x_train, x_all


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
            KNeighborsRegressor(n_neighbors=max(1, min(8, len(y_train)))),
        )
    elif method == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        model = ExtraTreesRegressor(n_estimators=200, random_state=117, n_jobs=-1)
    elif method == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(
            max_iter=200,
            random_state=117,
            early_stopping=False,
        )
    else:
        raise ValueError(f"Unsupported method: {method}")
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_all), dtype=float)


def evaluate_target(
    df: pd.DataFrame,
    *,
    target: str,
    split_manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_df, splits = _target_subset(df, target, split_manifest)
    metric_rows: list[dict[str, Any]] = []
    y = pd.to_numeric(target_df[target], errors="coerce").to_numpy(dtype=float)
    train_idx = splits["train"]
    train_std = float(np.std(y[train_idx])) if train_idx else 0.0
    if any(len(splits[split]) < MIN_SPLIT_ROWS for split in ("train", "val", "test")):
        return metric_rows, {
            "target": target,
            "row_count": int(len(target_df)),
            "train_rows": len(splits["train"]),
            "val_rows": len(splits["val"]),
            "test_rows": len(splits["test"]),
            "status": "blocked_insufficient_split_rows",
            "reason": "one or more splits is below the minimum row count",
            "phase117_candidate": False,
        }
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
        x_train, x_all = _one_hot_frame(target_df.iloc[train_idx], target_df, profile)
        y_train = y[train_idx]
        for method in MODEL_METHODS:
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
    non_shortcut = [
        row
        for row in metric_rows
        if row["split"] == "val" and row["profile"] not in {"mean", "series_shortcut"}
    ]
    best_val = min(non_shortcut, key=lambda row: row["rmse"])
    best_test = next(
        row
        for row in metric_rows
        if row["target"] == target
        and row["profile"] == best_val["profile"]
        and row["method"] == best_val["method"]
        and row["split"] == "test"
    )
    shortcut_val_candidates = [
        row for row in metric_rows if row["split"] == "val" and row["profile"] == "series_shortcut"
    ]
    shortcut_val = min(shortcut_val_candidates, key=lambda row: row["rmse"])
    shortcut_test = next(
        row
        for row in metric_rows
        if row["target"] == target
        and row["profile"] == shortcut_val["profile"]
        and row["method"] == shortcut_val["method"]
        and row["split"] == "test"
    )
    val_gain = float(mean_val["rmse"]) - float(best_val["rmse"])
    test_gain = float(mean_test["rmse"]) - float(best_test["rmse"])
    min_gain = float(mean_val["rmse"]) * MIN_RELATIVE_VAL_GAIN
    shortcut_blocks = (
        float(shortcut_val["rmse"]) <= float(best_val["rmse"]) * 1.02
        and float(shortcut_test["rmse"]) <= float(best_test["rmse"]) * 1.02
    )
    candidate = val_gain > min_gain and test_gain > 0.0 and not shortcut_blocks
    if candidate:
        status = "candidate_gap_ready_focused_review"
        reason = "validation-selected safe profile beats mean and preserves test gain"
    elif shortcut_blocks:
        status = "blocked_series_shortcut"
        reason = "test-series shortcut matches or beats the safe profile"
    elif val_gain <= min_gain:
        status = "blocked_no_validation_gain"
        reason = "best safe profile does not clear the validation gain threshold"
    else:
        status = "blocked_validation_test_reversal"
        reason = "validation gain does not preserve test gain"
    review = {
        "target": target,
        "row_count": int(len(target_df)),
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
        "phase117_candidate": candidate,
        "status": status,
        "reason": reason,
    }
    return metric_rows, review


def build_gate(review_rows: list[dict[str, Any]], *, source_info: dict[str, Any], split_manifest: dict[str, Any]) -> dict[str, Any]:
    candidate_rows = [row for row in review_rows if bool(row.get("phase117_candidate"))]
    if candidate_rows:
        selected = min(candidate_rows, key=lambda row: float(row["best_val_rmse"]))
        status = "phase117_battery_failure_databank_gap_ready_focused_review"
        next_action = "enter focused leakage/target review before any model mechanism"
    elif any(row.get("status") == "blocked_insufficient_split_rows" for row in review_rows):
        selected = None
        status = "phase117_battery_failure_databank_incomplete_insufficient_split_rows"
        next_action = "repair split or target availability before model design"
    else:
        selected = None
        status = "phase117_battery_failure_databank_closed_no_stable_guarded_gap"
        next_action = "close as external-data diagnostic or choose a different public registered target"
    return {
        "status": status,
        "source_name": "NLR/NREL Battery Failure Databank Revision 2",
        "source_url": source_info["source_url"],
        "source_byte_size": source_info["byte_size"],
        "source_sha256": source_info["sha256"],
        "row_count": sum(len(indices) for indices in split_manifest["splits"].values()),
        "group_count": split_manifest["n_groups"],
        "split_counts": {split: len(indices) for split, indices in split_manifest["splits"].items()},
        "leakage_safe_group_split": bool(split_manifest["leakage_safe"]),
        "reviewed_targets": len(review_rows),
        "candidate_targets": [row["target"] for row in candidate_rows],
        "selected_target": selected.get("target") if selected else None,
        "selected_profile": selected.get("best_profile") if selected else None,
        "selected_method": selected.get("best_method") if selected else None,
        "selected_validation_rmse": selected.get("best_val_rmse") if selected else None,
        "selected_test_rmse": selected.get("best_test_rmse") if selected else None,
        "phase117_focused_review_allowed": bool(candidate_rows),
        "phase117_model_mechanism_allowed": False,
        "phase117_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": next_action,
    }


def build_data_card(source_info: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_name": "NLR/NREL Battery Failure Databank Revision 2",
        "source_url": source_info["source_url"],
        "local_raw_path": source_info["path"],
        "byte_size": source_info["byte_size"],
        "sha256": source_info["sha256"],
        "public_reproducibility": "public_spreadsheet_download",
        "registration_story": "single-row abuse-test metadata and outcome targets",
        "target_family": "battery thermal-abuse energy and ejecta outcomes",
        "selection_rule": "validation-only target/profile selection against mean and strong tabular baselines",
        "training_policy": "no neural model training in Phase 117",
        "gate_status": gate["status"],
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_csv_value(row.get(key)).replace("\n", " ") for key, _ in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def build_markdown(gate: dict[str, Any], review_rows: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "# Phase 117 Battery Failure Databank Gate",
            "",
            "## Gate Decision",
            "",
            f"Status: `{gate['status']}`.",
            f"Focused review allowed: `{str(gate['phase117_focused_review_allowed']).lower()}`.",
            f"Model training allowed: `{str(gate['phase117_model_training_allowed']).lower()}`.",
            f"A100 training allowed now: `{str(gate['a100_training_allowed_now']).lower()}`.",
            f"A100-SXM4-80GB request now: `{str(gate['a100_80gb_request_now']).lower()}`.",
            "",
            "Phase 117 is a no-training external-data intake and strong-baseline review.",
            "",
            "## Target Review",
            "",
            _markdown_table(
                review_rows,
                [
                    ("target", "Target"),
                    ("best_profile", "Best profile"),
                    ("best_method", "Best method"),
                    ("val_gain_vs_mean", "Val gain"),
                    ("test_gain_vs_mean", "Test gain"),
                    ("status", "Status"),
                ],
            ),
            "",
            "## Next Action",
            "",
            gate["next_action"],
            "",
        ]
    )


def build_package(
    *,
    root: Path,
    output_dir: Path,
    raw_xlsx: Path,
    source_url: str = SOURCE_URL,
    force_download: bool = False,
) -> dict[str, Any]:
    source_info = ensure_source_file(raw_xlsx, source_url=source_url, force_download=force_download)
    df = load_source_table(raw_xlsx)
    field_table = build_field_table(df)
    split_manifest = split_by_group(field_table)
    schema_rows = build_schema_rows(df)
    metric_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    for target in TARGET_COLUMNS:
        target_metrics, review = evaluate_target(field_table, target=target, split_manifest=split_manifest)
        metric_rows.extend(target_metrics)
        review_rows.append(review)
    gate = build_gate(review_rows, source_info=source_info, split_manifest=split_manifest)
    data_card = build_data_card(source_info, gate)

    output_dir.mkdir(parents=True, exist_ok=True)
    field_path = output_dir / "phase117_battery_failure_databank_field_table.csv"
    split_path = output_dir / "phase117_battery_failure_databank_split_manifest.json"
    schema_path = output_dir / "phase117_battery_failure_databank_schema_table.csv"
    metric_path = output_dir / "phase117_battery_failure_databank_metric_table.csv"
    review_path = output_dir / "phase117_battery_failure_databank_target_review_table.csv"
    gate_path = output_dir / "phase117_battery_failure_databank_gate.json"
    card_path = output_dir / "phase117_battery_failure_databank_data_card.json"
    markdown_path = output_dir / "phase117_battery_failure_databank.md"
    manifest_path = output_dir / "phase117_battery_failure_databank_manifest.json"

    _write_csv(field_path, field_table.to_dict("records"), list(field_table.columns))
    _write_json(split_path, split_manifest)
    _write_csv(schema_path, schema_rows, SCHEMA_FIELDS)
    _write_csv(metric_path, metric_rows, METRIC_FIELDS)
    _write_csv(review_path, review_rows, REVIEW_FIELDS)
    _write_json(gate_path, gate)
    _write_json(card_path, data_card)
    markdown_path.write_text(build_markdown(gate, review_rows), encoding="utf-8")

    manifest = {
        "phase": 117,
        "objective": "battery_failure_databank_external_baseline_first_gate",
        "inputs": {
            "source_url": source_url,
            "raw_xlsx": _display_path(raw_xlsx, root),
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
            "source_rows": int(len(df)),
            "field_rows": int(len(field_table)),
            "schema_rows": len(schema_rows),
            "metric_rows": len(metric_rows),
            "review_rows": len(review_rows),
            "candidate_targets": len(gate["candidate_targets"]),
        },
        "gate": gate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase117_battery_failure_databank_gate"),
    )
    parser.add_argument("--raw-xlsx", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--source-url", default=SOURCE_URL)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    raw_xlsx = args.raw_xlsx if args.raw_xlsx.is_absolute() else root / args.raw_xlsx
    manifest = build_package(
        root=root,
        output_dir=output_dir,
        raw_xlsx=raw_xlsx,
        source_url=args.source_url,
        force_download=args.force_download,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
