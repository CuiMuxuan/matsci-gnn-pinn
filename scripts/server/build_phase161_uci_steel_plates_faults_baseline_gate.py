#!/usr/bin/env python3
"""Build Phase 161 baseline-first gate for UCI Steel Plates Faults.

This phase opens a fresh small manufacturing-defect source for the second-paper
candidate queue. It reviews leakage-aware steel/geometry grouped splits with
strong non-neural classification baselines only; it does not train neural
models or open A100/A800 training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SOURCE_URL = "https://cdn.uci-ics-mlr-prod.aws.uci.edu/198/steel%2Bplates%2Bfaults.zip"
SOURCE_DOI = "10.24432/C5J88N"
DEFAULT_RAW_PATH = Path(
    "data/raw/external/phase161_uci_steel_plates_faults/steel_plates_faults.zip"
)
DEFAULT_OUTPUT_DIR = Path("docs/results/phase161_uci_steel_plates_faults_baseline_gate")

EXPECTED_MIN_BYTES = 90_000
MIN_ROWS_FOR_REVIEW = 1_500
MIN_SPLIT_ROWS = 150
MIN_CLASS_COUNT_PER_SPLIT = 7
MIN_BALANCED_ACCURACY_GAIN = 0.20
MIN_MACRO_F1_GAIN = 0.15
MIN_TEST_BALANCED_ACCURACY_GAIN = 0.10
SHORTCUT_DOMINANCE_TOLERANCE = 0.98
MODEL_METHODS = ("logistic_regression", "knn", "extra_trees", "hist_gradient_boosting")

LABEL_COLUMNS = (
    "Pastry",
    "Z_Scratch",
    "K_Scatch",
    "Stains",
    "Dirtiness",
    "Bumps",
    "Other_Faults",
)

OVERVIEW_FIELDS = (
    "source_id",
    "source_url",
    "source_doi",
    "raw_path",
    "raw_bytes",
    "raw_sha256",
    "field_rows",
    "feature_columns",
    "target",
    "class_count",
    "class_counts_json",
    "group_column",
    "group_count",
    "train_rows_split",
    "val_rows_split",
    "test_rows_split",
)

METRIC_FIELDS = (
    "profile",
    "method",
    "role",
    "split",
    "balanced_accuracy",
    "macro_f1",
    "accuracy",
    "n_rows",
)

REVIEW_FIELDS = (
    "target",
    "selected_profile",
    "selected_method",
    "selected_validation_balanced_accuracy",
    "selected_test_balanced_accuracy",
    "selected_validation_macro_f1",
    "selected_test_macro_f1",
    "majority_validation_balanced_accuracy",
    "majority_test_balanced_accuracy",
    "majority_validation_macro_f1",
    "majority_test_macro_f1",
    "best_shortcut_profile",
    "best_shortcut_method",
    "best_shortcut_validation_balanced_accuracy",
    "best_shortcut_test_balanced_accuracy",
    "validation_balanced_accuracy_gain_over_majority",
    "test_balanced_accuracy_gain_over_majority",
    "validation_macro_f1_gain_over_majority",
    "test_macro_f1_gain_over_majority",
    "baseline_visible_gap",
    "shortcut_dominant",
    "phase162_focused_review_allowed",
    "status",
    "blocker",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())


def ensure_source(path: Path, *, source_url: str, allow_download: bool) -> dict[str, Any]:
    if not path.exists():
        if not allow_download:
            raise FileNotFoundError(f"Missing raw source and download disabled: {path}")
        _download(source_url, path)
    size = path.stat().st_size
    if size < EXPECTED_MIN_BYTES:
        raise ValueError(f"Raw source is unexpectedly small: {size} bytes")
    with zipfile.ZipFile(path) as archive:
        members = set(archive.namelist())
        required = {"Faults.NNA", "Faults27x7_var"}
        missing = required - members
        if missing:
            raise ValueError(f"Missing required UCI members: {sorted(missing)}")
        table_size = archive.getinfo("Faults.NNA").file_size
        var_size = archive.getinfo("Faults27x7_var").file_size
    return {
        "raw_path": str(path),
        "raw_bytes": size,
        "raw_sha256": _sha256(path),
        "zip_members": sorted(members),
        "faults_nna_bytes": table_size,
        "faults_var_bytes": var_size,
    }


def _stable_unit_hash(text: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    safe = denominator.replace(0.0, np.nan)
    return (numerator / safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _quantile_bin(values: pd.Series, *, bins: int) -> pd.Series:
    ranks = values.rank(method="first")
    try:
        out = pd.qcut(ranks, q=bins, labels=False, duplicates="drop")
    except ValueError:
        out = pd.Series(np.zeros(len(values), dtype=int), index=values.index)
    return out.fillna(0).astype(int).astype(str)


def load_steel_plates_table(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        variable_names = archive.read("Faults27x7_var").decode("utf-8").splitlines()
        with archive.open("Faults.NNA") as handle:
            frame = pd.read_csv(handle, sep=r"\s+", header=None)
    if len(variable_names) != frame.shape[1]:
        raise ValueError(
            f"Variable count {len(variable_names)} does not match table columns {frame.shape[1]}"
        )
    if tuple(variable_names[-7:]) != LABEL_COLUMNS:
        raise ValueError(f"Unexpected label columns: {variable_names[-7:]}")
    out = frame.copy()
    out.columns = variable_names
    for column in variable_names:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna().reset_index(drop=True)
    label_matrix = out[list(LABEL_COLUMNS)].to_numpy(dtype=int)
    if not np.all(label_matrix.sum(axis=1) == 1):
        raise ValueError("Each steel-plate row must have exactly one one-hot fault label")
    out.insert(0, "phase161_row_id", np.arange(len(out), dtype=int))
    out["target_fault_class"] = [LABEL_COLUMNS[int(index)] for index in np.argmax(label_matrix, axis=1)]

    width = out["X_Maximum"] - out["X_Minimum"]
    height = out["Y_Maximum"] - out["Y_Minimum"]
    bbox_area = width * height
    out["defect_width"] = width
    out["defect_height"] = height
    out["bbox_area"] = bbox_area
    out["area_density"] = _safe_divide(out["Pixels_Areas"], bbox_area)
    out["luminosity_mean"] = _safe_divide(out["Sum_of_Luminosity"], out["Pixels_Areas"])
    out["luminosity_range"] = out["Maximum_of_Luminosity"] - out["Minimum_of_Luminosity"]
    out["perimeter_sum"] = out["X_Perimeter"] + out["Y_Perimeter"]
    out["x_center"] = (out["X_Minimum"] + out["X_Maximum"]) / 2.0
    out["y_center"] = (out["Y_Minimum"] + out["Y_Maximum"]) / 2.0
    out["x_center_fraction"] = _safe_divide(out["x_center"], out["Length_of_Conveyer"])
    out["row_order_fraction"] = np.arange(len(out), dtype=float) / max(len(out) - 1, 1)

    area_bin = _quantile_bin(out["LogOfAreas"], bins=8)
    x_bin = _quantile_bin(out["X_Minimum"], bins=8)
    width_bin = _quantile_bin(out["defect_width"], bins=8)
    height_bin = _quantile_bin(out["defect_height"], bins=8)
    out["steel_geometry_context_key"] = [
        (
            f"S{int(row.TypeOfSteel_A300)}{int(row.TypeOfSteel_A400)}"
            f"|T{float(row.Steel_Plate_Thickness):.3g}"
            f"|L{float(row.Length_of_Conveyer):.3g}"
            f"|O{float(row.Outside_Global_Index):.3g}"
            f"|A{area_bin.iloc[index]}|X{x_bin.iloc[index]}"
        )
        for index, row in out.iterrows()
    ]
    out["steel_condition_key"] = [
        (
            f"S{int(row.TypeOfSteel_A300)}{int(row.TypeOfSteel_A400)}"
            f"|T{float(row.Steel_Plate_Thickness):.3g}"
            f"|L{float(row.Length_of_Conveyer):.3g}"
        )
        for _, row in out.iterrows()
    ]
    out["geometry_size_key"] = [
        f"A{area_bin.iloc[index]}|W{width_bin.iloc[index]}|H{height_bin.iloc[index]}|O{row.Outside_Global_Index:.3g}"
        for index, row in out.iterrows()
    ]
    return out


def split_by_group(df: pd.DataFrame, *, group_column: str = "steel_geometry_context_key") -> dict[str, Any]:
    groups = sorted(str(value) for value in df[group_column].dropna().unique())
    split_groups = {"train": set(), "val": set(), "test": set()}
    for group in groups:
        value = _stable_unit_hash(group, "phase161_split")
        if value < 0.60:
            split_groups["train"].add(group)
        elif value < 0.80:
            split_groups["val"].add(group)
        else:
            split_groups["test"].add(group)
    assignments = []
    for group in df[group_column].astype(str):
        if group in split_groups["train"]:
            assignments.append("train")
        elif group in split_groups["val"]:
            assignments.append("val")
        else:
            assignments.append("test")
    counts = {split: assignments.count(split) for split in ("train", "val", "test")}
    class_counts = {
        split: int(df.loc[[idx for idx, name in enumerate(assignments) if name == split], "target_fault_class"].nunique())
        for split in ("train", "val", "test")
    }
    if any(count < MIN_SPLIT_ROWS for count in counts.values()):
        raise ValueError(f"Split too small for Phase 161 review: {counts}")
    if any(count < MIN_CLASS_COUNT_PER_SPLIT for count in class_counts.values()):
        raise ValueError(f"One or more splits miss fault classes: {class_counts}")
    return {
        "group_column": group_column,
        "group_count": len(groups),
        "split_groups": {key: sorted(values) for key, values in split_groups.items()},
        "assignments": assignments,
        "counts": counts,
        "class_counts": class_counts,
    }


def profile_columns(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    raw_feature_columns = [
        column
        for column in df.columns
        if column
        not in {
            "phase161_row_id",
            "target_fault_class",
            "steel_geometry_context_key",
            "steel_condition_key",
            "geometry_size_key",
            *LABEL_COLUMNS,
        }
        and pd.api.types.is_numeric_dtype(df[column])
    ]
    return {
        "geometry_luminosity_full": {
            "role": "admissible",
            "columns": tuple(raw_feature_columns),
        },
        "shape_luminosity_core": {
            "role": "admissible",
            "columns": (
                "Pixels_Areas",
                "X_Perimeter",
                "Y_Perimeter",
                "Sum_of_Luminosity",
                "Minimum_of_Luminosity",
                "Maximum_of_Luminosity",
                "Edges_Index",
                "Empty_Index",
                "Square_Index",
                "Outside_X_Index",
                "Edges_X_Index",
                "Edges_Y_Index",
                "Outside_Global_Index",
                "LogOfAreas",
                "Log_X_Index",
                "Log_Y_Index",
                "Orientation_Index",
                "Luminosity_Index",
                "SigmoidOfAreas",
                "defect_width",
                "defect_height",
                "area_density",
                "luminosity_mean",
                "luminosity_range",
            ),
        },
        "steel_context_control": {
            "role": "shortcut_control",
            "columns": (
                "TypeOfSteel_A300",
                "TypeOfSteel_A400",
                "Steel_Plate_Thickness",
                "Length_of_Conveyer",
            ),
        },
        "location_only_control": {
            "role": "shortcut_control",
            "columns": (
                "X_Minimum",
                "X_Maximum",
                "Y_Minimum",
                "Y_Maximum",
                "x_center",
                "y_center",
                "x_center_fraction",
                "Length_of_Conveyer",
            ),
        },
        "size_only_control": {
            "role": "shortcut_control",
            "columns": (
                "Pixels_Areas",
                "X_Perimeter",
                "Y_Perimeter",
                "LogOfAreas",
                "Log_X_Index",
                "Log_Y_Index",
                "defect_width",
                "defect_height",
                "bbox_area",
                "area_density",
            ),
        },
        "row_order_control": {
            "role": "shortcut_control",
            "columns": ("row_order_fraction",),
        },
    }


def _numeric_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    return [column for column in columns if column in df.columns]


def _model(method: str):
    if method == "logistic_regression":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
    if method == "knn":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=7, weights="distance", algorithm="brute"),
        )
    if method == "extra_trees":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=256,
                min_samples_leaf=2,
                random_state=161,
                n_jobs=1,
                class_weight="balanced",
            ),
        )
    if method == "hist_gradient_boosting":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=160,
                learning_rate=0.06,
                l2_regularization=0.01,
                random_state=161,
            ),
        )
    raise ValueError(f"Unknown method: {method}")


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def evaluate_baselines(df: pd.DataFrame, assignments: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    y = df["target_fault_class"].to_numpy(dtype=str)
    split_indices = {
        split: np.array([idx for idx, name in enumerate(assignments) if name == split], dtype=int)
        for split in ("train", "val", "test")
    }
    train_labels = pd.Series(y[split_indices["train"]])
    majority_label = str(train_labels.value_counts().idxmax())
    metric_rows: list[dict[str, Any]] = []
    for split, indices in split_indices.items():
        metric_rows.append(
            {
                "profile": "train_majority",
                "method": "majority",
                "role": "majority_baseline",
                "split": split,
                "n_rows": int(len(indices)),
                **_classification_metrics(y[indices], np.full(len(indices), majority_label, dtype=object)),
            }
        )

    for profile_name, spec in profile_columns(df).items():
        columns = _numeric_columns(df, tuple(spec["columns"]))
        if not columns:
            continue
        x = df[columns].to_numpy(dtype=float)
        for method in MODEL_METHODS:
            if spec["role"] == "shortcut_control" and method == "hist_gradient_boosting":
                continue
            model = _model(method)
            model.fit(x[split_indices["train"]], y[split_indices["train"]])
            for split, indices in split_indices.items():
                pred = model.predict(x[indices])
                metric_rows.append(
                    {
                        "profile": profile_name,
                        "method": method,
                        "role": spec["role"],
                        "split": split,
                        "n_rows": int(len(indices)),
                        **_classification_metrics(y[indices], pred),
                    }
                )
    return metric_rows, build_review_rows(metric_rows)


def _metric_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {(row["profile"], row["method"], row["split"]): row for row in rows}


def build_review_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = _metric_lookup(metric_rows)
    majority_val = lookup[("train_majority", "majority", "val")]
    majority_test = lookup[("train_majority", "majority", "test")]
    admissible_val = [
        row
        for row in metric_rows
        if row["split"] == "val" and row["role"] == "admissible"
    ]
    shortcut_val = [
        row
        for row in metric_rows
        if row["split"] == "val" and row["role"] == "shortcut_control"
    ]
    selected = max(admissible_val, key=lambda row: (row["balanced_accuracy"], row["macro_f1"]))
    selected_test = lookup[(selected["profile"], selected["method"], "test")]
    best_shortcut = max(shortcut_val, key=lambda row: (row["balanced_accuracy"], row["macro_f1"]))
    best_shortcut_test = lookup[(best_shortcut["profile"], best_shortcut["method"], "test")]
    val_bal_gain = selected["balanced_accuracy"] - majority_val["balanced_accuracy"]
    test_bal_gain = selected_test["balanced_accuracy"] - majority_test["balanced_accuracy"]
    val_f1_gain = selected["macro_f1"] - majority_val["macro_f1"]
    test_f1_gain = selected_test["macro_f1"] - majority_test["macro_f1"]
    baseline_visible_gap = (
        val_bal_gain >= MIN_BALANCED_ACCURACY_GAIN
        and test_bal_gain >= MIN_TEST_BALANCED_ACCURACY_GAIN
        and val_f1_gain >= MIN_MACRO_F1_GAIN
    )
    shortcut_dominant = (
        best_shortcut["balanced_accuracy"]
        >= selected["balanced_accuracy"] * SHORTCUT_DOMINANCE_TOLERANCE
        or best_shortcut_test["balanced_accuracy"]
        >= selected_test["balanced_accuracy"] * SHORTCUT_DOMINANCE_TOLERANCE
    )
    focused_allowed = bool(baseline_visible_gap and not shortcut_dominant)
    blocker = ""
    if not baseline_visible_gap:
        blocker = "strong admissible baseline does not beat majority by required balanced-accuracy/macro-F1 margins"
    elif shortcut_dominant:
        blocker = "shortcut control is too close to or better than selected admissible profile"
    status = (
        "phase161_uci_steel_plates_faults_ready_focused_review"
        if focused_allowed
        else "phase161_uci_steel_plates_faults_closed_no_stable_guarded_gap"
    )
    return [
        {
            "target": "target_fault_class",
            "selected_profile": selected["profile"],
            "selected_method": selected["method"],
            "selected_validation_balanced_accuracy": selected["balanced_accuracy"],
            "selected_test_balanced_accuracy": selected_test["balanced_accuracy"],
            "selected_validation_macro_f1": selected["macro_f1"],
            "selected_test_macro_f1": selected_test["macro_f1"],
            "majority_validation_balanced_accuracy": majority_val["balanced_accuracy"],
            "majority_test_balanced_accuracy": majority_test["balanced_accuracy"],
            "majority_validation_macro_f1": majority_val["macro_f1"],
            "majority_test_macro_f1": majority_test["macro_f1"],
            "best_shortcut_profile": best_shortcut["profile"],
            "best_shortcut_method": best_shortcut["method"],
            "best_shortcut_validation_balanced_accuracy": best_shortcut["balanced_accuracy"],
            "best_shortcut_test_balanced_accuracy": best_shortcut_test["balanced_accuracy"],
            "validation_balanced_accuracy_gain_over_majority": val_bal_gain,
            "test_balanced_accuracy_gain_over_majority": test_bal_gain,
            "validation_macro_f1_gain_over_majority": val_f1_gain,
            "test_macro_f1_gain_over_majority": test_f1_gain,
            "baseline_visible_gap": baseline_visible_gap,
            "shortcut_dominant": shortcut_dominant,
            "phase162_focused_review_allowed": focused_allowed,
            "status": status,
            "blocker": blocker,
        }
    ]


def _csv_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _display_path(path: Path, root: Path | None = None) -> str:
    if root is None:
        return str(path).replace("\\", "/")
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _markdown_table(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[str]:
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    body = [
        "| " + " | ".join(_csv_value(row.get(field, "")) for field in fields)
        + " |"
        for row in rows
    ]
    return [header, sep, *body]


def build_gate(*, overview: dict[str, Any], review_rows: list[dict[str, Any]]) -> dict[str, Any]:
    review = review_rows[0]
    focused = bool(review["phase162_focused_review_allowed"])
    return {
        "status": review["status"],
        "source": "UCI Steel Plates Faults",
        "source_doi": SOURCE_DOI,
        "raw_sha256": overview["raw_sha256"],
        "field_rows": overview["field_rows"],
        "group_count": overview["group_count"],
        "class_count": overview["class_count"],
        "selected_target": "target_fault_class",
        "selected_profile": review["selected_profile"],
        "selected_method": review["selected_method"],
        "selected_validation_balanced_accuracy": review["selected_validation_balanced_accuracy"],
        "selected_test_balanced_accuracy": review["selected_test_balanced_accuracy"],
        "selected_validation_macro_f1": review["selected_validation_macro_f1"],
        "selected_test_macro_f1": review["selected_test_macro_f1"],
        "majority_validation_balanced_accuracy": review["majority_validation_balanced_accuracy"],
        "majority_test_balanced_accuracy": review["majority_test_balanced_accuracy"],
        "best_shortcut_profile": review["best_shortcut_profile"],
        "best_shortcut_validation_balanced_accuracy": review[
            "best_shortcut_validation_balanced_accuracy"
        ],
        "best_shortcut_test_balanced_accuracy": review["best_shortcut_test_balanced_accuracy"],
        "phase162_focused_review_allowed": focused,
        "phase161_model_mechanism_allowed": False,
        "phase161_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": (
            "run Phase 162 split/shortcut/class-balance focused review before any mechanism"
            if focused
            else "close this source as diagnostic or choose another baseline-first source"
        ),
    }


def build_markdown(
    *,
    gate: dict[str, Any],
    overview_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
) -> str:
    lines: list[str] = [
        "# Phase 161 UCI Steel Plates Faults Baseline Gate",
        "",
        "## Gate",
        f"- Status: `{gate['status']}`",
        f"- Phase 162 focused review allowed: `{_csv_value(gate['phase162_focused_review_allowed'])}`",
        f"- Model training allowed: `{_csv_value(gate['phase161_model_training_allowed'])}`",
        f"- A100 training allowed now: `{_csv_value(gate['a100_training_allowed_now'])}`",
        f"- A100-SXM4-80GB request now: `{_csv_value(gate['a100_80gb_request_now'])}`",
        "",
        "## Interpretation",
        (
            "This is a no-training baseline-first intake for a possible second-paper "
            "source. The split groups rows by coarse steel and defect-geometry context "
            "so closely related surface-defect contexts stay within a single split. "
            "A positive gate can only open a focused split/shortcut/class-balance review."
        ),
        "",
        "## Source Overview",
        *_markdown_table(overview_rows, OVERVIEW_FIELDS),
        "",
        "## Review",
        *_markdown_table(review_rows, REVIEW_FIELDS),
        "",
    ]
    return "\n".join(lines)


def build_package(
    *,
    root: Path,
    output_dir: Path,
    raw_path: Path,
    source_url: str,
    allow_download: bool,
) -> dict[str, Any]:
    root = root.resolve()
    output_dir = output_dir if output_dir.is_absolute() else root / output_dir
    raw_path = raw_path if raw_path.is_absolute() else root / raw_path
    source_info = ensure_source(raw_path, source_url=source_url, allow_download=allow_download)
    df = load_steel_plates_table(raw_path)
    if len(df) < MIN_ROWS_FOR_REVIEW:
        raise ValueError(f"Too few rows for Phase 161 review: {len(df)}")
    split = split_by_group(df)
    metric_rows, review_rows = evaluate_baselines(df, split["assignments"])
    # KNN can differ by tiny amounts across platforms when distance ties occur.
    # The gate JSON and review table remain authoritative; keep the tracked
    # metric table deterministic across local/A800 runs.
    tracked_metric_rows = [row for row in metric_rows if row.get("method") != "knn"]
    class_counts = {
        str(key): int(value)
        for key, value in df["target_fault_class"].value_counts().sort_index().items()
    }
    overview = {
        "source_id": "phase161_uci_steel_plates_faults",
        "source_url": source_url,
        "source_doi": SOURCE_DOI,
        "raw_path": _display_path(raw_path, root),
        "raw_bytes": source_info["raw_bytes"],
        "raw_sha256": source_info["raw_sha256"],
        "field_rows": int(len(df)),
        "feature_columns": int(
            sum(
                1
                for column in df.columns
                if column not in LABEL_COLUMNS
                and pd.api.types.is_numeric_dtype(df[column])
            )
        ),
        "target": "target_fault_class",
        "class_count": int(df["target_fault_class"].nunique()),
        "class_counts_json": class_counts,
        "group_column": split["group_column"],
        "group_count": int(split["group_count"]),
        "train_rows_split": int(split["counts"]["train"]),
        "val_rows_split": int(split["counts"]["val"]),
        "test_rows_split": int(split["counts"]["test"]),
    }
    gate = build_gate(overview=overview, review_rows=review_rows)

    overview_rows = [overview]
    overview_path = output_dir / "phase161_source_overview_table.csv"
    metric_path = output_dir / "phase161_baseline_metric_table.csv"
    review_path = output_dir / "phase161_baseline_review_table.csv"
    split_path = output_dir / "phase161_split_manifest.json"
    gate_path = output_dir / "phase161_uci_steel_plates_faults_baseline_gate.json"
    markdown_path = output_dir / "phase161_uci_steel_plates_faults_baseline_gate.md"
    manifest_path = output_dir / "phase161_uci_steel_plates_faults_baseline_manifest.json"

    _write_csv(overview_path, overview_rows, OVERVIEW_FIELDS)
    _write_csv(metric_path, tracked_metric_rows, METRIC_FIELDS)
    _write_csv(review_path, review_rows, REVIEW_FIELDS)
    _write_json(
        split_path,
        {
            "phase": 161,
            "source": "UCI Steel Plates Faults",
            "group_column": split["group_column"],
            "group_count": split["group_count"],
            "counts": split["counts"],
            "class_counts": split["class_counts"],
            "split_groups": split["split_groups"],
        },
    )
    _write_json(gate_path, gate)
    with markdown_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(build_markdown(gate=gate, overview_rows=overview_rows, review_rows=review_rows))
    source_info_for_manifest = {
        **source_info,
        "raw_path": _display_path(raw_path, root),
    }
    manifest = {
        "phase": 161,
        "description": "baseline-first intake for UCI steel plates faults",
        "source_info": source_info_for_manifest,
        "outputs": {
            "source_overview_table": _display_path(overview_path, root),
            "baseline_metric_table": _display_path(metric_path, root),
            "baseline_review_table": _display_path(review_path, root),
            "split_manifest": _display_path(split_path, root),
            "gate": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "field_rows": int(len(df)),
            "metric_rows": len(tracked_metric_rows),
            "review_rows": len(review_rows),
            "group_count": int(split["group_count"]),
            "class_count": int(df["target_fault_class"].nunique()),
        },
        "gate": gate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--source-url", default=SOURCE_URL)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_package(
        root=args.root,
        output_dir=args.output_dir,
        raw_path=args.raw_path,
        source_url=args.source_url,
        allow_download=args.allow_download,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
