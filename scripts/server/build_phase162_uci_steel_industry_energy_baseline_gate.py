#!/usr/bin/env python3
"""Build Phase 162 baseline-first gate for UCI steel-industry energy use.

This phase opens a fresh manufacturing-process source intake. It predicts
15-minute steel-plant electricity use from leakage-limited calendar/time
context while treating synchronous electrical quantities, CO2, load labels, and
row order as shortcut controls. It does not train neural models.
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SOURCE_URL = "https://archive.ics.uci.edu/static/public/851/steel+industry+energy+consumption.zip"
SOURCE_DOI = "10.24432/C52G8C"
DEFAULT_RAW_PATH = Path(
    "data/raw/external/phase162_uci_steel_industry_energy/steel_industry_energy_consumption.zip"
)
DEFAULT_OUTPUT_DIR = Path("docs/results/phase162_uci_steel_industry_energy_baseline_gate")

EXPECTED_MIN_BYTES = 450_000
MIN_ROWS_FOR_REVIEW = 30_000
MIN_SPLIT_ROWS = 4_000
MIN_RELATIVE_VAL_GAIN = 0.20
MIN_RELATIVE_TEST_GAIN = 0.10
SHORTCUT_DOMINANCE_TOLERANCE = 1.02
MODEL_METHODS = ("ridge", "extra_trees")

TARGET_COLUMN = "Usage_kWh"
RAW_NUMERIC_COLUMNS = (
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
)
RAW_CATEGORICAL_COLUMNS = ("WeekStatus", "Day_of_week", "Load_Type")

OVERVIEW_FIELDS = (
    "source_id",
    "source_url",
    "source_doi",
    "raw_path",
    "raw_bytes",
    "raw_sha256",
    "field_rows",
    "numeric_feature_columns",
    "categorical_feature_columns",
    "target",
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
    "rmse",
    "mae",
    "r2",
    "n_rows",
)

REVIEW_FIELDS = (
    "target",
    "selected_profile",
    "selected_method",
    "selected_validation_rmse",
    "selected_test_rmse",
    "mean_validation_rmse",
    "mean_test_rmse",
    "best_shortcut_profile",
    "best_shortcut_method",
    "best_shortcut_validation_rmse",
    "best_shortcut_test_rmse",
    "validation_relative_improvement_over_mean",
    "test_relative_improvement_over_mean",
    "baseline_visible_gap",
    "shortcut_dominant",
    "phase163_focused_review_allowed",
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
        required = {"Steel_industry_data.csv"}
        missing = required - members
        if missing:
            raise ValueError(f"Missing required UCI members: {sorted(missing)}")
        table_size = archive.getinfo("Steel_industry_data.csv").file_size
    return {
        "raw_path": str(path),
        "raw_bytes": size,
        "raw_sha256": _sha256(path),
        "zip_members": sorted(members),
        "steel_industry_csv_bytes": table_size,
    }


def _stable_unit_hash(text: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def load_steel_energy_table(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        with archive.open("Steel_industry_data.csv") as handle:
            out = pd.read_csv(handle)
    required = {"date", TARGET_COLUMN, *RAW_NUMERIC_COLUMNS, *RAW_CATEGORICAL_COLUMNS}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing steel-industry columns: {sorted(missing)}")
    for column in (TARGET_COLUMN, *RAW_NUMERIC_COLUMNS):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    timestamp = pd.to_datetime(out["date"], format="%d/%m/%Y %H:%M", errors="coerce")
    out = out.loc[timestamp.notna()].copy().reset_index(drop=True)
    timestamp = pd.to_datetime(out["date"], format="%d/%m/%Y %H:%M")
    out.insert(0, "phase162_row_id", np.arange(len(out), dtype=int))
    out["timestamp"] = timestamp
    out["month"] = timestamp.dt.month.astype(int)
    out["day_of_year"] = timestamp.dt.dayofyear.astype(int)
    out["hour"] = timestamp.dt.hour.astype(int)
    out["minute"] = timestamp.dt.minute.astype(int)
    out["week_of_year"] = timestamp.dt.isocalendar().week.astype(int)
    out["date_day"] = timestamp.dt.strftime("%Y-%m-%d")
    out["week_key"] = timestamp.dt.strftime("%Y-W%U")
    out["row_order_fraction"] = np.arange(len(out), dtype=float) / max(len(out) - 1, 1)
    out["nsm_sin"] = np.sin(2.0 * math.pi * out["NSM"] / 86400.0)
    out["nsm_cos"] = np.cos(2.0 * math.pi * out["NSM"] / 86400.0)
    out = out.dropna(subset=[TARGET_COLUMN, *RAW_NUMERIC_COLUMNS]).reset_index(drop=True)
    return out


def split_by_group(df: pd.DataFrame, *, group_column: str = "week_key") -> dict[str, Any]:
    groups = sorted(str(value) for value in df[group_column].dropna().unique())
    split_groups = {"train": set(), "val": set(), "test": set()}
    for group in groups:
        value = _stable_unit_hash(group, "phase162_week_split")
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
    if any(count < MIN_SPLIT_ROWS for count in counts.values()):
        raise ValueError(f"Split too small for Phase 162 review: {counts}")
    return {
        "group_column": group_column,
        "group_count": len(groups),
        "split_groups": {key: sorted(values) for key, values in split_groups.items()},
        "assignments": assignments,
        "counts": counts,
    }


def profile_columns() -> dict[str, dict[str, Any]]:
    return {
        "calendar_cycle_context": {
            "role": "admissible",
            "numeric": (
                "NSM",
                "nsm_sin",
                "nsm_cos",
                "month",
                "day_of_year",
                "hour",
                "minute",
                "week_of_year",
            ),
            "categorical": ("WeekStatus", "Day_of_week"),
        },
        "calendar_numeric_context": {
            "role": "admissible",
            "numeric": (
                "NSM",
                "nsm_sin",
                "nsm_cos",
                "month",
                "day_of_year",
                "hour",
                "minute",
                "week_of_year",
            ),
            "categorical": (),
        },
        "load_type_shortcut_control": {
            "role": "shortcut_control",
            "numeric": ("NSM", "nsm_sin", "nsm_cos"),
            "categorical": ("Load_Type",),
        },
        "direct_electrical_proxy_control": {
            "role": "shortcut_control",
            "numeric": (
                "Lagging_Current_Reactive.Power_kVarh",
                "Leading_Current_Reactive_Power_kVarh",
                "Lagging_Current_Power_Factor",
                "Leading_Current_Power_Factor",
            ),
            "categorical": (),
        },
        "co2_direct_control": {
            "role": "shortcut_control",
            "numeric": ("CO2(tCO2)",),
            "categorical": (),
        },
        "row_order_control": {
            "role": "shortcut_control",
            "numeric": ("row_order_fraction",),
            "categorical": (),
        },
    }


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _preprocessor(numeric_columns: tuple[str, ...], categorical_columns: tuple[str, ...]):
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                list(numeric_columns),
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                make_pipeline(SimpleImputer(strategy="most_frequent"), _one_hot_encoder()),
                list(categorical_columns),
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _model(method: str, *, numeric: tuple[str, ...], categorical: tuple[str, ...]):
    preprocessor = _preprocessor(numeric, categorical)
    if method == "ridge":
        estimator = Ridge(alpha=1.0, random_state=162)
    elif method == "knn":
        estimator = KNeighborsRegressor(n_neighbors=7, weights="distance", algorithm="brute")
    elif method == "extra_trees":
        estimator = ExtraTreesRegressor(
            n_estimators=160,
            min_samples_leaf=2,
            random_state=162,
            n_jobs=1,
        )
    elif method == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            max_iter=160,
            learning_rate=0.06,
            l2_regularization=0.01,
            random_state=162,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    return make_pipeline(preprocessor, estimator)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


def evaluate_baselines(df: pd.DataFrame, assignments: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    y = df[TARGET_COLUMN].to_numpy(dtype=float)
    split_indices = {
        split: np.array([idx for idx, name in enumerate(assignments) if name == split], dtype=int)
        for split in ("train", "val", "test")
    }
    mean_value = float(np.mean(y[split_indices["train"]]))
    metric_rows: list[dict[str, Any]] = []
    for split, indices in split_indices.items():
        metric_rows.append(
            {
                "profile": "train_mean",
                "method": "mean",
                "role": "mean_baseline",
                "split": split,
                "n_rows": int(len(indices)),
                **_metrics(y[indices], np.full(len(indices), mean_value, dtype=float)),
            }
        )

    for profile_name, spec in profile_columns().items():
        numeric = tuple(column for column in spec["numeric"] if column in df.columns)
        categorical = tuple(column for column in spec["categorical"] if column in df.columns)
        if not numeric and not categorical:
            continue
        columns = [*numeric, *categorical]
        x = df[columns]
        for method in MODEL_METHODS:
            if spec["role"] == "shortcut_control" and method == "hist_gradient_boosting":
                continue
            model = _model(method, numeric=numeric, categorical=categorical)
            model.fit(x.iloc[split_indices["train"]], y[split_indices["train"]])
            for split, indices in split_indices.items():
                pred = model.predict(x.iloc[indices])
                metric_rows.append(
                    {
                        "profile": profile_name,
                        "method": method,
                        "role": spec["role"],
                        "split": split,
                        "n_rows": int(len(indices)),
                        **_metrics(y[indices], pred),
                    }
                )
    return metric_rows, build_review_rows(metric_rows)


def _metric_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {(row["profile"], row["method"], row["split"]): row for row in rows}


def build_review_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = _metric_lookup(metric_rows)
    mean_val = lookup[("train_mean", "mean", "val")]["rmse"]
    mean_test = lookup[("train_mean", "mean", "test")]["rmse"]
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
    selected = min(admissible_val, key=lambda row: row["rmse"])
    selected_test = lookup[(selected["profile"], selected["method"], "test")]
    best_shortcut = min(shortcut_val, key=lambda row: row["rmse"])
    best_shortcut_test = lookup[(best_shortcut["profile"], best_shortcut["method"], "test")]
    val_gain = (mean_val - selected["rmse"]) / mean_val if mean_val else 0.0
    test_gain = (mean_test - selected_test["rmse"]) / mean_test if mean_test else 0.0
    baseline_visible_gap = val_gain >= MIN_RELATIVE_VAL_GAIN and test_gain >= MIN_RELATIVE_TEST_GAIN
    shortcut_dominant = (
        best_shortcut["rmse"] <= selected["rmse"] * SHORTCUT_DOMINANCE_TOLERANCE
        or best_shortcut_test["rmse"] <= selected_test["rmse"] * SHORTCUT_DOMINANCE_TOLERANCE
    )
    focused_allowed = bool(baseline_visible_gap and not shortcut_dominant)
    blocker = ""
    if not baseline_visible_gap:
        blocker = "leakage-limited admissible baseline does not beat mean by required validation/test margins"
    elif shortcut_dominant:
        blocker = "direct proxy/load-type/row-order shortcut is too close to or better than selected admissible profile"
    status = (
        "phase162_uci_steel_industry_energy_ready_focused_review"
        if focused_allowed
        else "phase162_uci_steel_industry_energy_closed_no_stable_guarded_gap"
    )
    return [
        {
            "target": TARGET_COLUMN,
            "selected_profile": selected["profile"],
            "selected_method": selected["method"],
            "selected_validation_rmse": selected["rmse"],
            "selected_test_rmse": selected_test["rmse"],
            "mean_validation_rmse": mean_val,
            "mean_test_rmse": mean_test,
            "best_shortcut_profile": best_shortcut["profile"],
            "best_shortcut_method": best_shortcut["method"],
            "best_shortcut_validation_rmse": best_shortcut["rmse"],
            "best_shortcut_test_rmse": best_shortcut_test["rmse"],
            "validation_relative_improvement_over_mean": val_gain,
            "test_relative_improvement_over_mean": test_gain,
            "baseline_visible_gap": baseline_visible_gap,
            "shortcut_dominant": shortcut_dominant,
            "phase163_focused_review_allowed": focused_allowed,
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
    focused = bool(review["phase163_focused_review_allowed"])
    return {
        "status": review["status"],
        "source": "UCI Steel Industry Energy Consumption",
        "source_doi": SOURCE_DOI,
        "raw_sha256": overview["raw_sha256"],
        "field_rows": overview["field_rows"],
        "group_count": overview["group_count"],
        "selected_target": TARGET_COLUMN,
        "selected_profile": review["selected_profile"],
        "selected_method": review["selected_method"],
        "selected_validation_rmse": review["selected_validation_rmse"],
        "selected_test_rmse": review["selected_test_rmse"],
        "mean_validation_rmse": review["mean_validation_rmse"],
        "mean_test_rmse": review["mean_test_rmse"],
        "best_shortcut_profile": review["best_shortcut_profile"],
        "best_shortcut_validation_rmse": review["best_shortcut_validation_rmse"],
        "best_shortcut_test_rmse": review["best_shortcut_test_rmse"],
        "phase163_focused_review_allowed": focused,
        "phase162_model_mechanism_allowed": False,
        "phase162_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": (
            "run Phase 163 split/shortcut focused review before any mechanism"
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
        "# Phase 162 UCI Steel Industry Energy Baseline Gate",
        "",
        "## Gate",
        f"- Status: `{gate['status']}`",
        f"- Phase 163 focused review allowed: `{_csv_value(gate['phase163_focused_review_allowed'])}`",
        f"- Model training allowed: `{_csv_value(gate['phase162_model_training_allowed'])}`",
        f"- A100 training allowed now: `{_csv_value(gate['a100_training_allowed_now'])}`",
        f"- A100-SXM4-80GB request now: `{_csv_value(gate['a100_80gb_request_now'])}`",
        "",
        "## Interpretation",
        (
            "This is a no-training baseline-first intake for a steel-industry energy "
            "source. The registered split holds out complete weeks. Synchronous "
            "electrical quantities, CO2, load labels, and row order are shortcut "
            "controls rather than model inputs for a publishable claim."
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
    df = load_steel_energy_table(raw_path)
    if len(df) < MIN_ROWS_FOR_REVIEW:
        raise ValueError(f"Too few rows for Phase 162 review: {len(df)}")
    split = split_by_group(df)
    metric_rows, review_rows = evaluate_baselines(df, split["assignments"])
    overview = {
        "source_id": "phase162_uci_steel_industry_energy",
        "source_url": source_url,
        "source_doi": SOURCE_DOI,
        "raw_path": _display_path(raw_path, root),
        "raw_bytes": source_info["raw_bytes"],
        "raw_sha256": source_info["raw_sha256"],
        "field_rows": int(len(df)),
        "numeric_feature_columns": int(
            sum(pd.api.types.is_numeric_dtype(df[column]) for column in df.columns)
        ),
        "categorical_feature_columns": int(len(RAW_CATEGORICAL_COLUMNS)),
        "target": TARGET_COLUMN,
        "group_column": split["group_column"],
        "group_count": int(split["group_count"]),
        "train_rows_split": int(split["counts"]["train"]),
        "val_rows_split": int(split["counts"]["val"]),
        "test_rows_split": int(split["counts"]["test"]),
    }
    gate = build_gate(overview=overview, review_rows=review_rows)

    overview_rows = [overview]
    overview_path = output_dir / "phase162_source_overview_table.csv"
    metric_path = output_dir / "phase162_baseline_metric_table.csv"
    review_path = output_dir / "phase162_baseline_review_table.csv"
    split_path = output_dir / "phase162_split_manifest.json"
    gate_path = output_dir / "phase162_uci_steel_industry_energy_baseline_gate.json"
    markdown_path = output_dir / "phase162_uci_steel_industry_energy_baseline_gate.md"
    manifest_path = output_dir / "phase162_uci_steel_industry_energy_baseline_manifest.json"

    _write_csv(overview_path, overview_rows, OVERVIEW_FIELDS)
    _write_csv(metric_path, metric_rows, METRIC_FIELDS)
    _write_csv(review_path, review_rows, REVIEW_FIELDS)
    _write_json(
        split_path,
        {
            "phase": 162,
            "source": "UCI Steel Industry Energy Consumption",
            "group_column": split["group_column"],
            "group_count": split["group_count"],
            "counts": split["counts"],
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
        "phase": 162,
        "description": "baseline-first intake for UCI steel-industry energy consumption",
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
            "metric_rows": len(metric_rows),
            "review_rows": len(review_rows),
            "group_count": int(split["group_count"]),
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
