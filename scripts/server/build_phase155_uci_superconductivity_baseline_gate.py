#!/usr/bin/env python3
"""Build Phase 155 baseline-first gate for UCI Superconductivity data."""

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
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SOURCE_URL = "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip"
SOURCE_DOI = "10.24432/C53P47"
DEFAULT_RAW_PATH = Path(
    "data/raw/external/phase155_uci_superconductivity/superconductivty_data.zip"
)
DEFAULT_OUTPUT_DIR = Path("docs/results/phase155_uci_superconductivity_baseline_gate")

EXPECTED_MIN_BYTES = 8_000_000
MIN_ROWS_FOR_REVIEW = 10_000
MIN_SPLIT_ROWS = 500
MIN_RELATIVE_VAL_GAIN = 0.15
MIN_RELATIVE_TEST_GAIN = 0.05
SHORTCUT_DOMINANCE_TOLERANCE = 1.02
MODEL_METHODS = ("knn", "extra_trees", "hist_gradient_boosting")

OVERVIEW_FIELDS = (
    "source_id",
    "source_url",
    "source_doi",
    "raw_path",
    "raw_bytes",
    "raw_sha256",
    "train_rows",
    "unique_rows",
    "feature_columns",
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
    "phase156_focused_review_allowed",
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
        required = {"train.csv", "unique_m.csv"}
        missing = required - members
        if missing:
            raise ValueError(f"Missing required UCI members: {sorted(missing)}")
        train_size = archive.getinfo("train.csv").file_size
        unique_size = archive.getinfo("unique_m.csv").file_size
    return {
        "raw_path": str(path),
        "raw_bytes": size,
        "raw_sha256": _sha256(path),
        "zip_members": sorted(members),
        "train_csv_bytes": train_size,
        "unique_m_csv_bytes": unique_size,
    }


def _stable_unit_hash(text: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def load_superconductivity_table(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        with archive.open("train.csv") as handle:
            train = pd.read_csv(handle)
        with archive.open("unique_m.csv") as handle:
            unique = pd.read_csv(handle)
    if len(train) != len(unique):
        raise ValueError(f"train.csv rows {len(train)} != unique_m.csv rows {len(unique)}")
    if "critical_temp" not in train.columns or "critical_temp" not in unique.columns:
        raise ValueError("Both UCI tables must contain critical_temp")
    if "material" not in unique.columns:
        raise ValueError("unique_m.csv must contain material")

    element_columns = [
        column
        for column in unique.columns
        if column not in {"critical_temp", "material"}
        and pd.api.types.is_numeric_dtype(unique[column])
    ]
    train_feature_columns = [
        column
        for column in train.columns
        if column != "critical_temp" and pd.api.types.is_numeric_dtype(train[column])
    ]
    out = train.copy()
    for column in element_columns:
        out[f"frac_{column}"] = pd.to_numeric(unique[column], errors="coerce").fillna(0.0)
    out["material"] = unique["material"].astype(str)
    out["target_critical_temp_K"] = pd.to_numeric(train["critical_temp"], errors="coerce")
    out = out.drop(columns=["critical_temp"])

    element_sets: list[str] = []
    dominant_elements: list[str] = []
    max_fractions: list[float] = []
    formula_lengths: list[int] = []
    for _, row in out.iterrows():
        fractions = {
            column[5:]: float(row[column])
            for column in out.columns
            if column.startswith("frac_") and float(row[column]) > 0.0
        }
        present = sorted(fractions)
        element_sets.append("-".join(present) if present else "unknown")
        if fractions:
            dominant = max(fractions.items(), key=lambda item: (item[1], item[0]))[0]
            max_fraction = max(fractions.values())
        else:
            dominant = "unknown"
            max_fraction = 0.0
        dominant_elements.append(dominant)
        max_fractions.append(max_fraction)
        formula_lengths.append(len(str(row["material"])))
    out["element_set_key"] = element_sets
    out["dominant_element"] = dominant_elements
    out["max_element_fraction"] = max_fractions
    out["formula_length"] = formula_lengths
    out["element_set_hash"] = [
        _stable_unit_hash(value, "phase155_element_set") for value in out["element_set_key"]
    ]
    out["dominant_element_hash"] = [
        _stable_unit_hash(value, "phase155_dominant") for value in out["dominant_element"]
    ]
    out["row_order_fraction"] = np.arange(len(out), dtype=float) / max(len(out) - 1, 1)
    out.attrs["phase155_train_feature_columns"] = train_feature_columns
    return out


def split_by_group(df: pd.DataFrame, *, group_column: str = "element_set_key") -> dict[str, Any]:
    groups = sorted(str(value) for value in df[group_column].dropna().unique())
    split_groups = {"train": set(), "val": set(), "test": set()}
    for group in groups:
        value = _stable_unit_hash(group, "phase155_split")
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
        raise ValueError(f"Split too small for review: {counts}")
    return {
        "group_column": group_column,
        "group_count": len(groups),
        "split_groups": {key: sorted(values) for key, values in split_groups.items()},
        "assignments": assignments,
        "counts": counts,
    }


def _numeric_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    return [column for column in columns if column in df.columns]


def profile_columns(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    train_feature_columns = tuple(df.attrs.get("phase155_train_feature_columns") or ())
    if not train_feature_columns:
        shortcut_derived_columns = {
            "element_set_hash",
            "dominant_element_hash",
            "row_order_fraction",
            "formula_length",
            "max_element_fraction",
        }
        train_feature_columns = tuple(
            column
            for column in df.columns
            if column not in {
                "material",
                "target_critical_temp_K",
                "element_set_key",
                "dominant_element",
                *shortcut_derived_columns,
            }
            and not column.startswith("frac_")
            and pd.api.types.is_numeric_dtype(df[column])
        )
    train_feature_columns = tuple(
        column
        for column in train_feature_columns
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column])
    )
    weighted_columns = tuple(
        column
        for column in train_feature_columns
        if column.startswith("wtd_") or column == "number_of_elements"
    )
    element_fraction_columns = tuple(column for column in df.columns if column.startswith("frac_"))
    return {
        "uci_feature_full": {"role": "admissible", "columns": train_feature_columns},
        "weighted_feature_core": {"role": "admissible", "columns": weighted_columns},
        "element_fraction_vector": {"role": "admissible", "columns": element_fraction_columns},
        "formula_shape_control": {
            "role": "shortcut_control",
            "columns": (
                "number_of_elements",
                "formula_length",
                "max_element_fraction",
            ),
        },
        "element_set_hash_control": {
            "role": "shortcut_control",
            "columns": (
                "element_set_hash",
                "dominant_element_hash",
                "number_of_elements",
            ),
        },
        "row_order_control": {"role": "shortcut_control", "columns": ("row_order_fraction",)},
    }


def _model(method: str):
    if method == "knn":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=7, weights="distance"),
        )
    if method == "extra_trees":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=96,
                min_samples_leaf=2,
                random_state=155,
                n_jobs=1,
            ),
        )
    if method == "hist_gradient_boosting":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                max_iter=120,
                learning_rate=0.08,
                l2_regularization=0.01,
                random_state=155,
            ),
        )
    raise ValueError(f"Unknown method: {method}")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


def evaluate_baselines(df: pd.DataFrame, assignments: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    y = df["target_critical_temp_K"].to_numpy(dtype=float)
    split_indices = {
        split: np.array([idx for idx, name in enumerate(assignments) if name == split], dtype=int)
        for split in ("train", "val", "test")
    }
    mean_value = float(np.mean(y[split_indices["train"]]))
    metric_rows: list[dict[str, Any]] = []
    for split, indices in split_indices.items():
        metrics = _metrics(y[indices], np.full(len(indices), mean_value, dtype=float))
        metric_rows.append(
            {
                "profile": "train_mean",
                "method": "mean",
                "role": "mean_baseline",
                "split": split,
                "n_rows": int(len(indices)),
                **metrics,
            }
        )

    profiles = profile_columns(df)
    for profile_name, spec in profiles.items():
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
                        **_metrics(y[indices], pred),
                    }
                )

    review_rows = build_review_rows(metric_rows)
    return metric_rows, review_rows


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
        blocker = "strong admissible baseline does not beat mean by required validation/test margins"
    elif shortcut_dominant:
        blocker = "shortcut control is too close to or better than selected admissible profile"
    status = (
        "phase155_uci_superconductivity_ready_focused_review"
        if focused_allowed
        else "phase155_uci_superconductivity_closed_no_stable_guarded_gap"
    )
    return [
        {
            "target": "target_critical_temp_K",
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
            "phase156_focused_review_allowed": focused_allowed,
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
        return f"{value:.12g}"
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
        "| " + " | ".join(_csv_value(row.get(field, "")) for field in fields) + " |"
        for row in rows
    ]
    return [header, sep, *body]


def build_gate(
    *,
    overview: dict[str, Any],
    review_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    review = review_rows[0]
    focused = bool(review["phase156_focused_review_allowed"])
    return {
        "status": review["status"],
        "source": "UCI Superconductivty Data",
        "source_doi": SOURCE_DOI,
        "raw_sha256": overview["raw_sha256"],
        "field_rows": overview["train_rows"],
        "group_count": overview["group_count"],
        "selected_target": "target_critical_temp_K",
        "selected_profile": review["selected_profile"],
        "selected_method": review["selected_method"],
        "selected_validation_rmse": review["selected_validation_rmse"],
        "selected_test_rmse": review["selected_test_rmse"],
        "mean_validation_rmse": review["mean_validation_rmse"],
        "mean_test_rmse": review["mean_test_rmse"],
        "best_shortcut_profile": review["best_shortcut_profile"],
        "best_shortcut_validation_rmse": review["best_shortcut_validation_rmse"],
        "best_shortcut_test_rmse": review["best_shortcut_test_rmse"],
        "phase156_focused_review_allowed": focused,
        "phase155_model_mechanism_allowed": False,
        "phase155_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": (
            "run Phase 156 split/shortcut focused review before any mechanism"
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
        "# Phase 155 UCI Superconductivity Baseline Gate",
        "",
        "## Gate",
        f"- Status: `{gate['status']}`",
        f"- Phase 156 focused review allowed: `{_csv_value(gate['phase156_focused_review_allowed'])}`",
        f"- Model training allowed: `{_csv_value(gate['phase155_model_training_allowed'])}`",
        f"- A100 training allowed now: `{_csv_value(gate['a100_training_allowed_now'])}`",
        f"- A100-SXM4-80GB request now: `{_csv_value(gate['a100_80gb_request_now'])}`",
        "",
        "## Interpretation",
        (
            "This is a no-training baseline-first intake for a possible second-paper "
            "positive mainline. A positive gate can only open a focused split/shortcut "
            "review; it does not open neural model training."
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
    df = load_superconductivity_table(raw_path)
    if len(df) < MIN_ROWS_FOR_REVIEW:
        raise ValueError(f"Too few rows for Phase 155 review: {len(df)}")
    split = split_by_group(df)
    metric_rows, review_rows = evaluate_baselines(df, split["assignments"])
    overview = {
        "source_id": "phase155_uci_superconductivity",
        "source_url": source_url,
        "source_doi": SOURCE_DOI,
        "raw_path": _display_path(raw_path, root),
        "raw_bytes": source_info["raw_bytes"],
        "raw_sha256": source_info["raw_sha256"],
        "train_rows": int(len(df)),
        "unique_rows": int(len(df)),
        "feature_columns": int(sum(1 for column in df.columns if pd.api.types.is_numeric_dtype(df[column]))),
        "target": "target_critical_temp_K",
        "group_column": split["group_column"],
        "group_count": int(split["group_count"]),
        "train_rows_split": int(split["counts"]["train"]),
        "val_rows_split": int(split["counts"]["val"]),
        "test_rows_split": int(split["counts"]["test"]),
    }
    gate = build_gate(overview=overview, review_rows=review_rows)

    overview_rows = [overview]
    overview_path = output_dir / "phase155_source_overview_table.csv"
    metric_path = output_dir / "phase155_baseline_metric_table.csv"
    review_path = output_dir / "phase155_baseline_review_table.csv"
    split_path = output_dir / "phase155_split_manifest.json"
    gate_path = output_dir / "phase155_uci_superconductivity_baseline_gate.json"
    markdown_path = output_dir / "phase155_uci_superconductivity_baseline_gate.md"
    manifest_path = output_dir / "phase155_uci_superconductivity_baseline_manifest.json"

    _write_csv(overview_path, overview_rows, OVERVIEW_FIELDS)
    _write_csv(metric_path, metric_rows, METRIC_FIELDS)
    _write_csv(review_path, review_rows, REVIEW_FIELDS)
    _write_json(
        split_path,
        {
            "phase": 155,
            "source": "UCI Superconductivty Data",
            "group_column": split["group_column"],
            "group_count": split["group_count"],
            "counts": split["counts"],
            "split_groups": split["split_groups"],
        },
    )
    _write_json(gate_path, gate)
    with markdown_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(build_markdown(gate=gate, overview_rows=overview_rows, review_rows=review_rows))
    manifest = {
        "phase": 155,
        "description": "baseline-first intake for UCI Superconductivty Data critical temperature",
        "source_info": source_info,
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
