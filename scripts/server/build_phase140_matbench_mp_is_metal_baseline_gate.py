#!/usr/bin/env python3
"""Build a Phase 140 baseline-first triage gate for Matbench MP is-metal.

This phase opens a larger public external source after the Phase 138/139 glass
branch closed. It downloads/reuses Matbench v0.1 ``matbench_mp_is_metal`` on the
server, parses structure JSON dictionaries through the Phase 130 structure
descriptor route, and reviews the binary ``is_metal`` target with strong
classification baselines and shortcut controls. It does not train a neural
model or open A100/A800 training.
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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SOURCE_URL = "https://ml.materialsproject.org/projects/matbench_mp_is_metal.json.gz"
DEFAULT_RAW_PATH = Path("data/raw/external/matbench_mp_is_metal/matbench_mp_is_metal.json.gz")
DEFAULT_OUTPUT_DIR = Path("docs/results/phase140_matbench_mp_is_metal_baseline_gate")
EXPECTED_MIN_BYTES = 100_000_000
EXPECTED_HEAD_BYTES = 136_698_078
TARGET_COLUMN = "is_metal"
DEFAULT_MAX_ROWS = 12_000
MIN_ROWS_FOR_REVIEW = 1_000
MIN_SPLIT_ROWS = 200
MIN_BALANCED_ACCURACY_GAIN = 0.05
SHORTCUT_DOMINANCE_MARGIN = 0.005


def _load_module(filename: str, module_name: str):
    script = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase130 = _load_module("build_phase130_matbench_log_kvrh_baseline_gate.py", "phase130_helpers")
phase138 = _load_module("build_phase138_matbench_glass_baseline_gate.py", "phase138_helpers")

FIELD_FIELDS = (
    "phase140_row_id",
    "composition",
    TARGET_COLUMN,
    *phase130.FIELD_FIELDS[3:],
)
PROFILE_COLUMNS: dict[str, dict[str, Any]] = {
    "composition_descriptors": {
        **phase130.PROFILE_COLUMNS["composition_descriptors"],
        "role": "admissible",
    },
    "common_element_fractions": {
        **phase130.PROFILE_COLUMNS["common_element_fractions"],
        "role": "admissible",
    },
    "lattice_descriptors": {
        **phase130.PROFILE_COLUMNS["lattice_descriptors"],
        "role": "admissible",
    },
    "composition_lattice_descriptors": {
        **phase130.PROFILE_COLUMNS["composition_lattice_descriptors"],
        "role": "admissible",
    },
    "composition_hash_shortcut": {
        **phase130.PROFILE_COLUMNS["composition_hash_shortcut"],
        "role": "negative_control",
    },
    "chemistry_family_shortcut": {
        **phase130.PROFILE_COLUMNS["chemistry_family_shortcut"],
        "role": "negative_control",
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
    "phase140_candidate",
    "status",
    "reason",
)
SCHEMA_FIELDS = tuple(phase130.SCHEMA_FIELDS)


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
            headers={"User-Agent": "Mozilla/5.0 Codex Phase140 matbench mp is metal gate"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
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


def _deterministic_row_cap(rows: list[list[Any]], max_rows: int | None, *, salt: str = "phase140") -> list[list[Any]]:
    if max_rows is None or max_rows <= 0 or len(rows) <= max_rows:
        return rows
    ranked = sorted(
        range(len(rows)),
        key=lambda index: hashlib.sha256(f"{salt}::{index}".encode("utf-8")).hexdigest(),
    )
    keep = set(ranked[:max_rows])
    return [row for index, row in enumerate(rows) if index in keep]


def load_matbench_payload(path: Path, *, max_rows: int | None = DEFAULT_MAX_ROWS) -> tuple[list[str], list[list[Any]], dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    columns = payload.get("columns")
    data = payload.get("data")
    if columns != ["structure", TARGET_COLUMN]:
        raise ValueError(f"Unexpected matbench_mp_is_metal columns: {columns}")
    if not isinstance(data, list):
        raise ValueError("Expected split-orient data list")
    capped = _deterministic_row_cap(data, max_rows)
    return list(columns), capped, {
        "raw_rows": len(data),
        "selected_rows": len(capped),
        "max_rows": max_rows,
        "row_cap_applied": max_rows is not None and max_rows > 0 and len(data) > max_rows,
        "row_cap_strategy": "stable_hash_by_source_row_index",
    }


def _target_to_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return int(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "metal"}:
            return 1
        if lowered in {"false", "0", "no", "nonmetal"}:
            return 0
    raise ValueError(f"Unsupported is_metal label: {value!r}")


def build_field_table(rows: list[list[Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    converted_rows = []
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(f"Unexpected row shape at {index}: {type(row).__name__}")
        converted_rows.append([row[0], float(_target_to_int(row[1]))])
    field_df, parse_audit = phase130.build_field_table(converted_rows)
    field_df = field_df.rename(
        columns={"phase130_row_id": "phase140_row_id", "log10_k_vrh": TARGET_COLUMN}
    )
    field_df["phase140_row_id"] = [f"P140-{index:05d}" for index in range(len(field_df))]
    field_df[TARGET_COLUMN] = field_df[TARGET_COLUMN].round().astype(int)
    return field_df.loc[:, list(FIELD_FIELDS)], parse_audit


def group_split(df: pd.DataFrame, *, group_column: str = "chemistry_family_key", salt: str = "phase140") -> dict[str, Any]:
    return phase130.group_split(df, group_column=group_column, salt=salt)


def _classification_metrics(y_true: np.ndarray, positive_proba: np.ndarray) -> dict[str, Any]:
    return phase138._classification_metrics(y_true.astype(int), positive_proba.astype(float))


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
            "phase140_candidate": False,
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
        x_train, x_all = phase130.phase126._one_hot_frame(df.iloc[train_idx], df, profile)
        y_train = y[train_idx]
        for method in PROFILE_METHODS.get(profile_name, MODEL_METHODS):
            proba = phase138._fit_predict_proba(method, x_train, y_train, x_all)
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
        reason = "safe structure/composition profile beats majority and shortcut controls"

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
        "phase140_candidate": candidate,
        "status": status,
        "reason": reason,
    }
    return metric_rows, review


def build_schema_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = phase130.build_schema_rows(df.rename(columns={"phase140_row_id": "phase130_row_id", TARGET_COLUMN: "log10_k_vrh"}))
    for row in rows:
        if row["column"] == "phase130_row_id":
            row["column"] = "phase140_row_id"
            row["role"] = "identifier"
        elif row["column"] == "log10_k_vrh":
            row["column"] = TARGET_COLUMN
            row["role"] = "target"
    return rows


def build_gate(
    *,
    source_info: dict[str, Any],
    split_manifest: dict[str, Any],
    review: dict[str, Any],
    load_audit: dict[str, Any],
    parse_audit: dict[str, Any],
) -> dict[str, Any]:
    focused_allowed = bool(review.get("phase140_candidate"))
    if focused_allowed:
        status = "phase140_matbench_mp_is_metal_triage_ready_focused_review"
        next_action = "enter focused split/shortcut review before any model mechanism"
    elif review.get("status") == "blocked_insufficient_rows_or_split":
        status = "phase140_matbench_mp_is_metal_incomplete_insufficient_split_rows"
        next_action = "increase row cap or choose another public target"
    else:
        status = "phase140_matbench_mp_is_metal_closed_no_stable_guarded_gap"
        next_action = "close as external-data diagnostic or choose another public target"
    return {
        "status": status,
        "source_name": "Matbench v0.1 matbench_mp_is_metal",
        "source_url": source_info["source_url"],
        "source_byte_size": source_info["byte_size"],
        "source_sha256": source_info["sha256"],
        "raw_row_count": load_audit["raw_rows"],
        "selected_raw_row_count": load_audit["selected_rows"],
        "parsed_row_count": parse_audit["parsed_rows"],
        "skipped_row_count": parse_audit["skipped_rows"],
        "row_cap_applied": load_audit["row_cap_applied"],
        "row_cap_max_rows": load_audit["max_rows"],
        "row_cap_strategy": load_audit["row_cap_strategy"],
        "row_count": review["row_count"],
        "group_count": split_manifest["n_groups"],
        "split_counts": {split: len(indices) for split, indices in split_manifest["splits"].items()},
        "leakage_safe_group_split": bool(split_manifest["leakage_safe"]),
        "selected_target": review["target"],
        "selected_profile": review.get("best_profile") if focused_allowed else None,
        "selected_method": review.get("best_method") if focused_allowed else None,
        "selected_validation_balanced_accuracy": review.get("best_val_balanced_accuracy") if focused_allowed else None,
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
        "phase140_focused_review_allowed": focused_allowed,
        "phase140_model_mechanism_allowed": False,
        "phase140_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": next_action,
        "reason": review.get("reason"),
    }


def build_data_card(source_info: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": "Matbench v0.1 matbench_mp_is_metal",
        "source_url": source_info["source_url"],
        "source_sha256": source_info["sha256"],
        "source_byte_size": source_info["byte_size"],
        "license_note": "Public Matbench benchmark dataset; cite Matbench/matminer in manuscripts.",
        "target": "binary is_metal label",
        "input": "pymatgen Structure JSON",
        "row_count": gate["row_count"],
        "raw_row_count": gate["raw_row_count"],
        "row_cap_applied": gate["row_cap_applied"],
        "row_cap_max_rows": gate["row_cap_max_rows"],
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
        "# Phase 140 Matbench MP Is-Metal Baseline Triage Gate",
        "",
        f"- Status: `{gate['status']}`",
        f"- Rows: `{gate['row_count']}` of raw `{gate['raw_row_count']}`",
        f"- Row cap applied: `{gate['row_cap_applied']}`",
        f"- Selected target: `{gate['selected_target']}`",
        f"- Focused review allowed: `{gate['phase140_focused_review_allowed']}`",
        f"- Model training allowed: `{gate['phase140_model_training_allowed']}`",
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
    max_rows: int | None = DEFAULT_MAX_ROWS,
) -> dict[str, Any]:
    source_info = ensure_source_file(raw_path, source_url=source_url, force_download=force_download)
    _, raw_rows, load_audit = load_matbench_payload(raw_path, max_rows=max_rows)
    field_df, parse_audit = build_field_table(raw_rows)
    split_manifest = group_split(field_df)
    schema_rows = build_schema_rows(field_df)
    metric_rows, review = evaluate_target(field_df, split_manifest)
    gate = build_gate(
        source_info=source_info,
        split_manifest=split_manifest,
        review=review,
        load_audit=load_audit,
        parse_audit=parse_audit,
    )
    data_card = build_data_card(source_info, gate)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "phase140_matbench_mp_is_metal"
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
    _write_json(split_path, split_manifest)
    _write_csv(schema_path, schema_rows, SCHEMA_FIELDS)
    _write_csv(metric_path, metric_rows, METRIC_FIELDS)
    _write_csv(review_path, [review], REVIEW_FIELDS)
    _write_json(gate_path, gate)
    _write_json(card_path, data_card)
    markdown_path.write_text(build_markdown(gate, review), encoding="utf-8")

    manifest = {
        "phase": 140,
        "objective": "matbench_mp_is_metal_large_source_classification_baseline_triage_no_training",
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
            "raw_rows": load_audit["raw_rows"],
            "selected_raw_rows": load_audit["selected_rows"],
            "field_rows": int(len(field_df)),
            "skipped_rows": parse_audit["skipped_rows"],
            "schema_rows": len(schema_rows),
            "metric_rows": len(metric_rows),
            "candidate_targets": 1 if gate["phase140_focused_review_allowed"] else 0,
        },
        "load_audit": load_audit,
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
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    raw_path = args.raw_path if args.raw_path.is_absolute() else root / args.raw_path
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    max_rows = args.max_rows if args.max_rows and args.max_rows > 0 else None
    manifest = build_package(
        root=root,
        raw_path=raw_path,
        output_dir=output_dir,
        source_url=args.source_url,
        force_download=args.force_download,
        max_rows=max_rows,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
