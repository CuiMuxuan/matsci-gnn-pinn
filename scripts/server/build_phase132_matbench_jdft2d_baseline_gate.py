#!/usr/bin/env python3
"""Build a Phase 132 baseline-first gate for Matbench JDFT2D.

This phase opens a fresh small public external data-source intake while the
current A100 endpoint is unavailable. It downloads Matbench v0.1
``matbench_jdft2d`` if needed, parses structure JSON dictionaries through the
Phase 130 structure descriptor path, and reviews ``exfoliation_en`` with
strong tabular baselines and shortcut controls. It does not train a neural
model or open A100 training.
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

import pandas as pd


SOURCE_URL = "https://ml.materialsproject.org/projects/matbench_jdft2d.json.gz"
DEFAULT_RAW_PATH = Path("data/raw/external/matbench_jdft2d/matbench_jdft2d.json.gz")
DEFAULT_OUTPUT_DIR = Path("docs/results/phase132_matbench_jdft2d_baseline_gate")
EXPECTED_MIN_BYTES = 100_000
EXPECTED_HEAD_BYTES = 267_131
TARGET_COLUMN = "exfoliation_en"


def _load_phase130_module():
    script = Path(__file__).with_name("build_phase130_matbench_log_kvrh_baseline_gate.py")
    spec = importlib.util.spec_from_file_location("phase130_helpers", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Phase 130 helper module from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase130 = _load_phase130_module()

FIELD_FIELDS = (
    "phase132_row_id",
    "composition",
    TARGET_COLUMN,
    *phase130.FIELD_FIELDS[3:],
)
REVIEW_FIELDS = tuple(
    "phase132_candidate" if field == "phase130_candidate" else field
    for field in phase130.REVIEW_FIELDS
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
            headers={"User-Agent": "Mozilla/5.0 Codex Phase132 matbench jdft2d gate"},
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
    if columns != ["structure", TARGET_COLUMN]:
        raise ValueError(f"Unexpected matbench_jdft2d columns: {columns}")
    if not isinstance(data, list):
        raise ValueError("Expected split-orient data list")
    return list(columns), data


def build_field_table(rows: list[list[Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    field_df, parse_audit = phase130.build_field_table(rows)
    field_df = field_df.rename(
        columns={"phase130_row_id": "phase132_row_id", "log10_k_vrh": TARGET_COLUMN}
    )
    field_df["phase132_row_id"] = [f"P132-{index:05d}" for index in range(len(field_df))]
    return field_df.loc[:, list(FIELD_FIELDS)], parse_audit


def group_split(df: pd.DataFrame, *, group_column: str = "chemistry_family_key", salt: str = "phase132") -> dict[str, Any]:
    return phase130.group_split(df, group_column=group_column, salt=salt)


def _as_phase130_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={"phase132_row_id": "phase130_row_id", TARGET_COLUMN: "log10_k_vrh"})


def evaluate_target(df: pd.DataFrame, split_manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metric_rows, review = phase130.evaluate_target(_as_phase130_frame(df), split_manifest)
    for row in metric_rows:
        row["target"] = TARGET_COLUMN
    review = dict(review)
    review["target"] = TARGET_COLUMN
    review["phase132_candidate"] = bool(review.pop("phase130_candidate", False))
    return metric_rows, review


def build_schema_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    schema = phase130.build_schema_rows(_as_phase130_frame(df))
    for row in schema:
        if row["column"] == "phase130_row_id":
            row["column"] = "phase132_row_id"
        elif row["column"] == "log10_k_vrh":
            row["column"] = TARGET_COLUMN
        if row["column"] == TARGET_COLUMN:
            row["role"] = "target"
        elif row["column"] == "phase132_row_id":
            row["role"] = "identifier"
    return schema


def build_gate(*, source_info: dict[str, Any], split_manifest: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    status = (
        "phase132_matbench_jdft2d_ready_focused_review"
        if review.get("phase132_candidate")
        else "phase132_matbench_jdft2d_closed_no_stable_guarded_gap"
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
        "phase132_focused_review_allowed": bool(review.get("phase132_candidate")),
        "phase132_model_mechanism_allowed": False,
        "phase132_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "reason": review.get("reason"),
    }


def build_markdown(gate: dict[str, Any], review: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Phase 132 Matbench JDFT2D Baseline Gate",
            "",
            f"- Status: `{gate['status']}`",
            f"- Target: `{gate['selected_target']}`",
            f"- Rows: `{gate['row_count']}`",
            f"- Group split: `{gate['group_split']}` with `{gate['group_count']}` groups",
            f"- Selected profile/method: `{gate['selected_profile']}` / `{gate['selected_method']}`",
            f"- Selected validation/test RMSE: `{gate['selected_validation_rmse']:.6g}` / `{gate['selected_test_rmse']:.6g}`",
            f"- Mean validation/test RMSE: `{gate['mean_validation_rmse']:.6g}` / `{gate['mean_test_rmse']:.6g}`",
            f"- Best negative control: `{gate['best_negative_profile']}` / `{gate['best_negative_method']}`",
            f"- Focused review allowed: `{gate['phase132_focused_review_allowed']}`",
            f"- Model training allowed: `{gate['phase132_model_training_allowed']}`",
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
    field_df, parse_audit = build_field_table(raw_rows)
    split_manifest = group_split(field_df)
    metric_rows, review = evaluate_target(field_df, split_manifest)
    schema_rows = build_schema_rows(field_df)
    gate = build_gate(source_info=source_info, split_manifest=split_manifest, review=review)
    gate.update(
        {
            "raw_row_count": parse_audit["raw_rows"],
            "parsed_row_count": parse_audit["parsed_rows"],
            "skipped_row_count": parse_audit["skipped_rows"],
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    field_path = output_dir / "phase132_matbench_jdft2d_field_table.csv"
    split_path = output_dir / "phase132_matbench_jdft2d_split_manifest.json"
    metric_path = output_dir / "phase132_matbench_jdft2d_metric_table.csv"
    review_path = output_dir / "phase132_matbench_jdft2d_target_review_table.csv"
    schema_path = output_dir / "phase132_matbench_jdft2d_schema_table.csv"
    data_card_path = output_dir / "phase132_matbench_jdft2d_data_card.json"
    gate_path = output_dir / "phase132_matbench_jdft2d_gate.json"
    markdown_path = output_dir / "phase132_matbench_jdft2d.md"
    manifest_path = output_dir / "phase132_matbench_jdft2d_manifest.json"

    _write_csv(field_path, field_df.to_dict("records"), FIELD_FIELDS)
    _write_json(split_path, split_manifest)
    _write_csv(metric_path, metric_rows, phase130.METRIC_FIELDS)
    _write_csv(review_path, [review], REVIEW_FIELDS)
    _write_csv(schema_path, schema_rows, phase130.SCHEMA_FIELDS)
    _write_json(
        data_card_path,
        {
            "dataset": "matbench_jdft2d",
            "source_url": source_url,
            "raw_path": _display_path(raw_path, root),
            "byte_size": source_info["byte_size"],
            "sha256": source_info["sha256"],
            "target": TARGET_COLUMN,
            "input": "pymatgen Structure JSON",
            "row_count": int(len(field_df)),
            "raw_row_count": parse_audit["raw_rows"],
            "skipped_row_count": parse_audit["skipped_rows"],
            "skipped_row_examples": parse_audit["skipped_row_examples"],
            "phase132_no_training": True,
            "a100_reproduction_pending_due_to_connectivity": True,
        },
    )
    _write_json(gate_path, gate)
    markdown_path.write_text(build_markdown(gate, review), encoding="utf-8")

    manifest = {
        "phase": 132,
        "objective": "matbench_jdft2d_baseline_first_gate_no_training",
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
            "raw_rows": parse_audit["raw_rows"],
            "field_rows": int(len(field_df)),
            "skipped_rows": parse_audit["skipped_rows"],
            "metric_rows": len(metric_rows),
            "schema_rows": len(schema_rows),
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
