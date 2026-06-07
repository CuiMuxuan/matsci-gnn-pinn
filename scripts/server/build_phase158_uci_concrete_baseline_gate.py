#!/usr/bin/env python3
"""Build Phase 158 baseline-first gate for UCI concrete strength data.

This phase opens a fresh small public source for the second-paper candidate
queue. It reviews leakage-safe concrete mix-design splits with strong tabular
baselines only; it does not train neural models or open A100/A800 training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
import urllib.request
import zipfile
from io import BytesIO, StringIO
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


SOURCE_URL = "https://cdn.uci-ics-mlr-prod.aws.uci.edu/165/concrete%2Bcompressive%2Bstrength.zip"
SOURCE_DOI = "10.24432/C5PK67"
DEFAULT_RAW_PATH = Path(
    "data/raw/external/phase158_uci_concrete/concrete_compressive_strength.zip"
)
DEFAULT_OUTPUT_DIR = Path("docs/results/phase158_uci_concrete_baseline_gate")

EXPECTED_MIN_BYTES = 30_000
MIN_ROWS_FOR_REVIEW = 800
MIN_SPLIT_ROWS = 100
MIN_RELATIVE_VAL_GAIN = 0.12
MIN_RELATIVE_TEST_GAIN = 0.05
SHORTCUT_DOMINANCE_TOLERANCE = 1.02
MODEL_METHODS = ("knn", "extra_trees", "hist_gradient_boosting")

RAW_INPUT_COLUMNS = (
    "cement_kg_m3",
    "blast_furnace_slag_kg_m3",
    "fly_ash_kg_m3",
    "water_kg_m3",
    "superplasticizer_kg_m3",
    "coarse_aggregate_kg_m3",
    "fine_aggregate_kg_m3",
    "age_day",
)
TARGET_COLUMN = "target_compressive_strength_mpa"

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
    "phase159_focused_review_allowed",
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
        if "Concrete_Data.xls" not in members and "Concrete_Data.csv" not in members:
            raise ValueError("Missing Concrete_Data.xls or Concrete_Data.csv in UCI concrete ZIP")
        readme_bytes = archive.getinfo("Concrete_Readme.txt").file_size if "Concrete_Readme.txt" in members else 0
        data_member = "Concrete_Data.xls" if "Concrete_Data.xls" in members else "Concrete_Data.csv"
        data_bytes = archive.getinfo(data_member).file_size
    return {
        "raw_path": str(path),
        "raw_bytes": size,
        "raw_sha256": _sha256(path),
        "zip_members": sorted(members),
        "data_member": data_member,
        "data_member_bytes": data_bytes,
        "readme_bytes": readme_bytes,
    }


def _stable_unit_hash(text: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def _stable_hash16(text: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{text}".encode("utf-8")).hexdigest()[:16]


def _sector_offset(sector_id: int, sector_size: int) -> int:
    return 512 + sector_id * sector_size


def _read_cfb_stream(data: bytes, names: tuple[str, ...] = ("Workbook", "Book")) -> bytes:
    """Read a named stream from a small Compound File Binary workbook."""

    if data[:8] != bytes.fromhex("d0cf11e0a1b11ae1"):
        raise ValueError("Concrete_Data.xls is not a Compound File Binary workbook")
    sector_size = 1 << struct.unpack_from("<H", data, 30)[0]
    num_fat_sectors = struct.unpack_from("<I", data, 44)[0]
    first_dir_sector = struct.unpack_from("<I", data, 48)[0]
    difat = list(struct.unpack_from("<109I", data, 76))
    fat_sector_ids = [sector for sector in difat if sector < 0xFFFFFFF0]
    fat: list[int] = []
    for sector_id in fat_sector_ids[:num_fat_sectors]:
        offset = _sector_offset(sector_id, sector_size)
        sector = data[offset : offset + sector_size]
        fat.extend(struct.unpack("<" + "I" * (sector_size // 4), sector))

    def read_chain(first_sector: int) -> bytes:
        out = bytearray()
        sector_id = first_sector
        seen: set[int] = set()
        while sector_id < 0xFFFFFFF0 and sector_id not in seen:
            seen.add(sector_id)
            offset = _sector_offset(sector_id, sector_size)
            out.extend(data[offset : offset + sector_size])
            sector_id = fat[sector_id]
        return bytes(out)

    directory = read_chain(first_dir_sector)
    streams: dict[str, tuple[int, int]] = {}
    for offset in range(0, len(directory), 128):
        entry = directory[offset : offset + 128]
        if len(entry) < 128:
            break
        name_len = struct.unpack_from("<H", entry, 64)[0]
        name = entry[: max(name_len - 2, 0)].decode("utf-16le", errors="ignore")
        first_sector = struct.unpack_from("<I", entry, 116)[0]
        stream_size = struct.unpack_from("<Q", entry, 120)[0]
        streams[name] = (first_sector, stream_size)

    for name in names:
        if name in streams:
            first_sector, stream_size = streams[name]
            return read_chain(first_sector)[:stream_size]
    raise ValueError(f"Workbook stream not found; available streams: {sorted(streams)}")


def _parse_biff_string(buffer: bytes | bytearray, offset: int) -> tuple[str, int]:
    char_count = struct.unpack_from("<H", buffer, offset)[0]
    offset += 2
    flags = buffer[offset]
    offset += 1
    is_utf16 = bool(flags & 0x01)
    has_ext = bool(flags & 0x04)
    has_rich = bool(flags & 0x08)
    rich_runs = 0
    ext_size = 0
    if has_rich:
        rich_runs = struct.unpack_from("<H", buffer, offset)[0]
        offset += 2
    if has_ext:
        ext_size = struct.unpack_from("<I", buffer, offset)[0]
        offset += 4
    byte_count = char_count * (2 if is_utf16 else 1)
    raw = buffer[offset : offset + byte_count]
    offset += byte_count
    text = raw.decode("utf-16le" if is_utf16 else "latin1")
    offset += rich_runs * 4 + ext_size
    return text, offset


def _decode_rk_number(encoded: int) -> float:
    divide_by_100 = bool(encoded & 0x01)
    is_integer = bool(encoded & 0x02)
    raw = encoded & 0xFFFFFFFC
    if is_integer:
        value = raw >> 2
        if value & (1 << 29):
            value -= 1 << 30
        decoded = float(value)
    else:
        decoded = struct.unpack("<d", struct.pack("<Q", raw << 32))[0]
    return decoded / 100.0 if divide_by_100 else decoded


def _read_uci_concrete_biff_table(xls_bytes: bytes) -> pd.DataFrame:
    """Parse the UCI BIFF8 workbook without depending on xlrd on the server."""

    workbook = _read_cfb_stream(xls_bytes)
    records: list[tuple[int, bytes]] = []
    offset = 0
    while offset + 4 <= len(workbook):
        record_type, length = struct.unpack_from("<HH", workbook, offset)
        offset += 4
        records.append((record_type, workbook[offset : offset + length]))
        offset += length

    shared_strings: list[str] = []
    index = 0
    while index < len(records):
        record_type, payload = records[index]
        if record_type == 0x00FC:
            blob = bytearray(payload)
            next_index = index + 1
            while next_index < len(records) and records[next_index][0] == 0x003C:
                blob.extend(records[next_index][1])
                next_index += 1
            _, unique_count = struct.unpack_from("<II", blob, 0)
            string_offset = 8
            for _ in range(unique_count):
                text, string_offset = _parse_biff_string(blob, string_offset)
                shared_strings.append(text)
            index = next_index
            continue
        index += 1

    cells: dict[tuple[int, int], Any] = {}
    for record_type, payload in records:
        if record_type == 0x0203 and len(payload) >= 14:
            row, col, _ = struct.unpack_from("<HHH", payload, 0)
            cells[(row, col)] = struct.unpack_from("<d", payload, 6)[0]
        elif record_type == 0x00FD and len(payload) >= 10:
            row, col, _, string_id = struct.unpack_from("<HHHI", payload, 0)
            cells[(row, col)] = shared_strings[string_id]
        elif record_type == 0x027E and len(payload) >= 10:
            row, col, _, value = struct.unpack_from("<HHHI", payload, 0)
            cells[(row, col)] = _decode_rk_number(value)
        elif record_type == 0x00BD and len(payload) >= 6:
            row, first_col = struct.unpack_from("<HH", payload, 0)
            last_col = struct.unpack_from("<H", payload, len(payload) - 2)[0]
            cell_offset = 4
            for col in range(first_col, last_col + 1):
                _, value = struct.unpack_from("<HI", payload, cell_offset)
                cell_offset += 6
                cells[(row, col)] = _decode_rk_number(value)

    if not cells:
        raise ValueError("No cells parsed from Concrete_Data.xls")
    max_row = max(row for row, _ in cells)
    max_col = max(col for _, col in cells)
    headers = [str(cells.get((0, col), "")).strip() for col in range(max_col + 1)]
    rows = [
        [cells.get((row, col), np.nan) for col in range(max_col + 1)]
        for row in range(1, max_row + 1)
    ]
    return pd.DataFrame(rows, columns=headers)


def _read_uci_concrete_table_from_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        if "Concrete_Data.csv" in archive.namelist():
            raw = archive.read("Concrete_Data.csv").decode("utf-8")
            return pd.read_csv(StringIO(raw))
        xls_bytes = archive.read("Concrete_Data.xls")
    try:
        return pd.read_excel(BytesIO(xls_bytes))
    except Exception:
        return _read_uci_concrete_biff_table(xls_bytes)


def _canonical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.shape[1] < 9:
        raise ValueError(f"Expected at least 9 concrete columns, found {frame.shape[1]}")
    column_map = dict(zip(frame.columns[:9], (*RAW_INPUT_COLUMNS, TARGET_COLUMN)))
    out = frame.rename(columns=column_map)[[*RAW_INPUT_COLUMNS, TARGET_COLUMN]].copy()
    for column in (*RAW_INPUT_COLUMNS, TARGET_COLUMN):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=[*RAW_INPUT_COLUMNS, TARGET_COLUMN]).reset_index(drop=True)
    return out


def load_concrete_table(path: Path) -> pd.DataFrame:
    out = _canonical_columns(_read_uci_concrete_table_from_zip(path))
    out.insert(0, "phase158_row_id", np.arange(len(out), dtype=int))
    binder = out["cement_kg_m3"] + out["blast_furnace_slag_kg_m3"] + out["fly_ash_kg_m3"]
    scm = out["blast_furnace_slag_kg_m3"] + out["fly_ash_kg_m3"]
    aggregate = out["coarse_aggregate_kg_m3"] + out["fine_aggregate_kg_m3"]
    paste = binder + out["water_kg_m3"] + out["superplasticizer_kg_m3"]
    total = binder + out["water_kg_m3"] + out["superplasticizer_kg_m3"] + aggregate
    safe_binder = binder.replace(0.0, np.nan)
    safe_fine = out["fine_aggregate_kg_m3"].replace(0.0, np.nan)
    out["binder_kg_m3"] = binder
    out["scm_kg_m3"] = scm
    out["aggregate_kg_m3"] = aggregate
    out["paste_kg_m3"] = paste
    out["total_mix_kg_m3"] = total
    out["water_binder_ratio"] = (out["water_kg_m3"] / safe_binder).fillna(0.0)
    out["superplasticizer_binder_ratio"] = (
        out["superplasticizer_kg_m3"] / safe_binder
    ).fillna(0.0)
    out["slag_binder_ratio"] = (out["blast_furnace_slag_kg_m3"] / safe_binder).fillna(0.0)
    out["fly_ash_binder_ratio"] = (out["fly_ash_kg_m3"] / safe_binder).fillna(0.0)
    out["cement_binder_ratio"] = (out["cement_kg_m3"] / safe_binder).fillna(0.0)
    out["aggregate_binder_ratio"] = (aggregate / safe_binder).fillna(0.0)
    out["coarse_fine_ratio"] = (out["coarse_aggregate_kg_m3"] / safe_fine).fillna(0.0)
    out["log_age_day"] = np.log1p(out["age_day"])
    out["sqrt_age_day"] = np.sqrt(out["age_day"].clip(lower=0.0))
    out["has_slag"] = (out["blast_furnace_slag_kg_m3"] > 0).astype(float)
    out["has_fly_ash"] = (out["fly_ash_kg_m3"] > 0).astype(float)
    out["has_superplasticizer"] = (out["superplasticizer_kg_m3"] > 0).astype(float)
    out["scm_fraction"] = (scm / safe_binder).fillna(0.0)
    out["mix_design_key"] = [
        "|".join(f"{float(row[column]):.3f}" for column in RAW_INPUT_COLUMNS[:-1])
        for _, row in out.iterrows()
    ]
    out["mix_design_hash"] = [
        _stable_unit_hash(value, "phase158_mix_hash") for value in out["mix_design_key"]
    ]
    out["mix_design_hash16"] = [
        _stable_hash16(value, "phase158_mix_hash16") for value in out["mix_design_key"]
    ]
    out["age_bucket"] = pd.cut(
        out["age_day"],
        bins=[-math.inf, 7, 14, 28, 56, 120, math.inf],
        labels=["le7", "le14", "le28", "le56", "le120", "gt120"],
    ).astype(str)
    out["row_order_fraction"] = np.arange(len(out), dtype=float) / max(len(out) - 1, 1)
    return out


def split_by_group(df: pd.DataFrame, *, group_column: str = "mix_design_key") -> dict[str, Any]:
    groups = sorted(str(value) for value in df[group_column].dropna().unique())
    split_groups = {"train": set(), "val": set(), "test": set()}
    for group in groups:
        value = _stable_unit_hash(group, "phase158_split")
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
        raise ValueError(f"Split too small for Phase 158 review: {counts}")
    return {
        "group_column": group_column,
        "group_count": len(groups),
        "split_groups": {key: sorted(values) for key, values in split_groups.items()},
        "assignments": assignments,
        "counts": counts,
    }


def profile_columns(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        "raw_mix_age": {
            "role": "admissible",
            "columns": tuple(column for column in RAW_INPUT_COLUMNS if column in df.columns),
        },
        "binder_ratio_age_core": {
            "role": "admissible",
            "columns": (
                "binder_kg_m3",
                "scm_kg_m3",
                "water_binder_ratio",
                "superplasticizer_binder_ratio",
                "slag_binder_ratio",
                "fly_ash_binder_ratio",
                "aggregate_binder_ratio",
                "log_age_day",
                "sqrt_age_day",
            ),
        },
        "full_concrete_features": {
            "role": "admissible",
            "columns": (
                *RAW_INPUT_COLUMNS,
                "binder_kg_m3",
                "scm_kg_m3",
                "aggregate_kg_m3",
                "paste_kg_m3",
                "total_mix_kg_m3",
                "water_binder_ratio",
                "superplasticizer_binder_ratio",
                "slag_binder_ratio",
                "fly_ash_binder_ratio",
                "cement_binder_ratio",
                "aggregate_binder_ratio",
                "coarse_fine_ratio",
                "log_age_day",
                "sqrt_age_day",
                "has_slag",
                "has_fly_ash",
                "has_superplasticizer",
                "scm_fraction",
            ),
        },
        "age_only_control": {
            "role": "shortcut_control",
            "columns": ("age_day", "log_age_day", "sqrt_age_day"),
        },
        "coarse_mix_presence_control": {
            "role": "shortcut_control",
            "columns": (
                "has_slag",
                "has_fly_ash",
                "has_superplasticizer",
                "age_day",
                "total_mix_kg_m3",
            ),
        },
        "mix_design_hash_control": {
            "role": "shortcut_control",
            "columns": ("mix_design_hash", "age_day"),
        },
        "row_order_control": {"role": "shortcut_control", "columns": ("row_order_fraction",)},
    }


def _numeric_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    return [column for column in columns if column in df.columns]


def _model(method: str):
    if method == "knn":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=7, weights="distance", algorithm="brute"),
        )
    if method == "extra_trees":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=128,
                min_samples_leaf=2,
                random_state=158,
                n_jobs=1,
            ),
        )
    if method == "hist_gradient_boosting":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                max_iter=140,
                learning_rate=0.06,
                l2_regularization=0.01,
                random_state=158,
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
        blocker = "strong admissible baseline does not beat mean by required validation/test margins"
    elif shortcut_dominant:
        blocker = "shortcut control is too close to or better than selected admissible profile"
    status = (
        "phase158_uci_concrete_ready_focused_review"
        if focused_allowed
        else "phase158_uci_concrete_closed_no_stable_guarded_gap"
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
            "phase159_focused_review_allowed": focused_allowed,
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
        return f"{value:.4g}"
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


def build_gate(*, overview: dict[str, Any], review_rows: list[dict[str, Any]]) -> dict[str, Any]:
    review = review_rows[0]
    focused = bool(review["phase159_focused_review_allowed"])
    return {
        "status": review["status"],
        "source": "UCI Concrete Compressive Strength",
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
        "phase159_focused_review_allowed": focused,
        "phase158_model_mechanism_allowed": False,
        "phase158_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": (
            "run Phase 159 split/shortcut focused review before any mechanism"
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
        "# Phase 158 UCI Concrete Baseline Gate",
        "",
        "## Gate",
        f"- Status: `{gate['status']}`",
        f"- Phase 159 focused review allowed: `{_csv_value(gate['phase159_focused_review_allowed'])}`",
        f"- Model training allowed: `{_csv_value(gate['phase158_model_training_allowed'])}`",
        f"- A100 training allowed now: `{_csv_value(gate['a100_training_allowed_now'])}`",
        f"- A100-SXM4-80GB request now: `{_csv_value(gate['a100_80gb_request_now'])}`",
        "",
        "## Interpretation",
        (
            "This is a no-training baseline-first intake for a possible second-paper "
            "positive mainline. The split groups rows by concrete mix design, excluding "
            "age, so the same formulation cannot appear across train/validation/test. "
            "A positive gate can only open a focused split/shortcut review."
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
    df = load_concrete_table(raw_path)
    if len(df) < MIN_ROWS_FOR_REVIEW:
        raise ValueError(f"Too few rows for Phase 158 review: {len(df)}")
    split = split_by_group(df)
    metric_rows, review_rows = evaluate_baselines(df, split["assignments"])
    overview = {
        "source_id": "phase158_uci_concrete",
        "source_url": source_url,
        "source_doi": SOURCE_DOI,
        "raw_path": _display_path(raw_path, root),
        "raw_bytes": source_info["raw_bytes"],
        "raw_sha256": source_info["raw_sha256"],
        "field_rows": int(len(df)),
        "feature_columns": int(sum(1 for column in df.columns if pd.api.types.is_numeric_dtype(df[column]))),
        "target": TARGET_COLUMN,
        "group_column": split["group_column"],
        "group_count": int(split["group_count"]),
        "train_rows_split": int(split["counts"]["train"]),
        "val_rows_split": int(split["counts"]["val"]),
        "test_rows_split": int(split["counts"]["test"]),
    }
    gate = build_gate(overview=overview, review_rows=review_rows)

    overview_rows = [overview]
    overview_path = output_dir / "phase158_source_overview_table.csv"
    metric_path = output_dir / "phase158_baseline_metric_table.csv"
    review_path = output_dir / "phase158_baseline_review_table.csv"
    split_path = output_dir / "phase158_split_manifest.json"
    gate_path = output_dir / "phase158_uci_concrete_baseline_gate.json"
    markdown_path = output_dir / "phase158_uci_concrete_baseline_gate.md"
    manifest_path = output_dir / "phase158_uci_concrete_baseline_manifest.json"

    _write_csv(overview_path, overview_rows, OVERVIEW_FIELDS)
    _write_csv(metric_path, metric_rows, METRIC_FIELDS)
    _write_csv(review_path, review_rows, REVIEW_FIELDS)
    _write_json(
        split_path,
        {
            "phase": 158,
            "source": "UCI Concrete Compressive Strength",
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
        "phase": 158,
        "description": "baseline-first intake for UCI concrete compressive strength",
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
