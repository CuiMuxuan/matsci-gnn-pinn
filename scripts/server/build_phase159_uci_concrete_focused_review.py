#!/usr/bin/env python3
"""Build Phase 159 focused review for the UCI concrete baseline gate.

This phase audits whether the Phase 158 concrete-strength baseline gap survives
alternate mix-design splits, shortcut controls, nearest-neighbor identity
controls, and target-distribution shifts. It may only open a later no-training
low-capacity mechanism gate; it must not train a neural model.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


DEFAULT_PHASE158_DIR = Path("docs/results/phase158_uci_concrete_baseline_gate")
DEFAULT_RAW_PATH = Path(
    "data/raw/external/phase158_uci_concrete/concrete_compressive_strength.zip"
)
DEFAULT_OUTPUT_DIR = Path("docs/results/phase159_uci_concrete_focused_review")

MIN_SPLIT_ROWS = 100
MIN_STABLE_SPLIT_PASS_RATE = 0.75
MAX_TARGET_DISTRIBUTION_SHIFT_Z = 0.85
SHORTCUT_DOMINANCE_TOLERANCE = 1.02
NEAREST_NEIGHBOR_DOMINANCE_TOLERANCE = 1.02
NEAR_DUPLICATE_DISTANCE = 0.05

SPLIT_SPECS = (
    ("phase158_registered_mix_design", "mix_design_hash", "phase158_split", True),
    ("mix_design_hash_0", "mix_design_hash", "phase159_mix_design_0", True),
    ("mix_design_hash_1", "mix_design_hash", "phase159_mix_design_1", True),
    ("mix_design_hash_2", "mix_design_hash", "phase159_mix_design_2", True),
    ("mix_design_hash_3", "mix_design_hash", "phase159_mix_design_3", True),
    ("mix_design_hash_4", "mix_design_hash", "phase159_mix_design_4", True),
    ("age_bucket_holdout", "group:age_bucket", "phase159_age_bucket", False),
    ("water_binder_bins", "bins:water_binder_ratio", "phase159_water_binder", False),
    ("binder_mass_bins", "bins:binder_kg_m3", "phase159_binder_mass", False),
)

SPLIT_FIELDS = (
    "split_id",
    "split_strategy",
    "gate_review_split",
    "split_viable",
    "reason",
    "train_rows",
    "val_rows",
    "test_rows",
    "group_count",
    "target_distribution_shift_z",
    "mean_validation_rmse",
    "mean_test_rmse",
    "best_admissible_profile",
    "best_admissible_method",
    "best_admissible_validation_rmse",
    "best_admissible_test_rmse",
    "validation_relative_improvement_over_mean",
    "test_relative_improvement_over_mean",
    "baseline_visible_gap",
    "best_shortcut_profile",
    "best_shortcut_method",
    "best_shortcut_validation_rmse",
    "best_shortcut_test_rmse",
    "shortcut_dominant",
    "nearest_neighbor_validation_rmse",
    "nearest_neighbor_test_rmse",
    "nearest_neighbor_dominant",
    "val_near_duplicate_fraction",
    "test_near_duplicate_fraction",
    "phase159_split_pass",
)

METRIC_FIELDS = (
    "split_id",
    "profile",
    "method",
    "role",
    "split",
    "rmse",
    "mae",
    "r2",
    "n_rows",
)

AUDIT_FIELDS = (
    "audit_id",
    "audit",
    "status",
    "severity",
    "value",
    "threshold",
    "reason",
)


def _load_phase158_module():
    script = Path(__file__).with_name("build_phase158_uci_concrete_baseline_gate.py")
    spec = importlib.util.spec_from_file_location("phase158_helpers", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Phase 158 helper module from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


p158 = _load_phase158_module()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _csv_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.8g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
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


def _split_from_keys(keys: list[str], *, salt: str, strategy: str) -> dict[str, Any]:
    groups = sorted(set(keys))
    if len(groups) < 3:
        return {
            "split_strategy": strategy,
            "split_salt": salt,
            "split_viable": False,
            "reason": "fewer than three groups",
            "assignments": [],
            "group_count": len(groups),
            "split_groups": {"train": [], "val": [], "test": []},
        }
    split_groups = {"train": set(), "val": set(), "test": set()}
    for group in groups:
        value = p158._stable_unit_hash(group, salt)
        if value < 0.60:
            split_groups["train"].add(group)
        elif value < 0.80:
            split_groups["val"].add(group)
        else:
            split_groups["test"].add(group)
    assignments = [
        "train" if key in split_groups["train"] else "val" if key in split_groups["val"] else "test"
        for key in keys
    ]
    counts = {split: assignments.count(split) for split in ("train", "val", "test")}
    return {
        "split_strategy": strategy,
        "split_salt": salt,
        "split_viable": min(counts.values()) >= MIN_SPLIT_ROWS,
        "reason": "ok" if min(counts.values()) >= MIN_SPLIT_ROWS else "one or more splits below minimum row count",
        "assignments": assignments,
        "counts": counts,
        "group_count": len(groups),
        "split_groups": {key: sorted(values) for key, values in split_groups.items()},
        "leakage_safe": sum(len(values) for values in split_groups.values()) == len(groups),
    }


def _mix_design_split(df: pd.DataFrame, salt: str) -> dict[str, Any]:
    keys = df["mix_design_key"].fillna("unknown").astype(str).tolist()
    return _split_from_keys(keys, salt=salt, strategy="group_hash_by_mix_design_key")


def _group_split(df: pd.DataFrame, column: str, salt: str) -> dict[str, Any]:
    if column not in df.columns:
        return {
            "split_strategy": f"group_hash_by_{column}",
            "split_salt": salt,
            "split_viable": False,
            "reason": f"missing group column: {column}",
            "assignments": [],
            "group_count": 0,
            "split_groups": {"train": [], "val": [], "test": []},
        }
    keys = df[column].fillna("unknown").astype(str).tolist()
    return _split_from_keys(keys, salt=salt, strategy=f"group_hash_by_{column}")


def _bin_split(df: pd.DataFrame, column: str, salt: str) -> dict[str, Any]:
    if column not in df.columns:
        return {
            "split_strategy": f"quartile_bins_by_{column}",
            "split_salt": salt,
            "split_viable": False,
            "reason": f"missing bin column: {column}",
            "assignments": [],
            "group_count": 0,
            "split_groups": {"train": [], "val": [], "test": []},
        }
    values = pd.to_numeric(df[column], errors="coerce")
    ranks = values.rank(method="first").to_numpy(dtype=float)
    n_rows = len(df)
    keys = [
        f"{column}:q{int(min(3, max(0, math.floor((rank - 1.0) / max(n_rows, 1) * 4.0))))}"
        for rank in ranks
    ]
    return _split_from_keys(keys, salt=salt, strategy=f"quartile_bins_by_{column}")


def _build_split_info(df: pd.DataFrame, spec: tuple[str, str, str, bool]) -> dict[str, Any]:
    split_id, strategy, salt, gate_review = spec
    if strategy == "mix_design_hash":
        info = _mix_design_split(df, salt)
    elif strategy.startswith("group:"):
        info = _group_split(df, strategy.split(":", 1)[1], salt)
    elif strategy.startswith("bins:"):
        info = _bin_split(df, strategy.split(":", 1)[1], salt)
    else:
        raise ValueError(f"Unsupported split strategy: {strategy}")
    info["split_id"] = split_id
    info["gate_review_split"] = gate_review
    return info


def _target_distribution_shift_z(df: pd.DataFrame, assignments: list[str]) -> float:
    y = df[p158.TARGET_COLUMN].to_numpy(dtype=float)
    indices = {
        split: np.array([idx for idx, label in enumerate(assignments) if label == split], dtype=int)
        for split in ("train", "val", "test")
    }
    train = y[indices["train"]]
    train_std = float(np.std(train))
    if train_std <= 0.0:
        return 0.0
    train_stats = {
        "mean": float(np.mean(train)),
        "median": float(np.median(train)),
        "q90": float(np.quantile(train, 0.9)),
    }
    max_shift = 0.0
    for split in ("val", "test"):
        values = y[indices[split]]
        if len(values) == 0:
            continue
        split_stats = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "q90": float(np.quantile(values, 0.9)),
        }
        for key, train_value in train_stats.items():
            max_shift = max(max_shift, abs(split_stats[key] - train_value) / train_std)
    return float(max_shift)


def _nearest_neighbor_control(df: pd.DataFrame, assignments: list[str]) -> dict[str, Any]:
    y = df[p158.TARGET_COLUMN].to_numpy(dtype=float)
    columns = list(p158.RAW_INPUT_COLUMNS)
    x = df[columns].to_numpy(dtype=float)
    indices = {
        split: np.array([idx for idx, label in enumerate(assignments) if label == split], dtype=int)
        for split in ("train", "val", "test")
    }
    train_idx = indices["train"]
    means = x[train_idx].mean(axis=0)
    stds = x[train_idx].std(axis=0)
    stds[stds == 0.0] = 1.0
    z = (x - means) / stds
    model = NearestNeighbors(n_neighbors=1, algorithm="brute", metric="manhattan")
    model.fit(z[train_idx])
    distances, neighbors = model.kneighbors(z)
    pred = y[train_idx][neighbors[:, 0]]

    def rmse(split: str) -> float:
        idx = indices[split]
        return float(math.sqrt(np.mean((y[idx] - pred[idx]) ** 2)))

    return {
        "nearest_neighbor_validation_rmse": rmse("val"),
        "nearest_neighbor_test_rmse": rmse("test"),
        "val_near_duplicate_fraction": float(np.mean(distances[indices["val"], 0] <= NEAR_DUPLICATE_DISTANCE)),
        "test_near_duplicate_fraction": float(np.mean(distances[indices["test"], 0] <= NEAR_DUPLICATE_DISTANCE)),
    }


def _review_for_split(
    df: pd.DataFrame,
    split_info: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    split_id = split_info["split_id"]
    base = {
        "split_id": split_id,
        "split_strategy": split_info.get("split_strategy"),
        "gate_review_split": bool(split_info.get("gate_review_split")),
        "split_viable": bool(split_info.get("split_viable")),
        "reason": split_info.get("reason"),
        "train_rows": split_info.get("counts", {}).get("train", 0),
        "val_rows": split_info.get("counts", {}).get("val", 0),
        "test_rows": split_info.get("counts", {}).get("test", 0),
        "group_count": split_info.get("group_count", 0),
    }
    if not split_info.get("split_viable"):
        return [], {**base, "phase159_split_pass": False}, split_info

    metric_rows, review_rows = p158.evaluate_baselines(df, split_info["assignments"])
    metric_rows = [{**row, "split_id": split_id} for row in metric_rows]
    review = review_rows[0]
    nearest = _nearest_neighbor_control(df, split_info["assignments"])
    nearest_dominant = (
        nearest["nearest_neighbor_validation_rmse"]
        <= review["selected_validation_rmse"] * NEAREST_NEIGHBOR_DOMINANCE_TOLERANCE
        or nearest["nearest_neighbor_test_rmse"]
        <= review["selected_test_rmse"] * NEAREST_NEIGHBOR_DOMINANCE_TOLERANCE
    )
    target_shift = _target_distribution_shift_z(df, split_info["assignments"])
    split_pass = bool(
        review["baseline_visible_gap"]
        and not review["shortcut_dominant"]
        and not nearest_dominant
        and target_shift <= MAX_TARGET_DISTRIBUTION_SHIFT_Z
    )
    row = {
        **base,
        "target_distribution_shift_z": target_shift,
        "mean_validation_rmse": review["mean_validation_rmse"],
        "mean_test_rmse": review["mean_test_rmse"],
        "best_admissible_profile": review["selected_profile"],
        "best_admissible_method": review["selected_method"],
        "best_admissible_validation_rmse": review["selected_validation_rmse"],
        "best_admissible_test_rmse": review["selected_test_rmse"],
        "validation_relative_improvement_over_mean": review[
            "validation_relative_improvement_over_mean"
        ],
        "test_relative_improvement_over_mean": review["test_relative_improvement_over_mean"],
        "baseline_visible_gap": review["baseline_visible_gap"],
        "best_shortcut_profile": review["best_shortcut_profile"],
        "best_shortcut_method": review["best_shortcut_method"],
        "best_shortcut_validation_rmse": review["best_shortcut_validation_rmse"],
        "best_shortcut_test_rmse": review["best_shortcut_test_rmse"],
        "shortcut_dominant": review["shortcut_dominant"],
        **nearest,
        "nearest_neighbor_dominant": nearest_dominant,
        "phase159_split_pass": split_pass,
    }
    return metric_rows, row, split_info


def build_split_reviews(df: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    metric_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    split_manifest: dict[str, Any] = {}
    for spec in SPLIT_SPECS:
        split_info = _build_split_info(df, spec)
        metrics, row, manifest_entry = _review_for_split(df, split_info)
        metric_rows.extend(metrics)
        split_rows.append(row)
        split_manifest[row["split_id"]] = {
            key: value for key, value in manifest_entry.items() if key != "assignments"
        }
    return metric_rows, split_rows, split_manifest


def _audit_row(
    *,
    audit_id: str,
    audit: str,
    status: str,
    severity: str,
    value: Any,
    threshold: Any,
    reason: str,
) -> dict[str, Any]:
    return {
        "audit_id": audit_id,
        "audit": audit,
        "status": status,
        "severity": severity,
        "value": value,
        "threshold": threshold,
        "reason": reason,
    }


def build_audit_rows(*, phase158_gate: dict[str, Any], split_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gate_ready = (
        phase158_gate.get("status") == "phase158_uci_concrete_ready_focused_review"
        and bool(phase158_gate.get("phase159_focused_review_allowed"))
    )
    rows.append(
        _audit_row(
            audit_id="P159-AUDIT-001",
            audit="phase158_gate_consistency",
            status="pass" if gate_ready else "block",
            severity="blocking" if not gate_ready else "info",
            value=phase158_gate.get("status"),
            threshold="phase158_uci_concrete_ready_focused_review",
            reason="Phase 159 requires a Phase 158 baseline-first gate that allowed focused review",
        )
    )
    gate_rows = [row for row in split_rows if row.get("gate_review_split") and row.get("split_viable")]
    passed = [row for row in gate_rows if row.get("phase159_split_pass")]
    stable_rate = float(len(passed) / len(gate_rows)) if gate_rows else 0.0
    rows.append(
        _audit_row(
            audit_id="P159-AUDIT-002",
            audit="stable_mix_design_split_pass_rate",
            status="pass" if stable_rate >= MIN_STABLE_SPLIT_PASS_RATE else "block",
            severity="blocking" if stable_rate < MIN_STABLE_SPLIT_PASS_RATE else "info",
            value=stable_rate,
            threshold=MIN_STABLE_SPLIT_PASS_RATE,
            reason="baseline-visible signal should survive most leakage-safe mix-design splits",
        )
    )
    registered = next(
        (row for row in split_rows if row.get("split_id") == "phase158_registered_mix_design"),
        None,
    )
    registered_pass = bool(registered and registered.get("phase159_split_pass"))
    rows.append(
        _audit_row(
            audit_id="P159-AUDIT-003",
            audit="registered_split_replay",
            status="pass" if registered_pass else "block",
            severity="blocking" if not registered_pass else "info",
            value=registered.get("phase159_split_pass") if registered else None,
            threshold=True,
            reason="the Phase 158 registered split must replay as a guarded candidate",
        )
    )
    shortcut_dominant = [row for row in gate_rows if row.get("shortcut_dominant")]
    rows.append(
        _audit_row(
            audit_id="P159-AUDIT-004",
            audit="shortcut_dominant_split_count",
            status="block" if shortcut_dominant else "pass",
            severity="blocking" if shortcut_dominant else "info",
            value=len(shortcut_dominant),
            threshold=0,
            reason="age-only, coarse mix, mix-design hash, or row-order controls must not dominate",
        )
    )
    nearest_dominant = [row for row in gate_rows if row.get("nearest_neighbor_dominant")]
    rows.append(
        _audit_row(
            audit_id="P159-AUDIT-005",
            audit="nearest_neighbor_dominant_split_count",
            status="block" if nearest_dominant else "pass",
            severity="blocking" if nearest_dominant else "info",
            value=len(nearest_dominant),
            threshold=0,
            reason="nearest-neighbor concrete mixture identity control must not explain the signal",
        )
    )
    shifted = [
        row
        for row in gate_rows
        if float(row.get("target_distribution_shift_z") or 0.0) > MAX_TARGET_DISTRIBUTION_SHIFT_Z
    ]
    rows.append(
        _audit_row(
            audit_id="P159-AUDIT-006",
            audit="target_distribution_imbalanced_split_count",
            status="block" if shifted else "pass",
            severity="blocking" if shifted else "info",
            value=len(shifted),
            threshold=0,
            reason="target mean/median/q90 shifts must not dominate focused-review interpretation",
        )
    )
    return rows


def build_gate(
    *,
    phase158_gate: dict[str, Any],
    split_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    gate_rows = [row for row in split_rows if row.get("gate_review_split") and row.get("split_viable")]
    passed = [row for row in gate_rows if row.get("phase159_split_pass")]
    stable_rate = float(len(passed) / len(gate_rows)) if gate_rows else 0.0
    blocking = [row for row in audit_rows if row["severity"] == "blocking" and row["status"] == "block"]
    mechanism_allowed = not blocking
    status = (
        "phase159_uci_concrete_focused_review_ready_low_capacity_mechanism_gate"
        if mechanism_allowed
        else "phase159_uci_concrete_focused_review_closed_split_sensitivity_or_shortcut"
    )
    registered = next(row for row in split_rows if row["split_id"] == "phase158_registered_mix_design")
    return {
        "status": status,
        "source": phase158_gate.get("source"),
        "source_doi": phase158_gate.get("source_doi"),
        "selected_target": phase158_gate.get("selected_target"),
        "phase158_selected_profile": phase158_gate.get("selected_profile"),
        "phase158_selected_method": phase158_gate.get("selected_method"),
        "phase158_validation_rmse": phase158_gate.get("selected_validation_rmse"),
        "phase158_test_rmse": phase158_gate.get("selected_test_rmse"),
        "registered_replay_profile": registered.get("best_admissible_profile"),
        "registered_replay_method": registered.get("best_admissible_method"),
        "registered_replay_validation_rmse": registered.get("best_admissible_validation_rmse"),
        "registered_replay_test_rmse": registered.get("best_admissible_test_rmse"),
        "viable_split_reviews": len(gate_rows),
        "passed_split_reviews": len(passed),
        "split_pass_rate": stable_rate,
        "shortcut_dominant_splits": sum(1 for row in gate_rows if row.get("shortcut_dominant")),
        "nearest_neighbor_dominant_splits": sum(
            1 for row in gate_rows if row.get("nearest_neighbor_dominant")
        ),
        "target_distribution_imbalanced_splits": sum(
            1
            for row in gate_rows
            if float(row.get("target_distribution_shift_z") or 0.0) > MAX_TARGET_DISTRIBUTION_SHIFT_Z
        ),
        "blocking_audits": [row["audit"] for row in blocking],
        "phase159_model_mechanism_allowed": mechanism_allowed,
        "phase159_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": (
            "run Phase 160 no-training low-capacity concrete mechanism gate before any neural training"
            if mechanism_allowed
            else "close Phase 158/159 as diagnostic or choose another second-paper source"
        ),
    }


def _markdown_table(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[str]:
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    body = [
        "| " + " | ".join(_csv_value(row.get(field)) for field in fields) + " |"
        for row in rows
    ]
    return [header, sep, *body]


def build_markdown(
    *,
    gate: dict[str, Any],
    split_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
) -> str:
    split_fields = (
        "split_id",
        "gate_review_split",
        "best_admissible_profile",
        "best_admissible_method",
        "best_admissible_validation_rmse",
        "best_admissible_test_rmse",
        "shortcut_dominant",
        "nearest_neighbor_dominant",
        "target_distribution_shift_z",
        "phase159_split_pass",
    )
    audit_fields = ("audit", "status", "severity", "value", "threshold", "reason")
    lines = [
        "# Phase 159 UCI Concrete Focused Review",
        "",
        "## Gate",
        f"- Status: `{gate['status']}`",
        f"- Viable split reviews: `{gate['viable_split_reviews']}`",
        f"- Split pass rate: `{_csv_value(gate['split_pass_rate'])}`",
        f"- Model mechanism allowed: `{_csv_value(gate['phase159_model_mechanism_allowed'])}`",
        f"- Model training allowed: `{_csv_value(gate['phase159_model_training_allowed'])}`",
        f"- A100 training allowed now: `{_csv_value(gate['a100_training_allowed_now'])}`",
        f"- A100-SXM4-80GB request now: `{_csv_value(gate['a100_80gb_request_now'])}`",
        "",
        "## Interpretation",
        (
            "This focused review checks whether the Phase 158 concrete source gate is "
            "stable enough to justify a later no-training low-capacity mechanism gate. "
            "It does not train a neural model or support a second-paper model claim by itself."
        ),
        "",
        "## Split Review",
        *_markdown_table(split_rows, split_fields),
        "",
        "## Audits",
        *_markdown_table(audit_rows, audit_fields),
        "",
    ]
    return "\n".join(lines)


def build_package(
    *,
    root: Path,
    phase158_dir: Path,
    raw_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    root = root.resolve()
    phase158_dir = phase158_dir if phase158_dir.is_absolute() else root / phase158_dir
    raw_path = raw_path if raw_path.is_absolute() else root / raw_path
    output_dir = output_dir if output_dir.is_absolute() else root / output_dir

    phase158_gate = _read_json(phase158_dir / "phase158_uci_concrete_baseline_gate.json")
    if not raw_path.exists():
        manifest_path = phase158_dir / "phase158_uci_concrete_baseline_manifest.json"
        if manifest_path.exists():
            source_path = _read_json(manifest_path).get("source_info", {}).get("raw_path")
            if source_path and Path(source_path).exists():
                raw_path = Path(source_path)
    df = p158.load_concrete_table(raw_path)
    metric_rows, split_rows, split_manifest = build_split_reviews(df)
    audit_rows = build_audit_rows(phase158_gate=phase158_gate, split_rows=split_rows)
    gate = build_gate(phase158_gate=phase158_gate, split_rows=split_rows, audit_rows=audit_rows)

    split_table_path = output_dir / "phase159_uci_concrete_split_review_table.csv"
    metric_table_path = output_dir / "phase159_uci_concrete_metric_table.csv"
    audit_table_path = output_dir / "phase159_uci_concrete_audit_table.csv"
    split_manifest_path = output_dir / "phase159_uci_concrete_split_manifest.json"
    gate_path = output_dir / "phase159_uci_concrete_focused_review_gate.json"
    markdown_path = output_dir / "phase159_uci_concrete_focused_review.md"
    manifest_path = output_dir / "phase159_uci_concrete_focused_review_manifest.json"

    _write_csv(split_table_path, split_rows, SPLIT_FIELDS)
    _write_csv(metric_table_path, metric_rows, METRIC_FIELDS)
    _write_csv(audit_table_path, audit_rows, AUDIT_FIELDS)
    _write_json(split_manifest_path, {"phase": 159, "splits": split_manifest})
    _write_json(gate_path, gate)
    with markdown_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(build_markdown(gate=gate, split_rows=split_rows, audit_rows=audit_rows))
    manifest = {
        "phase": 159,
        "description": "focused split/shortcut review for Phase 158 UCI concrete gate",
        "inputs": {
            "phase158_dir": _display_path(phase158_dir, root),
            "raw_path": _display_path(raw_path, root),
        },
        "outputs": {
            "split_review_table": _display_path(split_table_path, root),
            "metric_table": _display_path(metric_table_path, root),
            "audit_table": _display_path(audit_table_path, root),
            "split_manifest": _display_path(split_manifest_path, root),
            "gate": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "field_rows": int(len(df)),
            "split_review_rows": len(split_rows),
            "gate_review_rows": sum(1 for row in split_rows if row.get("gate_review_split")),
            "metric_rows": len(metric_rows),
            "audit_rows": len(audit_rows),
        },
        "gate": gate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--phase158-dir", type=Path, default=DEFAULT_PHASE158_DIR)
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_package(
        root=args.root,
        phase158_dir=args.phase158_dir,
        raw_path=args.raw_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
