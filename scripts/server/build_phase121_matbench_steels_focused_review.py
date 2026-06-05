#!/usr/bin/env python3
"""Build Phase 121 focused review for the Matbench steels gate.

This phase consumes only small Phase 120 artifacts. It checks whether the
Phase 120 composition-property gap is stable under alternate grouped splits and
whether shortcut profiles based on composition identity or alloy-family labels
explain the signal. It does not train a neural model or open A100 training.
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


DEFAULT_PHASE120_DIR = Path("docs/results/phase120_matbench_steels_baseline_gate")
DEFAULT_OUTPUT_DIR = Path("docs/results/phase121_matbench_steels_focused_review")
MIN_SPLIT_ROWS = 20
MIN_RELATIVE_VAL_GAIN = 0.05
MIN_STABLE_SPLIT_PASS_RATE = 0.80
NEGATIVE_DOMINANCE_TOLERANCE = 1.02


def _load_phase120_module():
    script = Path(__file__).with_name("build_phase120_matbench_steels_baseline_gate.py")
    spec = importlib.util.spec_from_file_location("phase120_helpers", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Phase 120 helper module from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase120 = _load_phase120_module()

PROFILE_COLUMNS: dict[str, dict[str, Any]] = {
    **{
        name: {**profile, "role": "negative_control" if name == "composition_hash_shortcut" else "admissible"}
        for name, profile in phase120.PROFILE_COLUMNS.items()
    },
    "major_elements_only": {
        "role": "admissible",
        "numeric": ("frac_Fe", "frac_Cr", "frac_Ni", "frac_Co", "frac_Mo"),
        "categorical": (),
    },
    "minor_elements_only": {
        "role": "admissible",
        "numeric": ("frac_C", "frac_Mn", "frac_Si", "frac_V", "frac_N", "frac_Nb", "frac_W", "frac_Al", "frac_Ti"),
        "categorical": (),
    },
    "alloy_family_shortcut": {
        "role": "negative_control",
        "numeric": (),
        "categorical": ("alloy_family_key",),
    },
    "dominant_element_shortcut": {
        "role": "negative_control",
        "numeric": (),
        "categorical": ("dominant_non_fe_element",),
    },
}
MODEL_METHODS = phase120.MODEL_METHODS
SPLIT_PLAN = (
    ("phase120_registered_split", "phase120_manifest", "phase120"),
    ("alloy_family_hash_0", "group:alloy_family_key", "phase121_alloy_0"),
    ("alloy_family_hash_1", "group:alloy_family_key", "phase121_alloy_1"),
    ("alloy_family_hash_2", "group:alloy_family_key", "phase121_alloy_2"),
    ("alloy_family_hash_3", "group:alloy_family_key", "phase121_alloy_3"),
    ("alloy_family_hash_4", "group:alloy_family_key", "phase121_alloy_4"),
    ("dominant_element_hash", "group:dominant_non_fe_element", "phase121_dominant"),
    ("fe_ni_co_bins", "bins:frac_Fe,frac_Ni,frac_Co", "phase121_fe_ni_co"),
    ("cr_mo_ti_bins", "bins:frac_Cr,frac_Mo,frac_Ti", "phase121_cr_mo_ti"),
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
    "split_strategy",
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


def _group_split(df: pd.DataFrame, column: str, salt: str) -> dict[str, Any]:
    if column not in df.columns:
        return _blocked_split(f"group_hash_by_{column}", "group column missing")
    groups = sorted(str(value) for value in df[column].fillna("missing").unique())
    return _split_from_keys(df[column].fillna("missing").astype(str).tolist(), groups, f"group_hash_by_{column}", salt)


def _bin_split(df: pd.DataFrame, columns: tuple[str, ...], salt: str) -> dict[str, Any]:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        return _blocked_split(f"bins_{'_'.join(columns)}", f"missing columns: {missing}")
    keys: list[str] = []
    ranks = {column: df[column].rank(method="first").to_numpy(dtype=float) for column in columns}
    n_rows = len(df)
    for row_index in range(n_rows):
        parts = []
        for column in columns:
            quartile = int(min(3, max(0, math.floor((ranks[column][row_index] - 1.0) / n_rows * 4.0))))
            parts.append(f"{column}:{quartile}")
        keys.append("|".join(parts))
    groups = sorted(set(keys))
    return _split_from_keys(keys, groups, f"quartile_bins_by_{'_'.join(columns)}", salt)


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
    ranked = sorted(groups, key=lambda item: hashlib.sha256(f"{salt}::{item}".encode("utf-8")).hexdigest())
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


def _phase120_split(split_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "split_strategy": split_manifest.get("split_strategy", "phase120_registered_split"),
        "split_salt": "phase120",
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
        if split_spec == "phase120_manifest":
            reviews[split_id] = _phase120_split(split_manifest)
        elif split_spec.startswith("group:"):
            reviews[split_id] = _group_split(df, split_spec.split(":", 1)[1], salt)
        elif split_spec.startswith("bins:"):
            columns = tuple(split_spec.split(":", 1)[1].split(","))
            reviews[split_id] = _bin_split(df, columns, salt)
        else:
            raise ValueError(f"Unsupported split spec: {split_spec}")
    return reviews


def _one_hot_frame(train: pd.DataFrame, all_rows: pd.DataFrame, profile: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return phase120._one_hot_frame(train, all_rows, profile)


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
        }
    y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
    train_idx = splits["train"]
    train_std = float(np.std(y[train_idx])) if train_idx else 0.0
    train_mean = float(np.mean(y[train_idx]))
    mean_pred = np.full_like(y, train_mean, dtype=float)
    mean_val = phase120._metrics(y[splits["val"]], mean_pred[splits["val"]], train_std)
    mean_test = phase120._metrics(y[splits["test"]], mean_pred[splits["test"]], train_std)

    rows: list[dict[str, Any]] = []
    for profile_name, profile in PROFILE_COLUMNS.items():
        x_train, x_all = _one_hot_frame(df.iloc[train_idx], df, profile)
        y_train = y[train_idx]
        for method in MODEL_METHODS:
            pred = phase120._fit_predict(method, x_train, y_train, x_all)
            split_metrics = {
                split: phase120._metrics(y[splits[split]], pred[splits[split]], train_std)
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
                    "split_strategy": split_info.get("split_strategy"),
                    "profile": profile_name,
                    "profile_role": profile["role"],
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
                    "relative_val_gain": val_gain / float(mean_val["rmse"]) if float(mean_val["rmse"]) > 0 else None,
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
        float(best_negative["val_rmse"]) <= float(best_admissible["val_rmse"]) * NEGATIVE_DOMINANCE_TOLERANCE
        and float(best_negative["test_gain_vs_mean"]) > 0.0
    )
    summary = {
        **base,
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


def build_audit_rows(*, phase120_gate: dict[str, Any], split_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    phase120_ready = phase120_gate.get("status") == "phase120_matbench_steels_gap_ready_focused_review"
    rows.append(
        {
            "audit": "phase120_gate_status",
            "status": "pass" if phase120_ready else "block",
            "severity": "blocking" if not phase120_ready else "info",
            "value": phase120_gate.get("status"),
            "threshold": "phase120_matbench_steels_gap_ready_focused_review",
            "reason": "focused review requires a Phase 120 focused-review gate",
        }
    )
    original = next((row for row in split_rows if row.get("split_id") == "phase120_registered_split"), None)
    if original and original.get("split_viable"):
        rows.append(
            {
                "audit": "original_split_admissible_gain",
                "status": "pass" if _is_true(original.get("split_pass")) else "block",
                "severity": "blocking" if not _is_true(original.get("split_pass")) else "info",
                "value": original.get("best_admissible_relative_val_gain"),
                "threshold": MIN_RELATIVE_VAL_GAIN,
                "reason": "Phase 120 selected composition target must preserve validation and test gain",
            }
        )
        rows.append(
            {
                "audit": "original_split_shortcut_dominance",
                "status": "block" if _is_true(original.get("negative_control_dominates")) else "pass",
                "severity": "blocking" if _is_true(original.get("negative_control_dominates")) else "info",
                "value": original.get("best_negative_profile"),
                "threshold": f"negative val RMSE > admissible * {NEGATIVE_DOMINANCE_TOLERANCE}",
                "reason": "composition identity or family shortcuts must not dominate the selected profile",
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
                "reason": "Phase 120 registered split must be reviewable",
            }
        )
    viable = [row for row in split_rows if _is_true(row.get("split_viable"))]
    passed = [row for row in viable if _is_true(row.get("split_pass"))]
    stable_rate = len(passed) / len(viable) if viable else 0.0
    rows.append(
        {
            "audit": "split_sensitivity_pass_rate",
            "status": "pass" if stable_rate >= MIN_STABLE_SPLIT_PASS_RATE else "block",
            "severity": "blocking" if stable_rate < MIN_STABLE_SPLIT_PASS_RATE else "info",
            "value": stable_rate,
            "threshold": MIN_STABLE_SPLIT_PASS_RATE,
            "reason": "composition-profile gain must survive deterministic alloy-family and binned split perturbations",
        }
    )
    shortcut_dominant = [row for row in viable if _is_true(row.get("negative_control_dominates"))]
    rows.append(
        {
            "audit": "shortcut_dominant_split_count",
            "status": "block" if shortcut_dominant else "pass",
            "severity": "blocking" if shortcut_dominant else "info",
            "value": len(shortcut_dominant),
            "threshold": 0,
            "reason": "no viable split may be dominated by composition identity or family shortcut profiles",
        }
    )
    return rows


def build_gate(*, phase120_gate: dict[str, Any], split_rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    blockers = [row for row in audit_rows if row["status"] == "block"]
    viable = [row for row in split_rows if _is_true(row.get("split_viable"))]
    passed = [row for row in viable if _is_true(row.get("split_pass"))]
    shortcut_dominant = [row for row in viable if _is_true(row.get("negative_control_dominates"))]
    original = next((row for row in split_rows if row.get("split_id") == "phase120_registered_split"), {})
    if phase120_gate.get("status") != "phase120_matbench_steels_gap_ready_focused_review":
        status = "phase121_matbench_steels_review_blocked_by_phase120"
        mechanism_allowed = False
        next_action = "complete or close Phase 120 before focused review"
    elif blockers:
        status = "phase121_matbench_steels_focused_review_closed_split_sensitivity_or_shortcut"
        mechanism_allowed = False
        next_action = "close the Phase 120 target as diagnostic; do not train"
    else:
        status = "phase121_matbench_steels_focused_review_ready_low_capacity_mechanism_gate"
        mechanism_allowed = True
        next_action = "design a separate no-training low-capacity mechanism gate; keep model training closed"
    return {
        "status": status,
        "phase120_status": phase120_gate.get("status"),
        "selected_target": phase120_gate.get("selected_target", "yield_strength_mpa"),
        "phase120_selected_profile": phase120_gate.get("selected_profile"),
        "phase120_selected_method": phase120_gate.get("selected_method"),
        "viable_split_reviews": len(viable),
        "passed_split_reviews": len(passed),
        "split_pass_rate": len(passed) / len(viable) if viable else 0.0,
        "shortcut_dominant_splits": len(shortcut_dominant),
        "blocking_audit_rows": len(blockers),
        "blocking_audits": [row["audit"] for row in blockers],
        "original_best_admissible_profile": original.get("best_admissible_profile"),
        "original_best_admissible_method": original.get("best_admissible_method"),
        "original_best_admissible_val_rmse": original.get("best_admissible_val_rmse"),
        "original_best_admissible_test_rmse": original.get("best_admissible_test_rmse"),
        "original_best_negative_profile": original.get("best_negative_profile"),
        "original_best_negative_method": original.get("best_negative_method"),
        "phase121_model_mechanism_allowed": mechanism_allowed,
        "phase121_low_capacity_mechanism_design_allowed": mechanism_allowed,
        "phase121_model_training_allowed": False,
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
        "# Phase 121 Matbench Steels Focused Review",
        "",
        f"- Status: `{gate['status']}`",
        f"- Split pass rate: `{gate['split_pass_rate']:.6g}`",
        f"- Blocking audits: `{', '.join(gate['blocking_audits']) or 'none'}`",
        f"- Low-capacity mechanism design allowed: `{gate['phase121_low_capacity_mechanism_design_allowed']}`",
        f"- Model training allowed: `{gate['phase121_model_training_allowed']}`",
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
                ("Val RMSE", "best_admissible_val_rmse"),
                ("Test RMSE", "best_admissible_test_rmse"),
                ("Shortcut dominates", "negative_control_dominates"),
            ],
        ),
    ]
    return "\n".join(lines) + "\n"


def build_package(*, root: Path, phase120_dir: Path, output_dir: Path) -> dict[str, Any]:
    field_path = phase120_dir / "phase120_matbench_steels_field_table.csv"
    split_path = phase120_dir / "phase120_matbench_steels_split_manifest.json"
    gate_path = phase120_dir / "phase120_matbench_steels_gate.json"
    df = pd.read_csv(field_path)
    split_manifest = _read_json(split_path)
    phase120_gate = _read_json(gate_path)
    target = str(phase120_gate.get("selected_target") or "yield_strength_mpa")

    split_infos = build_split_reviews(df, split_manifest)
    profile_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for split_id, split_info in split_infos.items():
        rows, summary = evaluate_split(df, target=target, split_id=split_id, split_info=split_info)
        profile_rows.extend(rows)
        split_rows.append(summary)
    audit_rows = build_audit_rows(phase120_gate=phase120_gate, split_rows=split_rows)
    gate = build_gate(phase120_gate=phase120_gate, split_rows=split_rows, audit_rows=audit_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "phase121_matbench_steels_focused_profile_table.csv"
    split_review_path = output_dir / "phase121_matbench_steels_split_sensitivity_table.csv"
    audit_path = output_dir / "phase121_matbench_steels_shortcut_audit_table.csv"
    split_manifest_path = output_dir / "phase121_matbench_steels_split_review_manifest.json"
    gate_out = output_dir / "phase121_matbench_steels_focused_review_gate.json"
    markdown_path = output_dir / "phase121_matbench_steels_focused_review.md"
    manifest_path = output_dir / "phase121_matbench_steels_focused_review_manifest.json"

    _write_csv(profile_path, profile_rows, PROFILE_FIELDS)
    _write_csv(split_review_path, split_rows, SPLIT_FIELDS)
    _write_csv(audit_path, audit_rows, AUDIT_FIELDS)
    _write_json(split_manifest_path, split_infos)
    _write_json(gate_out, gate)
    markdown_path.write_text(build_markdown(gate, audit_rows, split_rows), encoding="utf-8")

    manifest = {
        "phase": 121,
        "objective": "matbench_steels_focused_split_shortcut_review_no_training",
        "inputs": {
            "phase120_dir": _display_path(phase120_dir, root),
            "field_table": _display_path(field_path, root),
            "split_manifest": _display_path(split_path, root),
            "phase120_gate": _display_path(gate_path, root),
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
    parser.add_argument("--phase120-dir", type=Path, default=DEFAULT_PHASE120_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    phase120_dir = args.phase120_dir if args.phase120_dir.is_absolute() else root / args.phase120_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = build_package(root=root, phase120_dir=phase120_dir, output_dir=output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
