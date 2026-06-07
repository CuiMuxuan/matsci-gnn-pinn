#!/usr/bin/env python3
"""Build Phase 142 focused review for the Matbench experimental is-metal gate.

This phase consumes only small Phase 141 artifacts. It checks whether the
Phase 141 experimental metallicity classification gap is stable under alternate
grouped/binned splits and whether shortcut profiles, nearest-neighbor
composition identity, or class-balance shifts explain the signal. It does not
train a neural model or open A100/A800 training.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PHASE141_DIR = Path("docs/results/phase141_matbench_expt_is_metal_baseline_gate")
DEFAULT_OUTPUT_DIR = Path("docs/results/phase142_matbench_expt_is_metal_focused_review")
TARGET = "is_metal"


def _load_module(filename: str, module_name: str):
    script = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase141 = _load_module("build_phase141_matbench_expt_is_metal_baseline_gate.py", "phase141_helpers")
phase139 = _load_module("build_phase139_matbench_glass_focused_review.py", "phase139_review_helpers")

phase139.phase138 = phase141
phase139.TARGET = TARGET
phase139.PROFILE_COLUMNS = dict(phase141.PROFILE_COLUMNS)
phase139.MODEL_METHODS = phase141.MODEL_METHODS
phase139.PROFILE_METHODS = dict(phase141.PROFILE_METHODS)
phase139.NN_FEATURE_COLUMNS = tuple(
    column
    for column in (
        *(f"frac_{element}" for element in phase141.ELEMENTS),
        "element_count",
        "entropy_fraction",
        "max_fraction",
        "anion_fraction",
        "oxygen_fraction",
        "chalcogen_fraction",
        "halogen_fraction",
        "pnictogen_fraction",
        "alkali_fraction",
        "alkaline_earth_fraction",
        "transition_metal_fraction",
        "post_transition_fraction",
        "metalloid_fraction",
        "rare_earth_fraction",
        "mean_atomic_number",
        "max_atomic_number",
        "mean_electronegativity",
        "electronegativity_range",
    )
)
phase139.SPLIT_PLAN = (
    ("phase141_registered_split", "phase141_manifest", "phase141"),
    ("chemistry_family_hash_0", "group:chemistry_family_key", "phase142_family_0"),
    ("chemistry_family_hash_1", "group:chemistry_family_key", "phase142_family_1"),
    ("chemistry_family_hash_2", "group:chemistry_family_key", "phase142_family_2"),
    ("dominant_element_hash", "group:dominant_element", "phase142_dominant"),
    ("element_count_bins", "bins:element_count", "phase142_element_count"),
    ("entropy_bins", "bins:entropy_fraction", "phase142_entropy"),
    ("transition_metal_bins", "bins:transition_metal_fraction", "phase142_transition"),
    ("metalloid_bins", "bins:metalloid_fraction", "phase142_metalloid"),
    ("anion_fraction_bins", "bins:anion_fraction", "phase142_anion"),
    ("max_fraction_bins", "bins:max_fraction", "phase142_max_fraction"),
)

PROFILE_FIELDS = phase139.PROFILE_FIELDS
SPLIT_FIELDS = phase139.SPLIT_FIELDS
AUDIT_FIELDS = phase139.AUDIT_FIELDS
MIN_STABLE_SPLIT_PASS_RATE = phase139.MIN_STABLE_SPLIT_PASS_RATE
MAX_CLASS_BALANCE_SHIFT = phase139.MAX_CLASS_BALANCE_SHIFT


def _read_json(path: Path) -> dict[str, Any]:
    return phase139._read_json(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    phase139._write_json(path, payload)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    phase139._write_csv(path, rows, fields)


def _display_path(path: Path, root: Path | None = None) -> str:
    return phase139._display_path(path, root)


def _is_true(value: Any) -> bool:
    return phase139._is_true(value)


def build_split_reviews(df: pd.DataFrame, split_manifest: dict[str, Any]) -> dict[str, Any]:
    reviews: dict[str, Any] = {}
    for split_id, split_spec, salt in phase139.SPLIT_PLAN:
        if split_spec == "phase141_manifest":
            reviews[split_id] = phase139._registered_split(split_manifest)
        elif split_spec.startswith("group:"):
            reviews[split_id] = phase139._group_split(df, split_spec.split(":", 1)[1], salt)
        elif split_spec.startswith("bins:"):
            columns = tuple(split_spec.split(":", 1)[1].split(","))
            reviews[split_id] = phase139._bin_split(df, columns, salt)
        else:
            raise ValueError(f"Unsupported split spec: {split_spec}")
    return reviews


def evaluate_split(
    df: pd.DataFrame,
    *,
    target: str,
    split_id: str,
    split_info: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return phase139.evaluate_split(df, target=target, split_id=split_id, split_info=split_info)


def _audit_row(*, audit: str, status: str, severity: str, value: Any, threshold: Any, reason: str) -> dict[str, Any]:
    return phase139._audit_row(
        audit=audit,
        status=status,
        severity=severity,
        value=value,
        threshold=threshold,
        reason=reason,
    )


def build_audit_rows(*, phase141_gate: dict[str, Any], split_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    phase139_split_rows = [
        {
            **row,
            "split_id": (
                "phase138_registered_split"
                if row.get("split_id") == "phase141_registered_split"
                else row.get("split_id")
            ),
        }
        for row in split_rows
    ]
    rows = phase139.build_audit_rows(
        phase138_gate=_phase141_as_phase138_gate(phase141_gate),
        split_rows=phase139_split_rows,
    )
    replacements = {
        "phase138_gate_status": "phase141_gate_status",
        "original_split_shortcut_dominance": "original_split_shortcut_dominance",
        "original_split_nearest_neighbor_dominance": "original_split_nearest_neighbor_dominance",
        "original_split_class_balance": "original_split_class_balance",
        "split_sensitivity_pass_rate": "split_sensitivity_pass_rate",
        "shortcut_dominant_split_count": "shortcut_dominant_split_count",
        "nearest_neighbor_dominant_split_count": "nearest_neighbor_dominant_split_count",
        "class_balance_imbalanced_split_count": "class_balance_imbalanced_split_count",
    }
    for row in rows:
        row["audit"] = replacements.get(str(row.get("audit")), row.get("audit"))
        reason = str(row.get("reason") or "")
        reason = reason.replace("Phase 138", "Phase 141")
        reason = reason.replace("glass", "experimental is-metal")
        row["reason"] = reason
        if row.get("audit") == "phase141_gate_status":
            row["value"] = phase141_gate.get("status")
            row["threshold"] = "phase141_matbench_expt_is_metal_ready_focused_review"
    return rows


def _phase141_as_phase138_gate(phase141_gate: dict[str, Any]) -> dict[str, Any]:
    converted = dict(phase141_gate)
    converted["status"] = (
        "phase138_matbench_glass_ready_focused_review"
        if phase141_gate.get("status") == "phase141_matbench_expt_is_metal_ready_focused_review"
        else phase141_gate.get("status")
    )
    return converted


def build_gate(*, phase141_gate: dict[str, Any], split_rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    blockers = [row for row in audit_rows if row["status"] == "block"]
    viable = [row for row in split_rows if _is_true(row.get("split_viable"))]
    passed = [row for row in viable if _is_true(row.get("split_pass"))]
    shortcut_dominant = [row for row in viable if _is_true(row.get("negative_control_dominates"))]
    nearest_dominant = [row for row in viable if _is_true(row.get("nearest_neighbor_dominates"))]
    class_imbalanced = [
        row
        for row in viable
        if float(row.get("class_balance_shift") or 0.0) > MAX_CLASS_BALANCE_SHIFT
    ]
    original = next((row for row in split_rows if row.get("split_id") == "phase141_registered_split"), {})
    if phase141_gate.get("status") != "phase141_matbench_expt_is_metal_ready_focused_review":
        status = "phase142_matbench_expt_is_metal_review_blocked_by_phase141"
        mechanism_allowed = False
        next_action = "complete or close Phase 141 before focused review"
    elif blockers:
        status = "phase142_matbench_expt_is_metal_focused_review_closed_split_sensitivity_or_shortcut"
        mechanism_allowed = False
        next_action = "close the Phase 141 experimental is-metal target as diagnostic; do not train"
    else:
        status = "phase142_matbench_expt_is_metal_focused_review_ready_low_capacity_mechanism_gate"
        mechanism_allowed = True
        next_action = (
            "design a separate no-training low-capacity experimental is-metal mechanism gate; "
            "keep model training closed"
        )
    return {
        "status": status,
        "source_gate_status": phase141_gate.get("status"),
        "selected_target": phase141_gate.get("selected_target", TARGET),
        "viable_split_reviews": len(viable),
        "passed_split_reviews": len(passed),
        "split_pass_rate": (len(passed) / len(viable)) if viable else 0.0,
        "shortcut_dominant_splits": len(shortcut_dominant),
        "nearest_neighbor_dominant_splits": len(nearest_dominant),
        "class_balance_imbalanced_splits": len(class_imbalanced),
        "blocking_audits": [row["audit"] for row in blockers],
        "original_best_admissible_profile": original.get("best_admissible_profile"),
        "original_best_admissible_method": original.get("best_admissible_method"),
        "original_best_admissible_val_balanced_accuracy": original.get("best_admissible_val_balanced_accuracy"),
        "original_best_admissible_test_balanced_accuracy": original.get("best_admissible_test_balanced_accuracy"),
        "original_best_negative_profile": original.get("best_negative_profile"),
        "original_best_negative_method": original.get("best_negative_method"),
        "original_best_negative_val_balanced_accuracy": original.get("best_negative_val_balanced_accuracy"),
        "original_best_negative_test_balanced_accuracy": original.get("best_negative_test_balanced_accuracy"),
        "original_nearest_neighbor_val_balanced_accuracy": original.get("nearest_neighbor_val_balanced_accuracy"),
        "original_nearest_neighbor_test_balanced_accuracy": original.get("nearest_neighbor_test_balanced_accuracy"),
        "original_class_balance_shift": original.get("class_balance_shift"),
        "phase142_model_mechanism_allowed": mechanism_allowed,
        "phase142_low_capacity_mechanism_design_allowed": mechanism_allowed,
        "phase142_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": next_action,
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    return phase139._markdown_table(rows, columns)


def build_markdown(gate: dict[str, Any], audit_rows: list[dict[str, Any]], split_rows: list[dict[str, Any]]) -> str:
    blocking = [row for row in audit_rows if row["status"] == "block"]
    viable = [row for row in split_rows if _is_true(row.get("split_viable"))]
    lines = [
        "# Phase 142 Matbench Experimental Is-Metal Focused Review",
        "",
        f"- Status: `{gate['status']}`",
        f"- Split pass rate: `{gate['split_pass_rate']:.6g}`",
        f"- Blocking audits: `{', '.join(gate['blocking_audits']) or 'none'}`",
        f"- Low-capacity mechanism design allowed: `{gate['phase142_low_capacity_mechanism_design_allowed']}`",
        f"- Model training allowed: `{gate['phase142_model_training_allowed']}`",
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
                ("Val BA", "best_admissible_val_balanced_accuracy"),
                ("Test BA", "best_admissible_test_balanced_accuracy"),
                ("Shortcut", "negative_control_dominates"),
                ("NN", "nearest_neighbor_dominates"),
                ("Class shift", "class_balance_shift"),
            ],
        ),
    ]
    return "\n".join(lines) + "\n"


def build_package(*, root: Path, phase141_dir: Path, output_dir: Path) -> dict[str, Any]:
    field_path = phase141_dir / "phase141_matbench_expt_is_metal_field_table.csv"
    split_path = phase141_dir / "phase141_matbench_expt_is_metal_split_manifest.json"
    gate_path = phase141_dir / "phase141_matbench_expt_is_metal_gate.json"
    df = pd.read_csv(field_path)
    split_manifest = _read_json(split_path)
    phase141_gate = _read_json(gate_path)
    target = str(phase141_gate.get("selected_target") or TARGET)

    split_infos = build_split_reviews(df, split_manifest)
    profile_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for split_id, split_info in split_infos.items():
        rows, summary = evaluate_split(df, target=target, split_id=split_id, split_info=split_info)
        profile_rows.extend(rows)
        split_rows.append(summary)
    audit_rows = build_audit_rows(phase141_gate=phase141_gate, split_rows=split_rows)
    gate = build_gate(phase141_gate=phase141_gate, split_rows=split_rows, audit_rows=audit_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "phase142_matbench_expt_is_metal"
    profile_path = output_dir / f"{prefix}_focused_profile_table.csv"
    split_review_path = output_dir / f"{prefix}_split_sensitivity_table.csv"
    audit_path = output_dir / f"{prefix}_shortcut_audit_table.csv"
    split_manifest_path = output_dir / f"{prefix}_split_review_manifest.json"
    gate_out = output_dir / f"{prefix}_focused_review_gate.json"
    markdown_path = output_dir / f"{prefix}_focused_review.md"
    manifest_path = output_dir / f"{prefix}_focused_review_manifest.json"

    _write_csv(profile_path, profile_rows, PROFILE_FIELDS)
    _write_csv(split_review_path, split_rows, SPLIT_FIELDS)
    _write_csv(audit_path, audit_rows, AUDIT_FIELDS)
    _write_json(split_manifest_path, split_infos)
    _write_json(gate_out, gate)
    markdown_path.write_text(build_markdown(gate, audit_rows, split_rows), encoding="utf-8")

    manifest = {
        "phase": 142,
        "objective": "matbench_expt_is_metal_focused_split_shortcut_review_no_training",
        "inputs": {
            "phase141_dir": _display_path(phase141_dir, root),
            "field_table": _display_path(field_path, root),
            "split_manifest": _display_path(split_path, root),
            "gate_json": _display_path(gate_path, root),
        },
        "outputs": {
            "focused_profile_table": _display_path(profile_path, root),
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
            "audit_rows": len(audit_rows),
            "blocking_audits": len(gate["blocking_audits"]),
        },
        "gate": gate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--phase141-dir", type=Path, default=DEFAULT_PHASE141_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    phase141_dir = args.phase141_dir if args.phase141_dir.is_absolute() else root / args.phase141_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = build_package(root=root, phase141_dir=phase141_dir, output_dir=output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
