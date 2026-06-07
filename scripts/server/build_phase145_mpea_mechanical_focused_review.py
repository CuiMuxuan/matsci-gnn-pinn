#!/usr/bin/env python3
"""Build Phase 145 focused review for the Phase 144 MPEA hardness gate.

This phase consumes only Phase 144 small artifacts. It audits whether the
MPEA ``hardness_hv`` baseline gap is stable under formula, reference, process,
phase, and test-context split perturbations before any mechanism design or
model training is allowed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PHASE144_DIR = Path("docs/results/phase144_mpea_mechanical_baseline_gate")
DEFAULT_OUTPUT_DIR = Path("docs/results/phase145_mpea_mechanical_focused_review")

MIN_STABLE_SPLIT_PASS_RATE = 0.75
MAX_TARGET_DISTRIBUTION_SHIFT_Z = 0.75
MAX_FORMULA_CROSS_SPLIT_FRACTION = 0.60

SPLIT_SPECS = (
    ("registered_formula_hash", "registered", "phase144_registered"),
    ("formula_hash_salt_0", "formula_hash16", "phase145_formula_0"),
    ("formula_hash_salt_1", "formula_hash16", "phase145_formula_1"),
    ("formula_hash_salt_2", "formula_hash16", "phase145_formula_2"),
    ("chemistry_family", "chemistry_family_key", "phase145_chemistry_family"),
    ("reference_holdout", "reference_hash16", "phase145_reference"),
    ("formula_reference_holdout", "formula_reference_hash16", "phase145_formula_reference"),
    ("processing_method_holdout", "processing_method", "phase145_processing_method"),
    ("phase_family_holdout", "phase_family", "phase145_phase_family"),
    ("test_type_holdout", "test_type", "phase145_test_type"),
    ("microstructure_holdout", "microstructure", "phase145_microstructure"),
    ("process_phase_holdout", "process_phase_key", "phase145_process_phase"),
)

PROFILE_FIELDS = (
    "split_id",
    "group_column",
    "salt",
    "n_groups",
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
    "relative_val_gain_vs_mean",
    "best_negative_profile",
    "best_negative_method",
    "best_negative_val_rmse",
    "best_negative_test_rmse",
    "shortcut_blocks",
    "target_distribution_shift_z",
    "formula_cross_split_fraction",
    "reference_cross_split_fraction",
    "status",
    "reason",
    "phase145_split_pass",
)
SPLIT_FIELDS = (
    "split_id",
    "group_column",
    "salt",
    "n_groups",
    "train_rows",
    "val_rows",
    "test_rows",
    "target_distribution_shift_z",
    "formula_cross_split_fraction",
    "reference_cross_split_fraction",
    "phase145_split_pass",
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


def _load_phase144_module():
    script = Path(__file__).with_name("build_phase144_mpea_mechanical_baseline_gate.py")
    spec = importlib.util.spec_from_file_location("phase144_helpers", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Phase 144 helper module from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase144 = _load_phase144_module()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    phase144._write_json(path, payload)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    phase144._write_csv(path, rows, fields)


def _display_path(path: Path, root: Path | None = None) -> str:
    return phase144._display_path(path, root)


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _split_counts(split_manifest: dict[str, Any]) -> dict[str, int]:
    return {split: len(split_manifest.get("splits", {}).get(split, [])) for split in ("train", "val", "test")}


def _target_distribution_shift_z(df: pd.DataFrame, target: str, split_manifest: dict[str, Any]) -> float:
    splits = split_manifest["splits"]
    train = pd.to_numeric(df.iloc[splits["train"]][target], errors="coerce").dropna().to_numpy(dtype=float)
    if train.size == 0:
        return 0.0
    train_std = float(np.std(train))
    if train_std <= 0:
        return 0.0
    train_stats = {
        "mean": float(np.mean(train)),
        "median": float(np.median(train)),
        "q90": float(np.quantile(train, 0.9)),
    }
    max_shift = 0.0
    for split in ("val", "test"):
        values = pd.to_numeric(df.iloc[splits[split]][target], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        split_stats = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "q90": float(np.quantile(values, 0.9)),
        }
        for key, train_value in train_stats.items():
            max_shift = max(max_shift, abs(split_stats[key] - train_value) / train_std)
    return float(max_shift)


def _cross_split_fraction(df: pd.DataFrame, split_manifest: dict[str, Any], column: str) -> float:
    splits = split_manifest["splits"]
    group_to_splits: dict[str, set[str]] = {}
    for split, indices in splits.items():
        for value in df.iloc[indices][column].fillna("missing"):
            group_to_splits.setdefault(str(value), set()).add(split)
    if not group_to_splits:
        return 0.0
    crossed = sum(1 for values in group_to_splits.values() if len(values) > 1)
    return float(crossed / len(group_to_splits))


def _add_review_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["process_phase_key"] = (
        out["processing_method"].fillna("missing").astype(str)
        + "::"
        + out["phase_family"].fillna("missing").astype(str)
    )
    return out


def _registered_split(split_reviews: dict[str, Any], target: str) -> dict[str, Any]:
    if target not in split_reviews:
        raise KeyError(f"Phase 144 split manifest does not contain {target}")
    return split_reviews[target]


def build_split_reviews(
    field_df: pd.DataFrame,
    split_reviews: dict[str, Any],
    *,
    target: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    target_df = phase144._target_frame(_add_review_group_columns(field_df), target)
    profile_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    split_manifest_out: dict[str, Any] = {}
    for split_id, group_column, salt in SPLIT_SPECS:
        if group_column == "registered":
            split_manifest = _registered_split(split_reviews, target)
        else:
            split_manifest = phase144.split_by_group(target_df, group_column=group_column, salt=salt)
        _, review = phase144._evaluate_one_target(target_df, target, split_manifest)
        counts = _split_counts(split_manifest)
        target_shift = _target_distribution_shift_z(target_df, target, split_manifest)
        formula_cross = _cross_split_fraction(target_df, split_manifest, "formula_hash16")
        reference_cross = _cross_split_fraction(target_df, split_manifest, "reference_hash16")
        split_pass = bool(
            review.get("phase144_candidate")
            and not review.get("shortcut_blocks")
            and target_shift <= MAX_TARGET_DISTRIBUTION_SHIFT_Z
        )
        row = {
            "split_id": split_id,
            "group_column": group_column,
            "salt": salt,
            "n_groups": split_manifest.get("n_groups"),
            "train_rows": counts["train"],
            "val_rows": counts["val"],
            "test_rows": counts["test"],
            "mean_val_rmse": review.get("mean_val_rmse"),
            "mean_test_rmse": review.get("mean_test_rmse"),
            "best_profile": review.get("best_profile"),
            "best_method": review.get("best_method"),
            "best_val_rmse": review.get("best_val_rmse"),
            "best_test_rmse": review.get("best_test_rmse"),
            "val_gain_vs_mean": review.get("val_gain_vs_mean"),
            "test_gain_vs_mean": review.get("test_gain_vs_mean"),
            "relative_val_gain_vs_mean": review.get("relative_val_gain_vs_mean"),
            "best_negative_profile": review.get("best_negative_profile"),
            "best_negative_method": review.get("best_negative_method"),
            "best_negative_val_rmse": review.get("best_negative_val_rmse"),
            "best_negative_test_rmse": review.get("best_negative_test_rmse"),
            "shortcut_blocks": bool(review.get("shortcut_blocks")),
            "target_distribution_shift_z": target_shift,
            "formula_cross_split_fraction": formula_cross,
            "reference_cross_split_fraction": reference_cross,
            "status": review.get("status"),
            "reason": review.get("reason"),
            "phase145_split_pass": split_pass,
        }
        profile_rows.append(row)
        split_rows.append({field: row.get(field) for field in SPLIT_FIELDS})
        split_manifest_out[split_id] = split_manifest
    return profile_rows, split_rows, split_manifest_out


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


def build_audit_rows(
    *,
    phase144_gate: dict[str, Any],
    profile_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    original = next(row for row in profile_rows if row["split_id"] == "registered_formula_hash")
    phase144_ready = phase144_gate.get("status") == "phase144_mpea_mechanical_ready_focused_review"
    rows.append(
        _audit_row(
            audit_id="P145-AUDIT-001",
            audit="phase144_gate_consistency",
            status="pass" if phase144_ready else "block",
            severity="blocking" if not phase144_ready else "info",
            value=phase144_gate.get("status"),
            threshold="phase144_mpea_mechanical_ready_focused_review",
            reason="focused review requires a positive Phase 144 baseline-first gate",
        )
    )
    original_pass = bool(original.get("phase145_split_pass"))
    rows.append(
        _audit_row(
            audit_id="P145-AUDIT-002",
            audit="registered_split_replay",
            status="pass" if original_pass else "block",
            severity="blocking" if not original_pass else "info",
            value=original.get("status"),
            threshold="registered split must remain a guarded candidate",
            reason="Phase 145 must reproduce the Phase 144 selected split without training",
        )
    )
    viable = [
        row
        for row in profile_rows
        if min(int(row.get(key) or 0) for key in ("train_rows", "val_rows", "test_rows"))
        >= phase144.MIN_SPLIT_ROWS
    ]
    passed = [row for row in viable if row.get("phase145_split_pass")]
    stable_rate = float(len(passed) / len(viable)) if viable else 0.0
    rows.append(
        _audit_row(
            audit_id="P145-AUDIT-003",
            audit="stable_split_pass_rate",
            status="pass" if stable_rate >= MIN_STABLE_SPLIT_PASS_RATE else "block",
            severity="blocking" if stable_rate < MIN_STABLE_SPLIT_PASS_RATE else "info",
            value=stable_rate,
            threshold=MIN_STABLE_SPLIT_PASS_RATE,
            reason="guarded gain should survive most viable formula/reference/process split reviews",
        )
    )
    shortcut_dominant = [row for row in viable if bool(row.get("shortcut_blocks"))]
    rows.append(
        _audit_row(
            audit_id="P145-AUDIT-004",
            audit="shortcut_or_process_control_dominant_split_count",
            status="block" if shortcut_dominant else "pass",
            severity="blocking" if shortcut_dominant else "info",
            value=len(shortcut_dominant),
            threshold=0,
            reason="process-only, formula, reference, or dominant-element controls must not dominate viable splits",
        )
    )
    target_shifted = [
        row
        for row in viable
        if float(row.get("target_distribution_shift_z") or 0.0) > MAX_TARGET_DISTRIBUTION_SHIFT_Z
    ]
    rows.append(
        _audit_row(
            audit_id="P145-AUDIT-005",
            audit="target_distribution_imbalanced_split_count",
            status="block" if target_shifted else "pass",
            severity="blocking" if target_shifted else "info",
            value=len(target_shifted),
            threshold=0,
            reason="split target mean/median/q90 shifts must not dominate interpretation",
        )
    )
    formula_crossed = [
        row
        for row in viable
        if float(row.get("formula_cross_split_fraction") or 0.0) > MAX_FORMULA_CROSS_SPLIT_FRACTION
    ]
    rows.append(
        _audit_row(
            audit_id="P145-AUDIT-006",
            audit="formula_cross_split_fraction_count",
            status="block" if formula_crossed else "pass",
            severity="blocking" if formula_crossed else "info",
            value=len(formula_crossed),
            threshold=0,
            reason="focused review should not rely on many formula identities crossing train/val/test",
        )
    )
    return rows


def build_gate(
    *,
    phase144_gate: dict[str, Any],
    profile_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    viable = [
        row
        for row in profile_rows
        if min(int(row.get(key) or 0) for key in ("train_rows", "val_rows", "test_rows"))
        >= phase144.MIN_SPLIT_ROWS
    ]
    passed = [row for row in viable if row.get("phase145_split_pass")]
    stable_rate = float(len(passed) / len(viable)) if viable else 0.0
    blocking = [row for row in audit_rows if row["severity"] == "blocking" and row["status"] == "block"]
    if not blocking:
        status = "phase145_mpea_mechanical_focused_review_ready_low_capacity_mechanism_gate"
        mechanism_allowed = True
        next_action = "design a no-training low-capacity mechanism gate before any neural training"
    else:
        status = "phase145_mpea_mechanical_focused_review_closed_split_sensitivity_or_shortcut"
        mechanism_allowed = False
        next_action = "close Phase 144/145 as diagnostic; do not train"
    original = next(row for row in profile_rows if row["split_id"] == "registered_formula_hash")
    return {
        "status": status,
        "source_name": phase144_gate.get("source_name"),
        "selected_target": phase144_gate.get("selected_target"),
        "selected_target_label": phase144_gate.get("selected_target_label"),
        "original_selected_profile": phase144_gate.get("selected_profile"),
        "original_selected_method": phase144_gate.get("selected_method"),
        "original_validation_rmse": phase144_gate.get("selected_validation_rmse"),
        "original_test_rmse": phase144_gate.get("selected_test_rmse"),
        "original_best_negative_profile": phase144_gate.get("best_negative_profile"),
        "original_best_negative_method": phase144_gate.get("best_negative_method"),
        "original_replayed_status": original.get("status"),
        "original_target_distribution_shift_z": original.get("target_distribution_shift_z"),
        "viable_split_reviews": len(viable),
        "passed_split_reviews": len(passed),
        "split_pass_rate": stable_rate,
        "shortcut_dominant_splits": sum(1 for row in viable if bool(row.get("shortcut_blocks"))),
        "target_distribution_imbalanced_splits": sum(
            1
            for row in viable
            if float(row.get("target_distribution_shift_z") or 0.0) > MAX_TARGET_DISTRIBUTION_SHIFT_Z
        ),
        "formula_cross_split_fraction_splits": sum(
            1
            for row in viable
            if float(row.get("formula_cross_split_fraction") or 0.0) > MAX_FORMULA_CROSS_SPLIT_FRACTION
        ),
        "blocking_audits": [row["audit"] for row in blocking],
        "phase145_model_mechanism_allowed": mechanism_allowed,
        "phase145_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": next_action,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(_fmt(row.get(key)) for _, key in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def build_markdown(
    *,
    gate: dict[str, Any],
    profile_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "# Phase 145 MPEA Mechanical Focused Review",
            "",
            f"- Status: `{gate['status']}`",
            f"- Selected target: `{gate['selected_target']}`",
            f"- Viable split reviews: `{gate['viable_split_reviews']}`",
            f"- Split pass rate: `{gate['split_pass_rate']}`",
            f"- Model mechanism allowed: `{gate['phase145_model_mechanism_allowed']}`",
            f"- Model training allowed: `{gate['phase145_model_training_allowed']}`",
            f"- A100 80GB request now: `{gate['a100_80gb_request_now']}`",
            "",
            "## Split Review",
            "",
            _markdown_table(
                profile_rows,
                [
                    ("Split", "split_id"),
                    ("Status", "status"),
                    ("Best profile", "best_profile"),
                    ("Best method", "best_method"),
                    ("Val RMSE", "best_val_rmse"),
                    ("Test RMSE", "best_test_rmse"),
                    ("Negative", "best_negative_profile"),
                    ("Target shift", "target_distribution_shift_z"),
                    ("Formula cross", "formula_cross_split_fraction"),
                    ("Pass", "phase145_split_pass"),
                ],
            ),
            "",
            "## Audits",
            "",
            _markdown_table(
                audit_rows,
                [
                    ("Audit", "audit"),
                    ("Status", "status"),
                    ("Severity", "severity"),
                    ("Value", "value"),
                    ("Threshold", "threshold"),
                    ("Reason", "reason"),
                ],
            ),
        ]
    ) + "\n"


def build_package(*, root: Path, phase144_dir: Path, output_dir: Path) -> dict[str, Any]:
    field_table_path = phase144_dir / "phase144_mpea_mechanical_field_table.csv"
    split_manifest_path = phase144_dir / "phase144_mpea_mechanical_split_manifest.json"
    gate_path = phase144_dir / "phase144_mpea_mechanical_gate.json"
    review_path = phase144_dir / "phase144_mpea_mechanical_target_review_table.csv"
    field_df = pd.read_csv(field_table_path)
    split_reviews = _read_json(split_manifest_path)
    phase144_gate = _read_json(gate_path)
    target_reviews = pd.read_csv(review_path).to_dict("records")
    target = str(phase144_gate.get("selected_target") or "hardness_hv")

    profile_rows, split_rows, split_manifest = build_split_reviews(field_df, split_reviews, target=target)
    audit_rows = build_audit_rows(phase144_gate=phase144_gate, profile_rows=profile_rows)
    gate = build_gate(phase144_gate=phase144_gate, profile_rows=profile_rows, audit_rows=audit_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "phase145_mpea_mechanical"
    profile_path = output_dir / f"{prefix}_focused_profile_table.csv"
    split_path = output_dir / f"{prefix}_split_sensitivity_table.csv"
    audit_path = output_dir / f"{prefix}_shortcut_audit_table.csv"
    split_manifest_out = output_dir / f"{prefix}_split_review_manifest.json"
    gate_out = output_dir / f"{prefix}_focused_review_gate.json"
    markdown_path = output_dir / f"{prefix}_focused_review.md"
    manifest_path = output_dir / f"{prefix}_focused_review_manifest.json"

    _write_csv(profile_path, profile_rows, PROFILE_FIELDS)
    _write_csv(split_path, split_rows, SPLIT_FIELDS)
    _write_csv(audit_path, audit_rows, AUDIT_FIELDS)
    _write_json(split_manifest_out, split_manifest)
    _write_json(gate_out, gate)
    markdown_path.write_text(
        build_markdown(gate=gate, profile_rows=profile_rows, audit_rows=audit_rows),
        encoding="utf-8",
    )
    manifest = {
        "phase": 145,
        "objective": "mpea_mechanical_focused_split_shortcut_review_no_training",
        "inputs": {
            "phase144_dir": _display_path(phase144_dir, root),
            "field_table": _display_path(field_table_path, root),
            "phase144_split_manifest": _display_path(split_manifest_path, root),
            "phase144_gate": _display_path(gate_path, root),
            "phase144_target_review_table": _display_path(review_path, root),
        },
        "outputs": {
            "focused_profile_table": _display_path(profile_path, root),
            "split_sensitivity_table": _display_path(split_path, root),
            "shortcut_audit_table": _display_path(audit_path, root),
            "split_review_manifest": _display_path(split_manifest_out, root),
            "gate_json": _display_path(gate_out, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "phase144_target_review_rows": len(target_reviews),
            "focused_profile_rows": len(profile_rows),
            "split_rows": len(split_rows),
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
    parser.add_argument("--phase144-dir", type=Path, default=DEFAULT_PHASE144_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    phase144_dir = args.phase144_dir if args.phase144_dir.is_absolute() else root / args.phase144_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = build_package(root=root, phase144_dir=phase144_dir, output_dir=output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
