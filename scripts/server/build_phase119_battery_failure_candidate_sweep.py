#!/usr/bin/env python3
"""Build Phase 119 focused-review sweep for all Phase 117 candidates.

Phase 118 closed the selected Battery Failure target. This phase applies the
same leakage, shortcut, dependency, and split-sensitivity standard to every
Phase 117 candidate target so no unreviewed candidate can accidentally become a
model-training entry. It reads only small Phase 117/118 artifacts.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PHASE117_DIR = Path("docs/results/phase117_battery_failure_databank_gate")
DEFAULT_PHASE118_DIR = Path("docs/results/phase118_battery_failure_focused_review")
DEFAULT_OUTPUT_DIR = Path("docs/results/phase119_battery_failure_candidate_sweep")


def _load_phase118_module():
    script = Path(__file__).with_name("build_phase118_battery_failure_focused_review.py")
    spec = importlib.util.spec_from_file_location("phase118_review_helpers", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Phase 118 helper module from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase118 = _load_phase118_module()

TARGET_SWEEP_FIELDS = (
    "target",
    "row_count",
    "phase117_candidate",
    "phase118_selected_target",
    "viable_split_reviews",
    "passed_split_reviews",
    "split_pass_rate",
    "negative_control_dominant_splits",
    "dependency_risk_rows",
    "blocking_audit_rows",
    "blocking_audits",
    "original_best_admissible_profile",
    "original_best_admissible_method",
    "original_best_admissible_val_rmse",
    "original_best_admissible_test_rmse",
    "original_best_negative_profile",
    "original_best_negative_method",
    "focused_review_status",
    "mechanism_allowed",
    "reason",
)
PROFILE_FIELDS = phase118.PROFILE_FIELDS
SPLIT_FIELDS = phase118.SPLIT_FIELDS
DEPENDENCY_FIELDS = phase118.DEPENDENCY_FIELDS
AUDIT_FIELDS = ("target", *phase118.AUDIT_FIELDS)


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


def _candidate_targets(phase117_gate: dict[str, Any], review_table: Path) -> list[str]:
    targets = [str(target) for target in phase117_gate.get("candidate_targets", []) if str(target)]
    if targets:
        return targets
    if not review_table.exists():
        return []
    with review_table.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        row["target"]
        for row in rows
        if row.get("target") and _is_true(row.get("phase117_candidate"))
    ]


def _review_one_target(
    *,
    field_table: pd.DataFrame,
    target: str,
    phase117_gate: dict[str, Any],
    phase117_split: dict[str, Any],
    phase118_gate: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    target_df = phase118._target_subset(field_table, target)
    profile_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    split_manifests: dict[str, Any] = {}
    original_profiles = phase118.PROFILE_COLUMNS
    phase118.PROFILE_COLUMNS = {
        profile_name: {
            **profile,
            "numeric": tuple(column for column in profile["numeric"] if column != target),
        }
        for profile_name, profile in original_profiles.items()
    }
    try:
        for split_id, group_column, salt in phase118.SPLIT_PLAN:
            if group_column == "phase117_manifest":
                split_info = phase118._remap_phase117_split(target_df, phase117_split)
            else:
                split_info = phase118._hash_group_split(target_df, group_column=group_column, salt=salt)
            split_manifests[split_id] = split_info
            rows, summary = phase118._evaluate_split(
                target_df,
                target=target,
                split_id=split_id,
                split_info=split_info,
            )
            profile_rows.extend(rows)
            split_rows.append(summary)
    finally:
        phase118.PROFILE_COLUMNS = original_profiles

    dependency_rows = phase118._dependency_rows(target_df, target=target)
    audit_rows = phase118._audit_rows(
        phase117_gate=phase117_gate,
        target=target,
        profile_rows=profile_rows,
        split_rows=split_rows,
        dependency_rows=dependency_rows,
    )
    blockers = [row for row in audit_rows if row["status"] == "block"]
    viable_splits = [row for row in split_rows if _is_true(row.get("split_viable"))]
    passed_splits = [row for row in viable_splits if _is_true(row.get("split_pass"))]
    negative_dominant_splits = [
        row for row in viable_splits if _is_true(row.get("negative_control_dominates"))
    ]
    original = next(
        (row for row in split_rows if row.get("split_id") == "phase117_registered_split"),
        {},
    )
    selected_target = str((phase118_gate or {}).get("selected_target") or phase117_gate.get("selected_target"))
    if blockers:
        focused_status = "closed_focused_review_blocked"
        reason = "blocked by " + ", ".join(row["audit"] for row in blockers)
        mechanism_allowed = False
    else:
        focused_status = "ready_low_capacity_mechanism_gate"
        reason = "admissible gain survives focused leakage and split review"
        mechanism_allowed = True
    target_row = {
        "target": target,
        "row_count": int(len(target_df)),
        "phase117_candidate": True,
        "phase118_selected_target": target == selected_target,
        "viable_split_reviews": len(viable_splits),
        "passed_split_reviews": len(passed_splits),
        "split_pass_rate": len(passed_splits) / len(viable_splits) if viable_splits else 0.0,
        "negative_control_dominant_splits": len(negative_dominant_splits),
        "dependency_risk_rows": sum(
            1 for row in dependency_rows if row["status"] == "high_dependency_risk"
        ),
        "blocking_audit_rows": len(blockers),
        "blocking_audits": [row["audit"] for row in blockers],
        "original_best_admissible_profile": original.get("best_admissible_profile"),
        "original_best_admissible_method": original.get("best_admissible_method"),
        "original_best_admissible_val_rmse": original.get("best_admissible_val_rmse"),
        "original_best_admissible_test_rmse": original.get("best_admissible_test_rmse"),
        "original_best_negative_profile": original.get("best_negative_profile"),
        "original_best_negative_method": original.get("best_negative_method"),
        "focused_review_status": focused_status,
        "mechanism_allowed": mechanism_allowed,
        "reason": reason,
    }
    audit_rows_with_target = [{**row, "target": target} for row in audit_rows]
    return (
        profile_rows,
        split_rows,
        dependency_rows,
        audit_rows_with_target,
        split_manifests,
        target_row,
    )


def _build_gate(
    *,
    phase117_gate: dict[str, Any],
    phase118_gate: dict[str, Any] | None,
    target_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    allowed_targets = [row for row in target_rows if _is_true(row.get("mechanism_allowed"))]
    if phase117_gate.get("status") != "phase117_battery_failure_databank_gap_ready_focused_review":
        status = "phase119_battery_failure_candidate_sweep_blocked_by_phase117"
        next_action = "complete or close Phase 117 before sweeping candidates"
    elif allowed_targets:
        status = "phase119_battery_failure_candidate_sweep_ready_low_capacity_mechanism_gate"
        next_action = "design a separate no-training low-capacity mechanism gate for allowed targets"
    else:
        status = "phase119_battery_failure_candidate_sweep_closed_all_phase117_candidates"
        next_action = "close Battery Failure Databank as diagnostic or open a new external baseline-first intake"
    return {
        "status": status,
        "phase117_status": phase117_gate.get("status"),
        "phase118_status": (phase118_gate or {}).get("status"),
        "reviewed_candidate_targets": len(target_rows),
        "allowed_candidate_targets": [row["target"] for row in allowed_targets],
        "closed_candidate_targets": [
            row["target"] for row in target_rows if not _is_true(row.get("mechanism_allowed"))
        ],
        "selected_phase117_target": phase117_gate.get("selected_target"),
        "phase118_selected_target_closed": (phase118_gate or {}).get("status")
        == "phase118_battery_failure_focused_review_closed_leakage_or_split_sensitivity",
        "total_blocking_audit_rows": sum(int(row["blocking_audit_rows"]) for row in target_rows),
        "targets_with_target_family_dependency": [
            row["target"]
            for row in target_rows
            if "target_family_dependency" in row.get("blocking_audits", [])
        ],
        "targets_with_negative_control_dominance": [
            row["target"]
            for row in target_rows
            if "original_split_negative_control_dominance" in row.get("blocking_audits", [])
        ],
        "phase119_model_mechanism_allowed": bool(allowed_targets),
        "phase119_low_capacity_mechanism_design_allowed": bool(allowed_targets),
        "phase119_model_training_allowed": False,
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
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            elif isinstance(value, list):
                values.append(", ".join(str(item) for item in value) or "none")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def _build_markdown(gate: dict[str, Any], target_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase 119 Battery Failure Candidate Sweep",
        "",
        f"- Status: `{gate['status']}`",
        f"- Reviewed targets: `{gate['reviewed_candidate_targets']}`",
        f"- Allowed targets: `{', '.join(gate['allowed_candidate_targets']) or 'none'}`",
        f"- Model training allowed: `{gate['phase119_model_training_allowed']}`",
        f"- A100 training allowed now: `{gate['a100_training_allowed_now']}`",
        "",
        "## Target Sweep",
        "",
        _markdown_table(
            target_rows,
            [
                ("Target", "target"),
                ("Status", "focused_review_status"),
                ("Allowed", "mechanism_allowed"),
                ("Split pass", "split_pass_rate"),
                ("Blockers", "blocking_audits"),
                ("Reason", "reason"),
            ],
        ),
    ]
    return "\n".join(lines) + "\n"


def build_package(
    *,
    root: Path,
    phase117_dir: Path,
    phase118_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    field_table_path = phase117_dir / "phase117_battery_failure_databank_field_table.csv"
    split_manifest_path = phase117_dir / "phase117_battery_failure_databank_split_manifest.json"
    phase117_gate_path = phase117_dir / "phase117_battery_failure_databank_gate.json"
    phase117_review_path = phase117_dir / "phase117_battery_failure_databank_target_review_table.csv"
    phase118_gate_path = phase118_dir / "phase118_battery_failure_focused_review_gate.json"

    field_table = pd.read_csv(field_table_path)
    phase117_split = _read_json(split_manifest_path)
    phase117_gate = _read_json(phase117_gate_path)
    phase118_gate = _read_json(phase118_gate_path) if phase118_gate_path.exists() else None
    targets = _candidate_targets(phase117_gate, phase117_review_path)

    all_profile_rows: list[dict[str, Any]] = []
    all_split_rows: list[dict[str, Any]] = []
    all_dependency_rows: list[dict[str, Any]] = []
    all_audit_rows: list[dict[str, Any]] = []
    all_split_manifests: dict[str, Any] = {}
    target_rows: list[dict[str, Any]] = []
    for target in targets:
        profile_rows, split_rows, dependency_rows, audit_rows, split_manifests, target_row = _review_one_target(
            field_table=field_table,
            target=target,
            phase117_gate=phase117_gate,
            phase117_split=phase117_split,
            phase118_gate=phase118_gate,
        )
        all_profile_rows.extend(profile_rows)
        all_split_rows.extend(split_rows)
        all_dependency_rows.extend(dependency_rows)
        all_audit_rows.extend(audit_rows)
        all_split_manifests[target] = split_manifests
        target_rows.append(target_row)

    gate = _build_gate(
        phase117_gate=phase117_gate,
        phase118_gate=phase118_gate,
        target_rows=target_rows,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "phase119_battery_failure_candidate_sweep_table.csv"
    profile_path = output_dir / "phase119_battery_failure_candidate_profile_table.csv"
    split_path = output_dir / "phase119_battery_failure_candidate_split_sensitivity_table.csv"
    dependency_path = output_dir / "phase119_battery_failure_candidate_dependency_table.csv"
    audit_path = output_dir / "phase119_battery_failure_candidate_audit_table.csv"
    split_manifest_out = output_dir / "phase119_battery_failure_candidate_split_manifest.json"
    gate_out = output_dir / "phase119_battery_failure_candidate_sweep_gate.json"
    markdown_path = output_dir / "phase119_battery_failure_candidate_sweep.md"
    manifest_path = output_dir / "phase119_battery_failure_candidate_sweep_manifest.json"

    _write_csv(target_path, target_rows, TARGET_SWEEP_FIELDS)
    _write_csv(profile_path, all_profile_rows, PROFILE_FIELDS)
    _write_csv(split_path, all_split_rows, SPLIT_FIELDS)
    _write_csv(dependency_path, all_dependency_rows, DEPENDENCY_FIELDS)
    _write_csv(audit_path, all_audit_rows, AUDIT_FIELDS)
    _write_json(split_manifest_out, all_split_manifests)
    _write_json(gate_out, gate)
    markdown_path.write_text(_build_markdown(gate, target_rows), encoding="utf-8")

    manifest = {
        "phase": 119,
        "objective": "battery_failure_all_candidate_focused_review_sweep_no_training",
        "inputs": {
            "phase117_dir": _display_path(phase117_dir, root),
            "phase118_dir": _display_path(phase118_dir, root),
            "field_table": _display_path(field_table_path, root),
            "phase117_gate": _display_path(phase117_gate_path, root),
            "phase118_gate": _display_path(phase118_gate_path, root),
        },
        "outputs": {
            "target_sweep_table": _display_path(target_path, root),
            "profile_table": _display_path(profile_path, root),
            "split_sensitivity_table": _display_path(split_path, root),
            "dependency_table": _display_path(dependency_path, root),
            "audit_table": _display_path(audit_path, root),
            "split_manifest": _display_path(split_manifest_out, root),
            "gate_json": _display_path(gate_out, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "reviewed_candidate_targets": len(target_rows),
            "allowed_candidate_targets": len(gate["allowed_candidate_targets"]),
            "profile_rows": len(all_profile_rows),
            "split_rows": len(all_split_rows),
            "dependency_rows": len(all_dependency_rows),
            "audit_rows": len(all_audit_rows),
        },
        "gate": gate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--phase117-dir", type=Path, default=DEFAULT_PHASE117_DIR)
    parser.add_argument("--phase118-dir", type=Path, default=DEFAULT_PHASE118_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    phase117_dir = args.phase117_dir if args.phase117_dir.is_absolute() else root / args.phase117_dir
    phase118_dir = args.phase118_dir if args.phase118_dir.is_absolute() else root / args.phase118_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = build_package(
        root=root,
        phase117_dir=phase117_dir,
        phase118_dir=phase118_dir,
        output_dir=output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
