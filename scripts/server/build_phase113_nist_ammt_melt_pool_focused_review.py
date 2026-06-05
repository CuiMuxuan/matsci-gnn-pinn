#!/usr/bin/env python3
"""Build Phase 113 NIST AMMT melt-pool focused review package.

This no-training review consumes Phase 112 artifacts only. It closes or carries
forward the melt-pool target branch based on validation/test reversal and
shortcut evidence before any model mechanism is considered.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REVIEW_FIELDS = (
    "target",
    "phase112_status",
    "phase112_candidate",
    "selected_feature_profile",
    "selected_validation_method",
    "selected_validation_rmse",
    "selected_test_rmse",
    "mean_validation_rmse",
    "mean_test_rmse",
    "validation_relative_improvement_over_mean",
    "test_relative_improvement_over_mean",
    "layer_time_shortcut_detected",
    "focused_review_status",
    "mechanism_allowed",
    "reason",
)
BOUNDARY_FIELDS = (
    "boundary_id",
    "blocked_item",
    "reason",
    "phase113_model_mechanism_allowed",
    "phase113_model_training_allowed",
    "a100_training_allowed_now",
    "a100_80gb_request_now",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
        return f"{value:.6f}"
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


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _focused_review_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        candidate = _bool(row.get("phase112_candidate"))
        test_gain = _float(row.get("test_relative_improvement_over_mean"))
        shortcut = _bool(row.get("layer_time_shortcut_detected"))
        if not candidate:
            status = "not_phase112_candidate"
            allowed = False
            reason = row.get("status", "")
        elif test_gain <= 0.0:
            status = "blocked_validation_test_reversal"
            allowed = False
            reason = "validation-selected profile is worse than mean guard on test"
        elif shortcut:
            status = "blocked_layer_time_shortcut"
            allowed = False
            reason = "layer/time shortcut detected"
        else:
            status = "focused_review_passed_mechanism_design_allowed"
            allowed = True
            reason = "validation gain preserved on test and shortcut review did not block"
        output.append(
            {
                "target": row.get("target"),
                "phase112_status": row.get("status"),
                "phase112_candidate": candidate,
                "selected_feature_profile": row.get("selected_feature_profile"),
                "selected_validation_method": row.get("selected_validation_method"),
                "selected_validation_rmse": _float(row.get("selected_validation_rmse")),
                "selected_test_rmse": _float(row.get("selected_test_rmse")),
                "mean_validation_rmse": _float(row.get("mean_validation_rmse")),
                "mean_test_rmse": _float(row.get("mean_test_rmse")),
                "validation_relative_improvement_over_mean": _float(
                    row.get("validation_relative_improvement_over_mean")
                ),
                "test_relative_improvement_over_mean": test_gain,
                "layer_time_shortcut_detected": shortcut,
                "focused_review_status": status,
                "mechanism_allowed": allowed,
                "reason": reason,
            }
        )
    return output


def _boundary_rows(review_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reversal_targets = [
        row["target"]
        for row in review_rows
        if row["focused_review_status"] == "blocked_validation_test_reversal"
    ]
    shortcut_targets = [
        row["target"]
        for row in review_rows
        if row["focused_review_status"] == "blocked_layer_time_shortcut"
    ]
    rows = [
        {
            "boundary_id": "phase113_no_training_on_phase112_candidates",
            "blocked_item": "Phase 112 melt-pool selected/candidate targets",
            "reason": f"validation/test reversal targets: {', '.join(reversal_targets) or 'none'}",
            "phase113_model_mechanism_allowed": False,
            "phase113_model_training_allowed": False,
            "a100_training_allowed_now": False,
            "a100_80gb_request_now": False,
        },
        {
            "boundary_id": "phase113_no_a100_80gb_request",
            "blocked_item": "A100-SXM4-80GB escalation",
            "reason": "no model mechanism or seed-positive branch is open",
            "phase113_model_mechanism_allowed": False,
            "phase113_model_training_allowed": False,
            "a100_training_allowed_now": False,
            "a100_80gb_request_now": False,
        },
    ]
    if shortcut_targets:
        rows.append(
            {
                "boundary_id": "phase113_layer_time_shortcut_targets",
                "blocked_item": "Melt-pool targets solved by layer/time shortcut",
                "reason": ", ".join(shortcut_targets),
                "phase113_model_mechanism_allowed": False,
                "phase113_model_training_allowed": False,
                "a100_training_allowed_now": False,
                "a100_80gb_request_now": False,
            }
        )
    return rows


def _build_gate(phase112_gate: dict[str, Any], review_rows: list[dict[str, Any]]) -> dict[str, Any]:
    allowed = [row for row in review_rows if row["mechanism_allowed"]]
    reversal_count = sum(
        1 for row in review_rows if row["focused_review_status"] == "blocked_validation_test_reversal"
    )
    if phase112_gate.get("status") != "phase112_melt_pool_target_gap_ready_focused_review":
        status = "phase113_melt_pool_review_blocked_by_phase112"
        next_action = "complete or close Phase 112 before focused review"
    elif allowed:
        status = "phase113_melt_pool_focused_review_ready_mechanism_design"
        next_action = "design a separate no-training mechanism gate; keep model training closed"
    else:
        status = "phase113_melt_pool_focused_review_closed_validation_test_reversal"
        next_action = "close melt-pool target branch as diagnostic; do not train"
    return {
        "status": status,
        "phase112_status": phase112_gate.get("status"),
        "phase112_selected_target": phase112_gate.get("selected_target"),
        "phase112_selected_validation_rmse": phase112_gate.get("selected_validation_rmse"),
        "phase112_selected_test_rmse": phase112_gate.get("selected_test_rmse"),
        "phase112_mean_test_rmse_for_selected": next(
            (
                row["mean_test_rmse"]
                for row in review_rows
                if row["target"] == phase112_gate.get("selected_target")
            ),
            None,
        ),
        "mechanism_allowed_targets": [row["target"] for row in allowed],
        "validation_test_reversal_target_count": reversal_count,
        "phase113_model_mechanism_allowed": bool(allowed),
        "phase113_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": next_action,
    }


def _write_markdown(
    path: Path,
    gate: dict[str, Any],
    review_rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Phase 113 NIST AMMT Melt-Pool Focused Review",
        "",
        f"- Status: `{gate['status']}`",
        f"- Phase 112 selected target: `{gate['phase112_selected_target']}`",
        f"- Mechanism allowed targets: `{', '.join(gate['mechanism_allowed_targets']) or 'none'}`",
        "- Model training allowed: `false`",
        "- A100 training allowed now: `false`",
        "",
        "| Target | Focused review | Val gain vs mean | Test gain vs mean | Reason |",
        "|---|---|---:|---:|---|",
    ]
    for row in review_rows:
        lines.append(
            "| {target} | {focused_review_status} | {validation_relative_improvement_over_mean} | {test_relative_improvement_over_mean} | {reason} |".format(
                **{key: _csv_value(value) for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "| Boundary | Blocked item | Reason |",
            "|---|---|---|",
        ]
    )
    for row in boundary_rows:
        lines.append(f"| {row['boundary_id']} | {row['blocked_item']} | {row['reason']} |")
    lines.extend(["", f"Next action: {gate['next_action']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_package(
    *,
    root: Path,
    phase112_gate_path: Path,
    phase112_review_table: Path,
    output_dir: Path,
) -> dict[str, Any]:
    phase112_gate = _read_json(phase112_gate_path)
    review_rows = _focused_review_rows(_read_csv(phase112_review_table))
    boundaries = _boundary_rows(review_rows)
    gate = _build_gate(phase112_gate, review_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    review_path = output_dir / "phase113_nist_ammt_melt_pool_focused_review_table.csv"
    boundary_path = output_dir / "phase113_nist_ammt_melt_pool_boundary_table.csv"
    gate_path = output_dir / "phase113_nist_ammt_melt_pool_focused_review_gate.json"
    markdown_path = output_dir / "phase113_nist_ammt_melt_pool_focused_review.md"
    manifest_path = output_dir / "phase113_nist_ammt_melt_pool_focused_review_manifest.json"
    _write_csv(review_path, review_rows, REVIEW_FIELDS)
    _write_csv(boundary_path, boundaries, BOUNDARY_FIELDS)
    _write_json(gate_path, gate)
    _write_markdown(markdown_path, gate, review_rows, boundaries)
    manifest = {
        "phase": 113,
        "objective": "nist_ammt_melt_pool_focused_review_no_training",
        "inputs": {
            "phase112_gate": _display_path(phase112_gate_path, root),
            "phase112_review_table": _display_path(phase112_review_table, root),
        },
        "outputs": {
            "review_table": _display_path(review_path, root),
            "boundary_table": _display_path(boundary_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown_summary": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "review_rows": len(review_rows),
            "boundary_rows": len(boundaries),
            "mechanism_allowed_targets": len(gate["mechanism_allowed_targets"]),
        },
        "gate": gate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--phase112-gate",
        type=Path,
        default=Path(
            "docs/results/phase112_nist_ammt_melt_pool_target_gate/"
            "phase112_nist_ammt_melt_pool_target_gate.json"
        ),
    )
    parser.add_argument(
        "--phase112-review-table",
        type=Path,
        default=Path(
            "docs/results/phase112_nist_ammt_melt_pool_target_gate/"
            "phase112_nist_ammt_melt_pool_target_review_table.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase113_nist_ammt_melt_pool_focused_review"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    phase112_gate = args.phase112_gate if args.phase112_gate.is_absolute() else root / args.phase112_gate
    phase112_review = (
        args.phase112_review_table
        if args.phase112_review_table.is_absolute()
        else root / args.phase112_review_table
    )
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = build_package(
        root=root,
        phase112_gate_path=phase112_gate,
        phase112_review_table=phase112_review,
        output_dir=output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
