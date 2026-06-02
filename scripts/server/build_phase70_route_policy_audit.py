#!/usr/bin/env python3
"""Build a non-training route-policy audit for Candidate B.

Phase 70 implements the Phase 68 `P68-ROUTE-POLICY` action. It audits existing
main-result, route-guard, stress, and upper-bound evidence to decide whether a
validation-auditable route policy can be opened without training a new policy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


AUDIT_FIELDS = (
    "evidence_source",
    "dataset",
    "split",
    "route",
    "classification",
    "policy_role",
    "status",
    "route_policy_signal",
    "strong_baseline_relation",
    "metrics_summary",
    "reason",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Expected at least one row in {path}")
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _display_path(path: Path, root: Path | None = None) -> str:
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def _default_paths(root: Path) -> dict[str, Path]:
    phase60 = root / "docs/results/phase60_manuscript_evidence_package"
    return {
        "phase60_main": phase60 / "phase60_main_spot_size_seed_positive_table.csv",
        "phase60_route": phase60 / "phase60_route_guard_boundary_table.csv",
        "phase60_stress": phase60 / "phase60_stress_boundary_table.csv",
        "phase59_upper": root
        / "docs/results/phase59_residual_anatomy/phase59_broad21_density_residual_upper_bound.json",
        "phase68_manifest": root
        / "docs/results/phase68_validation_signal_scorecard/phase68_validation_signal_scorecard_manifest.json",
        "phase69_gate": root / "docs/results/phase69_spot_size_signal_probe/phase69_candidate_a_gate.json",
    }


def _float_value(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _main_policy_rows(main_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in main_rows:
        key = (row.get("dataset", ""), row.get("split", ""), row.get("route", ""))
        grouped.setdefault(key, []).append(row)
    audit_rows: list[dict[str, Any]] = []
    for (dataset, split, route), rows in sorted(grouped.items()):
        deltas = [_float_value(row, "delta_vs_best_strong") for row in rows]
        no_process_deltas = [_float_value(row, "delta_vs_no_process") for row in rows]
        strong_pass = all(delta is not None and delta <= 0 for delta in deltas)
        no_process_pass = all(delta is not None and delta <= 0 for delta in no_process_deltas)
        status = "preserve_main_floor" if strong_pass and no_process_pass else "main_floor_risk"
        audit_rows.append(
            {
                "evidence_source": "phase60_main",
                "dataset": dataset,
                "split": split,
                "route": route,
                "classification": "paper_positive_seed_robust" if strong_pass else "main_floor_risk",
                "policy_role": "must_preserve_before_route_policy",
                "status": status,
                "route_policy_signal": "preserve_only",
                "strong_baseline_relation": "beats strong baseline on all three metrics"
                if strong_pass
                else "does not beat strong baseline on all three metrics",
                "metrics_summary": "; ".join(
                    f"{row.get('metric')}: delta_vs_best={row.get('delta_vs_best_strong')}"
                    for row in rows
                ),
                "reason": "Route policy must preserve this fixed-sampling spot_size floor before any expansion.",
            }
        )
    return audit_rows


def _route_policy_rows(route_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    audit_rows: list[dict[str, Any]] = []
    for row in route_rows:
        claim_use = row.get("claim_use", "")
        classification = row.get("classification", "")
        notes = row.get("notes", "")
        if "no-process fallback" in claim_use:
            status = "fallback_positive_not_policy_upgrade"
            signal = "preserve_fallback"
            relation = "strong-baseline positive through no-process fallback"
            reason = "This can support route guarding, but it is not a process route-policy improvement."
        elif classification == "paper_claim_positive":
            status = "process_route_positive"
            signal = "candidate_policy_signal"
            relation = "strong-baseline positive"
            reason = "This would support route-policy expansion if replicated without test leakage."
        else:
            status = "route_boundary_no_policy_signal"
            signal = "blocked_for_policy_training"
            relation = "trails strongest reproducible baseline"
            reason = "This boundary axis cannot open a route-policy branch without a new validation-visible selector."
        if "neural reference is better" in notes:
            reason += " A neural reference appears in diagnostics, but the selected route remains boundary/fallback evidence."
        audit_rows.append(
            {
                "evidence_source": "phase60_route",
                "dataset": row.get("dataset"),
                "split": row.get("split"),
                "route": row.get("route"),
                "classification": classification,
                "policy_role": claim_use,
                "status": status,
                "route_policy_signal": signal,
                "strong_baseline_relation": relation,
                "metrics_summary": row.get("metrics_summary"),
                "reason": reason,
            }
        )
    return audit_rows


def _stress_policy_rows(stress_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    audit_rows: list[dict[str, Any]] = []
    for row in stress_rows:
        scenario = row.get("scenario", "")
        status = row.get("status", "")
        if scenario == "residual_upper_bound_gate":
            policy_status = "blocks_density_route_policy"
            signal = "blocks_policy_training"
            reason = "The validation-selected upper-bound correction is mean fallback, not a transferable route policy."
        elif status == "boundary":
            policy_status = "stress_boundary"
            signal = "boundary_not_policy_signal"
            reason = "Stress boundary must be preserved as a limitation unless a new validation-only selector appears."
        elif status == "pass":
            policy_status = "stress_support"
            signal = "supports_current_floor"
            reason = "Stress support preserves the current floor but does not by itself open route-policy training."
        else:
            policy_status = "diagnostic"
            signal = "diagnostic"
            reason = "Diagnostic stress row."
        audit_rows.append(
            {
                "evidence_source": "phase60_stress",
                "dataset": row.get("dataset"),
                "split": row.get("split"),
                "route": row.get("selected_variant"),
                "classification": scenario,
                "policy_role": row.get("manuscript_use"),
                "status": policy_status,
                "route_policy_signal": signal,
                "strong_baseline_relation": row.get("comparator"),
                "metrics_summary": f"{row.get('metric')}: candidate={row.get('candidate')}; delta={row.get('delta_vs_comparator')}",
                "reason": reason,
            }
        )
    return audit_rows


def build_audit_rows(
    main_rows: list[dict[str, str]],
    route_rows: list[dict[str, str]],
    stress_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    return [
        *_main_policy_rows(main_rows),
        *_route_policy_rows(route_rows),
        *_stress_policy_rows(stress_rows),
    ]


def build_candidate_gate(
    audit_rows: list[dict[str, Any]],
    upper: dict[str, Any],
    phase69_gate: dict[str, Any],
) -> dict[str, Any]:
    preserves_spot_size = all(
        row["status"] == "preserve_main_floor"
        for row in audit_rows
        if row["evidence_source"] == "phase60_main"
    )
    process_route_signals = [
        row for row in audit_rows if row["route_policy_signal"] == "candidate_policy_signal"
    ]
    boundary_blockers = [
        row
        for row in audit_rows
        if row["status"] in {"route_boundary_no_policy_signal", "stress_boundary", "blocks_density_route_policy"}
    ]
    upper_decision = upper.get("decision") or {}
    upper_blocks = (
        upper.get("uses_test_for_selection") is False
        and upper_decision.get("selected_beats_reference_rmse") is False
    )
    candidate_a_open = bool(phase69_gate.get("open_for_seed7_a100_gate"))
    open_low_capacity_policy = (
        preserves_spot_size
        and bool(process_route_signals)
        and not upper_blocks
        and candidate_a_open
    )
    if open_low_capacity_policy:
        status = "opened_for_low_capacity_policy_gate"
        next_action = "implement a non-trainable validation-only route policy before any trainable policy"
        reason = "Existing route evidence includes a process-route strong-baseline signal and no active upper-bound blocker."
    else:
        status = "blocked_no_validation_visible_route_policy_signal"
        next_action = "do not train Candidate B; continue manuscript v0 audit or data-registration probe"
        reason = (
            "The current route guard preserves spot_size and no-process line fallback, but boundary axes still trail "
            "strong baselines and the Phase 59 density upper-bound selects mean fallback."
        )
    return {
        "candidate": "Candidate B: validation-auditable route policy",
        "status": status,
        "open_low_capacity_policy_gate": open_low_capacity_policy,
        "a100_80gb_request_now": False,
        "preserves_spot_size_floor": preserves_spot_size,
        "process_route_signal_count": len(process_route_signals),
        "boundary_blocker_count": len(boundary_blockers),
        "phase59_upper_blocks_route_policy": upper_blocks,
        "candidate_a_open_for_seed7": candidate_a_open,
        "next_action": next_action,
        "reason": reason,
        "seed7_gate": (
            "if reopened in the future, seed 7 must preserve broad12/broad21 spot_size and improve a boundary axis "
            "without using test labels for route selection"
        ),
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(key, "")).replace("\n", " ") for key, _ in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def build_markdown(gate: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "# Phase 70 Route-Policy Non-Training Audit",
            "",
            "## Purpose",
            "",
            "Phase 70 implements the Phase 68 `P68-ROUTE-POLICY` action. It audits existing route evidence before any trainable policy or mixture-of-experts branch is allowed.",
            "",
            "## Candidate B Gate",
            "",
            f"Status: `{gate['status']}`.",
            f"Open low-capacity policy gate: `{str(gate['open_low_capacity_policy_gate']).lower()}`.",
            f"A100-SXM4-80GB request now: `{str(gate['a100_80gb_request_now']).lower()}`.",
            "",
            gate["reason"],
            "",
            "## Audit Rows",
            "",
            _markdown_table(
                rows,
                [
                    ("evidence_source", "Source"),
                    ("dataset", "Dataset"),
                    ("split", "Split"),
                    ("route", "Route"),
                    ("status", "Status"),
                    ("route_policy_signal", "Policy signal"),
                    ("reason", "Reason"),
                ],
            ),
            "",
            "## Next Action",
            "",
            gate["next_action"],
            "",
        ]
    )


def build_package(
    root: Path,
    output_dir: Path,
    paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    resolved = _default_paths(root)
    if paths:
        resolved.update(paths)
    main_rows = _read_csv(resolved["phase60_main"])
    route_rows = _read_csv(resolved["phase60_route"])
    stress_rows = _read_csv(resolved["phase60_stress"])
    upper = _read_json(resolved["phase59_upper"])
    phase68_manifest = _read_json(resolved["phase68_manifest"])
    phase69_gate = _read_json(resolved["phase69_gate"])
    rows = build_audit_rows(main_rows, route_rows, stress_rows)
    gate = build_candidate_gate(rows, upper, phase69_gate)

    output_dir.mkdir(parents=True, exist_ok=True)
    audit_csv = output_dir / "phase70_route_policy_audit_table.csv"
    gate_json = output_dir / "phase70_candidate_b_gate.json"
    markdown_path = output_dir / "phase70_route_policy_audit.md"
    manifest_path = output_dir / "phase70_route_policy_audit_manifest.json"

    _write_csv(audit_csv, rows, AUDIT_FIELDS)
    _write_json(gate_json, gate)
    markdown_path.write_text(build_markdown(gate, rows), encoding="utf-8")
    status_counts: dict[str, int] = {}
    signal_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        signal_counts[row["route_policy_signal"]] = signal_counts.get(row["route_policy_signal"], 0) + 1
    manifest = {
        "phase": 70,
        "objective": "route_policy_non_training_audit",
        "inputs": {key: _display_path(path, root) for key, path in sorted(resolved.items())},
        "outputs": {
            "audit_table": _display_path(audit_csv, root),
            "candidate_b_gate": _display_path(gate_json, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "audit_rows": len(rows),
            "status_counts": status_counts,
            "signal_counts": signal_counts,
        },
        "candidate_b_gate": gate,
        "phase68_decision": phase68_manifest.get("current_decision"),
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase70_route_policy_audit"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    manifest = build_package(root=root, output_dir=output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
