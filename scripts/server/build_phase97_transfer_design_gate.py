#!/usr/bin/env python3
"""Build the Phase 97 AM-Bench / external transfer design gate.

Phase 97 decides whether the Phase 96 fixed heat-kernel / Green's-function
signal has a physically defensible transfer route. It is a design gate only:
it can allow a later local baseline-first smoke gate, but it never opens A100
training by itself.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


TRANSFER_FIELDS = (
    "route_id",
    "source_mechanism",
    "transfer_target",
    "dataset_id",
    "route_family",
    "physical_mapping",
    "registration_status",
    "split_status",
    "baseline_status",
    "leakage_control",
    "phase98_local_smoke_allowed",
    "a100_training_allowed",
    "status",
    "priority",
    "next_action",
    "evidence",
)

PROTOCOL_FIELDS = (
    "protocol_id",
    "component",
    "requirement",
    "status",
    "pass_condition",
    "stop_condition",
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
    if isinstance(value, bool):
        return "true" if value else "false"
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
    phase81 = root / "docs/results/phase81_registered_target_intake_gate"
    phase92 = root / "docs/results/phase92_benchmark_review_intake"
    phase96 = root / "docs/results/phase96_pfhub_local_smoke_gate"
    return {
        "phase81_gate": phase81 / "phase81_registered_target_gate.json",
        "phase81_intake_table": phase81 / "phase81_registered_target_intake_table.csv",
        "phase81_data_card": phase81 / "phase81_registered_target_data_card.json",
        "phase92_gate": phase92 / "phase92_benchmark_review_intake_gate.json",
        "phase96_gate": phase96 / "phase96_pfhub_local_smoke_gate.json",
        "phase96_mechanism_table": phase96 / "phase96_mechanism_decision_table.csv",
    }


def _truthy(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _positive_mechanisms(phase96_gate: dict[str, Any], mechanism_rows: list[dict[str, str]]) -> list[str]:
    positive = phase96_gate.get("positive_mechanisms")
    if isinstance(positive, list) and positive:
        return [str(item) for item in positive]
    return [
        row["mechanism"]
        for row in mechanism_rows
        if _truthy(row.get("transfer_design_signal"))
    ]


def build_protocol_rows() -> list[dict[str, Any]]:
    return [
        {
            "protocol_id": "P97-PROT-001",
            "component": "physical_mapping",
            "requirement": "Every fixed-kernel transfer route must define what the Green's-function/source coordinates mean in the target data.",
            "status": "defined",
            "pass_condition": "source, target, and coordinate systems are compatible without independent test-time rescaling",
            "stop_condition": "feature can only be interpreted as a synthetic basis with no target-data registration",
        },
        {
            "protocol_id": "P97-PROT-002",
            "component": "baseline_contract",
            "requirement": "A later smoke gate must compare against mean/kNN/ExtraTrees or stronger sklearn, no-process Macro PINN, and the relevant route guard.",
            "status": "defined",
            "pass_condition": "all baseline artifacts or commands are available before Phase 98",
            "stop_condition": "candidate cannot be compared to the frozen floor or strongest baseline",
        },
        {
            "protocol_id": "P97-PROT-003",
            "component": "leakage_control",
            "requirement": "Route choice and feature mapping must be decided without test-set labels.",
            "status": "defined",
            "pass_condition": "train/validation-only selection and fixed split manifest",
            "stop_condition": "feature alignment or route selection depends on test performance",
        },
        {
            "protocol_id": "P97-PROT-004",
            "component": "compute_governance",
            "requirement": "Phase 97 may allow only local Phase 98 smoke, never A100 training.",
            "status": "defined",
            "pass_condition": "A100 flags remain false",
            "stop_condition": "any route tries to skip Phase 98 local baseline-first smoke",
        },
    ]


def _route_allows_phase98(row: dict[str, str]) -> bool:
    return (
        row.get("status") == "open_registered_target"
        and row.get("coordinate_registration_status") not in {"", "must_be_verified"}
        and not row.get("status", "").startswith("blocked")
        and row.get("split_status") not in {"", "must_define_train_val_test_split"}
        and row.get("baseline_entry_status") not in {"", "baseline_table_required"}
    )


def _row_for_current_spot_size(source_mechanism: str) -> dict[str, Any]:
    return {
        "route_id": "phase97_current_ambench_spot_size_process_kernel",
        "source_mechanism": source_mechanism,
        "transfer_target": "current broad12/broad21 spot_size holdout",
        "dataset_id": "mds2-2716",
        "route_family": "current_route_guard_reuse",
        "physical_mapping": "not_defined_for_source_kernel; spot_size scalars do not define source-path Green's-function coordinates",
        "registration_status": "blocked_no_source_path_mapping",
        "split_status": "broad12_broad21_process_splits_available",
        "baseline_status": "existing_phase55_60_74_route_guard_and_strong_baselines_available",
        "leakage_control": "do not select kernel route from test metrics; no Phase 98 until physical mapping exists",
        "phase98_local_smoke_allowed": False,
        "a100_training_allowed": False,
        "status": "blocked_no_physical_mapping",
        "priority": 4,
        "next_action": "do not inject heat-kernel/source-path features into broad spot_size without a registered source path or equivalent physical mapping",
        "evidence": "Phase 52/53/71 registration audits and Phase 96 synthetic-only target manifest.",
    }


def _row_from_registered_route(row: dict[str, str], source_mechanism: str) -> dict[str, Any]:
    allowed = _route_allows_phase98(row)
    status = "phase98_local_smoke_ready_no_a100" if allowed else row.get("status", "blocked")
    if allowed:
        next_action = "enter Phase 98 local baseline-first smoke for this registered target"
        leakage_control = "freeze split and registration mapping before local smoke"
    else:
        next_action = row.get("next_action") or "resolve registration/data-card blockers first"
        leakage_control = "blocked route cannot be used for feature selection or A100 training"
    return {
        "route_id": f"phase97_{row['route_id']}",
        "source_mechanism": source_mechanism,
        "transfer_target": row.get("target_family", ""),
        "dataset_id": row.get("dataset_id", ""),
        "route_family": row.get("route_family", ""),
        "physical_mapping": row.get("source_family", ""),
        "registration_status": row.get("coordinate_registration_status", ""),
        "split_status": row.get("split_status", ""),
        "baseline_status": row.get("baseline_entry_status", ""),
        "leakage_control": leakage_control,
        "phase98_local_smoke_allowed": allowed,
        "a100_training_allowed": False,
        "status": status,
        "priority": int(row.get("priority") or 99),
        "next_action": next_action,
        "evidence": row.get("evidence", ""),
    }


def _row_for_pfhub_appendix(source_mechanism: str) -> dict[str, Any]:
    return {
        "route_id": "phase97_pfhub_only_appendix_extension",
        "source_mechanism": source_mechanism,
        "transfer_target": "PFHub-style synthetic benchmark only",
        "dataset_id": "pfhub_style_local",
        "route_family": "synthetic_benchmark_appendix",
        "physical_mapping": "defined inside manufactured benchmark but not transferred to AM-Bench",
        "registration_status": "synthetic_registered_by_definition",
        "split_status": "fixed_train_validation_test_grids",
        "baseline_status": "phase96_local_baselines_available",
        "leakage_control": "can be reported only as local mechanism evidence, not as AM-Bench performance",
        "phase98_local_smoke_allowed": False,
        "a100_training_allowed": False,
        "status": "synthetic_appendix_only",
        "priority": 5,
        "next_action": "keep as appendix/local mechanism evidence unless a registered target appears",
        "evidence": "Phase 96 local smoke gate.",
    }


def build_transfer_rows(
    *,
    positive_mechanisms: list[str],
    phase81_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not positive_mechanisms:
        return rows
    source_mechanism = positive_mechanisms[0]
    rows.append(_row_for_current_spot_size(source_mechanism))
    rows.extend(_row_from_registered_route(row, source_mechanism) for row in phase81_rows)
    rows.append(_row_for_pfhub_appendix(source_mechanism))
    return sorted(rows, key=lambda row: int(row["priority"]))


def build_gate(
    *,
    phase96_gate: dict[str, Any],
    phase81_gate: dict[str, Any],
    transfer_rows: list[dict[str, Any]],
    positive_mechanisms: list[str],
) -> dict[str, Any]:
    phase96_allows = bool(phase96_gate.get("phase97_transfer_design_allowed"))
    allowed_rows = [row for row in transfer_rows if row["phase98_local_smoke_allowed"]]
    if not phase96_allows or not positive_mechanisms:
        status = "blocked_by_phase96_local_smoke"
        next_action = "close transfer design; no positive local smoke mechanism is available"
        phase98_allowed = False
    elif allowed_rows:
        status = "transfer_design_ready_no_a100"
        next_action = "enter Phase 98 local baseline-first smoke for the highest-priority registered route"
        phase98_allowed = True
    else:
        status = "blocked_no_registered_transfer_target"
        next_action = "resolve pad registration or add an external registered data card before Phase 98"
        phase98_allowed = False
    return {
        "status": status,
        "source_phase96_status": phase96_gate.get("status"),
        "source_phase81_status": phase81_gate.get("status"),
        "positive_mechanisms": positive_mechanisms,
        "phase98_local_smoke_allowed": phase98_allowed,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "submission_ready": False,
        "transfer_rows": len(transfer_rows),
        "phase98_candidate_rows": len(allowed_rows),
        "blocked_rows": sum(1 for row in transfer_rows if row["status"].startswith("blocked")),
        "synthetic_only_rows": sum(1 for row in transfer_rows if row["status"] == "synthetic_appendix_only"),
        "preferred_phase98_route": allowed_rows[0]["route_id"] if allowed_rows else "none",
        "next_action": next_action,
        "required_before_a100_training": [
            "Phase 98 local baseline-first smoke",
            "strong baseline deltas",
            "non-worse global/hot/gradient metrics",
            "fixed train/validation/test split and registration mapping",
            "server validation from a pushed commit",
        ],
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_csv_value(row.get(key)).replace("\n", " ") for key, _ in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def build_markdown(
    gate: dict[str, Any],
    transfer_rows: list[dict[str, Any]],
    protocol_rows: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "# Phase 97 AM-Bench / External Transfer Design Gate",
            "",
            "## Gate Decision",
            "",
            f"Status: `{gate['status']}`.",
            f"Phase 98 local smoke allowed: `{str(gate['phase98_local_smoke_allowed']).lower()}`.",
            f"A100 training allowed now: `{str(gate['a100_training_allowed_now']).lower()}`.",
            f"A100-SXM4-80GB request now: `{str(gate['a100_80gb_request_now']).lower()}`.",
            "",
            "Phase 97 is a transfer-design gate. It cannot turn Phase 96 synthetic evidence into A100 training.",
            "",
            "## Transfer Routes",
            "",
            _markdown_table(
                transfer_rows,
                [
                    ("route_id", "Route"),
                    ("transfer_target", "Target"),
                    ("registration_status", "Registration"),
                    ("status", "Status"),
                    ("phase98_local_smoke_allowed", "Phase 98"),
                    ("next_action", "Next action"),
                ],
            ),
            "",
            "## Protocol",
            "",
            _markdown_table(
                protocol_rows,
                [
                    ("protocol_id", "Protocol"),
                    ("component", "Component"),
                    ("pass_condition", "Pass"),
                    ("stop_condition", "Stop"),
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

    phase96_gate = _read_json(resolved["phase96_gate"])
    phase96_mechanisms = _read_csv(resolved["phase96_mechanism_table"])
    phase81_gate = _read_json(resolved["phase81_gate"])
    phase81_rows = _read_csv(resolved["phase81_intake_table"])
    phase81_data_card = _read_json(resolved["phase81_data_card"])
    phase92_gate = _read_json(resolved["phase92_gate"])

    positive_mechanisms = _positive_mechanisms(phase96_gate, phase96_mechanisms)
    transfer_rows = build_transfer_rows(
        positive_mechanisms=positive_mechanisms,
        phase81_rows=phase81_rows,
    )
    protocol_rows = build_protocol_rows()
    gate = build_gate(
        phase96_gate=phase96_gate,
        phase81_gate=phase81_gate,
        transfer_rows=transfer_rows,
        positive_mechanisms=positive_mechanisms,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    transfer_path = output_dir / "phase97_transfer_route_table.csv"
    protocol_path = output_dir / "phase97_transfer_design_protocol.csv"
    gate_path = output_dir / "phase97_transfer_design_gate.json"
    markdown_path = output_dir / "phase97_transfer_design_gate.md"
    manifest_path = output_dir / "phase97_transfer_design_gate_manifest.json"

    _write_csv(transfer_path, transfer_rows, TRANSFER_FIELDS)
    _write_csv(protocol_path, protocol_rows, PROTOCOL_FIELDS)
    _write_json(gate_path, gate)
    markdown_path.write_text(build_markdown(gate, transfer_rows, protocol_rows), encoding="utf-8")

    manifest = {
        "phase": 97,
        "objective": "ambench_external_transfer_design_gate",
        "inputs": {key: _display_path(path, root) for key, path in sorted(resolved.items())},
        "outputs": {
            "transfer_route_table": _display_path(transfer_path, root),
            "transfer_design_protocol": _display_path(protocol_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "transfer_rows": len(transfer_rows),
            "protocol_rows": len(protocol_rows),
            "phase98_candidate_rows": gate["phase98_candidate_rows"],
            "blocked_rows": gate["blocked_rows"],
        },
        "gate": gate,
        "phase96_gate": {
            "status": phase96_gate.get("status"),
            "phase97_transfer_design_allowed": phase96_gate.get("phase97_transfer_design_allowed"),
            "positive_mechanisms": phase96_gate.get("positive_mechanisms"),
        },
        "phase81_gate": {
            "status": phase81_gate.get("status"),
            "open_registered_target_count": phase81_gate.get("open_registered_target_count"),
            "preferred_next_route": phase81_gate.get("preferred_next_route"),
        },
        "phase81_data_card_status": phase81_data_card.get("gate_status"),
        "phase92_gate": {
            "status": phase92_gate.get("status"),
            "submission_ready": phase92_gate.get("submission_ready"),
            "benchmark_review_ready": phase92_gate.get("benchmark_review_ready"),
        },
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase97_transfer_design_gate"),
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
