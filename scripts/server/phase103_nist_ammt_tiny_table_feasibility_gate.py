#!/usr/bin/env python3
"""Phase 103 tiny registered-table feasibility gate.

This gate consumes the Phase 103 intake audit, schema scout, and member schema
sampler artifacts. It decides whether manual construction of a tiny registered
source/path-to-target table may begin. It does not extract training rows, build
splits, run baselines, or open training gates.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_ROLES = (
    "coordinate_transform",
    "trigger_timing",
    "source_command_path",
    "target_observation",
)

OPTIONAL_ROLES = ("split_key",)

ROLE_FIELDS = (
    "role",
    "required",
    "scout_hits",
    "sample_rows",
    "sampled_text_rows",
    "non_text_rows",
    "member_names",
    "header_lines",
    "status",
)


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


def _read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
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


def _roles_for_sample(row: dict[str, str]) -> set[str]:
    return {role for role in str(row.get("roles", "")).split(";") if role}


def _unique_join(values: list[str], limit: int = 5) -> str:
    seen: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
        if len(seen) >= limit:
            break
    return ";".join(seen)


def _role_evidence_rows(
    *,
    scout_gate: dict[str, Any] | None,
    deep_probe_gate: dict[str, Any] | None,
    join_probe_gate: dict[str, Any] | None,
    sample_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    role_hits = scout_gate.get("role_hits", {}) if scout_gate else {}
    rows: list[dict[str, Any]] = []
    for role in (*REQUIRED_ROLES, *OPTIONAL_ROLES):
        matching = [row for row in sample_rows if role in _roles_for_sample(row)]
        text_rows = [row for row in matching if row.get("sample_status") == "sampled_text"]
        non_text_rows = [
            row for row in matching if row.get("sample_status") == "non_text_preview_skipped"
        ]
        scout_hits = int(role_hits.get(role) or 0)
        deep_target_ready = (
            role == "target_observation"
            and deep_probe_gate is not None
            and bool(deep_probe_gate.get("target_observation_binary_schema_ready"))
        )
        deep_timing_ready = (
            role == "trigger_timing"
            and deep_probe_gate is not None
            and bool(deep_probe_gate.get("explicit_trigger_timing_ready"))
        )
        layer_join_ready = (
            role == "trigger_timing"
            and join_probe_gate is not None
            and bool(join_probe_gate.get("source_target_join_ready"))
        )
        if scout_hits == 0 and not (deep_target_ready or deep_timing_ready or layer_join_ready):
            status = "missing_scout_candidate"
        elif not matching:
            if deep_target_ready:
                status = "ready_for_manual_join_review_binary_schema"
            elif deep_timing_ready:
                status = "ready_for_manual_join_review_deep_timing"
            elif layer_join_ready:
                status = "ready_for_manual_join_review_layer_join"
            else:
                status = "missing_member_sample"
        elif not text_rows:
            if deep_target_ready:
                status = "ready_for_manual_join_review_binary_schema"
            else:
                status = "missing_text_schema_sample"
        else:
            status = "ready_for_manual_join_review"
        rows.append(
            {
                "role": role,
                "required": role in REQUIRED_ROLES,
                "scout_hits": scout_hits,
                "sample_rows": len(matching),
                "sampled_text_rows": len(text_rows),
                "non_text_rows": len(non_text_rows),
                "member_names": _unique_join([row.get("member_name", "") for row in matching]),
                "header_lines": _unique_join([row.get("header_line", "") for row in text_rows]),
                "status": status,
            }
        )
    return rows


def build_gate(
    *,
    intake_gate: dict[str, Any] | None,
    scout_gate: dict[str, Any] | None,
    sampler_gate: dict[str, Any] | None,
    deep_probe_gate: dict[str, Any] | None,
    join_probe_gate: dict[str, Any] | None,
    role_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    required_not_ready = [
        row
        for row in role_rows
        if row["required"] and not str(row["status"]).startswith("ready_for_manual_join_review")
    ]
    if intake_gate is None:
        status = "intake_audit_gate_required"
        next_action = "run Phase 103 intake audit and large-download audit first"
    elif not bool(intake_gate.get("metadata_ready")):
        status = "intake_audit_not_ready"
        next_action = "finish Metadata.zip intake before schema feasibility review"
    elif scout_gate is None:
        status = "schema_scout_gate_required"
        next_action = "run Phase 103 schema scout after large ZIP downloads complete"
    elif sampler_gate is None:
        status = "member_schema_sampler_gate_required"
        next_action = "run Phase 103 member schema sampler on scout candidates"
    elif required_not_ready:
        if scout_gate.get("status") != "schema_candidates_ready_manual_sampling_required":
            status = "schema_scout_not_ready"
            next_action = "finish schema/deep registration review before tiny-table feasibility review"
        elif sampler_gate.get("status") != "member_schema_samples_ready_manual_registration_required":
            status = "member_schema_sampler_not_ready"
            next_action = "sample enough candidate members or use deep probe evidence for blocked roles"
        else:
            status = "tiny_registered_table_feasibility_blocked"
            next_action = "inspect additional candidates or add a targeted schema sampler for blocked roles"
    else:
        status = "tiny_registered_table_construction_allowed_training_locked"
        next_action = "manually construct a tiny registered source/path-to-target sample table and split manifest"
    return {
        "status": status,
        "intake_status": intake_gate.get("status") if intake_gate else "missing",
        "schema_scout_status": scout_gate.get("status") if scout_gate else "missing",
        "member_schema_sampler_status": sampler_gate.get("status") if sampler_gate else "missing",
        "deep_registration_probe_status": deep_probe_gate.get("status") if deep_probe_gate else "missing",
        "join_probe_status": join_probe_gate.get("status") if join_probe_gate else "missing",
        "source_target_join_ready": (
            bool(join_probe_gate.get("source_target_join_ready")) if join_probe_gate else False
        ),
        "explicit_absolute_timing_ready": (
            bool(join_probe_gate.get("explicit_absolute_timing_ready")) if join_probe_gate else False
        ),
        "required_roles": list(REQUIRED_ROLES),
        "optional_roles": list(OPTIONAL_ROLES),
        "required_roles_ready": len(required_not_ready) == 0,
        "missing_or_blocked_required_roles": [row["role"] for row in required_not_ready],
        "role_statuses": {row["role"]: row["status"] for row in role_rows},
        "tiny_registered_table_construction_allowed": (
            status == "tiny_registered_table_construction_allowed_training_locked"
        ),
        "tiny_registered_table_ready": False,
        "leakage_safe_split_manifest_ready": False,
        "phase104_baseline_smoke_allowed": False,
        "phase105_model_mechanism_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": next_action,
        "required_before_baseline_smoke": [
            "tiny registered sample table generated",
            "coordinate transform values applied to sample rows",
            "trigger/timing join applied to sample rows",
            "source command/path rows joined to target observations",
            "leakage-safe split manifest generated",
            "CPU baseline smoke plan validated without test-label route selection",
        ],
    }


def _write_markdown(path: Path, gate: dict[str, Any], role_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 103 Tiny Registered-Table Feasibility Gate",
        "",
        f"- Status: `{gate['status']}`",
        f"- Tiny-table construction allowed: `{gate['tiny_registered_table_construction_allowed']}`",
        "- Phase 104 baseline smoke allowed: `false`",
        "- A100 training allowed now: `false`",
        f"- Next action: {gate['next_action']}",
        "",
        "| Role | Required | Scout hits | Sample rows | Text rows | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in role_rows:
        lines.append(
            "| {role} | {required} | {scout_hits} | {sample_rows} | {sampled_text_rows} | {status} |".format(
                **row
            )
        )
    lines.append("")
    lines.append(
        "This package is a no-training feasibility gate. It does not create the tiny registered "
        "table, leakage-safe split manifest, baselines, model mechanisms, or any A100 training claim."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_package(
    *,
    root: Path,
    output_dir: Path,
    intake_gate_path: Path,
    scout_gate_path: Path,
    sampler_gate_path: Path,
    deep_probe_gate_path: Path,
    join_probe_gate_path: Path,
    samples_csv_path: Path,
) -> dict[str, Any]:
    intake_gate = _read_json_if_exists(intake_gate_path)
    scout_gate = _read_json_if_exists(scout_gate_path)
    sampler_gate = _read_json_if_exists(sampler_gate_path)
    deep_probe_gate = _read_json_if_exists(deep_probe_gate_path)
    join_probe_gate = _read_json_if_exists(join_probe_gate_path)
    sample_rows = _read_csv_if_exists(samples_csv_path)
    role_rows = _role_evidence_rows(
        scout_gate=scout_gate,
        deep_probe_gate=deep_probe_gate,
        join_probe_gate=join_probe_gate,
        sample_rows=sample_rows,
    )
    gate = build_gate(
        intake_gate=intake_gate,
        scout_gate=scout_gate,
        sampler_gate=sampler_gate,
        deep_probe_gate=deep_probe_gate,
        join_probe_gate=join_probe_gate,
        role_rows=role_rows,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    roles_path = output_dir / "phase103_nist_ammt_tiny_table_feasibility_roles.csv"
    gate_path = output_dir / "phase103_nist_ammt_tiny_table_feasibility_gate.json"
    markdown_path = output_dir / "phase103_nist_ammt_tiny_table_feasibility_summary.md"
    manifest_path = output_dir / "phase103_nist_ammt_tiny_table_feasibility_manifest.json"
    _write_csv(roles_path, role_rows, ROLE_FIELDS)
    _write_json(gate_path, gate)
    _write_markdown(markdown_path, gate, role_rows)
    manifest = {
        "phase": 103,
        "objective": "nist_ammt_tiny_registered_table_feasibility_gate_no_training",
        "inputs": {
            "intake_gate": _display_path(intake_gate_path, root),
            "schema_scout_gate": _display_path(scout_gate_path, root),
            "member_schema_sampler_gate": _display_path(sampler_gate_path, root),
            "deep_registration_probe_gate": _display_path(deep_probe_gate_path, root),
            "join_probe_gate": _display_path(join_probe_gate_path, root),
            "member_schema_samples": _display_path(samples_csv_path, root),
        },
        "outputs": {
            "roles": _display_path(roles_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown_summary": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "sample_rows": len(sample_rows),
            "role_rows": len(role_rows),
        },
        "gate": gate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase103_nist_ammt_registered_intake"),
    )
    parser.add_argument(
        "--intake-gate",
        type=Path,
        default=Path(
            "docs/results/phase103_nist_ammt_registered_intake/"
            "phase103_nist_ammt_registered_intake_gate.json"
        ),
    )
    parser.add_argument(
        "--schema-scout-gate",
        type=Path,
        default=Path(
            "docs/results/phase103_nist_ammt_registered_intake/"
            "phase103_nist_ammt_schema_scout_gate.json"
        ),
    )
    parser.add_argument(
        "--member-schema-sampler-gate",
        type=Path,
        default=Path(
            "docs/results/phase103_nist_ammt_registered_intake/"
            "phase103_nist_ammt_member_schema_sampler_gate.json"
        ),
    )
    parser.add_argument(
        "--deep-registration-probe-gate",
        type=Path,
        default=Path(
            "docs/results/phase103_nist_ammt_registered_intake/"
            "phase103_nist_ammt_deep_registration_probe_gate.json"
        ),
    )
    parser.add_argument(
        "--join-probe-gate",
        type=Path,
        default=Path(
            "docs/results/phase103_nist_ammt_registered_intake/"
            "phase103_nist_ammt_join_probe_gate.json"
        ),
    )
    parser.add_argument(
        "--member-schema-samples",
        type=Path,
        default=Path(
            "docs/results/phase103_nist_ammt_registered_intake/"
            "phase103_nist_ammt_member_schema_samples.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    intake_gate = args.intake_gate
    scout_gate = args.schema_scout_gate
    sampler_gate = args.member_schema_sampler_gate
    deep_probe_gate = args.deep_registration_probe_gate
    join_probe_gate = args.join_probe_gate
    samples_csv = args.member_schema_samples
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    if not intake_gate.is_absolute():
        intake_gate = root / intake_gate
    if not scout_gate.is_absolute():
        scout_gate = root / scout_gate
    if not sampler_gate.is_absolute():
        sampler_gate = root / sampler_gate
    if not deep_probe_gate.is_absolute():
        deep_probe_gate = root / deep_probe_gate
    if not join_probe_gate.is_absolute():
        join_probe_gate = root / join_probe_gate
    if not samples_csv.is_absolute():
        samples_csv = root / samples_csv
    manifest = build_package(
        root=root,
        output_dir=output_dir,
        intake_gate_path=intake_gate,
        scout_gate_path=scout_gate,
        sampler_gate_path=sampler_gate,
        deep_probe_gate_path=deep_probe_gate,
        join_probe_gate_path=join_probe_gate,
        samples_csv_path=samples_csv,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
