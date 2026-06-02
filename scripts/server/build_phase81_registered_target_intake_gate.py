#!/usr/bin/env python3
"""Build the Phase 81 registered-target intake gate.

Phase 81 is a data-first gate. It inventories candidate registered-target
routes before any heat-kernel, Green's-function, GCN/attention, Bayesian PINN,
or other architecture branch can run A100 validation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


INTAKE_FIELDS = (
    "route_id",
    "route_family",
    "dataset_id",
    "public_record",
    "source_manifest",
    "target_family",
    "source_family",
    "required_files_pinned",
    "required_files_present",
    "local_files_present",
    "process_metadata_status",
    "split_status",
    "coordinate_registration_status",
    "registration_blocker",
    "baseline_entry_status",
    "model_gate_status",
    "paper_use",
    "status",
    "priority",
    "next_action",
    "evidence",
)

OPEN_STATUSES = {"open_registered_target"}
BLOCKED_STATUSES = {
    "blocked_missing_registration",
    "blocked_no_data_card",
    "blocked_for_registered_physics",
}
DIAGNOSTIC_STATUSES = {"diagnostic_only", "diagnostic_prior_unstable"}


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


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object YAML at {path}")
    return payload


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
    return {
        "mds2_2716_manifest": root / "configs/data/ambench_mds2_2716_sources.yaml",
        "mds2_2718_manifest": root / "configs/data/ambench_mds2_2718_sources.yaml",
        "phase71_manifest": root
        / "docs/results/phase71_data_registration_audit/phase71_data_registration_audit_manifest.json",
        "phase71_table": root
        / "docs/results/phase71_data_registration_audit/phase71_data_registration_audit_table.csv",
        "phase80_manifest": root
        / "docs/results/phase80_bounded_spot_size_local_surrogate_gate/phase80_bounded_spot_size_local_surrogate_gate_manifest.json",
    }


def _record_url(manifest: dict[str, Any]) -> str:
    record = manifest.get("record") or {}
    return str(record.get("doi") or record.get("pdr_landing_page") or "")


def _local_root(root: Path, manifest: dict[str, Any]) -> Path:
    local = Path(str(manifest.get("local_root") or "."))
    if not local.is_absolute():
        local = root / local
    return local


def _required_file_state(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    local_root = _local_root(root, manifest)
    required = list(manifest.get("required_files") or [])
    present = 0
    missing_ids: list[str] = []
    for item in required:
        rel = Path(str(item.get("relative_path") or ""))
        if rel and (local_root / rel).exists():
            present += 1
        else:
            missing_ids.append(str(item.get("id") or item.get("relative_path") or "unknown"))
    return {
        "required_count": len(required),
        "present_count": present,
        "missing_ids": missing_ids,
        "all_present": bool(required) and present == len(required),
    }


def _has_phase71_status(rows: list[dict[str, str]], status: str) -> bool:
    return any((row.get("status") or "") == status for row in rows)


def _single_track_status(phase71_rows: list[dict[str, str]], phase71_gate: dict[str, Any]) -> tuple[str, str, str]:
    if _has_phase71_status(phase71_rows, "aligned_single_track_source_path") or bool(
        phase71_gate.get("open_aligned_feature_gate")
    ):
        return (
            "open_registered_target",
            "registered_single_track_source_path",
            "build Phase 82 baseline smoke on the aligned single-track target",
        )
    return (
        "blocked_missing_registration",
        "no single-track scan-path group or camera-pixel to galvo-mm mapping is available",
        "do not build source-path features for Line_* thermography; acquire aligned scan-path metadata or registration",
    )


def _pad_status(phase71_rows: list[dict[str, str]]) -> tuple[str, str, str]:
    if _has_phase71_status(phase71_rows, "aligned_pad_target_available"):
        return (
            "open_registered_target",
            "registered_pad_thermography_xypt",
            "build Phase 82 baseline smoke on the registered pad thermography target",
        )
    return (
        "blocked_missing_registration",
        "pad thermography and pad XYPT exist, but current evidence has only independent-rescale diagnostics",
        "search for documented pad camera-to-galvo registration before fixed source-path features",
    )


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    return counts


def build_intake_rows(
    *,
    root: Path,
    manifest_paths: dict[str, Path],
    mds2_2716: dict[str, Any],
    mds2_2718: dict[str, Any],
    phase71_manifest: dict[str, Any],
    phase71_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    m2716_state = _required_file_state(root, mds2_2716)
    m2718_state = _required_file_state(root, mds2_2718)
    phase71_gate = phase71_manifest.get("candidate_c_gate") or {}
    single_status, single_blocker, single_next = _single_track_status(phase71_rows, phase71_gate)
    pad_status, pad_blocker, pad_next = _pad_status(phase71_rows)
    return [
        {
            "route_id": "ambench_mds2_2716_single_track_scan_path",
            "route_family": "registered_thermal_source_path",
            "dataset_id": str(mds2_2716.get("dataset_id") or "mds2-2716"),
            "public_record": _record_url(mds2_2716),
            "source_manifest": _display_path(manifest_paths["mds2_2716_manifest"], root),
            "target_family": "ThermalData/Line_* single-track thermography",
            "source_family": "single-track scan path or camera-to-galvo registration",
            "required_files_pinned": m2716_state["required_count"],
            "required_files_present": m2716_state["present_count"],
            "local_files_present": m2716_state["all_present"],
            "process_metadata_status": "process_metadata_available",
            "split_status": "broad12_broad21_process_splits_available",
            "coordinate_registration_status": single_blocker
            if single_status != "open_registered_target"
            else "coordinate_compatible",
            "registration_blocker": "" if single_status == "open_registered_target" else single_blocker,
            "baseline_entry_status": "existing_route_guard_baselines_available",
            "model_gate_status": "phase82_required_before_features"
            if single_status == "open_registered_target"
            else "feature_gate_blocked",
            "paper_use": "candidate_registered_target"
            if single_status == "open_registered_target"
            else "appendix_or_future_data_requirement",
            "status": single_status,
            "priority": 2 if single_status == "open_registered_target" else 3,
            "next_action": single_next,
            "evidence": "Phase 71 single-track registration audit plus mds2-2716 source manifest.",
        },
        {
            "route_id": "ambench_mds2_2716_pad_thermography_xypt",
            "route_family": "registered_pad_thermal_source_path",
            "dataset_id": str(mds2_2716.get("dataset_id") or "mds2-2716"),
            "public_record": _record_url(mds2_2716),
            "source_manifest": _display_path(manifest_paths["mds2_2716_manifest"], root),
            "target_family": "ThermalData/X_pad* or Y_pad* pad thermography",
            "source_family": "XYPT/Xpad or XYPT/Ypad scan strategy",
            "required_files_pinned": m2716_state["required_count"],
            "required_files_present": m2716_state["present_count"],
            "local_files_present": m2716_state["all_present"],
            "process_metadata_status": "process_metadata_available",
            "split_status": "pad_frame_or_region_split_possible",
            "coordinate_registration_status": pad_blocker
            if pad_status != "open_registered_target"
            else "coordinate_compatible",
            "registration_blocker": "" if pad_status == "open_registered_target" else pad_blocker,
            "baseline_entry_status": "diagnostic_phase53_baselines_exist",
            "model_gate_status": "phase82_required_before_features"
            if pad_status == "open_registered_target"
            else "diagnostic_only_until_registration",
            "paper_use": "candidate_registered_target"
            if pad_status == "open_registered_target"
            else "highest_priority_data_followup",
            "status": pad_status,
            "priority": 1,
            "next_action": pad_next,
            "evidence": "Phase 53/71 pad inventory shows pad thermography and pad XYPT exist, but registration is not paper-facing yet.",
        },
        {
            "route_id": "ambench_mds2_2718_exact_line_microstructure",
            "route_family": "registered_microstructure_context",
            "dataset_id": str(mds2_2718.get("dataset_id") or "mds2-2718"),
            "public_record": _record_url(mds2_2718),
            "source_manifest": _display_path(manifest_paths["mds2_2718_manifest"], root),
            "target_family": "single-track optical microscopy and melt-pool cross-section measurements",
            "source_family": "exact-line P3/P4 Line_0_1 TIFF panel",
            "required_files_pinned": m2718_state["required_count"],
            "required_files_present": m2718_state["present_count"],
            "local_files_present": m2718_state["all_present"],
            "process_metadata_status": "process_metadata_available_for_exact_line_images",
            "split_status": "limited_image_panel_not_broad_thermal_split",
            "coordinate_registration_status": "not_registered_to_thermal_pixels_or_source_path",
            "registration_blocker": "prior exact-line microstructure features were weak-positive but seed-unstable and not source-path registered",
            "baseline_entry_status": "prior_microstructure_diagnostic_baselines_exist",
            "model_gate_status": "blocked_for_registered_physics",
            "paper_use": "appendix_diagnostic_or_separate_microstructure_branch",
            "status": "diagnostic_prior_unstable",
            "priority": 4,
            "next_action": "do not open GCN/image-encoder training without stronger physical alignment or a separate data card",
            "evidence": "Phases 18-22 exact-line microstructure routes were weak-positive diagnostics but not stable model claims.",
        },
        {
            "route_id": "external_public_registered_thermal_process_dataset",
            "route_family": "external_registered_target",
            "dataset_id": "external_tbd",
            "public_record": "",
            "source_manifest": "",
            "target_family": "public registered thermal/process target",
            "source_family": "aligned scan path, source command, or camera-to-galvo calibration",
            "required_files_pinned": 0,
            "required_files_present": 0,
            "local_files_present": False,
            "process_metadata_status": "must_be_verified",
            "split_status": "must_define_train_val_test_split",
            "coordinate_registration_status": "must_be_verified",
            "registration_blocker": "no source manifest or registration evidence has been provided yet",
            "baseline_entry_status": "baseline_table_required",
            "model_gate_status": "data_card_required_before_model_gate",
            "paper_use": "future_registered_target_or_second_paper_branch",
            "status": "blocked_no_data_card",
            "priority": 2,
            "next_action": "create a public source manifest and registration data card before any model work",
            "evidence": "Phase 80 policy requires a registered target before opening another architecture branch.",
        },
    ]


def build_gate(rows: list[dict[str, Any]], phase71_manifest: dict[str, Any], phase80_manifest: dict[str, Any]) -> dict[str, Any]:
    open_rows = [row for row in rows if row["status"] in OPEN_STATUSES]
    blocked_rows = [row for row in rows if row["status"] in BLOCKED_STATUSES]
    diagnostic_rows = [row for row in rows if row["status"] in DIAGNOSTIC_STATUSES]
    if open_rows:
        status = "opened_for_phase82_baseline_smoke"
        reason = "At least one registered target is public, split-ready, and coordinate compatible."
        next_action = "build Phase 82 baseline smoke for the highest-priority open registered target"
        preferred = min(open_rows, key=lambda row: int(row["priority"]))["route_id"]
    else:
        status = "blocked_no_registered_target"
        reason = (
            "No current route has public reproducibility, split readiness, process metadata, and coordinate-compatible "
            "source/target registration at the same time."
        )
        next_action = (
            "do not run A100 model training; pursue pad camera-to-galvo registration or an external registered-target data card"
        )
        preferred = "ambench_mds2_2716_pad_thermography_xypt"
    phase80_gate = phase80_manifest.get("gate") or {}
    return {
        "candidate": "registered_target_intake",
        "status": status,
        "phase82_baseline_smoke_allowed": bool(open_rows),
        "phase83_registered_feature_gate_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "open_registered_target_count": len(open_rows),
        "blocked_route_count": len(blocked_rows),
        "diagnostic_route_count": len(diagnostic_rows),
        "preferred_next_route": preferred,
        "phase71_candidate_c_status": (phase71_manifest.get("candidate_c_gate") or {}).get("status"),
        "phase80_bounded_spot_size_status": phase80_gate.get("status"),
        "reason": reason,
        "next_action": next_action,
        "required_before_training": [
            "public source manifest with checksums",
            "coordinate-compatible source/target registration",
            "train/validation/test split manifest",
            "baseline-first smoke table",
            "local/no-training feature gate",
        ],
    }


def build_data_card(gate: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "phase": 81,
        "title": "Registered-target data intake card",
        "purpose": (
            "Record whether any data route can support paper-facing heat-kernel, Green's-function, "
            "GCN/attention, Bayesian PINN, or physical-parameterization work."
        ),
        "gate_status": gate["status"],
        "selection_rule": "No architecture training opens unless a route status is open_registered_target and Phase 82 baseline smoke passes.",
        "candidate_routes": rows,
        "paper_floor_preserved": "Phase 55/60/74 broad_process_v1 fixed-sampling spot_size remains the manuscript floor.",
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


def build_markdown(gate: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "# Phase 81 Registered-Target Intake Gate",
            "",
            "## Gate Decision",
            "",
            f"Status: `{gate['status']}`.",
            f"Phase 82 baseline smoke allowed: `{str(gate['phase82_baseline_smoke_allowed']).lower()}`.",
            f"Phase 83 registered feature gate allowed: `{str(gate['phase83_registered_feature_gate_allowed']).lower()}`.",
            f"A100 training allowed now: `{str(gate['a100_training_allowed_now']).lower()}`.",
            f"A100-SXM4-80GB request now: `{str(gate['a100_80gb_request_now']).lower()}`.",
            f"Preferred next route: `{gate['preferred_next_route']}`.",
            "",
            gate["reason"],
            "",
            "## Intake Routes",
            "",
            _markdown_table(
                rows,
                [
                    ("route_id", "Route"),
                    ("target_family", "Target"),
                    ("source_family", "Source"),
                    ("coordinate_registration_status", "Registration"),
                    ("status", "Status"),
                    ("paper_use", "Paper use"),
                    ("next_action", "Next action"),
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
    mds2_2716 = _read_yaml(resolved["mds2_2716_manifest"])
    mds2_2718 = _read_yaml(resolved["mds2_2718_manifest"])
    phase71_manifest = _read_json(resolved["phase71_manifest"])
    phase71_rows = _read_csv(resolved["phase71_table"])
    phase80_manifest = _read_json(resolved["phase80_manifest"])
    rows = build_intake_rows(
        root=root,
        manifest_paths=resolved,
        mds2_2716=mds2_2716,
        mds2_2718=mds2_2718,
        phase71_manifest=phase71_manifest,
        phase71_rows=phase71_rows,
    )
    gate = build_gate(rows, phase71_manifest, phase80_manifest)
    data_card = build_data_card(gate, rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "phase81_registered_target_intake_table.csv"
    gate_path = output_dir / "phase81_registered_target_gate.json"
    data_card_path = output_dir / "phase81_registered_target_data_card.json"
    markdown_path = output_dir / "phase81_registered_target_intake_gate.md"
    manifest_path = output_dir / "phase81_registered_target_intake_gate_manifest.json"

    _write_csv(table_path, rows, INTAKE_FIELDS)
    _write_json(gate_path, gate)
    _write_json(data_card_path, data_card)
    markdown_path.write_text(build_markdown(gate, rows), encoding="utf-8")
    manifest = {
        "phase": 81,
        "objective": "registered_target_data_intake_gate",
        "inputs": {key: _display_path(path, root) for key, path in sorted(resolved.items())},
        "outputs": {
            "intake_table": _display_path(table_path, root),
            "gate_json": _display_path(gate_path, root),
            "data_card": _display_path(data_card_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "intake_rows": len(rows),
            "status_counts": _status_counts(rows),
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
        default=Path("docs/results/phase81_registered_target_intake_gate"),
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
