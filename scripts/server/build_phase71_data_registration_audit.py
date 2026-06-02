#!/usr/bin/env python3
"""Build the Phase 71 data-registration audit for Candidate C.

Phase 71 implements the Phase 68 `P68-DATA-REGISTRATION` action. It does not
train a model. It audits whether heat-kernel, Green's-function, or source-path
features have a physically registered target/source pair before any Macro PINN
integration or A100 validation is allowed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


AUDIT_FIELDS = (
    "evidence_source",
    "target_family",
    "source_path_family",
    "coordinate_status",
    "unit_status",
    "coverage_status",
    "registration_status",
    "feature_route_status",
    "paper_use",
    "status",
    "blocker",
    "evidence",
)

OPEN_STATUSES = {
    "aligned_single_track_source_path",
    "aligned_pad_target_available",
    "registered_feature_gate_ready",
}

BLOCKING_STATUSES = {
    "blocked_single_track_source_path",
    "blocked_broad_source_path",
    "blocked_pad_registration",
    "phase60_blocks_candidate_c",
    "phase68_blocks_candidate_c",
}

DIAGNOSTIC_STATUSES = {
    "diagnostic_global_regression",
    "diagnostic_all_metric_regression",
    "diagnostic_only",
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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
    phase68 = root / "docs/results/phase68_validation_signal_scorecard"
    return {
        "phase52_doc": root / "docs/results/ambench_registered_source_path_feature_gate_v1.md",
        "phase53_doc": root / "docs/results/ambench_source_path_data_pivot_gate_v1.md",
        "phase60_next_gate": phase60 / "phase60_next_branch_gate_table.csv",
        "phase68_scorecard": phase68 / "phase68_candidate_signal_scorecard.csv",
        "phase68_manifest": phase68 / "phase68_validation_signal_scorecard_manifest.json",
    }


def _has_positive_phase52_registration(text: str) -> bool:
    lower = text.lower()
    return (
        "formal decision is positive" in lower
        and "coordinate.compatible=true" in lower
        and "single-track scan-path source aligned" in lower
    )


def _has_positive_phase53_inventory(text: str) -> bool:
    lower = text.lower()
    if "formal inventory decision" not in lower or "positive" not in lower:
        return False
    blockers = (
        "single-track scan-path groups | none" in lower,
        "hdf5 registration metadata keys | none found" in lower,
        "formal inventory decision:\n\n```text\nnegative" in lower,
        "negative:" in lower,
    )
    return not any(blockers)


def _has_registered_pad_target(text: str) -> bool:
    lower = text.lower()
    return (
        "pad thermography groups" in lower
        and "xypt/xpad" in lower
        and "documented camera-pixel to galvo-mm registration" in lower
        and "none found" not in lower
    )


def _phase52_rows(text: str) -> list[dict[str, Any]]:
    if _has_positive_phase52_registration(text):
        return [
            {
                "evidence_source": "phase52_registered_source_path_gate",
                "target_family": "single_track_thermography",
                "source_path_family": "single_track_scan_path",
                "coordinate_status": "coordinate.compatible=true",
                "unit_status": "registered_physical_units",
                "coverage_status": "aligned_target_available",
                "registration_status": "registered_compatible",
                "feature_route_status": "ready_for_fixed_feature_gate",
                "paper_use": "candidate_c_reopen_evidence",
                "status": "aligned_single_track_source_path",
                "blocker": "",
                "evidence": "Formal decision is positive with aligned single-track scan path.",
            }
        ]
    return [
        {
            "evidence_source": "phase52_registered_source_path_gate",
            "target_family": "single_track_thermography",
            "source_path_family": "pad_xypt_only",
            "coordinate_status": "camera_pixels_vs_galvo_mm",
            "unit_status": "unit_mismatch_without_registration",
            "coverage_status": "single_track_target_not_covered_by_pad_xypt",
            "registration_status": "not_registered",
            "feature_route_status": "do_not_build_source_path_features",
            "paper_use": "appendix_data_blocker",
            "status": "blocked_single_track_source_path",
            "blocker": "pad XYPT groups cannot be used as registered source paths for Line_* thermography without a documented mapping",
            "evidence": "Phase 52 formal decision is negative for Line_0_1.",
        }
    ]


def _phase53_inventory_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if _has_positive_phase53_inventory(text):
        rows.append(
            {
                "evidence_source": "phase53_source_path_inventory",
                "target_family": "single_track_or_registered_pad",
                "source_path_family": "registered_scan_path",
                "coordinate_status": "coordinate_metadata_available",
                "unit_status": "registered_physical_units",
                "coverage_status": "target_and_source_overlap",
                "registration_status": "registered_compatible",
                "feature_route_status": "ready_for_fixed_feature_gate",
                "paper_use": "candidate_c_reopen_evidence",
                "status": "registered_feature_gate_ready",
                "blocker": "",
                "evidence": "Phase 53 inventory reports aligned scan path and registration metadata.",
            }
        )
    else:
        rows.append(
            {
                "evidence_source": "phase53_source_path_inventory",
                "target_family": "broad_single_track_thermography",
                "source_path_family": "pad_xypt_only",
                "coordinate_status": "no_single_track_scan_path_groups",
                "unit_status": "camera_pixels_not_mapped_to_galvo_mm",
                "coverage_status": "broad12_broad21_source_path_unavailable",
                "registration_status": "not_registered",
                "feature_route_status": "do_not_run_broad_source_path_validation",
                "paper_use": "appendix_data_blocker",
                "status": "blocked_broad_source_path",
                "blocker": "no single-track scan-path groups and no HDF5 camera-pixel to galvo-mm registration metadata",
                "evidence": "Phase 53 formal inventory decision is negative.",
            }
        )
    if _has_registered_pad_target(text):
        rows.append(
            {
                "evidence_source": "phase53_pad_inventory",
                "target_family": "pad_thermography",
                "source_path_family": "pad_xypt",
                "coordinate_status": "documented_pad_registration",
                "unit_status": "registered_physical_units",
                "coverage_status": "pad_target_and_source_overlap",
                "registration_status": "registered_compatible",
                "feature_route_status": "ready_for_fixed_feature_gate",
                "paper_use": "candidate_c_reopen_evidence",
                "status": "aligned_pad_target_available",
                "blocker": "",
                "evidence": "Pad thermography has documented camera-pixel to galvo-mm registration.",
            }
        )
    else:
        rows.append(
            {
                "evidence_source": "phase53_pad_inventory",
                "target_family": "pad_thermography",
                "source_path_family": "pad_xypt",
                "coordinate_status": "independent_rescale_only",
                "unit_status": "galvo_mm_vs_camera_pixels",
                "coverage_status": "pad_tables_exist_but_unregistered",
                "registration_status": "not_paper_registered",
                "feature_route_status": "diagnostic_only",
                "paper_use": "appendix_data_blocker",
                "status": "blocked_pad_registration",
                "blocker": "pad thermography and pad XYPT exist, but no HDF5 registration metadata was found",
                "evidence": "Phase 53 reports pad diagnostics as independent-rescale only.",
            }
        )
    return rows


def _phase53_pad_diagnostic_rows(text: str) -> list[dict[str, Any]]:
    lower = text.lower()
    rows: list[dict[str, Any]] = []
    if "x_pad1" in lower:
        if "improves hot/gradient but worsens global" in lower or "worsens global rmse" in lower:
            status = "diagnostic_global_regression"
            blocker = "X_pad1 rescale diagnostic improves focused regions but worsens global RMSE"
        else:
            status = "diagnostic_only"
            blocker = "X_pad1 result is diagnostic-only without registration metadata"
        rows.append(
            {
                "evidence_source": "phase53_x_pad1_rescale_diagnostic",
                "target_family": "pad_thermography_x_pad1",
                "source_path_family": "pad_xypt_xpad",
                "coordinate_status": "coordinate.compatible=false",
                "unit_status": "independent_rescale",
                "coverage_status": "diagnostic_target_only",
                "registration_status": "not_paper_registered",
                "feature_route_status": "failed_combined_metric_gate",
                "paper_use": "appendix_negative_diagnostic",
                "status": status,
                "blocker": blocker,
                "evidence": "Phase 53 X_pad1 registered-path rescale diagnostic.",
            }
        )
    if "y_pad1" in lower:
        if "worsens global, hot, gradient" in lower or "worsens global/hot/gradient" in lower:
            status = "diagnostic_all_metric_regression"
            blocker = "Y_pad1 rescale diagnostic worsens global, hot, gradient, and/or coverage"
        else:
            status = "diagnostic_only"
            blocker = "Y_pad1 result is diagnostic-only without registration metadata"
        rows.append(
            {
                "evidence_source": "phase53_y_pad1_rescale_diagnostic",
                "target_family": "pad_thermography_y_pad1",
                "source_path_family": "pad_xypt_ypad",
                "coordinate_status": "coordinate.compatible=false",
                "unit_status": "independent_rescale",
                "coverage_status": "diagnostic_target_only",
                "registration_status": "not_paper_registered",
                "feature_route_status": "failed_combined_metric_gate",
                "paper_use": "appendix_negative_diagnostic",
                "status": status,
                "blocker": blocker,
                "evidence": "Phase 53 Y_pad1 registered-path rescale diagnostic.",
            }
        )
    return rows


def _candidate_c_phase60_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    for row in rows:
        if (row.get("branch") or "").startswith("Candidate C"):
            return row
    return None


def _candidate_c_phase68_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    for row in rows:
        if row.get("candidate_id") == "C":
            return row
    return None


def _phase_gate_rows(
    phase60_next_gate: list[dict[str, str]],
    phase68_scorecard: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    phase60_c = _candidate_c_phase60_row(phase60_next_gate)
    if phase60_c is not None:
        blocked = "blocked" in (phase60_c.get("status") or "").lower()
        rows.append(
            {
                "evidence_source": "phase60_next_branch_gate",
                "target_family": "candidate_c",
                "source_path_family": "heat_kernel_or_green_function_features",
                "coordinate_status": "requires_aligned_source_path",
                "unit_status": "requires_compatible_units",
                "coverage_status": "requires_feature_gate_before_seed_expansion",
                "registration_status": "blocked_by_prior_evidence" if blocked else "candidate_gate_open",
                "feature_route_status": phase60_c.get("focused_validation"),
                "paper_use": phase60_c.get("manuscript_rule"),
                "status": "phase60_blocks_candidate_c" if blocked else "registered_feature_gate_ready",
                "blocker": phase60_c.get("entry_condition") if blocked else "",
                "evidence": "Phase 60 next-branch gate for Candidate C.",
            }
        )
    phase68_c = _candidate_c_phase68_row(phase68_scorecard)
    if phase68_c is not None:
        blocked = "blocked" in (phase68_c.get("status") or "").lower()
        rows.append(
            {
                "evidence_source": "phase68_candidate_signal_scorecard",
                "target_family": "candidate_c",
                "source_path_family": "data_aligned_physics_features",
                "coordinate_status": "registration_signal_missing" if blocked else "registration_signal_available",
                "unit_status": "requires_unit_compatibility",
                "coverage_status": "requires_broad12_broad21_gate",
                "registration_status": phase68_c.get("status"),
                "feature_route_status": phase68_c.get("required_first_action"),
                "paper_use": phase68_c.get("manuscript_use"),
                "status": "phase68_blocks_candidate_c" if blocked else "registered_feature_gate_ready",
                "blocker": phase68_c.get("blocking_evidence") if blocked else "",
                "evidence": phase68_c.get("evidence_locator"),
            }
        )
    return rows


def build_audit_rows(
    phase52_text: str,
    phase53_text: str,
    phase60_next_gate: list[dict[str, str]],
    phase68_scorecard: list[dict[str, str]],
) -> list[dict[str, Any]]:
    return [
        *_phase52_rows(phase52_text),
        *_phase53_inventory_rows(phase53_text),
        *_phase53_pad_diagnostic_rows(phase53_text),
        *_phase_gate_rows(phase60_next_gate, phase68_scorecard),
    ]


def build_candidate_gate(rows: list[dict[str, Any]], phase68_manifest: dict[str, Any]) -> dict[str, Any]:
    aligned_rows = [row for row in rows if row["status"] in OPEN_STATUSES]
    blocking_rows = [row for row in rows if row["status"] in BLOCKING_STATUSES]
    diagnostic_rows = [row for row in rows if row["status"] in DIAGNOSTIC_STATUSES]
    open_gate = bool(aligned_rows)
    if open_gate:
        status = "opened_for_aligned_feature_gate"
        next_action = (
            "build a no-training fixed heat-kernel/Green's-function feature gate on the aligned target before Macro PINN integration"
        )
        reason = "At least one source/target pair is registered and compatible enough for a fixed-feature gate."
    else:
        status = "blocked_by_registration_data"
        next_action = "do not train Candidate C; continue manuscript v0 audit or external data-registration planning"
        reason = (
            "Current AM-Bench evidence has pad XYPT but no paper-facing single-track registration or pad camera-to-galvo mapping."
        )
    return {
        "candidate": "Candidate C: data-aligned heat-kernel or Green's-function features",
        "status": status,
        "open_aligned_feature_gate": open_gate,
        "fixed_feature_gate_allowed": open_gate,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "aligned_target_count": len(aligned_rows),
        "blocking_row_count": len(blocking_rows),
        "diagnostic_row_count": len(diagnostic_rows),
        "phase68_trainable_model_opened": bool(
            (phase68_manifest.get("current_decision") or {}).get("trainable_model_opened")
        ),
        "next_action": next_action,
        "reason": reason,
        "seed7_gate": (
            "if reopened in the future, seed 7 must be non-worse than broad_process_v1 on broad12 and broad21 "
            "for rmse, hot_q90_rmse, and gradient_q90_rmse"
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
            "# Phase 71 Data-Registration Audit",
            "",
            "## Purpose",
            "",
            "Phase 71 implements the Phase 68 `P68-DATA-REGISTRATION` action. It checks whether Candidate C can reopen heat-kernel, Green's-function, or source-path features before any A100 training.",
            "",
            "## Candidate C Gate",
            "",
            f"Status: `{gate['status']}`.",
            f"Open aligned feature gate: `{str(gate['open_aligned_feature_gate']).lower()}`.",
            f"Fixed feature gate allowed: `{str(gate['fixed_feature_gate_allowed']).lower()}`.",
            f"A100 training allowed now: `{str(gate['a100_training_allowed_now']).lower()}`.",
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
                    ("target_family", "Target"),
                    ("source_path_family", "Source path"),
                    ("registration_status", "Registration"),
                    ("feature_route_status", "Feature route"),
                    ("status", "Status"),
                    ("blocker", "Blocker"),
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

    phase52_text = _read_text(resolved["phase52_doc"])
    phase53_text = _read_text(resolved["phase53_doc"])
    phase60_next_gate = _read_csv(resolved["phase60_next_gate"])
    phase68_scorecard = _read_csv(resolved["phase68_scorecard"])
    phase68_manifest = _read_json(resolved["phase68_manifest"])
    rows = build_audit_rows(
        phase52_text=phase52_text,
        phase53_text=phase53_text,
        phase60_next_gate=phase60_next_gate,
        phase68_scorecard=phase68_scorecard,
    )
    gate = build_candidate_gate(rows, phase68_manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    audit_csv = output_dir / "phase71_data_registration_audit_table.csv"
    gate_json = output_dir / "phase71_candidate_c_gate.json"
    markdown_path = output_dir / "phase71_data_registration_audit.md"
    manifest_path = output_dir / "phase71_data_registration_audit_manifest.json"

    _write_csv(audit_csv, rows, AUDIT_FIELDS)
    _write_json(gate_json, gate)
    markdown_path.write_text(build_markdown(gate, rows), encoding="utf-8")
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    manifest = {
        "phase": 71,
        "objective": "data_registration_non_training_audit",
        "inputs": {key: _display_path(path, root) for key, path in sorted(resolved.items())},
        "outputs": {
            "audit_table": _display_path(audit_csv, root),
            "candidate_c_gate": _display_path(gate_json, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "audit_rows": len(rows),
            "status_counts": status_counts,
        },
        "candidate_c_gate": gate,
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
        default=Path("docs/results/phase71_data_registration_audit"),
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
