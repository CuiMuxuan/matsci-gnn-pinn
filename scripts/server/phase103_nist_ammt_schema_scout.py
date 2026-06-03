#!/usr/bin/env python3
"""Phase 103 NIST AMMT schema scout.

This utility scans only ZIP member metadata from the Phase 103 NIST AMMT intake
files. It does not extract files, build training tables, run baselines, or open
model-training gates. The purpose is to find candidate members for coordinate
registration, trigger timing, source command/path, and target observations after
the long-running downloads finish.
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path
from typing import Any


ROLE_KEYWORDS = {
    "coordinate_transform": (
        "transform",
        "coordinate",
        "pixel",
        "ammt",
        "homography",
        "calibration",
        "registration",
        "dotgrid",
        "grid",
    ),
    "trigger_timing": (
        "trigger",
        "timestamp",
        "time",
        "timing",
        "sync",
        "clock",
        "frame_time",
    ),
    "source_command_path": (
        "xypt",
        "command",
        "scan",
        "path",
        "trajectory",
        "galvo",
        "laser",
        "hatch",
        "build_command",
    ),
    "target_observation": (
        "melt",
        "pool",
        "monitor",
        "frame",
        "image",
        "insitu",
        "in_situ",
        "in-situ",
        "camera",
        "mpm",
        "measurement",
    ),
    "split_key": (
        "layer",
        "build",
        "part",
        "track",
        "scan",
        "frame",
        "sample",
        "index",
    ),
}

REQUIRED_ROLES = (
    "coordinate_transform",
    "trigger_timing",
    "source_command_path",
    "target_observation",
)

SUMMARY_FIELDS = (
    "file_id",
    "file_name",
    "expected_bytes",
    "actual_bytes",
    "size_ok",
    "zip_status",
    "member_count_scanned",
    "candidate_rows",
    "coordinate_transform_hits",
    "trigger_timing_hits",
    "source_command_path_hits",
    "target_observation_hits",
    "split_key_hits",
    "status",
)

CANDIDATE_FIELDS = (
    "file_id",
    "file_name",
    "member_name",
    "member_size",
    "extension",
    "roles",
    "coordinate_transform",
    "trigger_timing",
    "source_command_path",
    "target_observation",
    "split_key",
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


def _default_data_card(root: Path) -> Path:
    return (
        root
        / "docs/results/phase102_registered_source_manifest_gate/phase102_nist_ammt_data_card.json"
    )


def _normalized_name(member_name: str) -> str:
    lowered = member_name.lower()
    return (
        lowered.replace("\\", "/")
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("__", "_")
    )


def _roles_for_member(member_name: str) -> dict[str, bool]:
    normalized = _normalized_name(member_name)
    return {
        role: any(keyword in normalized for keyword in keywords)
        for role, keywords in ROLE_KEYWORDS.items()
    }


def _scan_one_zip(
    *,
    file_row: dict[str, Any],
    data_root: Path,
    max_members: int,
    max_candidates_per_role: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    file_id = str(file_row["file_id"])
    file_name = str(file_row["file_name"])
    expected_bytes = int(file_row["expected_bytes"])
    path = data_root / file_name
    actual_bytes = path.stat().st_size if path.exists() else 0
    size_ok = path.exists() and actual_bytes == expected_bytes

    role_counts = {role: 0 for role in ROLE_KEYWORDS}
    role_kept = {role: 0 for role in ROLE_KEYWORDS}
    candidates: list[dict[str, Any]] = []
    member_count = 0
    zip_status = "not_present"
    status = "missing"
    if path.exists():
        try:
            with zipfile.ZipFile(path) as archive:
                infos = archive.infolist()
            zip_status = "valid_zip"
            status = "ready_for_schema_scout" if size_ok else "size_mismatch"
            for info in infos[:max_members]:
                if info.is_dir():
                    continue
                member_count += 1
                roles = _roles_for_member(info.filename)
                matched_roles = [role for role, matched in roles.items() if matched]
                for role in matched_roles:
                    role_counts[role] += 1
                if not matched_roles:
                    continue
                keep = any(role_kept[role] < max_candidates_per_role for role in matched_roles)
                if not keep:
                    continue
                for role in matched_roles:
                    role_kept[role] += 1
                candidates.append(
                    {
                        "file_id": file_id,
                        "file_name": file_name,
                        "member_name": info.filename,
                        "member_size": info.file_size,
                        "extension": Path(info.filename).suffix.lower(),
                        "roles": ";".join(matched_roles),
                        **roles,
                    }
                )
        except zipfile.BadZipFile:
            zip_status = "bad_zip"
            status = "invalid_zip"

    summary = {
        "file_id": file_id,
        "file_name": file_name,
        "expected_bytes": expected_bytes,
        "actual_bytes": actual_bytes,
        "size_ok": size_ok,
        "zip_status": zip_status,
        "member_count_scanned": member_count,
        "candidate_rows": len(candidates),
        "status": status,
        **{f"{role}_hits": role_counts[role] for role in ROLE_KEYWORDS},
    }
    return summary, candidates


def build_gate(summary_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    role_hits = {
        role: sum(1 for row in candidate_rows if bool(row.get(role)))
        for role in ROLE_KEYWORDS
    }
    ready_files = [row for row in summary_rows if row["status"] == "ready_for_schema_scout"]
    missing_or_invalid = [
        row
        for row in summary_rows
        if row["status"] in {"missing", "invalid_zip", "size_mismatch"}
        and row["file_name"] != "Movies.zip"
    ]
    missing_roles = [role for role in REQUIRED_ROLES if role_hits[role] == 0]
    if missing_or_invalid:
        status = "large_intake_incomplete"
        next_action = "wait for Build Command Data.zip and In-situ Meas Data.zip to finish downloading"
    elif missing_roles:
        status = "schema_candidates_incomplete"
        next_action = "manually inspect package directories and metadata for missing registration roles"
    else:
        status = "schema_candidates_ready_manual_sampling_required"
        next_action = "sample candidate members and build a tiny registered source/path-to-target table"
    return {
        "status": status,
        "candidate_rows": len(candidate_rows),
        "ready_file_rows": len(ready_files),
        "missing_or_invalid_required_rows": len(missing_or_invalid),
        "role_hits": role_hits,
        "missing_required_roles": missing_roles,
        "phase104_baseline_smoke_allowed": False,
        "phase105_model_mechanism_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": next_action,
        "required_before_baseline_smoke": [
            "extract explicit coordinate transforms",
            "extract explicit trigger/timing join",
            "sample source command/path schema",
            "sample target observation schema",
            "generate tiny registered sample table",
            "generate leakage-safe split manifest",
        ],
    }


def build_package(
    *,
    root: Path,
    data_card_path: Path,
    data_root: Path,
    output_dir: Path,
    max_members: int = 100000,
    max_candidates_per_role: int = 200,
) -> dict[str, Any]:
    data_card = _read_json(data_card_path)
    summary_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for file_row in data_card["files"]:
        if str(file_row["file_name"]) == "Movies.zip":
            continue
        summary, candidates = _scan_one_zip(
            file_row=file_row,
            data_root=data_root,
            max_members=max_members,
            max_candidates_per_role=max_candidates_per_role,
        )
        summary_rows.append(summary)
        candidate_rows.extend(candidates)

    gate = build_gate(summary_rows, candidate_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "phase103_nist_ammt_schema_scout_summary.csv"
    candidates_path = output_dir / "phase103_nist_ammt_schema_scout_candidates.csv"
    gate_path = output_dir / "phase103_nist_ammt_schema_scout_gate.json"
    manifest_path = output_dir / "phase103_nist_ammt_schema_scout_manifest.json"
    _write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    _write_csv(candidates_path, candidate_rows, CANDIDATE_FIELDS)
    _write_json(gate_path, gate)
    manifest = {
        "phase": 103,
        "objective": "nist_ammt_schema_scout_no_extraction",
        "inputs": {
            "data_card": _display_path(data_card_path, root),
            "data_root": _display_path(data_root, root),
        },
        "outputs": {
            "summary": _display_path(summary_path, root),
            "candidates": _display_path(candidates_path, root),
            "gate_json": _display_path(gate_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "limits": {
            "max_members": max_members,
            "max_candidates_per_role": max_candidates_per_role,
        },
        "counts": {
            "summary_rows": len(summary_rows),
            "candidate_rows": len(candidate_rows),
        },
        "gate": gate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/nist_ammt/mds2_2044"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase103_nist_ammt_registered_intake"),
    )
    parser.add_argument("--data-card", type=Path)
    parser.add_argument("--max-members", type=int, default=100000)
    parser.add_argument("--max-candidates-per-role", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    data_root = args.data_root
    output_dir = args.output_dir
    data_card = args.data_card or _default_data_card(root)
    if not data_root.is_absolute():
        data_root = root / data_root
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    if not data_card.is_absolute():
        data_card = root / data_card
    manifest = build_package(
        root=root,
        data_card_path=data_card,
        data_root=data_root,
        output_dir=output_dir,
        max_members=args.max_members,
        max_candidates_per_role=args.max_candidates_per_role,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
