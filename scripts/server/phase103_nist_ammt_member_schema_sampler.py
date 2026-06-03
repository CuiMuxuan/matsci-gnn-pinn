#!/usr/bin/env python3
"""Phase 103 NIST AMMT member schema sampler.

This post-scout utility reads a small prefix from selected ZIP members that were
flagged by the Phase 103 schema scout. It never extracts full files, never builds
training tables, and never opens baseline or model-training gates. Its purpose is
to identify candidate headers/units/indices before a tiny registered table is
built.
"""

from __future__ import annotations

import argparse
import csv
import json
import string
import zipfile
from pathlib import Path
from typing import Any


REQUIRED_ROLES = (
    "coordinate_transform",
    "trigger_timing",
    "source_command_path",
    "target_observation",
)

TEXT_EXTENSIONS = {
    ".asc",
    ".cfg",
    ".csv",
    ".dat",
    ".ini",
    ".json",
    ".log",
    ".md",
    ".tab",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

SAMPLE_FIELDS = (
    "file_id",
    "file_name",
    "member_name",
    "member_size",
    "extension",
    "roles",
    "sample_status",
    "bytes_read",
    "line_count",
    "header_line",
    "preview_text",
)


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


def _role_set(row: dict[str, str]) -> set[str]:
    roles = str(row.get("roles") or "")
    return {role for role in roles.split(";") if role}


def _select_candidates(
    rows: list[dict[str, str]],
    roles: tuple[str, ...],
    max_per_role: int,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    counts = {role: 0 for role in roles}
    for row in rows:
        row_roles = _role_set(row)
        matched = [role for role in roles if role in row_roles]
        if not matched:
            continue
        if not any(counts[role] < max_per_role for role in matched):
            continue
        key = (str(row.get("file_name", "")), str(row.get("member_name", "")))
        if key in seen:
            continue
        selected.append(row)
        seen.add(key)
        for role in matched:
            counts[role] += 1
    return selected


def _looks_text(data: bytes, extension: str) -> bool:
    if extension in TEXT_EXTENSIONS:
        return True
    if not data:
        return True
    if b"\x00" in data:
        return False
    printable = set(bytes(string.printable, "ascii"))
    printable_count = sum(1 for byte in data if byte in printable or byte >= 128)
    return printable_count / max(len(data), 1) >= 0.85


def _decode_preview(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _sample_member(
    *,
    data_root: Path,
    row: dict[str, str],
    max_bytes: int,
) -> dict[str, Any]:
    file_name = str(row.get("file_name", ""))
    member_name = str(row.get("member_name", ""))
    zip_path = data_root / file_name
    extension = str(row.get("extension") or Path(member_name).suffix.lower())
    base = {
        "file_id": row.get("file_id", ""),
        "file_name": file_name,
        "member_name": member_name,
        "member_size": row.get("member_size", ""),
        "extension": extension,
        "roles": row.get("roles", ""),
        "sample_status": "not_sampled",
        "bytes_read": 0,
        "line_count": 0,
        "header_line": "",
        "preview_text": "",
    }
    if not zip_path.exists():
        return {**base, "sample_status": "zip_missing"}
    try:
        with zipfile.ZipFile(zip_path) as archive:
            with archive.open(member_name) as handle:
                data = handle.read(max_bytes)
    except KeyError:
        return {**base, "sample_status": "member_missing"}
    except zipfile.BadZipFile:
        return {**base, "sample_status": "bad_zip"}
    if not _looks_text(data, extension):
        return {**base, "sample_status": "non_text_preview_skipped", "bytes_read": len(data)}
    text = _decode_preview(data).replace("\r\n", "\n").replace("\r", "\n")
    lines = text.splitlines()
    preview = "\n".join(lines[:20])
    return {
        **base,
        "sample_status": "sampled_text",
        "bytes_read": len(data),
        "line_count": len(lines),
        "header_line": lines[0] if lines else "",
        "preview_text": preview,
    }


def build_gate(sample_rows: list[dict[str, Any]], roles: tuple[str, ...]) -> dict[str, Any]:
    sampled_roles: set[str] = set()
    for row in sample_rows:
        if row["sample_status"] not in {"sampled_text", "non_text_preview_skipped"}:
            continue
        sampled_roles.update(role for role in str(row["roles"]).split(";") if role)
    missing_required_roles = [role for role in REQUIRED_ROLES if role in roles and role not in sampled_roles]
    if not sample_rows:
        status = "schema_scout_candidates_required"
        next_action = "run Phase 103 schema scout after large ZIP downloads complete"
    elif missing_required_roles:
        status = "member_schema_samples_incomplete"
        next_action = "inspect additional scout candidates for missing registration roles"
    else:
        status = "member_schema_samples_ready_manual_registration_required"
        next_action = "manually inspect samples and build a tiny registered table candidate"
    return {
        "status": status,
        "sample_rows": len(sample_rows),
        "sampled_text_rows": sum(1 for row in sample_rows if row["sample_status"] == "sampled_text"),
        "non_text_rows": sum(1 for row in sample_rows if row["sample_status"] == "non_text_preview_skipped"),
        "missing_required_roles": missing_required_roles,
        "phase104_baseline_smoke_allowed": False,
        "phase105_model_mechanism_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": next_action,
    }


def build_package(
    *,
    root: Path,
    data_root: Path,
    candidates_csv: Path,
    output_dir: Path,
    roles: tuple[str, ...] = REQUIRED_ROLES,
    max_per_role: int = 8,
    max_bytes: int = 4096,
) -> dict[str, Any]:
    candidate_rows = _read_csv(candidates_csv) if candidates_csv.exists() else []
    selected = _select_candidates(candidate_rows, roles, max_per_role)
    sample_rows = [
        _sample_member(data_root=data_root, row=row, max_bytes=max_bytes)
        for row in selected
    ]
    gate = build_gate(sample_rows, roles)

    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "phase103_nist_ammt_member_schema_samples.csv"
    gate_path = output_dir / "phase103_nist_ammt_member_schema_sampler_gate.json"
    manifest_path = output_dir / "phase103_nist_ammt_member_schema_sampler_manifest.json"
    _write_csv(samples_path, sample_rows, SAMPLE_FIELDS)
    _write_json(gate_path, gate)
    manifest = {
        "phase": 103,
        "objective": "nist_ammt_candidate_member_schema_sampling_no_extraction",
        "inputs": {
            "candidate_csv": _display_path(candidates_csv, root),
            "data_root": _display_path(data_root, root),
        },
        "outputs": {
            "samples": _display_path(samples_path, root),
            "gate_json": _display_path(gate_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "limits": {
            "roles": list(roles),
            "max_per_role": max_per_role,
            "max_bytes": max_bytes,
        },
        "counts": {
            "candidate_rows": len(candidate_rows),
            "selected_rows": len(selected),
            "sample_rows": len(sample_rows),
        },
        "gate": gate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/nist_ammt/mds2_2044"))
    parser.add_argument(
        "--candidates-csv",
        type=Path,
        default=Path(
            "docs/results/phase103_nist_ammt_registered_intake/"
            "phase103_nist_ammt_schema_scout_candidates.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase103_nist_ammt_registered_intake"),
    )
    parser.add_argument("--role", action="append", dest="roles")
    parser.add_argument("--max-per-role", type=int, default=8)
    parser.add_argument("--max-bytes", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    data_root = args.data_root
    candidates_csv = args.candidates_csv
    output_dir = args.output_dir
    if not data_root.is_absolute():
        data_root = root / data_root
    if not candidates_csv.is_absolute():
        candidates_csv = root / candidates_csv
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    roles = tuple(args.roles) if args.roles else REQUIRED_ROLES
    manifest = build_package(
        root=root,
        data_root=data_root,
        candidates_csv=candidates_csv,
        output_dir=output_dir,
        roles=roles,
        max_per_role=args.max_per_role,
        max_bytes=args.max_bytes,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
