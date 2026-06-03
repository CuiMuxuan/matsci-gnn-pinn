#!/usr/bin/env python3
"""Phase 103 NIST AMMT deep registration probe.

This no-training probe inspects ZIP member names plus small binary/text prefixes
to refine the Phase 103 registration evidence. It does not extract full files,
build sample tables, run baselines, or open training gates.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import struct
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any


ZIP_FILES = (
    "Metadata.zip",
    "Build Command Data.zip",
    "In-situ Meas Data.zip",
)

TARGET_EXTENSIONS = {".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg"}
TEXT_EXTENSIONS = {".asc", ".cfg", ".csv", ".dat", ".gcode", ".ini", ".log", ".txt", ".xml"}
TARGET_KEYWORDS = (
    "camera",
    "frame",
    "image",
    "insitu",
    "in_situ",
    "in-situ",
    "layer_camera",
    "melt",
    "monitor",
    "pool",
)
TIMING_KEYWORDS = (
    "clock",
    "exposure",
    "fps",
    "frame_time",
    "sync",
    "time",
    "timestamp",
    "timing",
    "trigger",
)

SEQUENCE_FIELDS = (
    "file_name",
    "group_key",
    "directory",
    "extension",
    "count",
    "first_index",
    "last_index",
    "zero_padded_width",
    "min_member_size",
    "max_member_size",
    "example_first",
    "example_last",
    "target_observation_candidate",
    "implicit_sequence_index_ready",
)

TARGET_SAMPLE_FIELDS = (
    "file_name",
    "member_name",
    "member_size",
    "extension",
    "format",
    "width",
    "height",
    "bits_per_pixel",
    "channels",
    "header_status",
    "target_observation_binary_schema_ready",
)

TIMING_FIELDS = (
    "file_name",
    "member_name",
    "extension",
    "evidence_type",
    "matched_keyword",
    "header_line",
    "explicit_trigger_timing_ready",
)


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


def _normal(text: str) -> str:
    return (
        text.lower()
        .replace("\\", "/")
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def _target_name_candidate(member_name: str) -> bool:
    normalized = _normal(member_name)
    return any(keyword in normalized for keyword in TARGET_KEYWORDS)


def _timing_keyword(text: str) -> str:
    normalized = _normal(text)
    for keyword in TIMING_KEYWORDS:
        if keyword in normalized:
            return keyword
    return ""


def _numbered_group(member_name: str) -> tuple[str, int, int] | None:
    path = Path(member_name)
    match = re.match(r"^(.*?)(\d+)(\.[^.]+)$", path.name)
    if not match:
        return None
    prefix, digits, suffix = match.groups()
    group_name = f"{path.parent.as_posix()}/{prefix}{{index}}{suffix}"
    return group_name.lstrip("./"), int(digits), len(digits)


def _read_prefix(data_root: Path, file_name: str, member_name: str, max_bytes: int) -> bytes:
    with zipfile.ZipFile(data_root / file_name) as archive:
        with archive.open(member_name) as handle:
            return handle.read(max_bytes)


def _parse_bmp_header(data: bytes) -> dict[str, Any]:
    if len(data) < 30 or data[:2] != b"BM":
        return {"format": "", "header_status": "not_bmp"}
    dib_size = struct.unpack_from("<I", data, 14)[0]
    if dib_size < 12:
        return {"format": "bmp", "header_status": "unsupported_bmp_dib"}
    if dib_size == 12:
        width, height, _planes, bit_count = struct.unpack_from("<HHHH", data, 18)
    else:
        width, height, _planes, bit_count = struct.unpack_from("<iiHH", data, 18)
    channels = 1 if bit_count <= 8 else max(bit_count // 8, 1)
    return {
        "format": "bmp",
        "width": abs(int(width)),
        "height": abs(int(height)),
        "bits_per_pixel": int(bit_count),
        "channels": int(channels),
        "header_status": "parsed_binary_header",
    }


def _parse_png_header(data: bytes) -> dict[str, Any]:
    if len(data) < 29 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return {"format": "", "header_status": "not_png"}
    width, height = struct.unpack(">II", data[16:24])
    bit_depth = data[24]
    color_type = data[25]
    channel_map = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    channels = channel_map.get(color_type, 0)
    return {
        "format": "png",
        "width": int(width),
        "height": int(height),
        "bits_per_pixel": int(bit_depth * max(channels, 1)),
        "channels": int(channels),
        "header_status": "parsed_binary_header",
    }


def _parse_tiff_header(data: bytes) -> dict[str, Any]:
    if len(data) < 4:
        return {"format": "", "header_status": "not_tiff"}
    if data[:4] in {b"II*\x00", b"MM\x00*"}:
        return {"format": "tiff", "header_status": "recognized_binary_header"}
    return {"format": "", "header_status": "not_tiff"}


def _parse_binary_header(extension: str, data: bytes) -> dict[str, Any]:
    if extension == ".bmp":
        parsed = _parse_bmp_header(data)
    elif extension == ".png":
        parsed = _parse_png_header(data)
    elif extension in {".tif", ".tiff"}:
        parsed = _parse_tiff_header(data)
    elif extension in {".jpg", ".jpeg"} and data[:2] == b"\xff\xd8":
        parsed = {"format": "jpeg", "header_status": "recognized_binary_header"}
    else:
        parsed = {"format": "", "header_status": "unsupported_extension"}
    parsed.setdefault("width", "")
    parsed.setdefault("height", "")
    parsed.setdefault("bits_per_pixel", "")
    parsed.setdefault("channels", "")
    return parsed


def _decode_text_prefix(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _text_timing_evidence(
    *,
    data_root: Path,
    file_name: str,
    info: zipfile.ZipInfo,
    max_text_scan_bytes: int,
) -> dict[str, Any] | None:
    extension = Path(info.filename).suffix.lower()
    keyword = _timing_keyword(info.filename)
    if keyword:
        return {
            "file_name": file_name,
            "member_name": info.filename,
            "extension": extension,
            "evidence_type": "member_name",
            "matched_keyword": keyword,
            "header_line": "",
            "explicit_trigger_timing_ready": True,
        }
    if extension not in TEXT_EXTENSIONS or info.file_size <= 0:
        return None
    data = _read_prefix(data_root, file_name, info.filename, max_text_scan_bytes)
    text = _decode_text_prefix(data).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line for line in text.splitlines() if line.strip()]
    head = "\n".join(lines[:10])
    keyword = _timing_keyword(head)
    if not keyword:
        return None
    return {
        "file_name": file_name,
        "member_name": info.filename,
        "extension": extension,
        "evidence_type": "text_header",
        "matched_keyword": keyword,
        "header_line": lines[0] if lines else "",
        "explicit_trigger_timing_ready": True,
    }


def _collect_zip_infos(data_root: Path, max_members: int) -> list[tuple[str, zipfile.ZipInfo]]:
    rows: list[tuple[str, zipfile.ZipInfo]] = []
    for file_name in ZIP_FILES:
        zip_path = data_root / file_name
        if not zip_path.exists():
            continue
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist()[:max_members]:
                if not info.is_dir():
                    rows.append((file_name, info))
    return rows


def _build_sequence_rows(infos: list[tuple[str, zipfile.ZipInfo]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[tuple[int, int, zipfile.ZipInfo]]] = defaultdict(list)
    for file_name, info in infos:
        parsed = _numbered_group(info.filename)
        if parsed is None:
            continue
        group_key, index, width = parsed
        grouped[(file_name, group_key)].append((index, width, info))

    rows: list[dict[str, Any]] = []
    for (file_name, group_key), members in grouped.items():
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda item: item[0])
        first = members[0][2]
        last = members[-1][2]
        extension = Path(first.filename).suffix.lower()
        row = {
            "file_name": file_name,
            "group_key": group_key,
            "directory": Path(first.filename).parent.as_posix(),
            "extension": extension,
            "count": len(members),
            "first_index": members[0][0],
            "last_index": members[-1][0],
            "zero_padded_width": members[0][1],
            "min_member_size": min(info.file_size for _idx, _width, info in members),
            "max_member_size": max(info.file_size for _idx, _width, info in members),
            "example_first": first.filename,
            "example_last": last.filename,
            "target_observation_candidate": (
                extension in TARGET_EXTENSIONS and _target_name_candidate(group_key)
            ),
            "implicit_sequence_index_ready": True,
        }
        rows.append(row)
    rows.sort(key=lambda row: (str(row["file_name"]), str(row["group_key"])))
    return rows


def _select_target_members(
    sequence_rows: list[dict[str, Any]],
    max_target_groups: int,
    max_samples_per_group: int,
) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    target_groups = [
        row for row in sequence_rows if bool(row.get("target_observation_candidate"))
    ][:max_target_groups]
    for row in target_groups:
        examples = [str(row["example_first"])]
        if row["example_last"] != row["example_first"]:
            examples.append(str(row["example_last"]))
        for member_name in examples[:max_samples_per_group]:
            selected.append((str(row["file_name"]), member_name))
    return selected


def _build_target_rows(
    *,
    data_root: Path,
    selected_members: list[tuple[str, str]],
    max_binary_header_bytes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_name, member_name in selected_members:
        extension = Path(member_name).suffix.lower()
        data = _read_prefix(data_root, file_name, member_name, max_binary_header_bytes)
        parsed = _parse_binary_header(extension, data)
        ready = parsed["header_status"] in {
            "parsed_binary_header",
            "recognized_binary_header",
        }
        with zipfile.ZipFile(data_root / file_name) as archive:
            info = archive.getinfo(member_name)
        rows.append(
            {
                "file_name": file_name,
                "member_name": member_name,
                "member_size": info.file_size,
                "extension": extension,
                **parsed,
                "target_observation_binary_schema_ready": ready,
            }
        )
    return rows


def _build_timing_rows(
    *,
    data_root: Path,
    infos: list[tuple[str, zipfile.ZipInfo]],
    max_timing_rows: int,
    max_text_scan_bytes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_name, info in infos:
        evidence = _text_timing_evidence(
            data_root=data_root,
            file_name=file_name,
            info=info,
            max_text_scan_bytes=max_text_scan_bytes,
        )
        if evidence is None:
            continue
        rows.append(evidence)
        if len(rows) >= max_timing_rows:
            break
    return rows


def _build_gate(
    *,
    sequence_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    timing_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    target_ready = any(
        bool(row.get("target_observation_binary_schema_ready")) for row in target_rows
    )
    implicit_sequence_ready = any(
        bool(row.get("target_observation_candidate"))
        and bool(row.get("implicit_sequence_index_ready"))
        for row in sequence_rows
    )
    explicit_timing_ready = any(
        bool(row.get("explicit_trigger_timing_ready")) for row in timing_rows
    )
    missing_roles = []
    if not explicit_timing_ready:
        missing_roles.append("trigger_timing")
    if not target_ready:
        missing_roles.append("target_observation")
    if target_ready and not explicit_timing_ready:
        status = "target_binary_schema_ready_trigger_timing_missing"
        next_action = "find explicit trigger/timing join before constructing a tiny registered table"
    elif target_ready and explicit_timing_ready:
        status = "deep_probe_ready_manual_registration_required"
        next_action = "review timing candidates and construct a tiny registered table candidate"
    else:
        status = "deep_probe_incomplete"
        next_action = "inspect additional target/timing candidates without opening training"
    return {
        "status": status,
        "target_observation_binary_schema_ready": target_ready,
        "implicit_sequence_index_ready": implicit_sequence_ready,
        "explicit_trigger_timing_ready": explicit_timing_ready,
        "missing_or_blocked_required_roles": missing_roles,
        "sequence_group_rows": len(sequence_rows),
        "target_binary_sample_rows": len(target_rows),
        "timing_evidence_rows": len(timing_rows),
        "phase104_baseline_smoke_allowed": False,
        "phase105_model_mechanism_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": next_action,
        "required_before_baseline_smoke": [
            "explicit trigger/timing join identified",
            "target observation schema linked to source/path rows",
            "tiny registered source/path-to-target sample table generated",
            "leakage-safe split manifest generated",
            "baseline smoke plan validated without test-label route selection",
        ],
    }


def build_package(
    *,
    root: Path,
    data_root: Path,
    output_dir: Path,
    max_members: int = 100000,
    max_target_groups: int = 8,
    max_samples_per_group: int = 2,
    max_binary_header_bytes: int = 4096,
    max_timing_rows: int = 32,
    max_text_scan_bytes: int = 65536,
) -> dict[str, Any]:
    infos = _collect_zip_infos(data_root, max_members=max_members)
    sequence_rows = _build_sequence_rows(infos)
    target_members = _select_target_members(
        sequence_rows,
        max_target_groups=max_target_groups,
        max_samples_per_group=max_samples_per_group,
    )
    target_rows = _build_target_rows(
        data_root=data_root,
        selected_members=target_members,
        max_binary_header_bytes=max_binary_header_bytes,
    )
    timing_rows = _build_timing_rows(
        data_root=data_root,
        infos=infos,
        max_timing_rows=max_timing_rows,
        max_text_scan_bytes=max_text_scan_bytes,
    )
    gate = _build_gate(
        sequence_rows=sequence_rows,
        target_rows=target_rows,
        timing_rows=timing_rows,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_path = output_dir / "phase103_nist_ammt_deep_sequence_groups.csv"
    target_path = output_dir / "phase103_nist_ammt_deep_target_binary_samples.csv"
    timing_path = output_dir / "phase103_nist_ammt_deep_timing_evidence.csv"
    gate_path = output_dir / "phase103_nist_ammt_deep_registration_probe_gate.json"
    manifest_path = output_dir / "phase103_nist_ammt_deep_registration_probe_manifest.json"
    _write_csv(sequence_path, sequence_rows, SEQUENCE_FIELDS)
    _write_csv(target_path, target_rows, TARGET_SAMPLE_FIELDS)
    _write_csv(timing_path, timing_rows, TIMING_FIELDS)
    _write_json(gate_path, gate)
    manifest = {
        "phase": 103,
        "objective": "nist_ammt_deep_registration_probe_no_training",
        "inputs": {"data_root": _display_path(data_root, root)},
        "outputs": {
            "sequence_groups": _display_path(sequence_path, root),
            "target_binary_samples": _display_path(target_path, root),
            "timing_evidence": _display_path(timing_path, root),
            "gate_json": _display_path(gate_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "limits": {
            "max_members": max_members,
            "max_target_groups": max_target_groups,
            "max_samples_per_group": max_samples_per_group,
            "max_binary_header_bytes": max_binary_header_bytes,
            "max_timing_rows": max_timing_rows,
            "max_text_scan_bytes": max_text_scan_bytes,
        },
        "counts": {
            "zip_member_rows": len(infos),
            "sequence_group_rows": len(sequence_rows),
            "target_binary_sample_rows": len(target_rows),
            "timing_evidence_rows": len(timing_rows),
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
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase103_nist_ammt_registered_intake"),
    )
    parser.add_argument("--max-members", type=int, default=100000)
    parser.add_argument("--max-target-groups", type=int, default=8)
    parser.add_argument("--max-samples-per-group", type=int, default=2)
    parser.add_argument("--max-binary-header-bytes", type=int, default=4096)
    parser.add_argument("--max-timing-rows", type=int, default=32)
    parser.add_argument("--max-text-scan-bytes", type=int, default=65536)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    data_root = args.data_root
    output_dir = args.output_dir
    if not data_root.is_absolute():
        data_root = root / data_root
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    manifest = build_package(
        root=root,
        data_root=data_root,
        output_dir=output_dir,
        max_members=args.max_members,
        max_target_groups=args.max_target_groups,
        max_samples_per_group=args.max_samples_per_group,
        max_binary_header_bytes=args.max_binary_header_bytes,
        max_timing_rows=args.max_timing_rows,
        max_text_scan_bytes=args.max_text_scan_bytes,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
