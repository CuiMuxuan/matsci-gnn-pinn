#!/usr/bin/env python3
"""Phase 103 NIST AMMT registered data intake/audit.

This phase uses the Phase 102 data card to download and audit the minimum NIST
AMMT 3D Scan Strategies files needed to decide whether a registered source/path
target can be built. It does not run baselines or model training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


AUDIT_FIELDS = (
    "file_id",
    "file_name",
    "expected_bytes",
    "actual_bytes",
    "size_ok",
    "sha256",
    "download_status",
    "zip_status",
    "member_count",
    "registration_hits",
    "timing_hits",
    "command_hits",
    "target_hits",
    "status",
)

MEMBER_FIELDS = (
    "file_id",
    "member_name",
    "member_size",
    "registration_keyword",
    "timing_keyword",
    "command_keyword",
    "target_keyword",
)

KEYWORDS = {
    "registration": (
        "transform",
        "coordinate",
        "pixel",
        "ammt",
        "homography",
        "calibration",
        "registration",
    ),
    "timing": ("trigger", "time", "timestamp", "sync", "clock"),
    "command": ("xypt", "command", "scan", "path", "trajectory", "galvo"),
    "target": ("melt", "pool", "monitor", "frame", "image", "insitu", "in-situ"),
}


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


def _default_paths(root: Path) -> dict[str, Path]:
    phase102 = root / "docs/results/phase102_registered_source_manifest_gate"
    return {
        "phase102_gate": phase102 / "phase102_registered_source_manifest_gate.json",
        "phase102_data_card": phase102 / "phase102_nist_ammt_data_card.json",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _keyword_hit(text: str, family: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in KEYWORDS[family])


def _download_with_python(url: str, output: Path, timeout_seconds: int) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "gnnpinn-phase103-intake/1.0"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(output)
    return "downloaded_python"


def _download_with_external(
    *,
    backend: str,
    url: str,
    output: Path,
    timeout_seconds: int,
    retries: int,
    resume: bool,
) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    if backend == "python":
        return _download_with_python(url, output, timeout_seconds)
    if backend == "aria2c":
        command = [
            "aria2c",
            "--continue=true" if resume else "--continue=false",
            "--max-connection-per-server=8",
            "--split=8",
            "--min-split-size=16M",
            "--console-log-level=warn",
            "--summary-interval=0",
            "--auto-file-renaming=false",
            "--file-allocation=none",
            "--max-tries",
            str(retries),
            "--timeout",
            str(timeout_seconds),
            "--dir",
            str(output.parent),
            "--out",
            output.name,
            url,
        ]
    elif backend == "curl":
        command = [
            "curl",
            "-L",
            "--fail",
            "--retry",
            str(retries),
            "--connect-timeout",
            str(timeout_seconds),
            "--output",
            str(output),
        ]
        if resume:
            command.insert(2, "-C")
            command.insert(3, "-")
        command.append(url)
    elif backend == "wget":
        command = [
            "wget",
            "--tries",
            str(retries),
            "--timeout",
            str(timeout_seconds),
            "-O",
            str(output),
            url,
        ]
        if resume:
            command.insert(1, "-c")
    else:
        raise ValueError(f"Unsupported download backend: {backend}")
    outer_attempts = max(1, retries)
    for attempt in range(1, outer_attempts + 1):
        try:
            subprocess.run(command, check=True, stdout=sys.stderr)
            return f"downloaded_{backend}"
        except subprocess.CalledProcessError:
            if attempt >= outer_attempts:
                raise
            delay_seconds = min(60, 5 * attempt)
            print(
                f"{backend} download attempt {attempt}/{outer_attempts} failed for {output}; "
                f"retrying in {delay_seconds}s",
                flush=True,
            )
            time.sleep(delay_seconds)
    raise RuntimeError(f"unreachable download retry state for {output}")


def _inspect_zip_members(path: Path, max_members: int) -> tuple[str, list[dict[str, Any]]]:
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
    except zipfile.BadZipFile:
        return "bad_zip", []
    rows: list[dict[str, Any]] = []
    for info in infos[:max_members]:
        name = info.filename
        rows.append(
            {
                "member_name": name,
                "member_size": info.file_size,
                "registration_keyword": _keyword_hit(name, "registration"),
                "timing_keyword": _keyword_hit(name, "timing"),
                "command_keyword": _keyword_hit(name, "command"),
                "target_keyword": _keyword_hit(name, "target"),
            }
        )
    return "valid_zip", rows


def _file_required(row: dict[str, Any], large_downloads: bool) -> bool:
    if bool(row.get("required_for_phase103")):
        return True
    scope = str(row.get("download_scope", ""))
    return large_downloads and scope == "long_running_server_download_after_metadata_pass"


def audit_files(
    *,
    data_card: dict[str, Any],
    data_root: Path,
    download: bool,
    large_downloads: bool,
    backend: str,
    retries: int,
    timeout_seconds: int,
    resume: bool,
    max_members: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audit_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    for row in data_card["files"]:
        file_name = str(row["file_name"])
        file_id = str(row["file_id"])
        path = data_root / file_name
        expected_bytes = int(row["expected_bytes"])
        should_download = download and _file_required(row, large_downloads)
        actual_bytes_before = path.stat().st_size if path.exists() else 0
        needs_resume = path.exists() and actual_bytes_before != expected_bytes
        download_status = "not_requested"
        if should_download and (not path.exists() or needs_resume):
            download_status = _download_with_external(
                backend=backend,
                url=str(row["url"]),
                output=path,
                timeout_seconds=timeout_seconds,
                retries=retries,
                resume=resume,
            )
        elif path.exists():
            download_status = "present"
        elif _file_required(row, large_downloads):
            download_status = "missing_required"
        actual_bytes = path.stat().st_size if path.exists() else 0
        size_ok = path.exists() and actual_bytes == expected_bytes
        sha256 = _sha256(path) if path.exists() else ""
        zip_status = "not_present"
        members: list[dict[str, Any]] = []
        if path.exists() and path.suffix.lower() == ".zip":
            zip_status, members = _inspect_zip_members(path, max_members)
        for member in members:
            member_rows.append({"file_id": file_id, **member})
        registration_hits = sum(1 for member in members if member["registration_keyword"])
        timing_hits = sum(1 for member in members if member["timing_keyword"])
        command_hits = sum(1 for member in members if member["command_keyword"])
        target_hits = sum(1 for member in members if member["target_keyword"])
        if not path.exists() and _file_required(row, large_downloads):
            status = "missing_required"
        elif path.exists() and not size_ok:
            status = "size_mismatch"
        elif path.exists() and zip_status != "valid_zip":
            status = "invalid_zip"
        elif path.exists():
            status = "ready_for_schema_audit"
        else:
            status = "not_requested"
        audit_rows.append(
            {
                "file_id": file_id,
                "file_name": file_name,
                "expected_bytes": expected_bytes,
                "actual_bytes": actual_bytes,
                "size_ok": size_ok,
                "sha256": sha256,
                "download_status": download_status,
                "zip_status": zip_status,
                "member_count": len(members),
                "registration_hits": registration_hits,
                "timing_hits": timing_hits,
                "command_hits": command_hits,
                "target_hits": target_hits,
                "status": status,
            }
        )
    return audit_rows, member_rows


def build_gate(
    *,
    phase102_gate: dict[str, Any],
    audit_rows: list[dict[str, Any]],
    member_rows: list[dict[str, Any]],
    large_downloads: bool,
) -> dict[str, Any]:
    phase102_allows = bool(phase102_gate.get("phase103_intake_allowed"))
    metadata = next((row for row in audit_rows if row["file_name"] == "Metadata.zip"), None)
    metadata_ready = bool(metadata and metadata["status"] == "ready_for_schema_audit")
    registration_hits = sum(1 for row in member_rows if row["registration_keyword"])
    timing_hits = sum(1 for row in member_rows if row["timing_keyword"])
    command_hits = sum(1 for row in member_rows if row["command_keyword"])
    target_hits = sum(1 for row in member_rows if row["target_keyword"])
    required_missing = [row for row in audit_rows if row["status"] == "missing_required"]
    size_mismatches = [row for row in audit_rows if row["status"] == "size_mismatch"]
    if not phase102_allows:
        status = "blocked_by_phase102_gate"
        next_action = "repair Phase 102 source-manifest gate"
    elif not metadata_ready:
        status = "metadata_intake_incomplete"
        next_action = "download Metadata.zip and rerun Phase 103 audit"
    elif registration_hits and timing_hits:
        status = "metadata_ready_registration_schema_candidate"
        next_action = "inspect metadata members and then download command/measurement packages on server"
    else:
        status = "metadata_ready_registration_evidence_pending"
        next_action = "inspect metadata manually for transform/timing records before opening baseline smoke"
    return {
        "status": status,
        "source_phase102_status": phase102_gate.get("status"),
        "metadata_ready": metadata_ready,
        "large_downloads_requested": large_downloads,
        "registration_keyword_hits": registration_hits,
        "timing_keyword_hits": timing_hits,
        "command_keyword_hits": command_hits,
        "target_keyword_hits": target_hits,
        "audit_rows": len(audit_rows),
        "member_rows": len(member_rows),
        "required_missing_rows": len(required_missing),
        "size_mismatch_rows": len(size_mismatches),
        "phase104_baseline_smoke_allowed": False,
        "phase105_model_mechanism_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": next_action,
        "required_before_baseline_smoke": [
            "explicit coordinate transform files identified",
            "explicit trigger/timing join identified",
            "source command/path schema sampled",
            "target observation schema sampled",
            "tiny registered sample table and split manifest generated",
        ],
    }


def build_package(
    *,
    root: Path,
    output_dir: Path,
    data_root: Path,
    paths: dict[str, Path] | None = None,
    download: bool = False,
    large_downloads: bool = False,
    backend: str = "python",
    retries: int = 3,
    timeout_seconds: int = 300,
    resume: bool = True,
    max_members: int = 500,
) -> dict[str, Any]:
    resolved = _default_paths(root)
    if paths:
        resolved.update(paths)
    phase102_gate = _read_json(resolved["phase102_gate"])
    data_card = _read_json(resolved["phase102_data_card"])
    audit_rows, member_rows = audit_files(
        data_card=data_card,
        data_root=data_root,
        download=download,
        large_downloads=large_downloads,
        backend=backend,
        retries=retries,
        timeout_seconds=timeout_seconds,
        resume=resume,
        max_members=max_members,
    )
    gate = build_gate(
        phase102_gate=phase102_gate,
        audit_rows=audit_rows,
        member_rows=member_rows,
        large_downloads=large_downloads,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "phase103_nist_ammt_file_audit.csv"
    member_path = output_dir / "phase103_nist_ammt_zip_member_keywords.csv"
    gate_path = output_dir / "phase103_nist_ammt_registered_intake_gate.json"
    manifest_path = output_dir / "phase103_nist_ammt_registered_intake_manifest.json"
    _write_csv(audit_path, audit_rows, AUDIT_FIELDS)
    _write_csv(member_path, member_rows, MEMBER_FIELDS)
    _write_json(gate_path, gate)
    manifest = {
        "phase": 103,
        "objective": "nist_ammt_registered_data_intake_audit",
        "inputs": {key: _display_path(path, root) for key, path in sorted(resolved.items())},
        "outputs": {
            "file_audit": _display_path(audit_path, root),
            "zip_member_keywords": _display_path(member_path, root),
            "gate_json": _display_path(gate_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "data_root": _display_path(data_root, root),
        "download": {
            "requested": download,
            "large_downloads": large_downloads,
            "backend": backend,
            "retries": retries,
            "timeout_seconds": timeout_seconds,
            "resume": resume,
        },
        "counts": {
            "audit_rows": len(audit_rows),
            "member_rows": len(member_rows),
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
        "--data-root",
        type=Path,
        default=Path("data/raw/nist_ammt/mds2_2044"),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--large-downloads", action="store_true")
    parser.add_argument(
        "--download-backend",
        choices=("python", "curl", "wget", "aria2c"),
        default="python",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-members", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    data_root = args.data_root
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    if not data_root.is_absolute():
        data_root = root / data_root
    manifest = build_package(
        root=root,
        output_dir=output_dir,
        data_root=data_root,
        download=args.download,
        large_downloads=args.large_downloads,
        backend=args.download_backend,
        retries=args.retries,
        timeout_seconds=args.timeout_seconds,
        resume=not args.no_resume,
        max_members=args.max_members,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
