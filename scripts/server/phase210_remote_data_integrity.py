#!/usr/bin/env python3
"""Verify selected AMBench files against bundled official source manifests.

This phase performs byte-level integrity checks only. It does not open HDF5
payloads, derive temperatures, fit calibration parameters, or train a model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import string
from pathlib import Path
from typing import Any


DEFAULT_DATA_ROOT = Path(os.environ.get("MATSCI_DATA_ROOT", "/root/matsci-gnn-pinn-data"))
DEFAULT_MDS2716_METADATA = DEFAULT_DATA_ROOT / "evidence/phase207/mds2-2716_metadata.json"
DEFAULT_MDS2607_MANIFEST = (
    DEFAULT_DATA_ROOT / "raw/ambench/2022_3d_build/AMB2022-01/mds2-2607/manifest.json"
)
DEFAULT_MDS2715_LISTING = (
    DEFAULT_DATA_ROOT / "raw/ambench/2022_3d_build/AMB2022-01/mds2-2715/official/_filelisting.csv"
)

MDS2716_LOCAL_ROOT = "raw/ambench/2022_external_intake/AMB2022-03-thermography"
MDS2607_LOCAL_ROOT = "raw/ambench/2022_3d_build/AMB2022-01/mds2-2607"
MDS2715_LOCAL_ROOT = "raw/ambench/2022_3d_build/AMB2022-01/mds2-2715/official"

REQUIRED_RECORD_KEYS = (
    "mds2-2716:Thermography/AMB2022-03-718-AMMT-StaringCamera_Signal.h5",
    "mds2-2716:ScanStrategy/AMB2022-03-AMMT-718-Pad_XYPT.h5",
    "mds2-2607:Scan_Strategy/AMB2022-01-AMMT-XYPT_v1.h5",
    "mds2-2715:DataProcessingScripts/AMB2022_HDF5_Temperature_v1.m",
)

SHA256_CHUNK_BYTES = 1024 * 1024


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _valid_sha256(value: object) -> str | None:
    text = str(value).strip().lower()
    if len(text) != 64 or any(character not in string.hexdigits for character in text):
        return None
    return text


def _record(
    *,
    source_id: str,
    source_relative_path: str,
    expected_size_bytes: object,
    expected_sha256: object,
    local_root: Path,
    source_url: object | None,
) -> dict[str, Any] | None:
    digest = _valid_sha256(expected_sha256)
    try:
        size = int(expected_size_bytes)
    except (TypeError, ValueError):
        return None
    if not source_relative_path or digest is None or size < 0:
        return None
    relative_path = Path(source_relative_path)
    return {
        "record_key": f"{source_id}:{relative_path.as_posix()}",
        "source_id": source_id,
        "source_relative_path": relative_path.as_posix(),
        "local_path": str(local_root / relative_path),
        "expected_size_bytes": size,
        "expected_sha256": digest,
        "source_url": None if source_url is None else str(source_url),
    }


def records_from_mds2716_metadata(metadata: dict[str, Any], data_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for component in metadata.get("components", []):
        if not isinstance(component, dict):
            continue
        component_types = component.get("@type", [])
        if isinstance(component_types, str):
            component_types = [component_types]
        if "nrdp:DataFile" not in component_types:
            continue
        checksum = component.get("checksum", {})
        if not isinstance(checksum, dict):
            continue
        record = _record(
            source_id="mds2-2716",
            source_relative_path=str(component.get("filepath", "")),
            expected_size_bytes=component.get("size"),
            expected_sha256=checksum.get("hash"),
            local_root=data_root / MDS2716_LOCAL_ROOT,
            source_url=component.get("downloadURL"),
        )
        if record is not None:
            records.append(record)
    return sorted(records, key=lambda item: item["record_key"])


def records_from_mds2607_manifest(manifest: dict[str, Any], data_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in manifest.get("files", []):
        if not isinstance(item, dict):
            continue
        record = _record(
            source_id="mds2-2607",
            source_relative_path=str(item.get("relative_path", "")),
            expected_size_bytes=item.get("size_bytes"),
            expected_sha256=item.get("sha256"),
            local_root=data_root / MDS2607_LOCAL_ROOT,
            source_url=item.get("url"),
        )
        if record is not None:
            records.append(record)
    return sorted(records, key=lambda item: item["record_key"])


def records_from_mds2715_listing(path: Path, data_root: Path) -> list[dict[str, Any]]:
    """Parse the official headerless six-column NIST file listing.

    Columns are relative path, byte size, an unused field, media type, SHA-256,
    and download URL. Comment lines begin with ``#``.
    """

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(line for line in handle if line.strip() and not line.startswith("#"))
        for row in reader:
            if len(row) < 6:
                continue
            record = _record(
                source_id="mds2-2715",
                source_relative_path=row[0].strip(),
                expected_size_bytes=row[1].strip(),
                expected_sha256=row[4].strip(),
                local_root=data_root / MDS2715_LOCAL_ROOT,
                source_url=row[5].strip(),
            )
            if record is not None:
                records.append(record)
    return sorted(records, key=lambda item: item["record_key"])


def sha256_file(path: Path, chunk_bytes: int = SHA256_CHUNK_BYTES) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def verify_record(record: dict[str, Any]) -> dict[str, Any]:
    result = dict(record)
    path = Path(str(record["local_path"]))
    if not path.is_file():
        result.update(
            {
                "status": "not_selected",
                "actual_size_bytes": None,
                "actual_sha256": None,
                "size_matches": False,
                "sha256_matches": False,
            }
        )
        return result
    actual_size = path.stat().st_size
    actual_digest = sha256_file(path)
    size_matches = actual_size == int(record["expected_size_bytes"])
    sha256_matches = actual_digest == str(record["expected_sha256"])
    result.update(
        {
            "status": "verified" if size_matches and sha256_matches else "mismatch",
            "actual_size_bytes": actual_size,
            "actual_sha256": actual_digest,
            "size_matches": size_matches,
            "sha256_matches": sha256_matches,
        }
    )
    return result


def build_gate(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {str(record["record_key"]): record for record in records}
    selected = [record for record in records if record.get("status") != "not_selected"]
    mismatched = [record for record in selected if record.get("status") != "verified"]
    missing_required = [
        key for key in REQUIRED_RECORD_KEYS if by_key.get(key, {}).get("status") != "verified"
    ]
    blockers: list[str] = []
    if not selected:
        blockers.append("no_manifest_described_local_files")
    if mismatched:
        blockers.append("manifest_described_file_hash_or_size_mismatch")
    if missing_required:
        blockers.append("required_formula_input_not_verified")
    verified = not blockers
    return {
        "status": (
            "phase210_remote_data_integrity_complete_phase211_formula_identifiability"
            if verified
            else "phase210_remote_data_integrity_incomplete_or_mismatched"
        ),
        "phase211_formula_identifiability_allowed": verified,
        "raw_signal_integrity_verified": by_key.get(REQUIRED_RECORD_KEYS[0], {}).get("status")
        == "verified",
        "temperature_conversion_executed": False,
        "calibration_fitting_performed": False,
        "model_training_allowed": False,
        "blocking_audits": blockers,
        "missing_required_records": missing_required,
        "verified_selected_record_count": sum(
            record.get("status") == "verified" for record in selected
        ),
        "selected_record_count": len(selected),
        "not_selected_source_record_count": sum(
            record.get("status") == "not_selected" for record in records
        ),
    }


def build_payload(
    *,
    data_root: Path,
    mds2716_metadata: dict[str, Any],
    mds2607_manifest: dict[str, Any],
    mds2715_listing: Path,
) -> dict[str, Any]:
    records = [
        *records_from_mds2716_metadata(mds2716_metadata, data_root),
        *records_from_mds2607_manifest(mds2607_manifest, data_root),
        *records_from_mds2715_listing(mds2715_listing, data_root),
    ]
    verified_records = [verify_record(record) for record in records]
    source_summaries: list[dict[str, Any]] = []
    for source_id in sorted({record["source_id"] for record in verified_records}):
        rows = [record for record in verified_records if record["source_id"] == source_id]
        source_summaries.append(
            {
                "source_id": source_id,
                "manifest_record_count": len(rows),
                "selected_record_count": sum(row["status"] != "not_selected" for row in rows),
                "verified_record_count": sum(row["status"] == "verified" for row in rows),
                "mismatch_record_count": sum(row["status"] == "mismatch" for row in rows),
                "not_selected_record_count": sum(row["status"] == "not_selected" for row in rows),
            }
        )
    return {
        "phase": 210,
        "objective": "remote_byte_level_integrity_verification_before_formula_identifiability",
        "integrity_boundary": (
            "Only byte size and SHA-256 are read for manifest-described files. "
            "No HDF5 payload, temperature conversion, calibration fitting, or model training occurs."
        ),
        "data_root": str(data_root),
        "source_summaries": source_summaries,
        "verification_records": verified_records,
        "gate": build_gate(verified_records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--mds2716-metadata", type=Path, default=DEFAULT_MDS2716_METADATA)
    parser.add_argument("--mds2607-manifest", type=Path, default=DEFAULT_MDS2607_MANIFEST)
    parser.add_argument("--mds2715-listing", type=Path, default=DEFAULT_MDS2715_LISTING)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_payload(
        data_root=args.data_root,
        mds2716_metadata=_read_json(args.mds2716_metadata),
        mds2607_manifest=_read_json(args.mds2607_manifest),
        mds2715_listing=args.mds2715_listing,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
