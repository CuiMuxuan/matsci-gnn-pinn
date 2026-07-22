#!/usr/bin/env python3
"""Revalidate CPU-only AMB2022-03 gates and cross-section workbook provenance.

This phase coordinates existing metadata-only gates. It verifies a workbook
against a NIST headerless file listing, but never reads raw Signal values or
cross-section target values for fitting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_DATA_ROOT = Path(os.environ.get("MATSCI_DATA_ROOT", "/root/matsci-gnn-pinn-data"))
DEFAULT_WORKBOOK = (
    DEFAULT_DATA_ROOT
    / "raw/ambench/2022_external_intake/AMB2022-03-cross-sections"
    / "AMB2022-718-SH1-MeltPool_Cross-Section_Measurement_Results.xlsx"
)
DEFAULT_OFFICIAL_LISTING = DEFAULT_DATA_ROOT / "evidence/phase212/mds2-2718_filelisting.csv"
DEFAULT_PHASE192 = DEFAULT_DATA_ROOT / "derived/phase212/phase212_amb2022_03_calibration_intake.json"
DEFAULT_PHASE201 = (
    DEFAULT_DATA_ROOT / "derived/phase212/phase212_amb2022_03_thermalcal_metadata_audit.json"
)
DEFAULT_PHASE202 = DEFAULT_DATA_ROOT / "derived/phase212/phase212_amb2022_03_formula_contract.json"
SHA256_CHUNK_BYTES = 1024 * 1024


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def sha256_file(path: Path, chunk_bytes: int = SHA256_CHUNK_BYTES) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def workbook_listing_record(listing: Path, workbook_name: str) -> dict[str, Any]:
    with listing.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(line for line in handle if line.strip() and not line.startswith("#"))
        for row in reader:
            if len(row) < 6 or row[0].strip() != workbook_name:
                continue
            try:
                size = int(row[1].strip())
            except ValueError as exc:
                raise ValueError(f"Official workbook size is invalid in {listing}") from exc
            expected_hash = row[4].strip().lower()
            if len(expected_hash) != 64 or any(character not in "0123456789abcdef" for character in expected_hash):
                raise ValueError(f"Official workbook SHA-256 is invalid in {listing}")
            return {
                "filepath": row[0].strip(),
                "size": size,
                "media_type": row[3].strip(),
                "sha256": expected_hash,
                "source_url": row[5].strip(),
            }
    raise ValueError(f"Official workbook listing record not found: {workbook_name}")


def verify_workbook(workbook: Path, listing: Path) -> dict[str, Any]:
    official = workbook_listing_record(listing, workbook.name)
    actual_size = workbook.stat().st_size
    actual_hash = sha256_file(workbook)
    size_matches = actual_size == official["size"]
    sha256_matches = actual_hash == official["sha256"]
    return {
        "official_component": official,
        "local_component": {
            "path": str(workbook),
            "size": actual_size,
            "sha256": actual_hash,
        },
        "official_listing": {
            "dataset_id": "mds2-2718",
            "doi": "https://doi.org/10.18434/mds2-2718",
            "filelisting_path": str(listing),
            "filelisting_sha256": sha256_file(listing),
        },
        "size_matches": size_matches,
        "sha256_matches": sha256_matches,
        "status": "verified" if size_matches and sha256_matches else "mismatch",
    }


def build_payload(
    *,
    phase192: dict[str, Any],
    phase201: dict[str, Any],
    phase202: dict[str, Any],
    workbook_provenance: dict[str, Any],
) -> dict[str, Any]:
    phase192_gate = phase192.get("gate", {})
    phase201_gate = phase201.get("gate", {})
    phase202_gate = phase202.get("gate", {})
    intake = phase201.get("thermalcal_intake", {})
    contract = phase202.get("formula_contract", {})
    boundary_assertions = {
        "raw_signal_arrays_read": intake.get("raw_signal_arrays_read"),
        "cross_section_targets_read": intake.get("cross_section_targets_read"),
        "calibration_fitting_performed": intake.get("calibration_fitting_performed"),
        "temperature_conversion_executed": contract.get("temperature_conversion_executed"),
    }
    phase192_ready = (
        phase192_gate.get("status")
        == "phase192_amb2022_03_calibration_intake_ready_phase193_identifier_join_design"
        and not phase192_gate.get("blocking_audits")
    )
    phase201_ready = (
        phase201_gate.get("status")
        == "phase201_thermalcal_metadata_audit_ready_phase202_formula_contract_design"
        and not phase201_gate.get("blocking_audits")
    )
    phase202_boundary = (
        phase202_gate.get("status") == "phase202_formula_contract_complete_temperature_conversion_blocked"
        and phase202_gate.get("calibration_formula_execution_allowed") is False
        and phase202_gate.get("model_training_allowed") is False
    )
    blocked_formula_reasons = set(phase202_gate.get("blocking_audits", []))
    expected_formula_block = {
        "hdf5_formula_contains_undefined_symbol",
        "hdf5_formula_not_unambiguously_mapped_to_official_equation",
    }.issubset(blocked_formula_reasons)
    boundaries_preserved = boundary_assertions == {
        "raw_signal_arrays_read": False,
        "cross_section_targets_read": False,
        "calibration_fitting_performed": False,
        "temperature_conversion_executed": False,
    }
    complete = (
        phase192_ready
        and phase201_ready
        and phase202_boundary
        and expected_formula_block
        and workbook_provenance.get("status") == "verified"
        and boundaries_preserved
    )
    return {
        "phase": 212,
        "objective": "remote_cpu_gate_revalidation_without_temperature_conversion_or_training",
        "workbook_provenance": workbook_provenance,
        "gate_results": {
            "phase192_status": phase192_gate.get("status"),
            "phase192_blocking_audits": phase192_gate.get("blocking_audits", []),
            "phase201_status": phase201_gate.get("status"),
            "phase201_blocking_audits": phase201_gate.get("blocking_audits", []),
            "phase202_status": phase202_gate.get("status"),
            "phase202_blocking_audits": phase202_gate.get("blocking_audits", []),
            "temperature_conversion_allowed": phase202_gate.get("calibration_formula_execution_allowed"),
            "model_training_allowed": phase202_gate.get("model_training_allowed"),
        },
        "boundary_assertions": boundary_assertions,
        "gate": {
            "status": (
                "phase212_remote_cpu_gate_revalidation_complete_cpu_evidence_boundary_preserved"
                if complete
                else "phase212_remote_cpu_gate_revalidation_incomplete_or_boundary_broken"
            ),
            "phase212_complete": complete,
            "temperature_conversion_allowed": False,
            "calibration_fitting_allowed": False,
            "model_training_allowed": False,
            "gpu_required_now": False,
            "next_action": (
                "retain the formula and spatial-registration blocks; do not begin GPU training from this phase"
                if complete
                else "repair the failed integrity or evidence boundary before considering any downstream work"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--official-listing", type=Path, default=DEFAULT_OFFICIAL_LISTING)
    parser.add_argument("--phase192", type=Path, default=DEFAULT_PHASE192)
    parser.add_argument("--phase201", type=Path, default=DEFAULT_PHASE201)
    parser.add_argument("--phase202", type=Path, default=DEFAULT_PHASE202)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_payload(
        phase192=_read_json(args.phase192),
        phase201=_read_json(args.phase201),
        phase202=_read_json(args.phase202),
        workbook_provenance=verify_workbook(args.workbook, args.official_listing),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
