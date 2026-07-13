#!/usr/bin/env python3
"""Freeze the external-confirmation and simulation-evidence boundary after Phase 190."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_PHASE190 = Path(
    os.environ.get(
        "AMB2022_01_PHASE190_SPATIAL_ANALYSIS",
        "/root/matsci-gnn-pinn-ops/phase190_spatial_failure_analysis.json",
    )
)
DEFAULT_EXTERNAL_ROOT = Path(
    os.environ.get(
        "AMB2022_EXTERNAL_INTAKE_ROOT",
        "/root/matsci-gnn-pinn-data/raw/ambench/2022_external_intake",
    )
)
FILE_REQUIREMENTS = (
    {
        "id": "amb2022_02_readme",
        "relative_path": "AMB2022-02/2617_README.txt",
        "expected_size": 11419,
        "expected_sha256": "013a3d737844d9bd5a4d7651d3af9e7a5cacea0a7df38ef7140149edf2801eb0",
    },
    {
        "id": "amb2022_03_thermography_readme",
        "relative_path": "AMB2022-03-thermography/2716_README.txt",
        "expected_size": 12573,
        "expected_sha256": "ba44076ed51b69c0e4ca80ff0e2568eed2dc6459e85c9ad83b85860bee5760f2",
    },
    {
        "id": "amb2022_03_staring_signal",
        "relative_path": "AMB2022-03-thermography/Thermography/AMB2022-03-718-AMMT-StaringCamera_Signal.h5",
        "expected_size": 549979044,
        "expected_sha256": "f6fe21ec911707f72e7efda2932c77eae2b75d84765848878fe5beb6b728cd43",
    },
    {
        "id": "amb2022_03_pad_xypt",
        "relative_path": "AMB2022-03-thermography/ScanStrategy/AMB2022-03-AMMT-718-Pad_XYPT.h5",
        "expected_size": 406992,
        "expected_sha256": "7b7004753e150bc26632e9ce356e0440429160fa92cbff8fc8559202fdce2103",
    },
    {
        "id": "amb2022_03_cross_section_readme",
        "relative_path": "AMB2022-03-cross-sections/2718_README.txt",
        "expected_size": 7956,
        "expected_sha256": "8fa24c82e087a43ea453f9dbab7e56acfccb421a4b5afa620992142967a45274",
    },
    {
        "id": "amb2022_03_cross_section_workbook",
        "relative_path": "AMB2022-03-cross-sections/AMB2022-718-SH1-MeltPool_Cross-Section_Measurement_Results.xlsx",
        "expected_size": 25811,
        "expected_sha256": "2cfaac96aaca3dabb77b7029f842cdcc7e75c5a2cf3577d0734823246364a931",
    },
)
CONTRACT_FIELDS = (
    "evidence_id",
    "role",
    "scope",
    "available_observations",
    "target_status",
    "current_admissibility",
    "claim_boundary",
    "mandatory_next_audit",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def inspect_files(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for requirement in FILE_REQUIREMENTS:
        path = root / str(requirement["relative_path"])
        exists = path.is_file()
        observed_size = path.stat().st_size if exists else None
        observed_hash = _sha256(path) if exists else None
        rows.append(
            {
                **requirement,
                "path": str(path),
                "exists": exists,
                "observed_size": observed_size,
                "observed_sha256": observed_hash,
                "verified": bool(
                    exists
                    and observed_size == requirement["expected_size"]
                    and observed_hash == requirement["expected_sha256"]
                ),
            }
        )
    return rows


def build_contract_rows() -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": "amb2022_03_track_pad_calibration",
            "role": "local_physical_calibration",
            "scope": "Bare-plate single tracks and pads, not an independent 3D build holdout.",
            "available_observations": "Raw staring-camera signal, synchronized pad XYPT, and cross-section depth/width measurements.",
            "target_status": "Publicly available after checksum-verified intake.",
            "current_admissibility": "Calibration and simulation-anchor evidence only.",
            "claim_boundary": "Cannot establish external 3D-build generalization of B6/B7/B8 TAM/SCR predictions.",
            "mandatory_next_audit": "Create a deterministic track/pad identifier join before fitting or evaluating any calibration relation.",
        },
        {
            "evidence_id": "amb2022_02_v6_v8_scan_policy",
            "role": "prospective_independent_3d_scan_policy_confirmation",
            "scope": "IN718 3D builds V6/V7/V8 with custom scan strategies.",
            "available_observations": "Official XYPT and TAM/SCR submission templates are documented.",
            "target_status": "Processed TAM/SCR truth source unresolved; templates are not labels.",
            "current_admissibility": "Blocked as an external test set until a public truth record and coordinate/build join are verified.",
            "claim_boundary": "No AMB2022-02 performance or transfer claim may be made from scan commands or submission templates alone.",
            "mandatory_next_audit": "Locate the published TAM/SCR truth components, then verify V6/V7/V8 coordinate registration and target units.",
        },
        {
            "evidence_id": "simulation_counterfactual_stress_test",
            "role": "mechanism_stress_test",
            "scope": "Finite-element or reduced heat-transfer simulations matched only to frozen AMB2022-03 calibration cases.",
            "available_observations": "Synthetic trajectories, spatial fields, and perturbation responses after a documented calibration protocol.",
            "target_status": "Synthetic; not an independent experimental observation.",
            "current_admissibility": "Permitted as supporting falsification and sensitivity evidence after calibration auditing.",
            "claim_boundary": "Simulation cannot substitute for external experimental validation or upgrade the held-build claim.",
            "mandatory_next_audit": "Pre-register material parameters, scan perturbations, and a calibration-versus-stress-test split before generation.",
        },
    ]


def _phase190_ready(phase190: dict[str, Any]) -> bool:
    gate = phase190.get("gate", {})
    return (
        gate.get("status") == "phase190_spatial_failure_analysis_ready_phase191_external_confirmation_design"
        and bool(gate.get("phase191_external_confirmation_design_allowed"))
        and gate.get("model_training_allowed") is False
        and gate.get("post_b8_model_reselection_allowed") is False
    )


def build_gate(phase190: dict[str, Any], file_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(row["id"]): row for row in file_rows}
    calibration_ids = {
        "amb2022_03_thermography_readme",
        "amb2022_03_staring_signal",
        "amb2022_03_pad_xypt",
        "amb2022_03_cross_section_readme",
        "amb2022_03_cross_section_workbook",
    }
    missing_calibration_files = sorted(
        file_id for file_id in calibration_ids if not bool(by_id.get(file_id, {}).get("verified"))
    )
    blockers: list[str] = []
    if not _phase190_ready(phase190):
        blockers.append("phase190_spatial_failure_gate_not_ready")
    if missing_calibration_files:
        blockers.extend(f"missing_or_unverified_{file_id}" for file_id in missing_calibration_files)
    local_calibration_ready = not missing_calibration_files and _phase190_ready(phase190)
    return {
        "status": (
            "phase191_external_confirmation_design_ready_phase192_local_calibration_intake"
            if local_calibration_ready
            else "phase191_external_confirmation_design_waiting_for_calibration_intake"
        ),
        "phase192_local_calibration_intake_allowed": local_calibration_ready,
        "phase192_amb2022_02_truth_discovery_allowed": _phase190_ready(phase190),
        "simulation_stress_test_design_allowed": _phase190_ready(phase190),
        "independent_3d_temperature_confirmation_allowed": False,
        "model_training_allowed": False,
        "hyperparameter_search_allowed": False,
        "post_b8_model_reselection_allowed": False,
        "external_generalization_claim_allowed": False,
        "simulation_as_external_validation_allowed": False,
        "amb2022_02_templates_as_ground_truth_allowed": False,
        "blocking_audits": sorted(set(blockers)),
        "next_action": (
            "build the checksum-verified AMB2022-03 calibration intake and independently locate AMB2022-02 TAM/SCR truth"
            if local_calibration_ready
            else "finish checksum-verified AMB2022-03 intake before any calibration-table construction"
        ),
    }


def build_design(phase190: dict[str, Any], file_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "phase": 191,
        "objective": "external_confirmation_and_simulation_evidence_design",
        "file_evidence": file_rows,
        "contract": build_contract_rows(),
        "gate": build_gate(phase190, file_rows),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONTRACT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase190", type=Path, default=DEFAULT_PHASE190)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract-csv", type=Path, required=True)
    args = parser.parse_args()
    payload = build_design(_read_json(args.phase190), inspect_files(args.external_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(args.contract_csv, payload["contract"])
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
