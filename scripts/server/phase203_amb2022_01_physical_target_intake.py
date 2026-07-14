#!/usr/bin/env python3
"""Audit AMB2022-01 microstructure target availability before any spatial-target download."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_PHASE202 = Path(
    os.environ.get(
        "AMB2022_03_PHASE202_FORMULA_CONTRACT",
        "/root/matsci-gnn-pinn-ops/phase202_amb2022_03_formula_contract.json",
    )
)
DEFAULT_DESCRIPTION_TEXT = Path(
    os.environ.get(
        "AMB2022_01_MICROSTRUCTURE_DESCRIPTION_TEXT",
        "/root/matsci-gnn-pinn-ops/phase203_amb2022_01_microstructure_descriptions.txt",
    )
)
DEFAULT_FILE_LIST = Path(
    os.environ.get("AMB2022_01_MDS2692_FILE_LIST", "/root/mds2-2692/_filelisting.csv")
)
SAMPLE_SPECS = (
    {
        "sample_id": "AMB2022-718-AMMT-B7-P1-L7-L8-L9-O1",
        "build_id": "B7",
        "part_id": "P1",
        "plane": "XZ",
        "condition": "as_built",
        "role": "development_physical_observation_only",
    },
    {
        "sample_id": "AMB2022-718-AMMT-B6-P2-L7-L8-L9-O1",
        "build_id": "B6",
        "part_id": "P2",
        "plane": "XZ",
        "condition": "heat_treated",
        "role": "not_comparable_to_as_built_thermal_target",
    },
    {
        "sample_id": "AMB2022-718-AMMT-B8-P3-L9",
        "build_id": "B8",
        "part_id": "P3",
        "plane": "XY",
        "condition": "as_built",
        "role": "frozen_build_physical_observation_candidate",
    },
    {
        "sample_id": "AMB2022-718-AMMT-B8-P3-L10-W3",
        "build_id": "B8",
        "part_id": "P3",
        "plane": "XY",
        "condition": "as_built",
        "role": "frozen_build_physical_observation_candidate",
    },
)
SAMPLE_FIELDS = ("sample_id", "build_id", "part_id", "plane", "condition", "role", "description_confirmed")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_file_listing(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if line.startswith("# file path,")),
        None,
    )
    if header_index is None:
        raise ValueError("NIST file listing has no '# file path' CSV header")
    header = lines[header_index].removeprefix("# ")
    return list(csv.DictReader([header, *lines[header_index + 1 :]]))


def _phase202_ready(phase202: dict[str, Any]) -> bool:
    gate = phase202.get("gate", {})
    return (
        gate.get("status") == "phase202_formula_contract_complete_temperature_conversion_blocked"
        and bool(gate.get("phase203_calibration_documentation_resolution_allowed"))
        and gate.get("calibrated_temperature_descriptor_allowed") is False
        and gate.get("model_training_allowed") is False
    )


def build_sample_rows(description_text: str) -> list[dict[str, Any]]:
    return [
        {
            **spec,
            "description_confirmed": spec["sample_id"] in description_text,
        }
        for spec in SAMPLE_SPECS
    ]


def build_intake_audit(sample_rows: list[dict[str, Any]], file_rows: list[dict[str, str]]) -> dict[str, Any]:
    b8_leg9_tiffs = [
        row
        for row in file_rows
        if "b8-p3-leg 9" in str(row.get("# file path", row.get("file path", ""))).lower()
        and str(row.get("file path", row.get("# file path", ""))).lower().endswith(".tif")
    ]
    # The comment-prefixed header is normalized by csv.DictReader only if the source lacks comments.
    if not b8_leg9_tiffs:
        b8_leg9_tiffs = [
            row
            for row in file_rows
            if "b8-p3-leg 9" in " ".join(row.values()).lower()
            and ".tif" in " ".join(row.values()).lower()
        ]
    return {
        "description_sample_count": len(sample_rows),
        "description_confirmed_sample_ids": sorted(
            str(row["sample_id"]) for row in sample_rows if row["description_confirmed"]
        ),
        "description_missing_sample_ids": sorted(
            str(row["sample_id"]) for row in sample_rows if not row["description_confirmed"]
        ),
        "file_listing_row_count": len(file_rows),
        "b8_p3_leg9_tiff_component_count": len(b8_leg9_tiffs),
        "b8_p3_leg9_tiff_components": [
            {
                "path": next((value for value in row.values() if ".tif" in value.lower()), ""),
                "size_bytes": next((value for value in row.values() if value.isdigit()), ""),
            }
            for row in b8_leg9_tiffs
        ],
        "sample_to_build_identifier_confirmed": True,
        "physical_target_coordinate_transform_demonstrated": False,
        "physical_target_files_downloaded": False,
        "model_training_performed": False,
    }


def build_gate(phase202: dict[str, Any], sample_rows: list[dict[str, Any]], audit: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if not _phase202_ready(phase202):
        blockers.append("phase202_formula_contract_gate_not_ready")
    if any(not bool(row["description_confirmed"]) for row in sample_rows):
        blockers.append("microstructure_sample_identifier_missing_from_description")
    b8_candidates = [row for row in sample_rows if row["build_id"] == "B8" and row["condition"] == "as_built"]
    if len(b8_candidates) != 2:
        blockers.append("unexpected_b8_as_built_physical_target_count")
    if int(audit.get("b8_p3_leg9_tiff_component_count", 0)) < 1:
        blockers.append("b8_leg9_public_microstructure_component_missing")
    if audit.get("physical_target_coordinate_transform_demonstrated") is not False:
        blockers.append("coordinate_transform_claimed_before_protocol")
    if audit.get("physical_target_files_downloaded") is not False:
        blockers.append("physical_target_downloaded_before_spatial_protocol")
    if audit.get("model_training_performed") is not False:
        blockers.append("model_training_before_physical_target_mapping")
    blockers = sorted(set(blockers))
    ready = not blockers
    return {
        "status": (
            "phase203_physical_target_intake_ready_phase204_spatial_join_protocol_design"
            if ready
            else "phase203_physical_target_intake_incomplete_or_premature"
        ),
        "phase204_spatial_join_protocol_design_allowed": ready,
        "physical_target_download_allowed": False,
        "physical_target_evaluation_allowed": False,
        "calibrated_temperature_descriptor_allowed": False,
        "model_training_allowed": False,
        "hyperparameter_search_allowed": False,
        "simulation_as_external_validation_allowed": False,
        "blocking_audits": blockers,
        "next_action": (
            "freeze a sample-plane-to-nominal-build coordinate protocol before downloading or evaluating B8 physical target maps"
            if ready
            else "repair sample identifiers or public component evidence before any spatial-join design"
        ),
    }


def build_payload(phase202: dict[str, Any], description_text: str, file_rows: list[dict[str, str]]) -> dict[str, Any]:
    sample_rows = build_sample_rows(description_text)
    audit = build_intake_audit(sample_rows, file_rows)
    return {
        "phase": 203,
        "objective": "amb2022_01_public_physical_target_identifier_intake_without_target_download",
        "sample_manifest": sample_rows,
        "intake_audit": audit,
        "coordinate_boundary": (
            "Sample/build identifiers and public EBSD component availability do not establish a transform from the sample "
            "plane to AMB2022-01 nominal build/layer coordinates. No physical target is downloaded or evaluated in this phase."
        ),
        "gate": build_gate(phase202, sample_rows, audit),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase202", type=Path, default=DEFAULT_PHASE202)
    parser.add_argument("--description-text", type=Path, default=DEFAULT_DESCRIPTION_TEXT)
    parser.add_argument("--file-list", type=Path, default=DEFAULT_FILE_LIST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-csv", type=Path, required=True)
    args = parser.parse_args()
    payload = build_payload(_read_json(args.phase202), args.description_text.read_text(encoding="utf-8"), _read_file_listing(args.file_list))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(args.samples_csv, payload["sample_manifest"])
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
