#!/usr/bin/env python3
"""Build the Phase 102 registered source-manifest/data-card gate.

Phase 102 turns the Phase 101 "no real registered target" blocker into a
concrete public-source data-card decision. It does not download the dataset and
does not open model training. A pass only allows Phase 103 minimal intake/audit.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SOURCE_FIELDS = (
    "candidate_id",
    "source_family",
    "dataset_id",
    "title",
    "doi",
    "primary_url",
    "public_provenance",
    "source_manifest_status",
    "registration_evidence_status",
    "target_observation_status",
    "split_plan_status",
    "baseline_plan_status",
    "phase103_intake_allowed",
    "phase104_baseline_smoke_allowed",
    "a100_training_allowed",
    "a100_80gb_request_now",
    "status",
    "priority",
    "next_action",
    "evidence",
)

FILE_FIELDS = (
    "file_id",
    "candidate_id",
    "file_name",
    "url",
    "expected_bytes",
    "expected_gib",
    "content_type",
    "required_for_phase103",
    "download_order",
    "download_scope",
    "status",
    "notes",
)

REGISTRATION_FIELDS = (
    "evidence_id",
    "candidate_id",
    "component",
    "claimed_evidence",
    "current_status",
    "phase103_check",
    "pass_condition",
    "stop_condition",
)

QUEUE_FIELDS = (
    "queue_id",
    "candidate_id",
    "priority",
    "task",
    "allowed_location",
    "expected_input",
    "expected_output",
    "pass_condition",
    "stop_condition",
)

PROTOCOL_FIELDS = (
    "protocol_id",
    "component",
    "requirement",
    "current_status",
    "pass_condition",
    "stop_condition",
)

DATASET_BASE_URL = "https://data.nist.gov/od/ds/85196AB9232E7202E053245706813DFA2044"


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
    phase101 = root / "docs/results/phase101_registered_target_acquisition_gate"
    return {
        "phase101_gate": phase101 / "phase101_registered_target_acquisition_gate.json",
        "phase101_target_table": phase101 / "phase101_registered_target_acquisition_table.csv",
        "phase101_manual_queue": phase101 / "phase101_registered_target_manual_queue.csv",
    }


def _gib(byte_count: int) -> float:
    return byte_count / (1024.0**3)


def _file_url(file_name: str) -> str:
    return f"{DATASET_BASE_URL}/{file_name.replace(' ', '%20')}"


def build_nist_ammt_data_card() -> dict[str, Any]:
    files = [
        {
            "file_id": "p102_nist_metadata_zip",
            "file_name": "Metadata.zip",
            "expected_bytes": 2_489_233,
            "content_type": "application/zip",
            "required_for_phase103": True,
            "download_order": 1,
            "download_scope": "minimal_registration_metadata_intake",
            "notes": "Small first download for schema, transform, timing, and metadata audit.",
        },
        {
            "file_id": "p102_nist_build_command_data_zip",
            "file_name": "Build Command Data.zip",
            "expected_bytes": 7_419_446_651,
            "content_type": "application/zip",
            "required_for_phase103": False,
            "download_order": 2,
            "download_scope": "long_running_server_download_after_metadata_pass",
            "notes": "Large command/scan-strategy package; server download is allowed after metadata audit starts.",
        },
        {
            "file_id": "p102_nist_in_situ_meas_data_zip",
            "file_name": "In-situ Meas Data.zip",
            "expected_bytes": 9_170_420_366,
            "content_type": "application/zip",
            "required_for_phase103": False,
            "download_order": 3,
            "download_scope": "long_running_server_download_after_metadata_pass",
            "notes": "Large in-situ measurement package for target observations.",
        },
        {
            "file_id": "p102_nist_movies_zip",
            "file_name": "Movies.zip",
            "expected_bytes": 698_954_503,
            "content_type": "application/zip",
            "required_for_phase103": False,
            "download_order": 4,
            "download_scope": "optional_visual_context_after_registration_pass",
            "notes": "Optional movie data, not needed for the first registration audit.",
        },
    ]
    for row in files:
        row["candidate_id"] = "P102-CAND-NIST-AMMT-3D-SCAN"
        row["url"] = _file_url(str(row["file_name"]))
        row["expected_gib"] = _gib(int(row["expected_bytes"]))
        row["status"] = "manifested_not_downloaded"
    return {
        "candidate_id": "P102-CAND-NIST-AMMT-3D-SCAN",
        "dataset_id": "nist_mds2_2044",
        "title": "Process Monitoring Dataset from the Additive Manufacturing Metrology Testbed (AMMT): 3D Scan Strategies",
        "data_doi": "10.18434/M32044",
        "article_doi": "10.6028/jres.124.033",
        "pdr_landing_page": "https://data.nist.gov/pdr/lps/85196AB9232E7202E053245706813DFA2044",
        "doi_resolution_url": "https://data.nist.gov/od/id/mds2-2044",
        "dataset_file_index": f"{DATASET_BASE_URL}/",
        "publication_pdf": "https://nvlpubs.nist.gov/nistpubs/jres/124/jres.124.033.pdf",
        "public_provenance": "NIST PDR and NIST JRES official public dataset/article records",
        "known_file_count": len(files),
        "known_total_bytes": sum(int(row["expected_bytes"]) for row in files),
        "known_total_gib": _gib(sum(int(row["expected_bytes"]) for row in files)),
        "source_variables_expected": [
            "input/build command files",
            "scan strategy / XYPT-style source path commands",
            "trigger timing",
            "process-monitoring timestamps",
        ],
        "target_observations_expected": [
            "in-situ process-monitoring data",
            "melt-pool monitoring image frames",
            "movies or image-derived process-monitoring targets",
        ],
        "registration_evidence_expected": [
            "pixel-to-AMMT coordinate transformations",
            "trigger timing that links commands to measurement frames",
            "metadata that identifies units, coordinate frames, and build context",
        ],
        "phase102_decision": "source_manifest_ready_phase103_intake_allowed",
        "phase103_minimal_intake_first_file": "Metadata.zip",
        "phase103_large_server_download_allowed": True,
        "phase104_baseline_smoke_allowed_now": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "files": files,
        "notes": (
            "Phase 102 records official public provenance and file manifest only. "
            "Phase 103 must inspect downloaded metadata before any baseline smoke or training."
        ),
    }


def build_source_rows(
    *,
    phase101_gate: dict[str, Any],
    phase101_targets: list[dict[str, str]],
    data_card: dict[str, Any],
    large_server_download_allowed: bool,
) -> list[dict[str, Any]]:
    phase101_blocked = phase101_gate.get("status") == "blocked_no_real_registered_target"
    primary_allowed = bool(phase101_blocked and large_server_download_allowed)
    rows: list[dict[str, Any]] = [
        {
            "candidate_id": data_card["candidate_id"],
            "source_family": "external_registered_process_monitoring_dataset",
            "dataset_id": data_card["dataset_id"],
            "title": data_card["title"],
            "doi": data_card["data_doi"],
            "primary_url": data_card["pdr_landing_page"],
            "public_provenance": "official_nist_pdr_jres",
            "source_manifest_status": "ready_official_file_manifest",
            "registration_evidence_status": "claimed_by_official_dataset_description_pending_metadata_audit",
            "target_observation_status": "expected_in_situ_process_monitoring_data",
            "split_plan_status": "draft_command_or_build_segment_split_pending_schema",
            "baseline_plan_status": "draft_mean_knn_extratrees_macro_pinn_pending_intake",
            "phase103_intake_allowed": primary_allowed,
            "phase104_baseline_smoke_allowed": False,
            "a100_training_allowed": False,
            "a100_80gb_request_now": False,
            "status": (
                "source_manifest_ready_phase103_intake_allowed"
                if primary_allowed
                else "blocked_until_user_allows_server_download"
            ),
            "priority": 1,
            "next_action": "download Metadata.zip on the server, inspect transforms/timing/schema, then decide whether to download large packages",
            "evidence": "DOI 10.18434/M32044 resolves to NIST mds2-2044; PDR file index exposes Metadata, Build Command Data, In-situ Meas Data, and Movies ZIPs.",
        }
    ]
    for row in phase101_targets:
        target_id = row.get("target_id", "")
        if target_id == "phase101_ambench_mds2_2716_pad_thermography_xypt":
            rows.append(
                {
                    "candidate_id": "P102-CAND-AMBENCH-PAD-XYPT",
                    "source_family": "current_ambench_pad_route",
                    "dataset_id": row.get("dataset_id"),
                    "title": "Current AM-Bench pad thermography plus XYPT route",
                    "doi": "",
                    "primary_url": "",
                    "public_provenance": row.get("public_reproducibility"),
                    "source_manifest_status": row.get("source_manifest_status"),
                    "registration_evidence_status": "blocked_missing_camera_to_galvo_or_equivalent_mapping",
                    "target_observation_status": row.get("target_type"),
                    "split_plan_status": row.get("split_plan_status"),
                    "baseline_plan_status": row.get("baseline_plan_status"),
                    "phase103_intake_allowed": False,
                    "phase104_baseline_smoke_allowed": False,
                    "a100_training_allowed": False,
                    "a100_80gb_request_now": False,
                    "status": "blocked_registration_evidence_required",
                    "priority": 2,
                    "next_action": row.get("next_action"),
                    "evidence": row.get("evidence"),
                }
            )
        elif target_id == "phase101_p94_cand_exaca_sim":
            rows.append(
                {
                    "candidate_id": "P102-CAND-EXACA-SIM-DATA-CARD",
                    "source_family": "simulation_data_card_candidate",
                    "dataset_id": row.get("dataset_id"),
                    "title": "ExaCA generated simulation target data card",
                    "doi": "",
                    "primary_url": "https://github.com/LLNL/ExaCA",
                    "public_provenance": row.get("public_reproducibility"),
                    "source_manifest_status": "missing_generated_target_manifest",
                    "registration_evidence_status": "requires_generated_dataset_and_alignment_card",
                    "target_observation_status": "not_generated_in_this_repo",
                    "split_plan_status": "missing",
                    "baseline_plan_status": "missing",
                    "phase103_intake_allowed": False,
                    "phase104_baseline_smoke_allowed": False,
                    "a100_training_allowed": False,
                    "a100_80gb_request_now": False,
                    "status": "blocked_until_simulation_manifest",
                    "priority": 3,
                    "next_action": "create generated data manifest only after real registered route is closed or deprioritized",
                    "evidence": row.get("evidence"),
                }
            )
    return sorted(rows, key=lambda item: int(item["priority"]))


def build_registration_rows() -> list[dict[str, Any]]:
    candidate_id = "P102-CAND-NIST-AMMT-3D-SCAN"
    return [
        {
            "evidence_id": "P102-REG-001",
            "candidate_id": candidate_id,
            "component": "source_path_commands",
            "claimed_evidence": "input command files and scan strategy data are listed in the official source package",
            "current_status": "manifested_pending_download",
            "phase103_check": "inspect Metadata.zip and Build Command Data schemas for command coordinates, units, and timestamps",
            "pass_condition": "source/path rows expose physical coordinates and times without target-label fitting",
            "stop_condition": "commands are absent, undocumented, or cannot be mapped to target frames",
        },
        {
            "evidence_id": "P102-REG-002",
            "candidate_id": candidate_id,
            "component": "target_observations",
            "claimed_evidence": "in-situ process-monitoring data and melt-pool monitoring frames are listed in the official source package",
            "current_status": "manifested_pending_download",
            "phase103_check": "inspect In-situ Meas Data schemas after Metadata.zip confirms frame/timing references",
            "pass_condition": "target observations have frame ids or timestamps that can join to commands",
            "stop_condition": "target observations are movie-only or lack machine-readable registration keys",
        },
        {
            "evidence_id": "P102-REG-003",
            "candidate_id": candidate_id,
            "component": "coordinate_transform",
            "claimed_evidence": "dataset description indicates pixel-to-AMMT coordinate transformations",
            "current_status": "critical_pending_metadata_audit",
            "phase103_check": "locate transform files and verify coordinate frame, units, orientation, and invertibility",
            "pass_condition": "pixel coordinates can be mapped to AMMT/source coordinates without independent rescaling",
            "stop_condition": "transform files are missing or only usable through target-label calibration",
        },
        {
            "evidence_id": "P102-REG-004",
            "candidate_id": candidate_id,
            "component": "trigger_timing",
            "claimed_evidence": "dataset description indicates trigger timing records",
            "current_status": "critical_pending_metadata_audit",
            "phase103_check": "verify command/monitoring clocks and trigger alignment",
            "pass_condition": "source-path time can be joined to monitoring frames by explicit timestamps/triggers",
            "stop_condition": "timing requires manual visual alignment or test-label optimization",
        },
        {
            "evidence_id": "P102-REG-005",
            "candidate_id": candidate_id,
            "component": "split_safety",
            "claimed_evidence": "3D build command/process-monitoring structure should support command segment or time-block splits",
            "current_status": "draft_pending_schema",
            "phase103_check": "derive train/validation/test split keys from build segments, commands, or time blocks",
            "pass_condition": "split keys are defined before model selection and do not leak adjacent target frames",
            "stop_condition": "only random frame splits are feasible or split keys require test labels",
        },
    ]


def build_file_rows(data_card: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in data_card["files"]]


def build_phase103_queue(data_card: dict[str, Any], large_server_download_allowed: bool) -> list[dict[str, Any]]:
    location = "A100 server /root/matsci-gnn-pinn/data/raw/nist_ammt/mds2_2044"
    return [
        {
            "queue_id": "P102-INTAKE-001",
            "candidate_id": data_card["candidate_id"],
            "priority": 1,
            "task": "download Metadata.zip and write checksum/content-length audit",
            "allowed_location": location,
            "expected_input": "Metadata.zip from NIST PDR file index",
            "expected_output": "phase103 metadata inventory with transforms/timing/schema candidates",
            "pass_condition": "metadata contains coordinate transform/timing/schema files sufficient to plan joins",
            "stop_condition": "Metadata.zip is unavailable, corrupted, or lacks transform/timing/schema references",
        },
        {
            "queue_id": "P102-INTAKE-002",
            "candidate_id": data_card["candidate_id"],
            "priority": 2,
            "task": "start resumable long-running server downloads for Build Command Data and In-situ Meas Data only after metadata inventory starts",
            "allowed_location": location if large_server_download_allowed else "blocked_until_user_approval",
            "expected_input": "Build Command Data.zip and In-situ Meas Data.zip",
            "expected_output": "server download manifest with sizes, checksums if available, and extraction plan",
            "pass_condition": "large packages are present and command/target schemas can be sampled",
            "stop_condition": "server storage/network fails repeatedly or metadata audit blocks registration",
        },
        {
            "queue_id": "P102-INTAKE-003",
            "candidate_id": data_card["candidate_id"],
            "priority": 3,
            "task": "extract a tiny registered sample table",
            "allowed_location": location,
            "expected_input": "metadata plus sampled command/monitoring rows",
            "expected_output": "registered sample CSV/JSONL and split draft",
            "pass_condition": "source/path features map to target observations without independent rescaling",
            "stop_condition": "only unregistered or visually aligned data can be constructed",
        },
    ]


def build_protocol_rows() -> list[dict[str, Any]]:
    return [
        {
            "protocol_id": "P102-PROT-001",
            "component": "no_training",
            "requirement": "Phase 102 may only create source-manifest/data-card artifacts.",
            "current_status": "enforced",
            "pass_condition": "phase104_baseline_smoke_allowed and a100_training_allowed remain false",
            "stop_condition": "any model training, baseline smoke, or A100 training is started in Phase 102",
        },
        {
            "protocol_id": "P102-PROT-002",
            "component": "public_provenance",
            "requirement": "Candidate must have public DOI/PDR/article provenance.",
            "current_status": "satisfied_for_nist_ammt",
            "pass_condition": "DOI, landing page, file index, and article record are listed in the data card",
            "stop_condition": "private or non-reproducible source",
        },
        {
            "protocol_id": "P102-PROT-003",
            "component": "registration_evidence",
            "requirement": "Source/path, target observations, transforms, and timing must be auditable before baseline smoke.",
            "current_status": "manifested_pending_phase103_metadata_audit",
            "pass_condition": "Phase 103 finds explicit transform and timing files",
            "stop_condition": "registration depends on independent rescaling or target-label fitting",
        },
        {
            "protocol_id": "P102-PROT-004",
            "component": "large_downloads",
            "requirement": "Large downloads are allowed only as Phase 103 server intake with resumable manifests.",
            "current_status": "allowed_by_user_for_server",
            "pass_condition": "server download manifests record URL, size, checksum/content length, and extraction state",
            "stop_condition": "large files are downloaded locally or without audit manifests",
        },
    ]


def build_gate(
    *,
    phase101_gate: dict[str, Any],
    source_rows: list[dict[str, Any]],
    registration_rows: list[dict[str, Any]],
    queue_rows: list[dict[str, Any]],
    large_server_download_allowed: bool,
) -> dict[str, Any]:
    open_intake = [row for row in source_rows if row["phase103_intake_allowed"]]
    critical_pending = [
        row
        for row in registration_rows
        if str(row["current_status"]).startswith("critical_pending")
    ]
    if open_intake:
        status = "source_manifest_ready_phase103_intake_allowed"
        next_action = "enter Phase 103 minimal registered data intake/audit on the NIST AMMT metadata package"
    else:
        status = "blocked_no_source_manifest_ready"
        next_action = "provide a public registered source manifest or allow server intake"
    return {
        "status": status,
        "source_phase101_status": phase101_gate.get("status"),
        "preferred_candidate": open_intake[0]["candidate_id"] if open_intake else "none",
        "phase103_intake_allowed": bool(open_intake),
        "phase103_large_server_download_allowed": bool(large_server_download_allowed and open_intake),
        "phase104_baseline_smoke_allowed": False,
        "phase105_model_mechanism_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "source_rows": len(source_rows),
        "registration_rows": len(registration_rows),
        "critical_registration_checks_pending": len(critical_pending),
        "phase103_queue_rows": len(queue_rows),
        "submission_ready": False,
        "next_action": next_action,
        "required_before_baseline_smoke": [
            "download and inspect Metadata.zip",
            "verify explicit coordinate transforms and trigger timing",
            "construct a tiny registered source/path-to-target sample table",
            "fix train/validation/test split keys before model selection",
            "write baseline plan against mean, kNN, ExtraTrees, and no-process Macro PINN",
        ],
        "required_before_a100_training": [
            "Phase 103 intake/audit pass",
            "Phase 104 baseline-first smoke pass",
            "Phase 105 low-capacity mechanism pass",
            "server validation from a pushed commit",
        ],
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


def build_markdown(
    *,
    gate: dict[str, Any],
    source_rows: list[dict[str, Any]],
    file_rows: list[dict[str, Any]],
    registration_rows: list[dict[str, Any]],
    queue_rows: list[dict[str, Any]],
    protocol_rows: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "# Phase 102 Registered Source-Manifest/Data-Card Gate",
            "",
            "## Gate Decision",
            "",
            f"Status: `{gate['status']}`.",
            f"Preferred candidate: `{gate['preferred_candidate']}`.",
            f"Phase 103 intake allowed: `{str(gate['phase103_intake_allowed']).lower()}`.",
            f"Large server download allowed for Phase 103: `{str(gate['phase103_large_server_download_allowed']).lower()}`.",
            f"Phase 104 baseline smoke allowed: `{str(gate['phase104_baseline_smoke_allowed']).lower()}`.",
            f"A100 training allowed now: `{str(gate['a100_training_allowed_now']).lower()}`.",
            f"A100-SXM4-80GB request now: `{str(gate['a100_80gb_request_now']).lower()}`.",
            "",
            "Phase 102 only opens data intake/audit. It does not open baseline smoke or model training.",
            "",
            "## Candidate Sources",
            "",
            _markdown_table(
                source_rows,
                [
                    ("candidate_id", "Candidate"),
                    ("dataset_id", "Dataset"),
                    ("source_manifest_status", "Manifest"),
                    ("registration_evidence_status", "Registration"),
                    ("phase103_intake_allowed", "Phase 103"),
                    ("status", "Status"),
                ],
            ),
            "",
            "## NIST AMMT File Manifest",
            "",
            _markdown_table(
                file_rows,
                [
                    ("file_name", "File"),
                    ("expected_gib", "GiB"),
                    ("required_for_phase103", "Required first"),
                    ("download_scope", "Scope"),
                    ("status", "Status"),
                ],
            ),
            "",
            "## Registration Checks",
            "",
            _markdown_table(
                registration_rows,
                [
                    ("evidence_id", "Check"),
                    ("component", "Component"),
                    ("current_status", "Status"),
                    ("phase103_check", "Phase 103 check"),
                ],
            ),
            "",
            "## Phase 103 Queue",
            "",
            _markdown_table(
                queue_rows,
                [
                    ("queue_id", "Queue"),
                    ("task", "Task"),
                    ("allowed_location", "Location"),
                    ("pass_condition", "Pass"),
                ],
            ),
            "",
            "## Protocol",
            "",
            _markdown_table(
                protocol_rows,
                [
                    ("protocol_id", "Protocol"),
                    ("component", "Component"),
                    ("current_status", "Current"),
                    ("stop_condition", "Stop"),
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
    large_server_download_allowed: bool = True,
) -> dict[str, Any]:
    resolved = _default_paths(root)
    if paths:
        resolved.update(paths)

    phase101_gate = _read_json(resolved["phase101_gate"])
    phase101_targets = _read_csv(resolved["phase101_target_table"])
    phase101_queue = _read_csv(resolved["phase101_manual_queue"])
    data_card = build_nist_ammt_data_card()
    source_rows = build_source_rows(
        phase101_gate=phase101_gate,
        phase101_targets=phase101_targets,
        data_card=data_card,
        large_server_download_allowed=large_server_download_allowed,
    )
    file_rows = build_file_rows(data_card)
    registration_rows = build_registration_rows()
    queue_rows = build_phase103_queue(data_card, large_server_download_allowed)
    protocol_rows = build_protocol_rows()
    gate = build_gate(
        phase101_gate=phase101_gate,
        source_rows=source_rows,
        registration_rows=registration_rows,
        queue_rows=queue_rows,
        large_server_download_allowed=large_server_download_allowed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / "phase102_registered_source_candidate_table.csv"
    file_path = output_dir / "phase102_nist_ammt_file_manifest.csv"
    registration_path = output_dir / "phase102_registration_evidence_table.csv"
    queue_path = output_dir / "phase102_phase103_intake_queue.csv"
    protocol_path = output_dir / "phase102_protocol.csv"
    card_path = output_dir / "phase102_nist_ammt_data_card.json"
    gate_path = output_dir / "phase102_registered_source_manifest_gate.json"
    markdown_path = output_dir / "phase102_registered_source_manifest_gate.md"
    manifest_path = output_dir / "phase102_registered_source_manifest_gate_manifest.json"

    _write_csv(source_path, source_rows, SOURCE_FIELDS)
    _write_csv(file_path, file_rows, FILE_FIELDS)
    _write_csv(registration_path, registration_rows, REGISTRATION_FIELDS)
    _write_csv(queue_path, queue_rows, QUEUE_FIELDS)
    _write_csv(protocol_path, protocol_rows, PROTOCOL_FIELDS)
    _write_json(card_path, data_card)
    _write_json(gate_path, gate)
    markdown_path.write_text(
        build_markdown(
            gate=gate,
            source_rows=source_rows,
            file_rows=file_rows,
            registration_rows=registration_rows,
            queue_rows=queue_rows,
            protocol_rows=protocol_rows,
        ),
        encoding="utf-8",
    )

    manifest = {
        "phase": 102,
        "objective": "registered_source_manifest_data_card_gate",
        "inputs": {key: _display_path(path, root) for key, path in sorted(resolved.items())},
        "outputs": {
            "source_candidate_table": _display_path(source_path, root),
            "nist_ammt_file_manifest": _display_path(file_path, root),
            "registration_evidence_table": _display_path(registration_path, root),
            "phase103_intake_queue": _display_path(queue_path, root),
            "protocol": _display_path(protocol_path, root),
            "nist_ammt_data_card": _display_path(card_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "source_rows": len(source_rows),
            "file_rows": len(file_rows),
            "registration_rows": len(registration_rows),
            "phase103_queue_rows": len(queue_rows),
            "protocol_rows": len(protocol_rows),
            "phase101_manual_queue_rows_read": len(phase101_queue),
        },
        "gate": gate,
        "source_gates": {
            "phase101_status": phase101_gate.get("status"),
            "phase101_phase102_baseline_smoke_allowed": phase101_gate.get(
                "phase102_baseline_smoke_allowed"
            ),
        },
        "server_download_policy": {
            "large_server_download_allowed": large_server_download_allowed,
            "local_large_download_allowed": False,
            "first_file": data_card["phase103_minimal_intake_first_file"],
            "known_total_gib": data_card["known_total_gib"],
        },
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase102_registered_source_manifest_gate"),
    )
    parser.add_argument(
        "--disallow-large-server-download",
        action="store_true",
        help="Record Phase 103 large server download as blocked even if the source manifest exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    manifest = build_package(
        root=root,
        output_dir=output_dir,
        large_server_download_allowed=not args.disallow_large_server_download,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
