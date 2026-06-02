#!/usr/bin/env python3
"""Build the Phase 89 literature and venue gap-resolution package.

Phase 89 converts the Phase 61 literature gap register and Phase 88 remaining
work table into a writing-ready evidence handoff. It verifies stable sources
for the AM-Bench/PINN/process-conditioning framing and explicitly leaves target
venue alignment unresolved until a venue, author guide, or accepted benchmark
paper set is provided.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


EVIDENCE_FIELDS = (
    "evidence_id",
    "gap_id",
    "claim_area",
    "source_title",
    "authors_or_owner",
    "venue_or_source",
    "year",
    "doi",
    "stable_url",
    "source_type",
    "trust_state",
    "verification_trail",
    "supports_claim",
    "allowed_claim_strength",
    "limitations",
    "writing_ready",
)

GAP_FIELDS = (
    "gap_id",
    "location",
    "original_claim_needing_support",
    "resolution_status",
    "resolved_by_evidence_ids",
    "blocks_submission_after_phase89",
    "blocks_experimental_claim",
    "unresolved_reason",
    "next_action",
)

HANDOFF_FIELDS = (
    "handoff_id",
    "target_section",
    "claim_anchor",
    "allowed_claim",
    "evidence_ids",
    "source_locator",
    "allowed_strength",
    "wording_guard",
    "unresolved_dependency",
)

MANUAL_QUEUE_FIELDS = (
    "queue_id",
    "category",
    "needed_input",
    "reason",
    "blocks_submission",
    "suggested_user_action",
)


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
    if isinstance(value, bool):
        return "true" if value else "false"
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
    phase61 = root / "docs/results/phase61_manuscript_draft_package"
    phase88 = root / "docs/results/phase88_fallback_manuscript_finalization"
    return {
        "phase61_literature_gaps": phase61 / "phase61_literature_gap_register.csv",
        "phase88_remaining_work": phase88 / "phase88_remaining_work_table.csv",
        "phase88_gate": phase88 / "phase88_fallback_finalization_gate.json",
        "phase88_manifest": phase88 / "phase88_fallback_manuscript_finalization_manifest.json",
        "mds2_2716_manifest": root / "configs/data/ambench_mds2_2716_sources.yaml",
        "mds2_2718_manifest": root / "configs/data/ambench_mds2_2718_sources.yaml",
    }


def build_evidence_rows() -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": "P89-EVD-AMBench-001",
            "gap_id": "LIT_GAP-61-001",
            "claim_area": "AM-Bench public benchmark context",
            "source_title": "Additive Manufacturing Benchmark Test Series (AM-Bench)",
            "authors_or_owner": "National Institute of Standards and Technology",
            "venue_or_source": "NIST official AM-Bench website",
            "year": "2026 access",
            "doi": "",
            "stable_url": "https://www.nist.gov/ambench",
            "source_type": "official_project_page",
            "trust_state": "verified_official_url",
            "verification_trail": "Phase 89 web verification: NIST identifies AM Bench as a NIST-led benchmark series with controlled AM measurement data and public archiving.",
            "supports_claim": "AM-Bench is an appropriate public benchmark ecosystem for additive-manufacturing model validation and generalization claims.",
            "allowed_claim_strength": "official_benchmark_context",
            "limitations": "Use for dataset/benchmark context, not as evidence that this model is universally robust.",
            "writing_ready": True,
        },
        {
            "evidence_id": "P89-EVD-AMBench-002",
            "gap_id": "LIT_GAP-61-001",
            "claim_area": "AM-Bench public data access",
            "source_title": "AM Bench Data Management Systems",
            "authors_or_owner": "National Institute of Standards and Technology",
            "venue_or_source": "NIST official AM-Bench data-management page",
            "year": "2023 page status; 2026 access",
            "doi": "",
            "stable_url": "https://www.nist.gov/ambench/am-bench-data-management-systems",
            "source_type": "official_data_management_page",
            "trust_state": "verified_official_url",
            "verification_trail": "Phase 89 web verification: NIST states AM Bench 2022 data are served through the AM Bench website, NIST PDR, measurement catalog, SciServer, and related mechanisms.",
            "supports_claim": "AM-Bench data and metadata are publicly discoverable through NIST data-management systems, including the Public Data Repository.",
            "allowed_claim_strength": "official_data_availability",
            "limitations": "Use with specific PDR records for dataset-level claims.",
            "writing_ready": True,
        },
        {
            "evidence_id": "P89-EVD-MDS2-2716",
            "gap_id": "LIT_GAP-61-001",
            "claim_area": "AM-Bench thermography and scan-strategy data",
            "source_title": "AM Bench 2022 Measurement Results Data: In-situ Thermography and Scan Strategy for Laser-scanned Single Tracks and Pads on Bare In718",
            "authors_or_owner": "National Institute of Standards and Technology",
            "venue_or_source": "NIST Public Data Repository",
            "year": "2024 metadata update",
            "doi": "10.18434/mds2-2716",
            "stable_url": "https://data.nist.gov/od/id/mds2-2716",
            "source_type": "official_dataset_record",
            "trust_state": "verified_project_manifest_and_stable_url",
            "verification_trail": "Project manifest `configs/data/ambench_mds2_2716_sources.yaml` pins DOI, PDR page, checksums, and thermography/scan-strategy files.",
            "supports_claim": "The project experiments are based on a public NIST PDR thermography/scan-strategy dataset for AMB2022-03 single tracks and pads.",
            "allowed_claim_strength": "dataset_specific",
            "limitations": "Current repository evidence still lacks single-track scan-path registration and pad camera-to-galvo mapping.",
            "writing_ready": True,
        },
        {
            "evidence_id": "P89-EVD-MDS2-2718",
            "gap_id": "LIT_GAP-61-001",
            "claim_area": "AM-Bench optical microscopy data",
            "source_title": "AM Bench 2022 Measurement Results Data: Optical Microscopy of Laser-scanned Single Tracks and Pads (AMB2022-03)",
            "authors_or_owner": "National Institute of Standards and Technology",
            "venue_or_source": "NIST Public Data Repository",
            "year": "2022 metadata update",
            "doi": "10.18434/mds2-2718",
            "stable_url": "https://data.nist.gov/od/id/mds2-2718",
            "source_type": "official_dataset_record",
            "trust_state": "verified_project_manifest_and_stable_url",
            "verification_trail": "Project manifest `configs/data/ambench_mds2_2718_sources.yaml` pins DOI, PDR page, TIFF/XLSX checksums, and exact-line microscopy assets.",
            "supports_claim": "Optical microscopy is a public auxiliary AM-Bench source used here only as diagnostic microstructure evidence.",
            "allowed_claim_strength": "appendix_dataset_context",
            "limitations": "Do not claim stable microstructure-conditioned model performance from the existing diagnostic branch.",
            "writing_ready": True,
        },
        {
            "evidence_id": "P89-EVD-PINN-001",
            "gap_id": "LIT_GAP-61-002",
            "claim_area": "PINN foundation and inverse problems",
            "source_title": "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations",
            "authors_or_owner": "Raissi, Perdikaris, and Karniadakis",
            "venue_or_source": "Journal of Computational Physics",
            "year": "2019",
            "doi": "10.1016/j.jcp.2018.10.045",
            "stable_url": "https://www.sciencedirect.com/science/article/pii/S0021999118307125",
            "source_type": "peer_reviewed_article",
            "trust_state": "verified_doi_primary_url",
            "verification_trail": "Phase 89 web verification matched title, venue, and DOI on the publisher page.",
            "supports_claim": "PINNs support forward and inverse PDE problem settings and are a suitable conceptual baseline for physics-informed thermal modeling.",
            "allowed_claim_strength": "foundational_method_context",
            "limitations": "Foundational support only; does not validate this AM-Bench architecture.",
            "writing_ready": True,
        },
        {
            "evidence_id": "P89-EVD-PINN-002",
            "gap_id": "LIT_GAP-61-002",
            "claim_area": "physics-informed ML scope and limitations",
            "source_title": "Physics-informed machine learning",
            "authors_or_owner": "Karniadakis, Kevrekidis, Lu, Perdikaris, Wang, and Yang",
            "venue_or_source": "Nature Reviews Physics",
            "year": "2021",
            "doi": "10.1038/s42254-021-00314-5",
            "stable_url": "https://www.nature.com/articles/s42254-021-00314-5",
            "source_type": "peer_reviewed_review",
            "trust_state": "verified_doi_primary_url",
            "verification_trail": "Phase 89 web verification matched title, authors, journal, year, pages, and DOI on the publisher page.",
            "supports_claim": "Physics-informed ML is relevant to forward/inverse problems, high-dimensional settings, and data-physics integration, while limitations should be stated explicitly.",
            "allowed_claim_strength": "review_context",
            "limitations": "Use for broad framing, not as a direct performance claim.",
            "writing_ready": True,
        },
        {
            "evidence_id": "P89-EVD-PINN-003",
            "gap_id": "LIT_GAP-61-002",
            "claim_area": "PINN training pathologies",
            "source_title": "Understanding and Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks",
            "authors_or_owner": "Wang, Teng, and Perdikaris",
            "venue_or_source": "SIAM Journal on Scientific Computing",
            "year": "2021",
            "doi": "10.1137/20M1318043",
            "stable_url": "https://epubs.siam.org/doi/10.1137/20M1318043",
            "source_type": "peer_reviewed_article",
            "trust_state": "verified_doi_primary_url",
            "verification_trail": "Phase 89 web verification matched title, authors, SIAM journal page, and DOI.",
            "supports_claim": "PINN optimization can suffer from gradient-flow pathologies, supporting careful wording around sparse/heterogeneous thermal data tradeoffs.",
            "allowed_claim_strength": "limitation_context",
            "limitations": "Do not imply this exact pathology is proven for every failed branch; cite as representative training-risk literature.",
            "writing_ready": True,
        },
        {
            "evidence_id": "P89-EVD-FILM-001",
            "gap_id": "LIT_GAP-61-002",
            "claim_area": "feature-wise conditioning",
            "source_title": "FiLM: Visual Reasoning with a General Conditioning Layer",
            "authors_or_owner": "Perez, Strub, de Vries, Dumoulin, and Courville",
            "venue_or_source": "Proceedings of the AAAI Conference on Artificial Intelligence",
            "year": "2018",
            "doi": "10.1609/aaai.v32i1.11671",
            "stable_url": "https://ojs.aaai.org/index.php/AAAI/article/view/11671",
            "source_type": "peer_reviewed_conference_article",
            "trust_state": "verified_doi_primary_url",
            "verification_trail": "Phase 89 web verification matched title, authors, AAAI venue, year, and DOI.",
            "supports_claim": "Feature-wise affine modulation is a recognized conditioning mechanism; this supports the process-conditioning mechanism description, not the AM-Bench result itself.",
            "allowed_claim_strength": "conditioning_mechanism_context",
            "limitations": "Original FiLM target is visual reasoning; use as method-mechanism support only.",
            "writing_ready": True,
        },
    ]


def _evidence_ids_by_gap(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for row in rows:
        if row.get("writing_ready"):
            grouped.setdefault(str(row["gap_id"]), []).append(str(row["evidence_id"]))
    return grouped


def build_gap_rows(
    literature_gaps: list[dict[str, str]],
    evidence_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped = _evidence_ids_by_gap(evidence_rows)
    rows: list[dict[str, Any]] = []
    for gap in literature_gaps:
        gap_id = gap.get("gap_id", "")
        evidence_ids = grouped.get(gap_id, [])
        if gap_id == "LIT_GAP-61-003":
            status = "unresolved_user_input_required"
            blocks_submission = True
            unresolved_reason = "No target venue, author guide, or accepted-paper benchmark set has been provided."
            next_action = "User should provide target venue or 3-10 benchmark papers before final formatting/citation-density claims."
        elif evidence_ids:
            status = "resolved_writing_ready"
            blocks_submission = False
            unresolved_reason = ""
            next_action = "Use the writing handoff in Phase 90 manuscript integration."
        else:
            status = "unresolved_missing_verified_evidence"
            blocks_submission = True
            unresolved_reason = "No writing-ready evidence record was assigned to this gap."
            next_action = "Run another literature verification pass with exact search strings."
        rows.append(
            {
                "gap_id": gap_id,
                "location": gap.get("location"),
                "original_claim_needing_support": gap.get("claim_needing_support"),
                "resolution_status": status,
                "resolved_by_evidence_ids": ";".join(evidence_ids),
                "blocks_submission_after_phase89": blocks_submission,
                "blocks_experimental_claim": False,
                "unresolved_reason": unresolved_reason,
                "next_action": next_action,
            }
        )
    return rows


def build_handoff_rows(evidence_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence_by_id = {row["evidence_id"]: row for row in evidence_rows}

    def locator(ids: list[str]) -> str:
        return "; ".join(
            f"{evidence_by_id[eid]['source_title']} ({evidence_by_id[eid]['stable_url']})"
            for eid in ids
        )

    handoffs = [
        {
            "handoff_id": "P89-HANDOFF-DATASET",
            "target_section": "Introduction/Dataset",
            "claim_anchor": "AM-Bench benchmark and data source framing",
            "allowed_claim": "AM-Bench can be described as a public NIST-led additive-manufacturing benchmark ecosystem, with the project experiments tied to the mds2-2716 thermography/scan-strategy record and diagnostic microscopy tied to mds2-2718.",
            "evidence_ids": [
                "P89-EVD-AMBench-001",
                "P89-EVD-AMBench-002",
                "P89-EVD-MDS2-2716",
                "P89-EVD-MDS2-2718",
            ],
            "allowed_strength": "dataset_context",
            "wording_guard": "Do not imply that all AM-Bench modalities were used or that registration blockers are solved.",
            "unresolved_dependency": "",
        },
        {
            "handoff_id": "P89-HANDOFF-PINN",
            "target_section": "Related Work/Methods",
            "claim_anchor": "PINN and physics-informed ML framing",
            "allowed_claim": "PINNs are established for forward and inverse PDE problems, but physics-informed ML has known optimization and data-integration limitations that motivate gated validation rather than architecture-only claims.",
            "evidence_ids": [
                "P89-EVD-PINN-001",
                "P89-EVD-PINN-002",
                "P89-EVD-PINN-003",
            ],
            "allowed_strength": "framing_and_limitation_context",
            "wording_guard": "Tie limitations to representative literature and project diagnostics; avoid claiming the literature proves this exact failure mode.",
            "unresolved_dependency": "",
        },
        {
            "handoff_id": "P89-HANDOFF-CONDITIONING",
            "target_section": "Methods/Model",
            "claim_anchor": "process-conditioned route mechanism",
            "allowed_claim": "Feature-wise modulation is a recognized conditioning mechanism; the paper may describe the `spot_size` branch as a process-conditioned Macro PINN route guarded by the project's empirical route-selection evidence.",
            "evidence_ids": ["P89-EVD-FILM-001"],
            "allowed_strength": "method_mechanism_context",
            "wording_guard": "Do not claim FiLM literature alone validates AM-Bench performance; performance must cite Phase 55/60/74 artifacts.",
            "unresolved_dependency": "",
        },
        {
            "handoff_id": "P89-HANDOFF-VENUE",
            "target_section": "All final manuscript sections",
            "claim_anchor": "target venue style and citation density",
            "allowed_claim": "No final venue-specific style claim is allowed yet.",
            "evidence_ids": [],
            "allowed_strength": "none_until_user_input",
            "wording_guard": "Keep final section order, citation density, and caption style provisional.",
            "unresolved_dependency": "target venue, author guide, or 3-10 accepted benchmark papers",
        },
    ]
    rows: list[dict[str, Any]] = []
    for row in handoffs:
        ids = list(row["evidence_ids"])
        rows.append(
            {
                **row,
                "evidence_ids": ";".join(ids),
                "source_locator": locator(ids) if ids else "",
            }
        )
    return rows


def build_manual_queue_rows(gap_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unresolved = [row for row in gap_rows if row["blocks_submission_after_phase89"]]
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(unresolved, start=1):
        rows.append(
            {
                "queue_id": f"P89-MANUAL-{index:03d}",
                "category": "target_venue_alignment" if row["gap_id"] == "LIT_GAP-61-003" else "literature_verification",
                "needed_input": row["unresolved_reason"],
                "reason": row["original_claim_needing_support"],
                "blocks_submission": True,
                "suggested_user_action": row["next_action"],
            }
        )
    return rows


def build_gate(
    gap_rows: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    phase88_gate: dict[str, Any],
) -> dict[str, Any]:
    resolved = [row for row in gap_rows if row["resolution_status"] == "resolved_writing_ready"]
    unresolved_submission = [row for row in gap_rows if row["blocks_submission_after_phase89"]]
    venue_unresolved = any(row["gap_id"] == "LIT_GAP-61-003" for row in unresolved_submission)
    all_nonvenue_lit_resolved = all(
        row["resolution_status"] == "resolved_writing_ready"
        for row in gap_rows
        if row["gap_id"] != "LIT_GAP-61-003"
    )
    if all_nonvenue_lit_resolved and venue_unresolved:
        status = "literature_core_resolved_venue_unresolved"
        next_action = "proceed to manuscript evidence integration, but request target venue before final formatting and submission readiness"
    elif not unresolved_submission:
        status = "literature_and_venue_resolved"
        next_action = "proceed to manuscript v1 integration and final polish"
    else:
        status = "literature_gap_resolution_incomplete"
        next_action = "resolve remaining evidence gaps before final manuscript integration"
    return {
        "status": status,
        "phase88_status": phase88_gate.get("status"),
        "experimental_claim_complete": bool(phase88_gate.get("experimental_claim_complete")),
        "submission_ready": False if unresolved_submission else bool(phase88_gate.get("experimental_claim_complete")),
        "core_literature_ready": all_nonvenue_lit_resolved,
        "venue_alignment_ready": not venue_unresolved,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "verified_evidence_rows": sum(1 for row in evidence_rows if row["writing_ready"]),
        "resolved_gap_rows": len(resolved),
        "unresolved_submission_blockers": len(unresolved_submission),
        "next_action": next_action,
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
    gate: dict[str, Any],
    gap_rows: list[dict[str, Any]],
    handoff_rows: list[dict[str, Any]],
    manual_rows: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "# Phase 89 Literature and Venue Gap Resolution",
            "",
            "## Gate Decision",
            "",
            f"Status: `{gate['status']}`.",
            f"Core literature ready: `{str(gate['core_literature_ready']).lower()}`.",
            f"Venue alignment ready: `{str(gate['venue_alignment_ready']).lower()}`.",
            f"Submission ready: `{str(gate['submission_ready']).lower()}`.",
            f"A100 training allowed now: `{str(gate['a100_training_allowed_now']).lower()}`.",
            "",
            "Phase 89 resolves the non-venue literature support needed for manuscript integration. It does not claim final submission readiness because the target venue remains unspecified.",
            "",
            "## Gap Resolution",
            "",
            _markdown_table(
                gap_rows,
                [
                    ("gap_id", "Gap"),
                    ("resolution_status", "Status"),
                    ("resolved_by_evidence_ids", "Evidence"),
                    ("blocks_submission_after_phase89", "Blocks submission"),
                    ("next_action", "Next action"),
                ],
            ),
            "",
            "## Writing Handoff",
            "",
            _markdown_table(
                handoff_rows,
                [
                    ("handoff_id", "Handoff"),
                    ("target_section", "Section"),
                    ("allowed_strength", "Strength"),
                    ("wording_guard", "Guard"),
                    ("unresolved_dependency", "Dependency"),
                ],
            ),
            "",
            "## Manual Queue",
            "",
            _markdown_table(
                manual_rows,
                [
                    ("queue_id", "Queue"),
                    ("category", "Category"),
                    ("needed_input", "Needed input"),
                    ("blocks_submission", "Blocks submission"),
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

    literature_gaps = _read_csv(resolved["phase61_literature_gaps"])
    remaining_work_rows = _read_csv(resolved["phase88_remaining_work"])
    phase88_gate = _read_json(resolved["phase88_gate"])
    phase88_manifest = _read_json(resolved["phase88_manifest"])

    evidence_rows = build_evidence_rows()
    gap_rows = build_gap_rows(literature_gaps, evidence_rows)
    handoff_rows = build_handoff_rows(evidence_rows)
    manual_rows = build_manual_queue_rows(gap_rows)
    gate = build_gate(gap_rows, evidence_rows, phase88_gate)

    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "phase89_evidence_register.csv"
    gap_path = output_dir / "phase89_gap_resolution_table.csv"
    handoff_path = output_dir / "phase89_writing_handoff.csv"
    manual_path = output_dir / "phase89_manual_verification_queue.csv"
    gate_path = output_dir / "phase89_literature_venue_gap_resolution_gate.json"
    markdown_path = output_dir / "phase89_literature_venue_gap_resolution.md"
    manifest_path = output_dir / "phase89_literature_venue_gap_resolution_manifest.json"

    _write_csv(evidence_path, evidence_rows, EVIDENCE_FIELDS)
    _write_csv(gap_path, gap_rows, GAP_FIELDS)
    _write_csv(handoff_path, handoff_rows, HANDOFF_FIELDS)
    _write_csv(manual_path, manual_rows, MANUAL_QUEUE_FIELDS)
    _write_json(gate_path, gate)
    markdown_path.write_text(build_markdown(gate, gap_rows, handoff_rows, manual_rows), encoding="utf-8")

    manifest = {
        "phase": 89,
        "objective": "literature_venue_gap_resolution",
        "inputs": {key: _display_path(path, root) for key, path in sorted(resolved.items())},
        "outputs": {
            "evidence_register": _display_path(evidence_path, root),
            "gap_resolution_table": _display_path(gap_path, root),
            "writing_handoff": _display_path(handoff_path, root),
            "manual_verification_queue": _display_path(manual_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "phase61_gap_rows": len(literature_gaps),
            "phase88_remaining_work_rows": len(remaining_work_rows),
            "evidence_rows": len(evidence_rows),
            "writing_ready_evidence_rows": sum(1 for row in evidence_rows if row["writing_ready"]),
            "gap_resolution_rows": len(gap_rows),
            "writing_handoff_rows": len(handoff_rows),
            "manual_queue_rows": len(manual_rows),
        },
        "gate": gate,
        "phase88_gate": {
            "status": phase88_gate.get("status"),
            "experimental_claim_complete": phase88_gate.get("experimental_claim_complete"),
            "submission_ready": phase88_gate.get("submission_ready"),
            "open_submission_blockers": phase88_gate.get("open_submission_blockers"),
        },
        "phase88_manifest_gate": phase88_manifest.get("gate"),
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase89_literature_venue_gap_resolution"),
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
