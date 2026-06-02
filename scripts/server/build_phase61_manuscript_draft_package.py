#!/usr/bin/env python3
"""Build a manuscript draft package from the Phase 60 evidence package.

The output is a draft packet, not a final polished manuscript. It converts the
current verified result tables into Results, Methods, captions, and a
claim-to-evidence crosswalk so later writing can preserve the evidence boundary.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CROSSWALK_FIELDS = (
    "claim_anchor_id",
    "claim_summary",
    "manuscript_location",
    "support_type",
    "support_locator",
    "evidence_register_key",
    "allowed_claim_strength",
    "verification_state",
    "owner_skill",
    "open_risk",
    "draft_sentence",
)

GAP_FIELDS = (
    "gap_id",
    "location",
    "claim_needing_support",
    "evidence_type_needed",
    "suggested_search_or_material",
    "blocks_current_phase61_draft",
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
    if isinstance(value, float):
        return f"{value:.6f}"
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


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _float_value(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _default_paths(root: Path) -> dict[str, Path]:
    package = root / "docs/results/phase60_manuscript_evidence_package"
    return {
        "main": package / "phase60_main_spot_size_seed_positive_table.csv",
        "route": package / "phase60_route_guard_boundary_table.csv",
        "stress": package / "phase60_stress_boundary_table.csv",
        "appendix": package / "phase60_appendix_negative_diagnostic_table.csv",
        "next_gate": package / "phase60_next_branch_gate_table.csv",
        "phase60_markdown": package / "phase60_manuscript_evidence_package.md",
        "phase60_manifest": package / "phase60_manuscript_evidence_package_manifest.json",
    }


def _row_by_dataset_metric(
    main_rows: list[dict[str, str]], dataset: str, metric: str
) -> dict[str, str]:
    for row in main_rows:
        if row.get("dataset") == dataset and row.get("metric") == metric:
            return row
    raise KeyError(f"Missing {dataset}/{metric} in main table")


def _route_summary(route_rows: list[dict[str, str]]) -> dict[str, list[str]]:
    summary: dict[str, list[str]] = {
        "process_conditioning_positive": [],
        "no_process_fallback_positive": [],
        "route_guard_only": [],
    }
    for row in route_rows:
        label = f"{row.get('dataset')}:{row.get('split')}"
        claim_use = row.get("claim_use") or ""
        classification = row.get("classification") or ""
        if "no-process fallback" in claim_use:
            summary["no_process_fallback_positive"].append(label)
        elif classification == "paper_claim_positive":
            summary["process_conditioning_positive"].append(label)
        else:
            summary["route_guard_only"].append(label)
    return summary


def _stress_by_scenario(stress_rows: list[dict[str, str]], scenario: str) -> list[dict[str, str]]:
    return [row for row in stress_rows if row.get("scenario") == scenario]


def _claim(
    claim_anchor_id: str,
    claim_summary: str,
    manuscript_location: str,
    support_type: str,
    support_locator: str,
    evidence_register_key: str,
    allowed_claim_strength: str,
    verification_state: str,
    open_risk: str,
    draft_sentence: str,
) -> dict[str, str]:
    return {
        "claim_anchor_id": claim_anchor_id,
        "claim_summary": claim_summary,
        "manuscript_location": manuscript_location,
        "support_type": support_type,
        "support_locator": support_locator,
        "evidence_register_key": evidence_register_key,
        "allowed_claim_strength": allowed_claim_strength,
        "verification_state": verification_state,
        "owner_skill": "paper-writing-workflow",
        "open_risk": open_risk,
        "draft_sentence": draft_sentence,
    }


def build_claim_crosswalk(
    paths: dict[str, Path],
    manifest: dict[str, Any],
    main_rows: list[dict[str, str]],
    route_rows: list[dict[str, str]],
    stress_rows: list[dict[str, str]],
    appendix_rows: list[dict[str, str]],
    next_gate_rows: list[dict[str, str]],
    root: Path,
) -> list[dict[str, str]]:
    broad12_rmse = _row_by_dataset_metric(main_rows, "broad12", "Test RMSE")
    broad12_hot = _row_by_dataset_metric(main_rows, "broad12", "Hot q90 RMSE")
    broad12_grad = _row_by_dataset_metric(main_rows, "broad12", "Gradient q90 RMSE")
    broad21_rmse = _row_by_dataset_metric(main_rows, "broad21", "Test RMSE")
    broad21_hot = _row_by_dataset_metric(main_rows, "broad21", "Hot q90 RMSE")
    broad21_grad = _row_by_dataset_metric(main_rows, "broad21", "Gradient q90 RMSE")
    boundary_rows = _stress_by_scenario(stress_rows, "alternate_density_stress")
    upper_rows = _stress_by_scenario(stress_rows, "residual_upper_bound_gate")
    stronger_rows = _stress_by_scenario(stress_rows, "stronger_baseline_stress")
    panel_rows = _stress_by_scenario(stress_rows, "auxiliary_process_panel")
    route_summary = _route_summary(route_rows)
    main_path = _display_path(paths["main"], root)
    route_path = _display_path(paths["route"], root)
    stress_path = _display_path(paths["stress"], root)
    appendix_path = _display_path(paths["appendix"], root)
    gate_path = _display_path(paths["next_gate"], root)
    manifest_path = _display_path(paths["phase60_manifest"], root)
    return [
        _claim(
            "C61-MAIN-001",
            "The main manuscript claim is fixed-sampling broad12/broad21 spot-size transfer under broad_process_v1, not density-invariant robustness.",
            "Results: Fixed-sampling spot-size transfer",
            "result",
            f"{main_path}; {manifest_path}",
            "phase60_main_table; phase60_manifest",
            "moderate",
            "writing_ready",
            "Claim must remain limited to fixed-sampling broad12/broad21 spot_size.",
            "The main performance result is the fixed-sampling `spot_size` transfer setting, where `broad_process_v1` is evaluated on broad12 and broad21 with seeds 7, 1, and 2.",
        ),
        _claim(
            "C61-RESULT-001",
            "On broad12 spot_size, broad_process_v1 beats the best strong baseline and no-process model on global, hot-zone, and gradient-band metrics.",
            "Results: Fixed-sampling spot-size transfer",
            "result",
            f"{main_path}; rows dataset=broad12",
            "phase60_main_table",
            "moderate",
            "writing_ready",
            "Do not generalize beyond broad12 spot_size.",
            f"On broad12, the route achieved {_fmt(broad12_rmse.get('broad_process_v1_mean'))} test RMSE, {_fmt(broad12_hot.get('broad_process_v1_mean'))} hot q90 RMSE, and {_fmt(broad12_grad.get('broad_process_v1_mean'))} gradient q90 RMSE, all below the corresponding best strong baselines.",
        ),
        _claim(
            "C61-RESULT-002",
            "On broad21 spot_size, broad_process_v1 beats the best strong baseline and no-process model on global, hot-zone, and gradient-band metrics.",
            "Results: Fixed-sampling spot-size transfer",
            "result",
            f"{main_path}; rows dataset=broad21",
            "phase60_main_table",
            "moderate",
            "writing_ready",
            "Do not generalize beyond broad21 spot_size.",
            f"On broad21, the route achieved {_fmt(broad21_rmse.get('broad_process_v1_mean'))} test RMSE, {_fmt(broad21_hot.get('broad_process_v1_mean'))} hot q90 RMSE, and {_fmt(broad21_grad.get('broad_process_v1_mean'))} gradient q90 RMSE, again below the corresponding best strong baselines.",
        ),
        _claim(
            "C61-STRESS-001",
            "Stronger random-forest and histogram-gradient-boosting baselines did not invalidate the fixed-sampling floor.",
            "Results: Stress tests",
            "result",
            f"{stress_path}; scenario=stronger_baseline_stress; rows={len(stronger_rows)}",
            "phase60_stress_boundary_table",
            "moderate",
            "writing_ready",
            "Only applies to the implemented reproducible stress baseline family.",
            "The stronger-baseline stress rows preserve the fixed-sampling floor: all six broad12/broad21 metric checks remain in the `pass` state.",
        ),
        _claim(
            "C61-STRESS-002",
            "The auxiliary broad15 panel supports the fixed-sampling spot-size route at seed 7.",
            "Results: Stress tests",
            "result",
            f"{stress_path}; scenario=auxiliary_process_panel; rows={len(panel_rows)}",
            "phase60_stress_boundary_table",
            "cautious",
            "writing_ready",
            "Single seed auxiliary panel; do not treat as seed-robust external validation.",
            "The auxiliary broad15 process panel is consistent with the fixed-sampling result, with all three reported metrics marked as `pass` at seed 7.",
        ),
        _claim(
            "C61-BOUNDARY-001",
            "Alternate-density broad21 spot_size is a density-sensitive route boundary.",
            "Results: Boundary tests",
            "result",
            f"{stress_path}; scenario=alternate_density_stress; boundary rows={len([r for r in boundary_rows if r.get('status') == 'boundary'])}",
            "phase60_stress_boundary_table",
            "strong",
            "writing_ready",
            "Boundary must not be softened into robustness.",
            "The alternate-density broad21 rows are marked as `boundary` for test RMSE, hot q90 RMSE, and gradient q90 RMSE, so the manuscript must not claim density-invariant robustness.",
        ),
        _claim(
            "C61-BOUNDARY-002",
            "The Phase 59 no-test-leakage upper-bound probe blocks model expansion from the density failure.",
            "Results: Boundary tests",
            "result",
            f"{stress_path}; scenario=residual_upper_bound_gate; {manifest_path}",
            "phase60_stress_boundary_table; phase60_manifest",
            "strong",
            "writing_ready",
            "Only blocks density-failure-driven expansion, not all future model work.",
            "The residual upper-bound gate selected `blend:broad_process_v1->mean:alpha=1` and records `block_density_failure_driven_model_expansion`, indicating that the density failure should be written as a route boundary rather than as a new model signal.",
        ),
        _claim(
            "C61-ROUTE-001",
            "Laser power, scan speed, full process, and line/no-process fallback are route-boundary evidence rather than universal process-conditioning wins.",
            "Results: Route-guard boundaries",
            "result",
            f"{route_path}; route_guard_only={len(route_summary['route_guard_only'])}; no_process_fallback={len(route_summary['no_process_fallback_positive'])}",
            "phase60_route_guard_table",
            "moderate",
            "writing_ready",
            "Must not claim strong-baseline wins for laser_power, scan_speed, or full process.",
            "The route-guard table keeps laser power, scan speed, and full process in boundary positions, while line positives are explicitly no-process fallback evidence.",
        ),
        _claim(
            "C61-APPX-001",
            "The appendix should carry Phases 33-53 negatives plus Phase 58/59 density-sensitive diagnostics.",
            "Appendix: Negative diagnostics",
            "result",
            f"{appendix_path}; rows={len(appendix_rows)}",
            "phase60_appendix_table",
            "moderate",
            "writing_ready",
            "Appendix wording should not bury limitations that affect main claim scope.",
            "The appendix table contains fourteen diagnostic or boundary rows, including the Phase 58 density stress and Phase 59 residual upper-bound gate.",
        ),
        _claim(
            "C61-GATE-001",
            "Next model branches remain gated: Candidate A is paused, Candidate B is blocked by Phase 59, and Candidate C is blocked by registration data.",
            "Discussion: Next-branch gate",
            "result",
            f"{gate_path}; rows={len(next_gate_rows)}",
            "phase60_next_branch_gate_table",
            "moderate",
            "writing_ready",
            "A future validation-visible signal can reopen a candidate branch.",
            "The next-branch table keeps Candidate A paused, Candidate B blocked by the Phase 59 density gate, and Candidate C blocked by registration data.",
        ),
        _claim(
            "C61-METHOD-001",
            "Route selection, seed expansion, and model-branch promotion are governed by no-test-leakage and frozen-floor rules.",
            "Methods: Claim governance",
            "code/result",
            manifest_path,
            "phase60_manifest",
            "moderate",
            "writing_ready",
            "Methods should not imply all historical branches used the Phase 60 gate before it existed.",
            "Route selection, seed expansion, and future branch promotion are described under a frozen-floor rule that requires preserving the broad12/broad21 `spot_size` floor before seed expansion.",
        ),
    ]


def build_literature_gaps() -> list[dict[str, str]]:
    return [
        {
            "gap_id": "LIT_GAP-61-001",
            "location": "Introduction or Dataset paragraph",
            "claim_needing_support": "AM-Bench is a public additive-manufacturing benchmark suitable for thermal/process generalization studies.",
            "evidence_type_needed": "verified dataset citation or official NIST AM-Bench source",
            "suggested_search_or_material": "NIST AM-Bench official dataset page or peer-reviewed AM-Bench description",
            "blocks_current_phase61_draft": "no",
        },
        {
            "gap_id": "LIT_GAP-61-002",
            "location": "Related work",
            "claim_needing_support": "Physics-informed neural networks and process-conditioned neural models have known tradeoffs under sparse or heterogeneous thermal data.",
            "evidence_type_needed": "verified literature review and representative primary papers",
            "suggested_search_or_material": "PINN thermal modeling, additive manufacturing thermal surrogate, process-conditioned neural operator papers",
            "blocks_current_phase61_draft": "no",
        },
        {
            "gap_id": "LIT_GAP-61-003",
            "location": "Target-venue adaptation",
            "claim_needing_support": "Final section order, citation density, and caption style match the target journal or conference.",
            "evidence_type_needed": "target venue author guide or 3-10 benchmark manuscripts",
            "suggested_search_or_material": "user-provided target venue or accepted examples",
            "blocks_current_phase61_draft": "no",
        },
    ]


def _sentence(claims: dict[str, dict[str, str]], claim_id: str) -> str:
    return f"{claims[claim_id]['draft_sentence']} [{claim_id}]"


def build_results_draft(claim_rows: list[dict[str, str]]) -> str:
    claims = {row["claim_anchor_id"]: row for row in claim_rows}
    lines = [
        "# Phase 61 Results Draft",
        "",
        "## Writing Gate",
        "",
        "Mode: section_draft",
        "Active gate: draft_ready for result claims supported by Phase 60; needs_verification for literature-framed introduction claims.",
        "Evidence status: internal result tables are writing-ready; external literature support is listed as LIT_GAP in the package manifest.",
        "",
        "## Fixed-Sampling Spot-Size Transfer",
        "",
        _sentence(claims, "C61-MAIN-001"),
        _sentence(claims, "C61-RESULT-001"),
        _sentence(claims, "C61-RESULT-002"),
        "These results support a bounded main claim: the route-guarded process-conditioned Macro physics-informed neural network (Macro PINN) has a stable `spot_size` branch under the fixed broad12/broad21 sampling and seed protocol. [C61-MAIN-001]",
        "",
        "## Stress Tests And Claim Boundaries",
        "",
        _sentence(claims, "C61-STRESS-001"),
        _sentence(claims, "C61-STRESS-002"),
        _sentence(claims, "C61-BOUNDARY-001"),
        _sentence(claims, "C61-BOUNDARY-002"),
        "The combined interpretation is therefore not that the route guard is universally robust, but that the fixed-sampling `spot_size` floor is strong enough for the main result while the density-sensitive broad21 case must be discussed as a boundary. [C61-MAIN-001; C61-BOUNDARY-001; C61-BOUNDARY-002]",
        "",
        "## Route-Guard Boundary Axes",
        "",
        _sentence(claims, "C61-ROUTE-001"),
        "This wording keeps the route-guard contribution separate from unsupported claims of universal process-conditioning superiority. [C61-ROUTE-001]",
        "",
        "## Appendix And Negative-Evidence Discipline",
        "",
        _sentence(claims, "C61-APPX-001"),
        _sentence(claims, "C61-GATE-001"),
        "The appendix should be used to show that negative branches were pruned by explicit gates rather than omitted after informal inspection. [C61-APPX-001; C61-GATE-001]",
        "",
        "## User-Review Items",
        "",
        "- Confirm whether this section should remain as a compact Results subsection or be split into Results and Discussion.",
        "- Provide target venue examples before final style alignment.",
        "- Resolve LIT_GAP items before drafting Introduction or Related Work claims.",
        "",
    ]
    return "\n".join(lines)


def build_methods_draft(claim_rows: list[dict[str, str]]) -> str:
    claims = {row["claim_anchor_id"]: row for row in claim_rows}
    lines = [
        "# Phase 61 Methods Draft",
        "",
        "## Evidence-Governed Experiment Selection",
        "",
        "The manuscript should describe experiments as a staged evidence-governed workflow rather than as an unconstrained architecture sweep. [C61-METHOD-001]",
        _sentence(claims, "C61-METHOD-001"),
        "The fixed floor for later branches is the `broad_process_v1` route on broad12 and broad21 `spot_size`, evaluated with seeds 7, 1, and 2. [C61-MAIN-001]",
        "",
        "## Route-Guarded Macro PINN Evaluation",
        "",
        "The route guard separates process-conditioned and no-process fallback behavior by split axis, so axes that do not pass the strong-baseline gate are not promoted as process-conditioning wins. [C61-ROUTE-001]",
        "The main promoted branch is the `spot_size` route, while laser power, scan speed, and full process remain route-boundary cases. [C61-ROUTE-001]",
        "",
        "## Metric Protocol",
        "",
        "All main comparisons should report test root mean squared error (RMSE), hot q90 RMSE, and gradient q90 RMSE. [C61-RESULT-001; C61-RESULT-002]",
        "The hot q90 and gradient q90 metrics are retained in the main text because Phase 35-59 diagnostics repeatedly showed that global RMSE alone can hide region-specific tradeoffs. [C61-APPX-001]",
        "",
        "## Stress And Residual-Gate Protocol",
        "",
        "Stress testing uses stronger tabular baselines, an auxiliary process-balanced panel, and alternate sampling density as separate checks instead of merging them into the main fixed-sampling claim. [C61-STRESS-001; C61-STRESS-002; C61-BOUNDARY-001]",
        "The residual upper-bound probe is reported as a no-test-leakage gate: it uses train/validation evidence to decide whether the density failure supports a new model branch and reports that the winning validation-visible correction is fallback to the mean reference. [C61-BOUNDARY-002]",
        "",
        "## Reproducibility Artifacts",
        "",
        "The manuscript methods should point to the generated package tables, scripts, and manifests rather than only to prose descriptions. [C61-METHOD-001]",
        "The Phase 60 and Phase 61 packages provide the source tables, draft text, captions, and claim anchors needed to audit each manuscript claim. [C61-METHOD-001]",
        "",
        "## User-Review Items",
        "",
        "- Confirm whether training hyperparameters should be summarized in this manuscript section or delegated to supplementary material.",
        "- Add external dataset and PINN literature citations before finalizing any broad field-motivation claims.",
        "",
    ]
    return "\n".join(lines)


def build_caption_package() -> str:
    lines = [
        "# Phase 61 Table And Figure Caption Drafts",
        "",
        "## Main Table",
        "",
        "**Table 1. Fixed-sampling `spot_size` transfer under the route-guarded Macro PINN.** The table reports three-seed means and standard deviations for broad12 and broad21, comparing `broad_process_v1` with no-process Macro PINN and the best strong baseline on test RMSE, hot q90 RMSE, and gradient q90 RMSE. This table supports the main fixed-sampling `spot_size` claim only. [C61-MAIN-001; C61-RESULT-001; C61-RESULT-002]",
        "",
        "## Route-Guard Table",
        "",
        "**Table 2. Route-guard boundary classification across process axes.** The table separates the seed-robust `spot_size` process-conditioned branch from no-process fallback and route-guard-only axes, preventing laser power, scan speed, full process, or line fallback results from being written as universal process-conditioning wins. [C61-ROUTE-001]",
        "",
        "## Stress And Boundary Table",
        "",
        "**Table 3. Stress tests and residual-boundary checks for the fixed `spot_size` floor.** Stronger-baseline stress and the auxiliary broad15 panel support the fixed-sampling floor, while alternate-density broad21 and the Phase 59 residual upper-bound probe define the density-sensitive boundary. [C61-STRESS-001; C61-STRESS-002; C61-BOUNDARY-001; C61-BOUNDARY-002]",
        "",
        "## Appendix Table",
        "",
        "**Table S1. Negative diagnostic and route-boundary ledger.** This supplementary table records model branches and data-alignment paths that did not pass the manuscript gate, including Phases 33-53 and the Phase 58/59 density-sensitive diagnostics. [C61-APPX-001]",
        "",
        "## Next-Branch Gate Table",
        "",
        "**Table S2. Gates for future model branches.** Candidate A is paused pending a new validation-visible `spot_size` signal, Candidate B is blocked by the Phase 59 density gate, and Candidate C remains blocked by scan-path registration data. [C61-GATE-001]",
        "",
        "## Figure Caption From Existing Phase 56 Asset",
        "",
        "**Figure 1. Seed-stable `spot_size` transfer across broad12 and broad21.** Bars summarize the three required error metrics for `broad_process_v1`, no-process Macro PINN, and the strongest classical baseline under the fixed `spot_size` split. The figure should be paired with Table 1 and should not be reused to imply density-invariant or universal process-axis robustness. [C61-MAIN-001; C61-BOUNDARY-001]",
        "",
    ]
    return "\n".join(lines)


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(key, "")).replace("\n", " ") for key, _ in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def build_package_markdown(
    manifest: dict[str, Any],
    claim_rows: list[dict[str, str]],
    gaps: list[dict[str, str]],
    output_paths: dict[str, str],
) -> str:
    gate = manifest.get("model_expansion_gate") or {}
    lines = [
        "# AM-Bench Phase 61 Manuscript Draft Package",
        "",
        "## Purpose",
        "",
        "Phase 61 turns the Phase 60 evidence package into manuscript-facing draft text, captions, and a claim-to-evidence crosswalk. It does not add training evidence.",
        "",
        "## Writing Stage Gate",
        "",
        "Mode: `section_draft`.",
        "Active gate: `draft_ready` for Results, Methods, captions, and internal claim mapping; `needs_verification` for Introduction and Related Work literature claims.",
        "Evidence status: Phase 60 result tables are writing-ready; literature gaps are listed below.",
        "",
        "## Model-Expansion Boundary",
        "",
        f"Phase 60 gate decision: `{gate.get('decision')}`.",
        f"Selected upper-bound variant: `{gate.get('selected_variant')}`.",
        "",
        "## Package Outputs",
        "",
        _markdown_table(
            [{"artifact": key, "path": value} for key, value in output_paths.items()],
            [("artifact", "Artifact"), ("path", "Path")],
        ),
        "",
        "## Claim Anchor Overview",
        "",
        _markdown_table(
            claim_rows,
            [
                ("claim_anchor_id", "Anchor"),
                ("manuscript_location", "Location"),
                ("allowed_claim_strength", "Strength"),
                ("support_locator", "Support"),
                ("open_risk", "Open risk"),
            ],
        ),
        "",
        "## Literature And Target-Style Gaps",
        "",
        _markdown_table(
            gaps,
            [
                ("gap_id", "Gap"),
                ("location", "Location"),
                ("claim_needing_support", "Claim needing support"),
                ("evidence_type_needed", "Evidence needed"),
                ("blocks_current_phase61_draft", "Blocks current draft"),
            ],
        ),
        "",
        "## Next Step",
        "",
        "Use the draft sections as the Phase 61 manuscript base. Before Introduction, Related Work, or final polishing, resolve the listed literature and target-style gaps.",
        "",
    ]
    return "\n".join(lines)


def build_package(
    root: Path,
    output_dir: Path,
    paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    resolved = _default_paths(root)
    if paths:
        resolved.update(paths)
    main_rows = _read_csv(resolved["main"])
    route_rows = _read_csv(resolved["route"])
    stress_rows = _read_csv(resolved["stress"])
    appendix_rows = _read_csv(resolved["appendix"])
    next_gate_rows = _read_csv(resolved["next_gate"])
    phase60_manifest = _read_json(resolved["phase60_manifest"])
    claim_rows = build_claim_crosswalk(
        resolved,
        phase60_manifest,
        main_rows,
        route_rows,
        stress_rows,
        appendix_rows,
        next_gate_rows,
        root,
    )
    gaps = build_literature_gaps()
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "phase61_results_draft.md"
    methods_path = output_dir / "phase61_methods_draft.md"
    captions_path = output_dir / "phase61_table_figure_captions.md"
    crosswalk_path = output_dir / "phase61_claim_evidence_crosswalk.csv"
    gaps_path = output_dir / "phase61_literature_gap_register.csv"
    package_path = output_dir / "phase61_manuscript_draft_package.md"
    manifest_path = output_dir / "phase61_manuscript_draft_package_manifest.json"

    results_path.write_text(build_results_draft(claim_rows), encoding="utf-8")
    methods_path.write_text(build_methods_draft(claim_rows), encoding="utf-8")
    captions_path.write_text(build_caption_package(), encoding="utf-8")
    _write_csv(crosswalk_path, claim_rows, CROSSWALK_FIELDS)
    _write_csv(gaps_path, gaps, GAP_FIELDS)

    output_paths = {
        "results_draft": _display_path(results_path, root),
        "methods_draft": _display_path(methods_path, root),
        "captions": _display_path(captions_path, root),
        "claim_evidence_crosswalk": _display_path(crosswalk_path, root),
        "literature_gap_register": _display_path(gaps_path, root),
        "package_markdown": _display_path(package_path, root),
        "manifest": _display_path(manifest_path, root),
    }
    package_path.write_text(
        build_package_markdown(phase60_manifest, claim_rows, gaps, output_paths),
        encoding="utf-8",
    )
    counts = {
        "claim_anchor_rows": len(claim_rows),
        "literature_gap_rows": len(gaps),
        "result_draft_files": 2,
        "caption_files": 1,
    }
    manifest = {
        "phase": 61,
        "objective": "manuscript_results_methods_caption_draft_package",
        "writing_stage_gate": {
            "mode": "section_draft",
            "active_gate": "draft_ready_for_internal_results_methods; needs_verification_for_literature_context",
            "evidence_status": "phase60_tables_writing_ready",
            "blocked_work": [
                "final Introduction or Related Work without verified literature",
                "target-venue style claims without target examples or author guide",
            ],
        },
        "inputs": {key: _display_path(path, root) for key, path in sorted(resolved.items())},
        "outputs": output_paths,
        "counts": counts,
        "claim_boundary": phase60_manifest.get("claim_boundary"),
        "model_expansion_gate": phase60_manifest.get("model_expansion_gate"),
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase61_manuscript_draft_package"),
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
