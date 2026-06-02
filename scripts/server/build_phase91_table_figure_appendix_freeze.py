#!/usr/bin/env python3
"""Build the Phase 91 table, figure, and appendix freeze package.

Phase 91 freezes the manuscript-facing quantitative package after Phase 90
claim integration. It does not add experiment results or venue-specific
formatting. Its job is to make every table, figure, caption, and appendix row
traceable to the Phase 55/60/74/88/89/90 evidence boundary.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MAIN_FIELDS = (
    "freeze_id",
    "dataset",
    "split",
    "route",
    "metric",
    "broad_process_v1_mean",
    "broad_process_v1_std",
    "no_process_mean",
    "best_strong_baseline",
    "delta_vs_best_strong",
    "n_seeds",
    "manuscript_role",
    "claim_anchor",
    "freeze_status",
)

ROUTE_FIELDS = (
    "freeze_id",
    "dataset",
    "split",
    "classification",
    "route",
    "claim_use",
    "metrics_summary",
    "manuscript_role",
    "wording_guard",
    "freeze_status",
)

STRESS_FIELDS = (
    "freeze_id",
    "scenario",
    "dataset",
    "split",
    "metric",
    "status",
    "candidate",
    "comparator",
    "delta_vs_comparator",
    "manuscript_use",
    "evidence",
    "freeze_status",
)

APPENDIX_FIELDS = (
    "freeze_id",
    "appendix_id",
    "phase",
    "branch",
    "status",
    "artifact",
    "manuscript_use",
    "reason",
    "freeze_status",
)

FIGURE_FIELDS = (
    "item_id",
    "item_type",
    "manuscript_label",
    "source_artifact",
    "caption_source",
    "claim_anchor",
    "manuscript_role",
    "freeze_status",
    "venue_dependency",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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
    if isinstance(value, float):
        return f"{value:.9f}"
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
    phase56 = root / "docs/results/phase56_manuscript_package"
    phase60 = root / "docs/results/phase60_manuscript_evidence_package"
    phase61 = root / "docs/results/phase61_manuscript_draft_package"
    phase88 = root / "docs/results/phase88_fallback_manuscript_finalization"
    phase90 = root / "docs/results/phase90_manuscript_v1_claim_integration"
    return {
        "phase56_figure_svg": phase56 / "phase56_spot_size_seed_validation_figure.svg",
        "phase56_figure_png": phase56 / "phase56_spot_size_seed_validation_figure.png",
        "phase60_main": phase60 / "phase60_main_spot_size_seed_positive_table.csv",
        "phase60_route": phase60 / "phase60_route_guard_boundary_table.csv",
        "phase60_stress": phase60 / "phase60_stress_boundary_table.csv",
        "phase60_manifest": phase60 / "phase60_manuscript_evidence_package_manifest.json",
        "phase61_captions": phase61 / "phase61_table_figure_captions.md",
        "phase88_appendix": phase88 / "phase88_appendix_diagnostic_table.csv",
        "phase88_gate": phase88 / "phase88_fallback_finalization_gate.json",
        "phase90_manifest": phase90 / "phase90_manuscript_v1_claim_integration_manifest.json",
        "phase90_gate": phase90 / "phase90_manuscript_v1_claim_integration_gate.json",
        "phase90_audit": phase90 / "phase90_claim_evidence_audit.csv",
        "phase90_manuscript": phase90 / "phase90_manuscript_v1_claim_integrated.md",
    }


def freeze_main_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    frozen: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        frozen.append(
            {
                "freeze_id": f"P91-MAIN-{index:03d}",
                "dataset": row.get("dataset"),
                "split": row.get("split"),
                "route": row.get("route"),
                "metric": row.get("metric"),
                "broad_process_v1_mean": row.get("broad_process_v1_mean"),
                "broad_process_v1_std": row.get("broad_process_v1_std"),
                "no_process_mean": row.get("no_process_mean"),
                "best_strong_baseline": row.get("best_strong_baseline"),
                "delta_vs_best_strong": row.get("delta_vs_best_strong"),
                "n_seeds": row.get("n_seeds"),
                "manuscript_role": "main_table_only_positive_performance_claim",
                "claim_anchor": "C61-MAIN-001;C61-RESULT-001;C61-RESULT-002",
                "freeze_status": "frozen_main_claim_row",
            }
        )
    return frozen


def freeze_route_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    frozen: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        classification = row.get("classification", "")
        if row.get("claim_use") == "route guard / no-process fallback evidence":
            role = "route_guard_no_process_fallback"
            guard = "Do not write this row as process-conditioning improvement."
        elif classification == "route_guard_positive":
            role = "route_guard_boundary"
            guard = "Do not write this row as a strong-baseline-positive main claim."
        else:
            role = "route_guard_context"
            guard = "Use only under the route-guard boundary table."
        frozen.append(
            {
                "freeze_id": f"P91-ROUTE-{index:03d}",
                "dataset": row.get("dataset"),
                "split": row.get("split"),
                "classification": classification,
                "route": row.get("route"),
                "claim_use": row.get("claim_use"),
                "metrics_summary": row.get("metrics_summary"),
                "manuscript_role": role,
                "wording_guard": guard,
                "freeze_status": "frozen_route_guard_row",
            }
        )
    return frozen


def freeze_stress_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    frozen: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        status = row.get("status", "")
        if status == "pass":
            freeze_status = "frozen_support_row"
        elif status == "boundary":
            freeze_status = "frozen_boundary_row"
        else:
            freeze_status = "frozen_blocks_model_expansion_row"
        frozen.append(
            {
                "freeze_id": f"P91-STRESS-{index:03d}",
                "scenario": row.get("scenario"),
                "dataset": row.get("dataset"),
                "split": row.get("split"),
                "metric": row.get("metric"),
                "status": status,
                "candidate": row.get("candidate"),
                "comparator": row.get("comparator"),
                "delta_vs_comparator": row.get("delta_vs_comparator"),
                "manuscript_use": row.get("manuscript_use"),
                "evidence": row.get("evidence"),
                "freeze_status": freeze_status,
            }
        )
    return frozen


def freeze_appendix_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    frozen: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        frozen.append(
            {
                "freeze_id": f"P91-APPX-{index:03d}",
                "appendix_id": row.get("appendix_id"),
                "phase": row.get("phase"),
                "branch": row.get("branch"),
                "status": row.get("status"),
                "artifact": row.get("artifact"),
                "manuscript_use": row.get("manuscript_use"),
                "reason": row.get("reason"),
                "freeze_status": "frozen_appendix_diagnostic_row",
            }
        )
    return frozen


def build_figure_rows(paths: dict[str, Path], captions_text: str, root: Path) -> list[dict[str, Any]]:
    main_caption = "Table 1. Fixed-sampling `spot_size` transfer under the route-guarded Macro PINN."
    route_caption = "Table 2. Route-guard boundary classification across process axes."
    stress_caption = "Table 3. Stress tests and residual-boundary checks for the fixed `spot_size` floor."
    appendix_caption = "Table S1. Negative diagnostic and route-boundary ledger."
    next_branch_caption = "Table S2. Gates for future model branches."
    figure_caption = "Figure 1. Seed-stable `spot_size` transfer across broad12 and broad21."
    expected = [
        main_caption,
        route_caption,
        stress_caption,
        appendix_caption,
        next_branch_caption,
        figure_caption,
    ]
    missing = [caption for caption in expected if caption not in captions_text]
    if missing:
        raise ValueError(f"Missing expected caption text: {missing}")
    return [
        {
            "item_id": "P91-TABLE-001",
            "item_type": "main_table",
            "manuscript_label": "Table 1",
            "source_artifact": _display_path(paths["phase60_main"], root),
            "caption_source": main_caption,
            "claim_anchor": "C61-MAIN-001;C61-RESULT-001;C61-RESULT-002",
            "manuscript_role": "main positive performance table",
            "freeze_status": "frozen",
            "venue_dependency": "label/style may change after target venue is provided",
        },
        {
            "item_id": "P91-TABLE-002",
            "item_type": "route_guard_table",
            "manuscript_label": "Table 2",
            "source_artifact": _display_path(paths["phase60_route"], root),
            "caption_source": route_caption,
            "claim_anchor": "C61-ROUTE-001",
            "manuscript_role": "route-boundary table",
            "freeze_status": "frozen",
            "venue_dependency": "label/style may change after target venue is provided",
        },
        {
            "item_id": "P91-TABLE-003",
            "item_type": "stress_boundary_table",
            "manuscript_label": "Table 3",
            "source_artifact": _display_path(paths["phase60_stress"], root),
            "caption_source": stress_caption,
            "claim_anchor": "C61-STRESS-001;C61-STRESS-002;C61-BOUNDARY-001;C61-BOUNDARY-002",
            "manuscript_role": "stress support and density-boundary table",
            "freeze_status": "frozen",
            "venue_dependency": "label/style may change after target venue is provided",
        },
        {
            "item_id": "P91-TABLE-S001",
            "item_type": "appendix_table",
            "manuscript_label": "Table S1",
            "source_artifact": _display_path(paths["phase88_appendix"], root),
            "caption_source": appendix_caption,
            "claim_anchor": "C61-APPX-001",
            "manuscript_role": "appendix negative diagnostic ledger",
            "freeze_status": "frozen",
            "venue_dependency": "supplement numbering may change after target venue is provided",
        },
        {
            "item_id": "P91-TABLE-S002",
            "item_type": "future_gate_table",
            "manuscript_label": "Table S2",
            "source_artifact": _display_path(paths["phase60_manifest"], root),
            "caption_source": next_branch_caption,
            "claim_anchor": "C61-GATE-001",
            "manuscript_role": "future model branch gate summary",
            "freeze_status": "frozen",
            "venue_dependency": "supplement numbering may change after target venue is provided",
        },
        {
            "item_id": "P91-FIG-001",
            "item_type": "figure",
            "manuscript_label": "Figure 1",
            "source_artifact": f"{_display_path(paths['phase56_figure_svg'], root)}; {_display_path(paths['phase56_figure_png'], root)}",
            "caption_source": figure_caption,
            "claim_anchor": "C61-MAIN-001;C61-BOUNDARY-001",
            "manuscript_role": "seed-stability visualization for fixed spot_size claim",
            "freeze_status": "frozen_existing_asset",
            "venue_dependency": "final size/DPI/format may change after target venue is provided",
        },
    ]


def build_gate(
    main_rows: list[dict[str, Any]],
    route_rows: list[dict[str, Any]],
    stress_rows: list[dict[str, Any]],
    appendix_rows: list[dict[str, Any]],
    figure_rows: list[dict[str, Any]],
    phase90_gate: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    figure_assets_exist = paths["phase56_figure_svg"].exists() and paths["phase56_figure_png"].exists()
    main_ok = len(main_rows) == 6 and all(
        row["freeze_status"] == "frozen_main_claim_row" for row in main_rows
    )
    route_ok = len(route_rows) >= 8
    stress_ok = any(row["status"] == "boundary" for row in stress_rows) and any(
        row["status"] == "pass" for row in stress_rows
    )
    appendix_ok = len(appendix_rows) >= 18 and any(row["phase"] == "75" for row in appendix_rows)
    figures_ok = figure_assets_exist and len(figure_rows) == 6
    core_ready = all([main_ok, route_ok, stress_ok, appendix_ok, figures_ok])
    submission_ready = False
    if core_ready:
        status = "table_figure_appendix_frozen_venue_unresolved"
        next_action = "enter Phase 92 internal benchmark review or provide target venue before final formatting"
    else:
        status = "table_figure_appendix_freeze_incomplete"
        next_action = "repair missing table, figure, or appendix freeze inputs"
    return {
        "status": status,
        "phase90_status": phase90_gate.get("status"),
        "core_claims_integrated": bool(phase90_gate.get("core_claims_integrated")),
        "table_figure_appendix_frozen": core_ready,
        "venue_alignment_ready": False,
        "submission_ready": submission_ready,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "main_rows": len(main_rows),
        "route_rows": len(route_rows),
        "stress_rows": len(stress_rows),
        "appendix_rows": len(appendix_rows),
        "figure_caption_rows": len(figure_rows),
        "figure_assets_exist": figure_assets_exist,
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
    figure_rows: list[dict[str, Any]],
    main_rows: list[dict[str, Any]],
    route_rows: list[dict[str, Any]],
    appendix_rows: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "# Phase 91 Table, Figure, and Appendix Freeze",
            "",
            "## Gate Decision",
            "",
            f"Status: `{gate['status']}`.",
            f"Frozen: `{str(gate['table_figure_appendix_frozen']).lower()}`.",
            f"Submission ready: `{str(gate['submission_ready']).lower()}`.",
            f"A100 training allowed now: `{str(gate['a100_training_allowed_now']).lower()}`.",
            "",
            "Phase 91 freezes manuscript-facing quantitative artifacts without adding results or venue-specific formatting.",
            "",
            "## Table/Figure Manifest",
            "",
            _markdown_table(
                figure_rows,
                [
                    ("item_id", "Item"),
                    ("manuscript_label", "Label"),
                    ("item_type", "Type"),
                    ("manuscript_role", "Role"),
                    ("freeze_status", "Status"),
                ],
            ),
            "",
            "## Main Table Check",
            "",
            _markdown_table(
                main_rows[:6],
                [
                    ("freeze_id", "Row"),
                    ("dataset", "Dataset"),
                    ("metric", "Metric"),
                    ("delta_vs_best_strong", "Delta vs best strong"),
                    ("freeze_status", "Status"),
                ],
            ),
            "",
            "## Route Guard Check",
            "",
            _markdown_table(
                route_rows,
                [
                    ("freeze_id", "Row"),
                    ("dataset", "Dataset"),
                    ("split", "Split"),
                    ("manuscript_role", "Role"),
                    ("wording_guard", "Guard"),
                ],
            ),
            "",
            "## Appendix Coverage",
            "",
            f"Frozen appendix rows: `{len(appendix_rows)}`.",
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

    main_input = _read_csv(resolved["phase60_main"])
    route_input = _read_csv(resolved["phase60_route"])
    stress_input = _read_csv(resolved["phase60_stress"])
    appendix_input = _read_csv(resolved["phase88_appendix"])
    captions_text = _read_text(resolved["phase61_captions"])
    phase60_manifest = _read_json(resolved["phase60_manifest"])
    phase88_gate = _read_json(resolved["phase88_gate"])
    phase90_manifest = _read_json(resolved["phase90_manifest"])
    phase90_gate = _read_json(resolved["phase90_gate"])
    phase90_audit = _read_csv(resolved["phase90_audit"])
    _read_text(resolved["phase90_manuscript"])

    main_rows = freeze_main_rows(main_input)
    route_rows = freeze_route_rows(route_input)
    stress_rows = freeze_stress_rows(stress_input)
    appendix_rows = freeze_appendix_rows(appendix_input)
    figure_rows = build_figure_rows(resolved, captions_text, root)
    gate = build_gate(main_rows, route_rows, stress_rows, appendix_rows, figure_rows, phase90_gate, resolved)

    output_dir.mkdir(parents=True, exist_ok=True)
    main_path = output_dir / "phase91_main_table_freeze.csv"
    route_path = output_dir / "phase91_route_guard_table_freeze.csv"
    stress_path = output_dir / "phase91_stress_boundary_table_freeze.csv"
    appendix_path = output_dir / "phase91_appendix_diagnostic_freeze.csv"
    figure_path = output_dir / "phase91_table_figure_caption_manifest.csv"
    gate_path = output_dir / "phase91_table_figure_appendix_freeze_gate.json"
    markdown_path = output_dir / "phase91_table_figure_appendix_freeze.md"
    manifest_path = output_dir / "phase91_table_figure_appendix_freeze_manifest.json"

    _write_csv(main_path, main_rows, MAIN_FIELDS)
    _write_csv(route_path, route_rows, ROUTE_FIELDS)
    _write_csv(stress_path, stress_rows, STRESS_FIELDS)
    _write_csv(appendix_path, appendix_rows, APPENDIX_FIELDS)
    _write_csv(figure_path, figure_rows, FIGURE_FIELDS)
    _write_json(gate_path, gate)
    markdown_path.write_text(
        build_markdown(gate, figure_rows, main_rows, route_rows, appendix_rows),
        encoding="utf-8",
    )

    manifest = {
        "phase": 91,
        "objective": "table_figure_appendix_freeze",
        "inputs": {key: _display_path(path, root) for key, path in sorted(resolved.items())},
        "outputs": {
            "main_table_freeze": _display_path(main_path, root),
            "route_guard_table_freeze": _display_path(route_path, root),
            "stress_boundary_table_freeze": _display_path(stress_path, root),
            "appendix_diagnostic_freeze": _display_path(appendix_path, root),
            "table_figure_caption_manifest": _display_path(figure_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "main_rows": len(main_rows),
            "route_rows": len(route_rows),
            "stress_rows": len(stress_rows),
            "appendix_rows": len(appendix_rows),
            "figure_caption_rows": len(figure_rows),
            "phase90_audit_rows": len(phase90_audit),
        },
        "gate": gate,
        "phase60_claim_boundary": phase60_manifest.get("claim_boundary"),
        "phase88_gate": {
            "status": phase88_gate.get("status"),
            "experimental_claim_complete": phase88_gate.get("experimental_claim_complete"),
        },
        "phase90_gate": phase90_manifest.get("gate"),
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase91_table_figure_appendix_freeze"),
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
