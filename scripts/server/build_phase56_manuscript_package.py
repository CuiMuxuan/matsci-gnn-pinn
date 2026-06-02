#!/usr/bin/env python3
"""Build manuscript-facing tables and figures from Phase 54/55 reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = (
    ("rmse", "Test RMSE"),
    ("hot_q90_rmse", "Hot q90 RMSE"),
    ("gradient_q90_rmse", "Gradient q90 RMSE"),
)

NEGATIVE_DIAGNOSTICS = [
    {
        "phase": "33",
        "branch": "Fixed Fourier spacetime representation",
        "target": "broad12 broad_process_v1, all grouped splits",
        "result": "Negative",
        "paper_use": "Appendix diagnostic: fixed coordinate basis degraded all broad12 splits.",
        "evidence": "docs/results/ambench_multiline_process_fourier_spacetime_v1.md",
    },
    {
        "phase": "34",
        "branch": "Sparse closure and learned residual correction",
        "target": "broad12 spot_size",
        "result": "Negative",
        "paper_use": "Appendix diagnostic: closure/residual heads over-traded hot/gradient metrics.",
        "evidence": "docs/results/ambench_multiline_process_residual_correction_v1.md",
    },
    {
        "phase": "35",
        "branch": "Train-split region-weighted data loss",
        "target": "broad12 spot_size",
        "result": "Negative after seed check",
        "paper_use": "Appendix diagnostic: single-seed region gains did not survive paired seeds.",
        "evidence": "docs/results/ambench_multiline_process_region_weighted_loss_v1.md",
    },
    {
        "phase": "36",
        "branch": "Structured process-neighborhood RBF features",
        "target": "broad12/broad21 laser_power/spot_size diagnostics",
        "result": "Unstable",
        "paper_use": "Appendix diagnostic: split-local process-neighborhood signal did not transfer.",
        "evidence": "docs/results/ambench_multiline_process_process_graph_rbf_v1.md",
    },
    {
        "phase": "37",
        "branch": "Strong-baseline residualized Macro PINN",
        "target": "broad12 spot_size and laser_power",
        "result": "Negative",
        "paper_use": "Appendix diagnostic: ExtraTrees residualization left no useful neural residual.",
        "evidence": "docs/results/ambench_multiline_process_target_residual_v1.md",
    },
    {
        "phase": "38",
        "branch": "Residual Macro PINN backbone",
        "target": "broad12 spot_size and laser_power",
        "result": "Negative",
        "paper_use": "Appendix diagnostic: backbone changes improved one metric only by hurting others.",
        "evidence": "docs/results/ambench_multiline_process_residual_backbone_v1.md",
    },
    {
        "phase": "39-40",
        "branch": "Process-conditioned output affine calibration",
        "target": "laser_power broad12/broad21",
        "result": "Non-transferable",
        "paper_use": "Appendix diagnostic: broad12 local positive failed broad21 transfer.",
        "evidence": "docs/results/ambench_multiline_process_output_affine_v1.md",
    },
    {
        "phase": "41-43",
        "branch": "Derived process features and process encoder",
        "target": "laser_power broad12/broad21",
        "result": "Non-transferable",
        "paper_use": "Appendix diagnostic: representation signals were broad21-positive but broad12-negative.",
        "evidence": "docs/results/ambench_multiline_process_process_encoder_v1.md",
    },
    {
        "phase": "44",
        "branch": "Process-condition group-balanced objective",
        "target": "laser_power broad12/broad21",
        "result": "Negative",
        "paper_use": "Appendix diagnostic: region gains on broad21 sacrificed global RMSE and broad12 failed.",
        "evidence": "docs/results/ambench_multiline_process_group_balance_v1.md",
    },
    {
        "phase": "45",
        "branch": "Baseline-guarded expert stack gate",
        "target": "laser_power broad12/broad21",
        "result": "Negative",
        "paper_use": "Appendix diagnostic: existing experts did not contain a validation-selectable transferable stack.",
        "evidence": "docs/results/ambench_multiline_process_baseline_guarded_expert_plan_v1.md",
    },
    {
        "phase": "46-51",
        "branch": "Bayesian inverse closure and moving-source inversion",
        "target": "synthetic and AM-Bench Line_0_1 sparse/dense gates",
        "result": "Synthetic-positive, AM-Bench-negative",
        "paper_use": "Limitation/negative control: interpretable source inversion needs better AM-Bench path registration.",
        "evidence": "docs/results/ambench_dense_source_parameter_transfer_v1.md",
    },
    {
        "phase": "52-53",
        "branch": "Registered source-path features and data pivot",
        "target": "Line_0_1 and pad thermography diagnostics",
        "result": "Data-incompatible",
        "paper_use": "Limitation: current bundle lacks single-track scan-path/camera registration metadata.",
        "evidence": "docs/results/ambench_source_path_data_pivot_gate_v1.md",
    },
]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _phase55_main_rows(phase55: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in phase55["datasets"]:
        aggregates = dataset["aggregates"]
        gate_metrics = dataset["aggregate_gate"]["metrics"]
        route = dataset.get("route") or {}
        route_label = "{}/{}".format(
            route.get("selected_conditioning_mode") or "",
            route.get("selected_feature_normalization") or "",
        ).strip("/")
        for metric_key, metric_label in METRICS:
            candidate = aggregates["broad_process_v1"][metric_key]
            no_process = aggregates["no_process"][metric_key]
            gate = gate_metrics[metric_key]
            rows.append(
                {
                    "dataset": dataset["label"],
                    "split": dataset["split"],
                    "route": route_label,
                    "metric": metric_label,
                    "broad_process_v1_mean": candidate["mean"],
                    "broad_process_v1_std": candidate["pstdev"],
                    "no_process_mean": no_process["mean"],
                    "no_process_std": no_process["pstdev"],
                    "best_strong_baseline": gate["best_strong_baseline"],
                    "best_strong_baseline_method": gate.get("best_strong_baseline_method"),
                    "delta_vs_best_strong": gate["delta_vs_best_strong"],
                    "delta_vs_no_process": gate["delta_vs_no_process"],
                    "n_seeds": aggregates["broad_process_v1"]["n"],
                    "gate": dataset["status"],
                }
            )
    return rows


def _route_guard_rows(phase54: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in phase54["datasets"]:
        for split, payload in dataset["splits"].items():
            if split == "spot_size":
                continue
            route = payload.get("route") or {}
            route_label = "{}/{}".format(
                route.get("selected_conditioning_mode") or "",
                route.get("selected_feature_normalization") or "",
            ).strip("/")
            metric_bits = []
            for metric_key, metric_label in METRICS:
                metric = payload["metrics"].get(metric_key) or {}
                metric_bits.append(
                    "{label}: {candidate} vs {method} {baseline} (d={delta})".format(
                        label=metric_label,
                        candidate=_fmt(metric.get("candidate")),
                        method=(metric.get("best_strong_baseline") or {}).get("method", ""),
                        baseline=_fmt((metric.get("best_strong_baseline") or {}).get("value")),
                        delta=_fmt(metric.get("candidate_delta_vs_best_strong")),
                    )
                )
            rows.append(
                {
                    "dataset": dataset["label"],
                    "split": split,
                    "classification": payload["classification"],
                    "route": route_label,
                    "claim_use": _claim_use(payload),
                    "metrics_summary": "; ".join(metric_bits),
                    "notes": " ".join(payload.get("notes") or []),
                }
            )
    return rows


def _claim_use(payload: dict[str, Any]) -> str:
    classification = payload.get("classification")
    route = payload.get("route") or {}
    selected = route.get("selected_conditioning_mode")
    if classification == "paper_claim_positive" and selected == "none":
        return "route guard / no-process fallback evidence"
    if classification == "route_guard_positive":
        return "route-guard-only boundary evidence"
    return str(classification)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(key)) for key, _ in columns) + " |")
    return "\n".join(lines)


def _write_svg_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1120, 620
    margin_left, margin_right = 120, 40
    margin_top, margin_bottom = 70, 95
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_value = max(
        float(row[key])
        for row in rows
        for key in ("broad_process_v1_mean", "no_process_mean", "best_strong_baseline")
        if row.get(key) is not None
    )
    x_max = ((int(max_value) // 50) + 2) * 50

    def sx(value: float) -> float:
        return margin_left + plot_w * value / x_max

    row_h = plot_h / len(rows)
    bar_h = row_h * 0.18
    colors = {
        "broad": "#2F6F73",
        "no_process": "#A64B3C",
        "baseline": "#6B7280",
    }
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111827}.axis{stroke:#9CA3AF;stroke-width:1}.tick{font-size:12px;fill:#4B5563}.label{font-size:13px}.title{font-size:18px;font-weight:700}.panel{font-size:15px;font-weight:700}.legend{font-size:13px}</style>',
        f'<text x="{margin_left}" y="32" class="title">Phase 55 spot-size route validation across broad12 and broad21</text>',
        f'<text x="{margin_left}" y="54" class="label">Lower RMSE is better; bars show three-seed means. Route: broad_process_v1 FiLM/global-standard.</text>',
    ]
    for tick in range(0, x_max + 1, 50):
        x = sx(tick)
        lines.append(f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{height-margin_bottom}" class="axis" opacity="0.35"/>')
        lines.append(f'<text x="{x:.1f}" y="{height-margin_bottom+22}" text-anchor="middle" class="tick">{tick}</text>')
    lines.append(f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-margin_right}" y2="{height-margin_bottom}" class="axis"/>')
    lines.append(f'<text x="{margin_left + plot_w/2:.1f}" y="{height-30}" text-anchor="middle" class="label">RMSE (deg C)</text>')

    for idx, row in enumerate(rows):
        y = margin_top + idx * row_h + row_h * 0.5
        label = f"{row['dataset']} {row['metric']}"
        lines.append(f'<text x="{margin_left-10}" y="{y+4:.1f}" text-anchor="end" class="label">{label}</text>')
        for offset, key, color_key, name in (
            (-bar_h * 1.2, "best_strong_baseline", "baseline", "best strong"),
            (0, "no_process_mean", "no_process", "no-process"),
            (bar_h * 1.2, "broad_process_v1_mean", "broad", "broad_process_v1"),
        ):
            value = float(row[key])
            x0 = margin_left
            x1 = sx(value)
            yy = y + offset - bar_h / 2
            lines.append(f'<rect x="{x0:.1f}" y="{yy:.1f}" width="{x1-x0:.1f}" height="{bar_h:.1f}" fill="{colors[color_key]}"/>')
            lines.append(f'<text x="{x1+5:.1f}" y="{yy+bar_h*0.78:.1f}" class="tick">{value:.1f}</text>')
    legend_x = margin_left
    legend_y = height - 68
    for i, (name, color_key) in enumerate((
        ("best strong baseline", "baseline"),
        ("no-process Macro PINN", "no_process"),
        ("broad_process_v1", "broad"),
    )):
        x = legend_x + i * 245
        lines.append(f'<rect x="{x}" y="{legend_y}" width="16" height="10" fill="{colors[color_key]}"/>')
        lines.append(f'<text x="{x+22}" y="{legend_y+10}" class="legend">{name}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_png_from_svg(svg_path: Path, png_path: Path) -> bool:
    try:
        import cairosvg  # type: ignore
    except Exception:
        return False
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), output_width=2240, output_height=1240)
    return True


def _write_png_figure(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return False

    labels = [f"{row['dataset']} {row['metric']}" for row in rows]
    y = np.arange(len(rows), dtype=float)
    height = 0.22
    series = [
        ("best strong baseline", "best_strong_baseline", "#6B7280", -height),
        ("no-process Macro PINN", "no_process_mean", "#A64B3C", 0.0),
        ("broad_process_v1", "broad_process_v1_mean", "#2F6F73", height),
    ]
    fig, ax = plt.subplots(figsize=(11.2, 6.2), dpi=200)
    for name, key, color, offset in series:
        values = [float(row[key]) for row in rows]
        ax.barh(y + offset, values, height=height, label=name, color=color)
        for yy, value in zip(y + offset, values, strict=True):
            ax.text(value + 3, yy, f"{value:.1f}", va="center", ha="left", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE (deg C)")
    ax.set_title("Phase 55 spot-size route validation across broad12 and broad21", fontsize=10, pad=10)
    ax.grid(axis="x", color="#D1D5DB", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=8)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def _write_markdown_package(
    path: Path,
    main_rows: list[dict[str, Any]],
    route_rows: list[dict[str, Any]],
    appendix_rows: list[dict[str, Any]],
    figure_svg: Path,
    figure_png: Path,
    png_written: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# Phase 56 Manuscript-Facing Table/Figure Package

## Status

Package complete.

This package converts Phase 55/54 reports into manuscript-facing artifacts. It does not introduce new training results. The main paper claim is a seed-robust broad-data process route guard with a stable `spot_size -> FiLM/global-standard` branch.

## Main Table: Seed-Robust Spot-Size Positive

{_markdown_table(main_rows, [
    ('dataset', 'Dataset'),
    ('split', 'Split'),
    ('route', 'Route'),
    ('metric', 'Metric'),
    ('broad_process_v1_mean', 'broad_process_v1 mean'),
    ('broad_process_v1_std', 'std'),
    ('best_strong_baseline', 'best strong baseline'),
    ('best_strong_baseline_method', 'baseline method'),
    ('no_process_mean', 'no-process mean'),
    ('delta_vs_best_strong', 'delta vs strong'),
    ('delta_vs_no_process', 'delta vs no-process'),
])}

## Route-Guard Boundary Table

{_markdown_table(route_rows, [
    ('dataset', 'Dataset'),
    ('split', 'Split'),
    ('classification', 'Class'),
    ('route', 'Route'),
    ('claim_use', 'Paper use'),
    ('metrics_summary', 'Metrics summary'),
])}

## Negative Diagnostic Appendix

{_markdown_table(appendix_rows, [
    ('phase', 'Phase'),
    ('branch', 'Branch'),
    ('target', 'Target'),
    ('result', 'Result'),
    ('paper_use', 'Paper use'),
    ('evidence', 'Evidence'),
])}

## Figure

- Editable SVG: `{figure_svg.as_posix()}`
- PNG preview: `{figure_png.as_posix() if png_written else 'not generated; cairosvg unavailable'}`

Suggested caption:

> Seed-robust `spot_size` validation of the broad-data process route guard. Bars show three-seed mean RMSE on broad12 and broad21 for global test points, hot-zone q90 points, and gradient q90 points. The `broad_process_v1` FiLM/global-standard route is compared with the no-process Macro PINN and the best strong classical baseline selected independently for each metric. Lower is better.

## Manuscript Placement

- Main results table: use the main table above.
- Method/model contribution figure: use the figure with a short route-guard schematic in the text, or pair it with the Phase 54 claim-boundary table.
- Supplementary appendix: include the route-guard boundary and negative diagnostic tables.

## Source Trace

- `outputs/reports/phase55_spot_size_route_seed_check_summary.json`
- `outputs/reports/phase54_process_route_claim_boundary_summary.json`
- `docs/results/ambench_multiline_process_spot_size_seed_validation_v1.md`
- `docs/results/ambench_multiline_process_route_claim_boundary_v1.md`
"""
    path.write_text(text, encoding="utf-8")


def build_package(root: Path, output_dir: Path) -> dict[str, Any]:
    phase55 = _read_json(root / "outputs/reports/phase55_spot_size_route_seed_check_summary.json")
    phase54 = _read_json(root / "outputs/reports/phase54_process_route_claim_boundary_summary.json")
    main_rows = _phase55_main_rows(phase55)
    route_rows = _route_guard_rows(phase54)
    appendix_rows = list(NEGATIVE_DIAGNOSTICS)
    output_dir.mkdir(parents=True, exist_ok=True)

    main_csv = output_dir / "phase56_main_spot_size_seed_positive_table.csv"
    route_csv = output_dir / "phase56_route_guard_boundary_table.csv"
    appendix_csv = output_dir / "phase56_negative_diagnostic_appendix_table.csv"
    figure_svg = output_dir / "phase56_spot_size_seed_validation_figure.svg"
    figure_png = output_dir / "phase56_spot_size_seed_validation_figure.png"
    package_md = output_dir / "phase56_manuscript_table_figure_package.md"

    _write_csv(main_csv, main_rows, list(main_rows[0].keys()))
    _write_csv(route_csv, route_rows, list(route_rows[0].keys()))
    _write_csv(appendix_csv, appendix_rows, list(appendix_rows[0].keys()))
    _write_svg_figure(figure_svg, main_rows)
    png_written = _write_png_from_svg(figure_svg, figure_png)
    if not png_written:
        png_written = _write_png_figure(figure_png, main_rows)
    _write_markdown_package(
        package_md,
        main_rows,
        route_rows,
        appendix_rows,
        figure_svg,
        figure_png,
        png_written,
    )
    return {
        "package_markdown": str(package_md),
        "main_table_csv": str(main_csv),
        "route_guard_table_csv": str(route_csv),
        "negative_appendix_csv": str(appendix_csv),
        "figure_svg": str(figure_svg),
        "figure_png": str(figure_png) if png_written else None,
        "n_main_rows": len(main_rows),
        "n_route_guard_rows": len(route_rows),
        "n_appendix_rows": len(appendix_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase56_manuscript_package"),
    )
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()

    manifest = build_package(args.root, args.output_dir)
    if args.manifest_output:
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
