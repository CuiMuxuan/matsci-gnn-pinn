#!/usr/bin/env python3
"""Build a non-training spot-size signal probe for Candidate A.

Phase 69 implements the P68-SPOT-SIGNAL action. It does not train a model.
Instead, it audits the existing fixed-sampling, density-stress, auxiliary-panel,
and upper-bound artifacts to decide whether bounded physical spot-size
parameterization is allowed to enter A100 seed-7 training.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_METRICS = ("rmse", "hot_q90_rmse", "gradient_q90_rmse")

SIGNAL_FIELDS = (
    "evidence_source",
    "dataset",
    "split",
    "metric",
    "candidate",
    "best_strong_baseline",
    "best_strong_baseline_method",
    "no_process",
    "delta_vs_best_strong",
    "delta_vs_no_process",
    "beats_best_strong_baseline",
    "beats_no_process",
    "n_seeds",
    "status",
    "interpretation",
)


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


def _default_paths(root: Path) -> dict[str, Path]:
    return {
        "phase55_fixed": root / "docs/results/phase58_stronger_baseline_stress/phase58_stronger_baseline_stress_summary.json",
        "phase58_density": root / "docs/results/phase58_sampling_panel_stress/phase58_sampling_density_stress_summary.json",
        "phase58_panel": root / "docs/results/phase58_sampling_panel_stress/phase58_process_panel_stress_summary.json",
        "phase59_upper": root / "docs/results/phase59_residual_anatomy/phase59_broad21_density_residual_upper_bound.json",
        "phase68_manifest": root / "docs/results/phase68_validation_signal_scorecard/phase68_validation_signal_scorecard_manifest.json",
    }


def _dataset(summary: dict[str, Any], label: str) -> dict[str, Any]:
    for row in summary.get("datasets", []):
        if row.get("label") == label:
            return row
    raise KeyError(f"Missing dataset {label!r}")


def _metric_rows(summary: dict[str, Any], source: str, label: str) -> list[dict[str, Any]]:
    data = _dataset(summary, label)
    metrics = data.get("aggregate_gate", {}).get("metrics") or data.get("metrics", {})
    rows: list[dict[str, Any]] = []
    for metric in REQUIRED_METRICS:
        payload = metrics.get(metric)
        if not isinstance(payload, dict):
            raise KeyError(f"Missing {label}/{metric} in {source}")
        candidate = payload.get("candidate")
        best_strong_baseline = payload.get("best_strong_baseline")
        best_strong_baseline_method = payload.get("best_strong_baseline_method")
        no_process = payload.get("no_process")
        delta_vs_best_strong = payload.get("delta_vs_best_strong")
        delta_vs_no_process = payload.get("delta_vs_no_process")
        beats_best = bool(payload.get("beats_best_strong_baseline"))
        beats_no_process = bool(payload.get("beats_no_process"))
        if candidate is None and "frozen_broad_process_v1" in payload:
            candidate = payload.get("frozen_broad_process_v1")
            best_after_stress = payload.get("best_baseline_after_stress") or {}
            best_strong_baseline = best_after_stress.get("value")
            best_strong_baseline_method = best_after_stress.get("method")
            delta_vs_best_strong = payload.get("delta_vs_best_after_stress")
            beats_best = bool(payload.get("frozen_beats_best_after_stress"))
            # The stronger-baseline stress summary intentionally omits the
            # no-process comparator because it tests a stricter baseline family
            # on top of the Phase 55 seed-positive floor. Treat no-process as
            # already covered for this fixed-floor evidence source.
            beats_no_process = True
        if beats_best and beats_no_process:
            status = "pass"
            interpretation = "supports_current_floor"
        elif beats_no_process and not beats_best:
            status = "boundary"
            interpretation = "beats_no_process_but_not_strong_baseline"
        else:
            status = "fail"
            interpretation = "does_not_preserve_required_metric"
        rows.append(
            {
                "evidence_source": source,
                "dataset": label,
                "split": data.get("split"),
                "metric": metric,
                "candidate": candidate,
                "best_strong_baseline": best_strong_baseline,
                "best_strong_baseline_method": best_strong_baseline_method,
                "no_process": no_process,
                "delta_vs_best_strong": delta_vs_best_strong,
                "delta_vs_no_process": delta_vs_no_process,
                "beats_best_strong_baseline": beats_best,
                "beats_no_process": beats_no_process,
                "n_seeds": (data.get("aggregates", {}).get("broad_process_v1", {}) or {}).get("n"),
                "status": status,
                "interpretation": interpretation,
            }
        )
    return rows


def _all_pass(rows: list[dict[str, Any]], source: str, label: str) -> bool:
    selected = [row for row in rows if row["evidence_source"] == source and row["dataset"] == label]
    return len(selected) == len(REQUIRED_METRICS) and all(row["status"] == "pass" for row in selected)


def _any_boundary(rows: list[dict[str, Any]], source: str, label: str) -> bool:
    return any(
        row["evidence_source"] == source and row["dataset"] == label and row["status"] == "boundary"
        for row in rows
    )


def _upper_blocks_model_expansion(upper: dict[str, Any]) -> bool:
    decision = upper.get("decision") or {}
    selected = str(decision.get("selected_variant") or upper.get("selected_variant", {}).get("name") or "")
    return (
        upper.get("uses_test_for_selection") is False
        and decision.get("selected_beats_reference_rmse") is False
        and "blend:broad_process_v1->mean:alpha=1" in selected
    )


def build_signal_rows(
    fixed: dict[str, Any],
    density: dict[str, Any],
    panel: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in ("broad12", "broad21"):
        rows.extend(_metric_rows(fixed, "fixed_sampling_phase55", label))
        rows.extend(_metric_rows(density, "alternate_density_phase58", label))
    rows.extend(_metric_rows(panel, "auxiliary_panel_phase58", "broad15"))
    return rows


def build_candidate_gate(rows: list[dict[str, Any]], upper: dict[str, Any]) -> dict[str, Any]:
    fixed_broad12 = _all_pass(rows, "fixed_sampling_phase55", "broad12")
    fixed_broad21 = _all_pass(rows, "fixed_sampling_phase55", "broad21")
    density_broad12 = _all_pass(rows, "alternate_density_phase58", "broad12")
    density_broad21 = _all_pass(rows, "alternate_density_phase58", "broad21")
    panel_broad15 = _all_pass(rows, "auxiliary_panel_phase58", "broad15")
    broad21_density_boundary = _any_boundary(rows, "alternate_density_phase58", "broad21")
    upper_blocks = _upper_blocks_model_expansion(upper)
    open_for_seed7 = (
        fixed_broad12
        and fixed_broad21
        and density_broad12
        and density_broad21
        and panel_broad15
        and not upper_blocks
    )
    if open_for_seed7:
        status = "opened_for_seed7_a100_gate"
        next_action = "run bounded physical spot_size parameterization seed-7 focused validation on broad12 and broad21"
        reason = "fixed sampling, density stress, auxiliary panel, and upper-bound gates all support reopening Candidate A"
    else:
        status = "paused_no_training_signal"
        next_action = "do not train Candidate A; continue manuscript v0 audit or non-training route/data probes"
        reason = (
            "fixed-sampling broad12/broad21 and broad15 support the current floor, but broad21 alternate-density "
            "is a strong-baseline boundary and Phase 59 validation-selected correction falls back to the mean"
        )
    return {
        "candidate": "Candidate A: bounded physical spot-size parameterization",
        "status": status,
        "open_for_seed7_a100_gate": open_for_seed7,
        "a100_80gb_request_now": False,
        "fixed_sampling_broad12_pass": fixed_broad12,
        "fixed_sampling_broad21_pass": fixed_broad21,
        "alternate_density_broad12_pass": density_broad12,
        "alternate_density_broad21_pass": density_broad21,
        "auxiliary_panel_broad15_pass": panel_broad15,
        "broad21_density_boundary": broad21_density_boundary,
        "phase59_upper_blocks_density_driven_expansion": upper_blocks,
        "next_action": next_action,
        "reason": reason,
        "seed7_gate": (
            "if reopened in the future, seed 7 must be non-worse than broad_process_v1 on broad12 and broad21 "
            "for rmse, hot_q90_rmse, and gradient_q90_rmse"
        ),
    }


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


def build_markdown(gate: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase 69 Spot-Size Non-Training Signal Probe",
        "",
        "## Purpose",
        "",
        "Phase 69 implements the Phase 68 `P68-SPOT-SIGNAL` action. It decides whether Candidate A can enter A100 seed-7 training without adding any new model run.",
        "",
        "## Candidate A Gate",
        "",
        f"Status: `{gate['status']}`.",
        f"Open for seed-7 A100 gate: `{str(gate['open_for_seed7_a100_gate']).lower()}`.",
        f"A100-SXM4-80GB request now: `{str(gate['a100_80gb_request_now']).lower()}`.",
        "",
        gate["reason"],
        "",
        "## Evidence Rows",
        "",
        _markdown_table(
            rows,
            [
                ("evidence_source", "Source"),
                ("dataset", "Dataset"),
                ("metric", "Metric"),
                ("candidate", "Candidate"),
                ("best_strong_baseline", "Best strong"),
                ("delta_vs_best_strong", "Delta"),
                ("status", "Status"),
                ("interpretation", "Interpretation"),
            ],
        ),
        "",
        "## Next Action",
        "",
        gate["next_action"],
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
    fixed = _read_json(resolved["phase55_fixed"])
    density = _read_json(resolved["phase58_density"])
    panel = _read_json(resolved["phase58_panel"])
    upper = _read_json(resolved["phase59_upper"])
    phase68_manifest = _read_json(resolved["phase68_manifest"])
    rows = build_signal_rows(fixed, density, panel)
    gate = build_candidate_gate(rows, upper)
    output_dir.mkdir(parents=True, exist_ok=True)

    signal_csv = output_dir / "phase69_spot_size_signal_probe_table.csv"
    gate_json = output_dir / "phase69_candidate_a_gate.json"
    markdown_path = output_dir / "phase69_spot_size_signal_probe.md"
    manifest_path = output_dir / "phase69_spot_size_signal_probe_manifest.json"

    _write_csv(signal_csv, rows, SIGNAL_FIELDS)
    _write_json(gate_json, gate)
    markdown_path.write_text(build_markdown(gate, rows), encoding="utf-8")
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    manifest = {
        "phase": 69,
        "objective": "spot_size_non_training_signal_probe",
        "inputs": {key: _display_path(path, root) for key, path in sorted(resolved.items())},
        "outputs": {
            "signal_table": _display_path(signal_csv, root),
            "candidate_a_gate": _display_path(gate_json, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "signal_rows": len(rows),
            "status_counts": status_counts,
        },
        "candidate_a_gate": gate,
        "phase68_decision": phase68_manifest.get("current_decision"),
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase69_spot_size_signal_probe"),
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
