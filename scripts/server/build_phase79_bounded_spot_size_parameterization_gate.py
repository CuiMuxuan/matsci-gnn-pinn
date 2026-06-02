#!/usr/bin/env python3
"""Build the Phase 79 bounded spot-size parameterization safety gate.

Phase 79 is a non-training gate for the next Candidate A variant. It asks
whether a bounded, interpretable process/spot-size parameterization has enough
evidence margin to enter A100 seed-7 validation, or whether it must first be
tested as a local surrogate candidate.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_METRICS = ("rmse", "hot_q90_rmse", "gradient_q90_rmse")

GATE_FIELDS = (
    "evidence_source",
    "dataset",
    "metric",
    "candidate",
    "best_strong_baseline",
    "no_process",
    "margin_vs_best_strong",
    "margin_vs_no_process",
    "density_debt_vs_best_strong",
    "safety_ratio",
    "status",
    "interpretation",
    "evidence_locator",
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
    if isinstance(value, bool):
        return "true" if value else "false"
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
        "phase55_fixed": root / "outputs/reports/phase55_spot_size_route_seed_check_summary.json",
        "phase58_density": root
        / "docs/results/phase58_sampling_panel_stress/phase58_sampling_density_stress_summary.json",
        "phase69_gate": root / "docs/results/phase69_spot_size_signal_probe/phase69_candidate_a_gate.json",
        "phase75_manifest": root
        / "docs/results/phase75_bayesian_inverse_closure_gate/phase75_bayesian_inverse_closure_gate_manifest.json",
    }


def _dataset(summary: dict[str, Any], label: str) -> dict[str, Any]:
    for item in summary.get("datasets", []):
        if item.get("label") == label:
            return item
    raise KeyError(f"Missing dataset {label!r}")


def _metric_payload(summary: dict[str, Any], label: str, metric: str) -> dict[str, Any]:
    data = _dataset(summary, label)
    metrics = data.get("aggregate_gate", {}).get("metrics") or data.get("metrics") or {}
    payload = metrics.get(metric)
    if not isinstance(payload, dict):
        raise KeyError(f"Missing {label}/{metric}")
    return payload


def _margin_row(
    *,
    evidence_source: str,
    summary: dict[str, Any],
    label: str,
    metric: str,
    density_summary: dict[str, Any] | None,
    root: Path,
    source_path: Path,
) -> dict[str, Any]:
    payload = _metric_payload(summary, label, metric)
    candidate = float(payload["candidate"])
    best = float(payload["best_strong_baseline"])
    no_process = payload.get("no_process")
    no_process_value = float(no_process) if no_process is not None else None
    margin_vs_best = best - candidate
    margin_vs_no_process = (
        no_process_value - candidate if no_process_value is not None else None
    )
    debt = 0.0
    if density_summary is not None:
        density_payload = _metric_payload(density_summary, label, metric)
        debt = max(0.0, float(density_payload["candidate"]) - float(density_payload["best_strong_baseline"]))
    if debt <= 0.0:
        safety_ratio = 0.0
    elif margin_vs_best > 0.0:
        safety_ratio = debt / margin_vs_best
    else:
        safety_ratio = None
    if margin_vs_best <= 0.0:
        status = "floor_not_preserved"
        interpretation = "current floor does not beat the strongest baseline"
    elif debt > margin_vs_best:
        status = "density_debt_exceeds_floor_margin"
        interpretation = "density boundary debt is larger than the fixed-sampling safety margin"
    elif debt > 0.0:
        status = "density_debt_within_floor_margin"
        interpretation = "density boundary exists but is smaller than the fixed-sampling safety margin"
    else:
        status = "margin_preserved"
        interpretation = "fixed-sampling floor preserves this metric against strong baselines"
    return {
        "evidence_source": evidence_source,
        "dataset": label,
        "metric": metric,
        "candidate": candidate,
        "best_strong_baseline": best,
        "no_process": no_process_value,
        "margin_vs_best_strong": margin_vs_best,
        "margin_vs_no_process": margin_vs_no_process,
        "density_debt_vs_best_strong": debt,
        "safety_ratio": safety_ratio,
        "status": status,
        "interpretation": interpretation,
        "evidence_locator": _display_path(source_path, root),
    }


def build_candidate_design() -> dict[str, Any]:
    return {
        "candidate_id": "bounded_spot_size_parameterization_v1",
        "candidate_family": "bounded process/spot-size parameterization",
        "mechanism": (
            "Add a constrained, low-capacity spot-size response inside the existing "
            "`broad_process_v1` FiLM/global-standard route, with identity initialization "
            "and bounded modulation so the frozen `spot_size` floor cannot be silently replaced."
        ),
        "allowed_changes": [
            "bounded scalar or two-parameter spot-size response",
            "identity-initialized modulation around the existing FiLM route",
            "train/validation-selected bounds only",
            "explicit report of global, hot q90, and gradient q90 deltas",
        ],
        "forbidden_shortcuts": [
            "test-set route selection",
            "unbounded mixture-of-experts",
            "density-failure-driven A100 training without a local gate",
            "claiming density-invariant robustness from fixed-sampling evidence",
        ],
        "gate_contract": (
            "A100 seed-7 validation is allowed only if fixed-sampling margins remain "
            "positive and density/stress debt does not exceed the frozen floor margin. "
            "Otherwise the candidate must first pass a local surrogate gate."
        ),
    }


def build_margin_rows(
    *,
    fixed: dict[str, Any],
    density: dict[str, Any],
    root: Path,
    fixed_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in ("broad12", "broad21"):
        for metric in REQUIRED_METRICS:
            rows.append(
                _margin_row(
                    evidence_source="phase55_fixed_sampling_floor",
                    summary=fixed,
                    label=label,
                    metric=metric,
                    density_summary=density,
                    root=root,
                    source_path=fixed_path,
                )
            )
    return rows


def _minimum_positive_margin(rows: list[dict[str, Any]]) -> float:
    margins = [float(row["margin_vs_best_strong"]) for row in rows]
    return min(margins) if margins else 0.0


def _maximum_density_debt(rows: list[dict[str, Any]]) -> float:
    debts = [float(row["density_debt_vs_best_strong"]) for row in rows]
    return max(debts) if debts else 0.0


def build_gate(
    rows: list[dict[str, Any]],
    *,
    phase69_gate: dict[str, Any],
    phase75_manifest: dict[str, Any],
) -> dict[str, Any]:
    fixed_floor_preserved = all(float(row["margin_vs_best_strong"]) > 0.0 for row in rows)
    debt_rows = [row for row in rows if float(row["density_debt_vs_best_strong"]) > 0.0]
    debt_exceeds_margin = any(
        row["status"] == "density_debt_exceeds_floor_margin" for row in rows
    )
    candidate_a_open = bool(phase69_gate.get("open_for_seed7_a100_gate"))
    prior_phase75_open = bool(
        (phase75_manifest.get("gate_status") or {}).get("phase76_seed7_allowed")
    )
    a100_seed7_allowed = (
        fixed_floor_preserved
        and not debt_rows
        and candidate_a_open
        and not prior_phase75_open
    )
    local_surrogate_allowed = fixed_floor_preserved and not a100_seed7_allowed
    if a100_seed7_allowed:
        status = "opened_for_phase76_seed7"
        next_action = "run bounded_spot_size_parameterization_v1 seed-7 focused validation on A100"
        reason = "fixed floor and stress margins preserve all required metrics"
    elif local_surrogate_allowed:
        status = "local_surrogate_required_before_a100"
        next_action = (
            "build a local surrogate gate for bounded_spot_size_parameterization_v1; "
            "do not run broad12/broad21 A100 training yet"
        )
        reason = (
            "fixed-sampling margins are positive, but density debt and Phase 69 still block "
            "direct A100 seed-7 validation"
        )
    else:
        status = "blocked_no_safe_margin"
        next_action = "close this candidate or redesign it with new validation-visible evidence"
        reason = "the frozen floor does not preserve enough strong-baseline margin"
    return {
        "candidate": "bounded_spot_size_parameterization_v1",
        "status": status,
        "a100_seed7_allowed": a100_seed7_allowed,
        "local_surrogate_allowed": local_surrogate_allowed,
        "a100_80gb_request_now": False,
        "fixed_floor_preserved": fixed_floor_preserved,
        "candidate_a_phase69_open": candidate_a_open,
        "phase75_candidate_open": prior_phase75_open,
        "density_debt_row_count": len(debt_rows),
        "density_debt_exceeds_floor_margin": debt_exceeds_margin,
        "minimum_fixed_margin_vs_best_strong": _minimum_positive_margin(rows),
        "maximum_density_debt_vs_best_strong": _maximum_density_debt(rows),
        "reason": reason,
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


def build_markdown(design: dict[str, Any], gate: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "# Phase 79 Bounded Spot-Size Parameterization Gate",
            "",
            "## Candidate",
            "",
            f"Candidate: `{design['candidate_id']}`.",
            "",
            design["mechanism"],
            "",
            "## Gate Decision",
            "",
            f"Status: `{gate['status']}`.",
            f"A100 seed-7 allowed: `{str(gate['a100_seed7_allowed']).lower()}`.",
            f"Local surrogate allowed: `{str(gate['local_surrogate_allowed']).lower()}`.",
            f"A100-SXM4-80GB request now: `{str(gate['a100_80gb_request_now']).lower()}`.",
            "",
            gate["reason"],
            "",
            "## Margin Rows",
            "",
            _markdown_table(
                rows,
                [
                    ("dataset", "Dataset"),
                    ("metric", "Metric"),
                    ("margin_vs_best_strong", "Fixed margin"),
                    ("density_debt_vs_best_strong", "Density debt"),
                    ("safety_ratio", "Debt / margin"),
                    ("status", "Status"),
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
    fixed = _read_json(resolved["phase55_fixed"])
    density = _read_json(resolved["phase58_density"])
    phase69_gate = _read_json(resolved["phase69_gate"])
    phase75_manifest = _read_json(resolved["phase75_manifest"])
    design = build_candidate_design()
    rows = build_margin_rows(
        fixed=fixed,
        density=density,
        root=root,
        fixed_path=resolved["phase55_fixed"],
    )
    gate = build_gate(rows, phase69_gate=phase69_gate, phase75_manifest=phase75_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)

    design_path = output_dir / "phase79_candidate_design.json"
    table_path = output_dir / "phase79_bounded_spot_size_parameterization_gate_table.csv"
    gate_path = output_dir / "phase79_bounded_spot_size_parameterization_gate.json"
    markdown_path = output_dir / "phase79_bounded_spot_size_parameterization_gate.md"
    manifest_path = output_dir / "phase79_bounded_spot_size_parameterization_gate_manifest.json"

    _write_json(design_path, design)
    _write_csv(table_path, rows, GATE_FIELDS)
    _write_json(gate_path, gate)
    markdown_path.write_text(build_markdown(design, gate, rows), encoding="utf-8")
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[str(row["status"])] = status_counts.get(str(row["status"]), 0) + 1
    manifest = {
        "phase": 79,
        "objective": "bounded_spot_size_parameterization_safety_gate",
        "inputs": {key: _display_path(path, root) for key, path in sorted(resolved.items())},
        "outputs": {
            "candidate_design": _display_path(design_path, root),
            "gate_table": _display_path(table_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "gate_rows": len(rows),
            "status_counts": status_counts,
        },
        "gate": gate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase79_bounded_spot_size_parameterization_gate"),
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
