#!/usr/bin/env python3
"""Build the Phase 75 Bayesian inverse-closure local identifiability gate.

Phase 75 is a gate package, not a broad12/broad21 training run. It tests one
candidate, `bayesian_inverse_closure_v1`, with a synthetic known-parameter
probe and a small local AM-Bench probe before any A100 seed-7 validation can be
opened.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any


LOCAL_TABLE_SNAPSHOT = "phase75_line0_1_temperature_medium_probe_snapshot.csv"
LOCAL_SPLIT_SNAPSHOT = "phase75_line0_1_temperature_medium_probe_split_snapshot.json"
JSON_FLOAT_DIGITS = 4

GATE_FIELDS = (
    "gate_id",
    "gate_scope",
    "input_source",
    "status",
    "rmse_gain",
    "hot_q90_rmse_gain",
    "gradient_q90_rmse_gain",
    "coverage_ok",
    "parameter_or_source_recovery_ok",
    "region_preservation_ok",
    "a100_training_allowed",
    "decision",
    "evidence_locator",
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_stable_json_value(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _stable_json_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, JSON_FLOAT_DIGITS)
    if isinstance(value, dict):
        return {key: _stable_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stable_json_value(item) for item in value]
    return value


def _stable_probe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Persist only the stable, gate-relevant part of a Phase 46 probe.

    The raw Phase 46 `runs` array includes posterior coefficients and per-run
    floating point diagnostics that can drift by a few ulps across BLAS/Python
    builds. Phase 75 needs the gate summary and decision, not a full numerical
    trace, so the committed artifact is intentionally compact and reproducible.
    """

    keys = (
        "label",
        "feature_names",
        "n_points",
        "split_sizes",
        "initial_size",
        "acquisition_size",
        "repeats",
        "strategies",
        "active_strategy",
        "feature_mode",
        "calibration_mode",
        "summary",
        "decision",
    )
    compact = {key: payload[key] for key in keys if key in payload}
    compact["raw_run_count"] = len(payload.get("runs") or [])
    compact["raw_runs_persisted"] = False
    compact["artifact_stability_note"] = (
        "Raw per-run posterior traces are omitted from the committed Phase 75 "
        "probe artifact because they are not used for the gate decision and "
        "can show cross-platform floating-point drift."
    )
    return compact


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


def _materialize_snapshot(*, source: Path, snapshot: Path, label: str) -> Path:
    """Return a committed gate input snapshot, creating it from source if needed."""

    if snapshot.exists():
        return snapshot
    if not source.exists():
        raise FileNotFoundError(
            f"Missing {label}: neither snapshot {snapshot} nor source {source} exists"
        )
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, snapshot)
    return snapshot


def _load_phase46_module(root: Path):
    module_path = root / "scripts/server/phase46_bayesian_inverse_closure_probe.py"
    module_spec = importlib.util.spec_from_file_location("phase46_probe", module_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def _default_paths(root: Path) -> dict[str, Path]:
    return {
        "phase74_manifest": root
        / "docs/results/phase74_manuscript_v0_claim_audit/phase74_manuscript_v0_claim_audit_manifest.json",
        "local_table": root
        / "data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_temperature_medium_probe.csv",
        "local_split": root / "outputs/data_splits/ambench_line_0_1_temperature_medium_probe_split.json",
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


def _probe_args(
    phase46: Any,
    *,
    mode: str,
    seed: int,
    repeats: int,
    table: Path | None = None,
    split_manifest: Path | None = None,
    target: str | None = None,
    synthetic_grid: int = 12,
    synthetic_frames: int = 5,
    synthetic_noise_std: float = 4.0,
    initial_size: int = 48,
    acquisition_size: int = 72,
) -> argparse.Namespace:
    argv = [
        "--mode",
        mode,
        "--feature-mode",
        "heat_kernel",
        "--strategy",
        "region_quota_uncertainty",
        "--strategy",
        "validation_selected_region_policy",
        "--active-strategy",
        "validation_selected_region_policy",
        "--calibration-mode",
        "conformal90",
        "--require-region-preservation",
        "--seed",
        str(seed),
        "--repeats",
        str(repeats),
        "--initial-size",
        str(initial_size),
        "--acquisition-size",
        str(acquisition_size),
    ]
    if mode == "synthetic":
        argv.extend(
            [
                "--synthetic-grid",
                str(synthetic_grid),
                "--synthetic-frames",
                str(synthetic_frames),
                "--synthetic-noise-std",
                str(synthetic_noise_std),
            ]
        )
    else:
        if table is None or split_manifest is None or target is None:
            raise ValueError("table, split_manifest, and target are required for table probes")
        argv.extend(
            [
                "--table",
                str(table),
                "--split-manifest",
                str(split_manifest),
                "--target",
                target,
            ]
        )
    return phase46.build_parser().parse_args(argv)


def _decision_gains(payload: dict[str, Any]) -> dict[str, Any]:
    decision = payload.get("decision") or {}
    return {
        "rmse_gain": float(decision.get("rmse_gain_vs_random", 0.0)),
        "hot_q90_rmse_gain": float(decision.get("hot_q90_rmse_gain_vs_random", 0.0)),
        "gradient_q90_rmse_gain": float(decision.get("gradient_q90_rmse_gain_vs_random", 0.0)),
        "coverage_ok": bool(decision.get("calibration_ok")),
        "parameter_or_source_recovery_ok": bool(decision.get("source_recovery_ok", True)),
        "region_preservation_ok": bool(decision.get("region_preservation_ok")),
        "status": str(decision.get("status") or "inconclusive"),
    }


def _gate_row(
    *,
    gate_id: str,
    gate_scope: str,
    input_source: str,
    payload: dict[str, Any],
    evidence_locator: str,
    allow_training_if_positive: bool,
) -> dict[str, Any]:
    gains = _decision_gains(payload)
    positive = gains["status"] == "positive"
    return {
        "gate_id": gate_id,
        "gate_scope": gate_scope,
        "input_source": input_source,
        "status": gains["status"],
        "rmse_gain": gains["rmse_gain"],
        "hot_q90_rmse_gain": gains["hot_q90_rmse_gain"],
        "gradient_q90_rmse_gain": gains["gradient_q90_rmse_gain"],
        "coverage_ok": gains["coverage_ok"],
        "parameter_or_source_recovery_ok": gains["parameter_or_source_recovery_ok"],
        "region_preservation_ok": gains["region_preservation_ok"],
        "a100_training_allowed": bool(positive and allow_training_if_positive),
        "decision": (
            "passes_this_gate"
            if positive
            else "fails_this_gate_due_to_metric_transfer_or_recovery"
        ),
        "evidence_locator": evidence_locator,
    }


def build_candidate_design() -> dict[str, Any]:
    return {
        "candidate_id": "bayesian_inverse_closure_v1",
        "candidate_family": "lightweight Bayesian inverse-closure PINN gate",
        "paper_facing_hypothesis": (
            "A low-dimensional Bayesian inverse closure can provide uncertainty-aware "
            "adaptive sampling and interpretable source/closure coefficients, but it "
            "must preserve global, hot-zone, and gradient-band errors before broad-data training."
        ),
        "mechanism": {
            "feature_family": "moving heat-kernel / Green's-function proxy closure basis",
            "inference": "closed-form Bayesian linear regression over interpretable closure coefficients",
            "adaptive_sampling": "validation-selected region policy over uncertainty, source-prior, and gradient proxies",
            "uncertainty": "posterior predictive standard deviation with conformal90 validation calibration",
        },
        "deterministic_controls": [
            "random sparse acquisition with the same selected sample budget",
            "region_quota_uncertainty acquisition",
            "uncertainty_source acquisition",
            "existing broad_process_v1 route guard before any broad12/broad21 promotion",
        ],
        "go_no_go_rule": (
            "Synthetic known-parameter recovery must be positive, and the local AM-Bench "
            "gate must preserve or improve RMSE, hot q90 RMSE, and gradient q90 RMSE. "
            "A synthetic-only positive is not enough for Phase 76 A100 validation."
        ),
        "a100_40gb_policy": "Use A100-SXM4-40GB only after the Phase 75 local gate passes.",
        "a100_80gb_policy": (
            "Request A100-SXM4-80GB only if a Phase 75/76-passing branch is blocked by "
            "measured A100-SXM4-40GB memory or runtime."
        ),
    }


def build_markdown(
    gate_status: dict[str, Any],
    rows: list[dict[str, Any]],
    candidate_design: dict[str, Any],
) -> str:
    lines = [
        "# Phase 75 Bayesian Inverse-Closure Local Identifiability Gate",
        "",
        "## Candidate",
        "",
        f"Candidate: `{candidate_design['candidate_id']}`.",
        "",
        candidate_design["paper_facing_hypothesis"],
        "",
        "## Gate Decision",
        "",
        f"Status: `{gate_status['status']}`.",
        f"Synthetic gate passed: `{str(gate_status['synthetic_gate_passed']).lower()}`.",
        f"Local AM-Bench gate passed: `{str(gate_status['local_gate_passed']).lower()}`.",
        f"Phase 76 A100 seed-7 validation allowed: `{str(gate_status['phase76_seed7_allowed']).lower()}`.",
        f"A100-SXM4-80GB request now: `{str(gate_status['a100_80gb_request_now']).lower()}`.",
        "",
        gate_status["reason"],
        "",
        "## Gate Rows",
        "",
        _markdown_table(
            rows,
            [
                ("gate_id", "Gate"),
                ("gate_scope", "Scope"),
                ("status", "Status"),
                ("rmse_gain", "RMSE gain"),
                ("hot_q90_rmse_gain", "Hot q90 gain"),
                ("gradient_q90_rmse_gain", "Gradient q90 gain"),
                ("a100_training_allowed", "A100 allowed"),
            ],
        ),
        "",
        "## Next Action",
        "",
        gate_status["next_action"],
        "",
    ]
    return "\n".join(lines)


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


def build_package(
    root: Path,
    output_dir: Path,
    paths: dict[str, Path] | None = None,
    *,
    seed: int = 75,
    repeats: int = 2,
) -> dict[str, Any]:
    resolved = _default_paths(root)
    if paths:
        resolved.update(paths)
    phase74_manifest = _read_json(resolved["phase74_manifest"])
    phase46 = _load_phase46_module(root)
    candidate_design = build_candidate_design()
    output_dir.mkdir(parents=True, exist_ok=True)
    local_table = _materialize_snapshot(
        source=resolved["local_table"],
        snapshot=output_dir / LOCAL_TABLE_SNAPSHOT,
        label="local AM-Bench gate table",
    )
    local_split = _materialize_snapshot(
        source=resolved["local_split"],
        snapshot=output_dir / LOCAL_SPLIT_SNAPSHOT,
        label="local AM-Bench split manifest",
    )

    synthetic_args = _probe_args(
        phase46,
        mode="synthetic",
        seed=seed,
        repeats=repeats,
        synthetic_grid=12,
        synthetic_frames=5,
        synthetic_noise_std=4.0,
        initial_size=48,
        acquisition_size=72,
    )
    synthetic_payload = phase46.run(synthetic_args)
    local_args = _probe_args(
        phase46,
        mode="table",
        seed=seed,
        repeats=repeats,
        table=local_table,
        split_manifest=local_split,
        target="temperature_C",
        initial_size=96,
        acquisition_size=192,
    )
    local_payload = phase46.run(local_args)

    design_path = output_dir / "phase75_candidate_design.json"
    synthetic_path = output_dir / "phase75_synthetic_identifiability_probe.json"
    local_path = output_dir / "phase75_local_ambench_probe.json"
    gate_table_path = output_dir / "phase75_bayesian_inverse_closure_gate_table.csv"
    markdown_path = output_dir / "phase75_bayesian_inverse_closure_gate.md"
    manifest_path = output_dir / "phase75_bayesian_inverse_closure_gate_manifest.json"

    _write_json(design_path, candidate_design)
    _write_json(synthetic_path, _stable_probe_payload(synthetic_payload))
    _write_json(local_path, _stable_probe_payload(local_payload))
    rows = [
        _gate_row(
            gate_id="P75-SYNTH",
            gate_scope="synthetic_known_parameter_identifiability",
            input_source="synthetic_heat_source_heat_kernel",
            payload=synthetic_payload,
            evidence_locator=_display_path(synthetic_path, root),
            allow_training_if_positive=False,
        ),
        _gate_row(
            gate_id="P75-LOCAL",
            gate_scope="local_ambench_line0_1_region_preservation",
            input_source=_display_path(local_table, root),
            payload=local_payload,
            evidence_locator=_display_path(local_path, root),
            allow_training_if_positive=True,
        ),
    ]
    _write_csv(gate_table_path, rows, GATE_FIELDS)

    synthetic_passed = rows[0]["status"] == "positive"
    local_passed = rows[1]["status"] == "positive"
    phase76_allowed = bool(synthetic_passed and local_passed)
    gate_status = {
        "status": "opened_for_phase76_seed7" if phase76_allowed else "blocked_by_local_ambench_gate",
        "synthetic_gate_passed": synthetic_passed,
        "local_gate_passed": local_passed,
        "phase76_seed7_allowed": phase76_allowed,
        "a100_training_allowed_now": phase76_allowed,
        "a100_80gb_request_now": False,
        "reason": (
            "Synthetic known-parameter recovery and local AM-Bench region preservation both pass."
            if phase76_allowed
            else (
                "The candidate is identifiable on synthetic heat-source data, but the local AM-Bench "
                "gate still shifts error between global, hot q90, and gradient q90 metrics. Do not "
                "run broad12/broad21 A100 validation yet."
            )
        ),
        "next_action": (
            "Run Phase 76 seed-7 broad12/broad21 focused validation on A100-SXM4-40GB."
            if phase76_allowed
            else "Close this candidate as a Phase 75 appendix diagnostic or redesign the local gate before any broad12/broad21 training."
        ),
    }
    markdown_path.write_text(build_markdown(gate_status, rows, candidate_design), encoding="utf-8")

    manifest = {
        "phase": 75,
        "objective": "bayesian_inverse_closure_local_identifiability_gate",
        "candidate": candidate_design["candidate_id"],
        "inputs": {
            **{key: _display_path(path, root) for key, path in sorted(resolved.items())},
            "local_table_snapshot": _display_path(local_table, root),
            "local_split_snapshot": _display_path(local_split, root),
        },
        "outputs": {
            "candidate_design": _display_path(design_path, root),
            "synthetic_probe": _display_path(synthetic_path, root),
            "local_ambench_probe": _display_path(local_path, root),
            "gate_table": _display_path(gate_table_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "gate_rows": len(rows),
            "positive_gate_rows": sum(1 for row in rows if row["status"] == "positive"),
            "negative_gate_rows": sum(1 for row in rows if row["status"] != "positive"),
        },
        "gate_status": gate_status,
        "probe_config": {
            "seed": seed,
            "repeats": repeats,
            "feature_mode": "heat_kernel",
            "active_strategy": "validation_selected_region_policy",
            "calibration_mode": "conformal90",
            "require_region_preservation": True,
        },
        "phase74_writing_stage_gate": phase74_manifest.get("writing_stage_gate"),
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase75_bayesian_inverse_closure_gate"),
    )
    parser.add_argument("--seed", type=int, default=75)
    parser.add_argument("--repeats", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    manifest = build_package(root=root, output_dir=output_dir, seed=args.seed, repeats=args.repeats)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
