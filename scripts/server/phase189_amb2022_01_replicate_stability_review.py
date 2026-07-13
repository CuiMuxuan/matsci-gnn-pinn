#!/usr/bin/env python3
"""Review the frozen Phase 188 AMB2022-01 candidate comparison without retraining."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any


DEFAULT_PHASE188 = Path(
    os.environ.get(
        "AMB2022_01_PHASE188_TRAINING",
        "/root/matsci-gnn-pinn-ops/phase188_bounded_gpu_training.json",
    )
)
DEFAULT_METRICS = Path(
    os.environ.get(
        "AMB2022_01_PHASE188_METRICS",
        "/root/matsci-gnn-pinn-ops/phase188_metrics.csv",
    )
)
DEFAULT_AUDITS = Path(
    os.environ.get(
        "AMB2022_01_PHASE188_AUDITS",
        "/root/matsci-gnn-pinn-ops/phase188_training_audit.csv",
    )
)
DEFAULT_PHASE186_METRICS = Path(
    os.environ.get(
        "AMB2022_01_PHASE186_METRICS",
        "/root/matsci-gnn-pinn-ops/phase186_feature_ablation_metrics.csv",
    )
)

TARGETS = ("tam_s", "scr_C_per_s")
SPLITS = ("train", "val", "test")
REVIEW_SPLITS = ("val", "test")
SEEDS = (1871, 1872, 1873)
DATA_ONLY_VARIANT = "small_data_only_mlp"
CANDIDATE_VARIANT = "physics_regularized_history_mlp"
PHASE186_REFERENCE_VARIANT = "heat_kernel_history_ridge"
SUMMARY_FIELDS = (
    "target",
    "split",
    "data_only_rmse_mean",
    "data_only_rmse_std",
    "candidate_rmse_mean",
    "candidate_rmse_std",
    "candidate_rmse_cv",
    "paired_gain_candidate_vs_data_only_mean",
    "paired_gain_candidate_vs_data_only_min",
    "paired_gain_candidate_vs_data_only_max",
    "all_fixed_seed_gains_positive",
    "phase186_heat_kernel_ridge_rmse",
    "candidate_gain_vs_phase186_heat_kernel_ridge",
    "candidate_beats_phase186_heat_kernel_ridge",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def _seed_values(
    rows: list[dict[str, Any]],
    *,
    target: str,
    split: str,
    variant_id: str,
) -> dict[int, float] | None:
    subset = [
        row
        for row in rows
        if row.get("target") == target
        and row.get("split") == split
        and row.get("variant_id") == variant_id
    ]
    if len(subset) != len(SEEDS):
        return None
    try:
        values = {int(row["seed"]): float(row["rmse"]) for row in subset}
    except (KeyError, TypeError, ValueError):
        return None
    return values if set(values) == set(SEEDS) else None


def _phase186_reference(
    rows: list[dict[str, Any]], *, target: str, split: str
) -> float | None:
    subset = [
        row
        for row in rows
        if row.get("target") == target
        and row.get("split") == split
        and row.get("variant_id") == PHASE186_REFERENCE_VARIANT
    ]
    if len(subset) != 1:
        return None
    try:
        return float(subset[0]["rmse"])
    except (KeyError, TypeError, ValueError):
        return None


def build_summary(
    metric_rows: list[dict[str, Any]], phase186_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    blockers: list[str] = []
    summary_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        for split in REVIEW_SPLITS:
            data_only = _seed_values(
                metric_rows,
                target=target,
                split=split,
                variant_id=DATA_ONLY_VARIANT,
            )
            candidate = _seed_values(
                metric_rows,
                target=target,
                split=split,
                variant_id=CANDIDATE_VARIANT,
            )
            if data_only is None:
                blockers.append(f"{target}_{split}_data_only_seed_contract")
            if candidate is None:
                blockers.append(f"{target}_{split}_candidate_seed_contract")
            if data_only is None or candidate is None:
                continue
            data_only_values = [data_only[seed] for seed in SEEDS]
            candidate_values = [candidate[seed] for seed in SEEDS]
            data_only_mean, data_only_std = _mean_std(data_only_values)
            candidate_mean, candidate_std = _mean_std(candidate_values)
            paired_gains = [data_only[seed] - candidate[seed] for seed in SEEDS]
            reference = _phase186_reference(phase186_rows, target=target, split=split)
            if reference is None:
                blockers.append(f"{target}_{split}_phase186_heat_kernel_reference_missing")
            summary_rows.append(
                {
                    "target": target,
                    "split": split,
                    "data_only_rmse_mean": data_only_mean,
                    "data_only_rmse_std": data_only_std,
                    "candidate_rmse_mean": candidate_mean,
                    "candidate_rmse_std": candidate_std,
                    "candidate_rmse_cv": candidate_std / max(candidate_mean, 1e-12),
                    "paired_gain_candidate_vs_data_only_mean": sum(paired_gains) / len(paired_gains),
                    "paired_gain_candidate_vs_data_only_min": min(paired_gains),
                    "paired_gain_candidate_vs_data_only_max": max(paired_gains),
                    "all_fixed_seed_gains_positive": all(gain > 0.0 for gain in paired_gains),
                    "phase186_heat_kernel_ridge_rmse": reference,
                    "candidate_gain_vs_phase186_heat_kernel_ridge": (
                        reference - candidate_mean if reference is not None else None
                    ),
                    "candidate_beats_phase186_heat_kernel_ridge": (
                        candidate_mean < reference if reference is not None else None
                    ),
                }
            )
    return summary_rows, blockers


def _complete_metric_contract(metric_rows: list[dict[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for target in TARGETS:
        for split in SPLITS:
            for variant_id in (DATA_ONLY_VARIANT, CANDIDATE_VARIANT):
                if _seed_values(
                    metric_rows,
                    target=target,
                    split=split,
                    variant_id=variant_id,
                ) is None:
                    blockers.append(f"{target}_{split}_{variant_id}_seed_contract")
    return blockers


def _candidate_monotonic_audit(audit_rows: list[dict[str, Any]]) -> tuple[float | None, list[str]]:
    blockers: list[str] = []
    candidate_rows = [row for row in audit_rows if row.get("variant_id") == CANDIDATE_VARIANT]
    if len(candidate_rows) != len(SEEDS):
        return None, ["candidate_monotonic_audit_seed_contract"]
    try:
        by_seed = {
            int(row["seed"]): float(row["monotonic_violation_fraction_test"])
            for row in candidate_rows
            if row.get("monotonic_violation_fraction_test") not in (None, "")
        }
    except (KeyError, TypeError, ValueError):
        return None, ["candidate_monotonic_audit_not_numeric"]
    if set(by_seed) != set(SEEDS):
        return None, ["candidate_monotonic_audit_seed_contract"]
    maximum = max(by_seed.values())
    if maximum > 0.25:
        blockers.append("candidate_per_seed_monotonic_violation_guard")
    return maximum, blockers


def build_gate(
    phase188: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    phase186_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    phase188_gate = phase188.get("gate", {})
    phase188_ready = (
        phase188_gate.get("status") == "phase188_bounded_gpu_training_complete_phase189_replicate_review"
        and bool(phase188_gate.get("training_complete"))
        and bool(phase188_gate.get("phase189_replicate_review_allowed"))
        and phase188_gate.get("model_training_allowed") is False
    )
    summary_rows, summary_blockers = build_summary(metric_rows, phase186_rows)
    blockers = list(_complete_metric_contract(metric_rows))
    blockers.extend(summary_blockers)
    if not phase188_ready:
        blockers.append("phase188_completion_gate_not_ready")
    expected_summary_keys = {(target, split) for target in TARGETS for split in REVIEW_SPLITS}
    observed_summary_keys = {(row["target"], row["split"]) for row in summary_rows}
    if observed_summary_keys != expected_summary_keys:
        blockers.append("replicate_summary_incomplete")
    directional_passes: dict[str, bool] = {}
    phase186_comparison: dict[str, bool | None] = {}
    for row in summary_rows:
        key = f"{row['target']}_{row['split']}"
        directional_passes[key] = bool(row["all_fixed_seed_gains_positive"])
        phase186_comparison[key] = row["candidate_beats_phase186_heat_kernel_ridge"]
        if not row["all_fixed_seed_gains_positive"]:
            blockers.append(f"{key}_candidate_not_directionally_replicated")
    max_monotonic_violation, monotonic_blockers = _candidate_monotonic_audit(audit_rows)
    blockers.extend(monotonic_blockers)
    blockers = sorted(set(blockers))
    ready = not blockers
    return {
        "status": (
            "phase189_replicate_stability_review_ready_phase190_spatial_failure_analysis"
            if ready
            else "phase189_replicate_stability_review_incomplete_or_not_stable"
        ),
        "phase190_spatial_failure_analysis_allowed": ready,
        "model_training_allowed": False,
        "hyperparameter_search_allowed": False,
        "post_b8_model_reselection_allowed": False,
        "a800_training_allowed_now": False,
        "fixed_seed_directional_passes": directional_passes,
        "candidate_beats_phase186_heat_kernel_ridge": phase186_comparison,
        "candidate_max_monotonic_violation_fraction_test": max_monotonic_violation,
        "compound_candidate_effect_claim_allowed": ready,
        "isolated_heat_kernel_effect_claim_allowed": False,
        "isolated_monotonic_prior_effect_claim_allowed": False,
        "external_generalization_claim_allowed": False,
        "absolute_wall_clock_trajectory_claim_allowed": False,
        "raw_frame_event_causal_training_allowed": False,
        "blocking_audits": blockers,
        "next_action": (
            "perform spatial failure analysis and prepare an external-build confirmation plan; do not tune after B8"
            if ready
            else "audit incomplete metrics, seed directionality, or monotonic compliance without rerunning selected models"
        ),
    }


def build_review(
    phase188: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    phase186_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary_rows, _ = build_summary(metric_rows, phase186_rows)
    return {
        "phase": 189,
        "objective": "post_b8_replicate_stability_and_claim_boundary_review",
        "comparison_boundary": (
            "The candidate jointly adds causal heat-kernel features and a TAM monotonic prior; "
            "this review does not attribute its effect to either component alone."
        ),
        "summary": summary_rows,
        "gate": build_gate(phase188, metric_rows, audit_rows, phase186_rows),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase188", type=Path, default=DEFAULT_PHASE188)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--audits", type=Path, default=DEFAULT_AUDITS)
    parser.add_argument("--phase186-metrics", type=Path, default=DEFAULT_PHASE186_METRICS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    args = parser.parse_args()
    payload = build_review(
        _read_json(args.phase188),
        _read_csv(args.metrics),
        _read_csv(args.audits),
        _read_csv(args.phase186_metrics),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(args.summary_csv, payload["summary"])
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
