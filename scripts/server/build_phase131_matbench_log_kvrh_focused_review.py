#!/usr/bin/env python3
"""Build Phase 131 focused review for the Matbench log KVRH gate.

This phase consumes only small Phase 130 artifacts. It checks whether the
Phase 130 bulk-modulus gap is stable under alternate grouped/binned splits and
whether shortcut profiles, nearest-neighbor composition/lattice identity, or
target-distribution imbalance explain the signal. It does not train a neural
model or open A100 training.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PHASE130_DIR = Path("docs/results/phase130_matbench_log_kvrh_baseline_gate")
DEFAULT_OUTPUT_DIR = Path("docs/results/phase131_matbench_log_kvrh_focused_review")
MIN_SPLIT_ROWS = 100
MIN_RELATIVE_VAL_GAIN = 0.05
MIN_STABLE_SPLIT_PASS_RATE = 0.75
NEGATIVE_DOMINANCE_TOLERANCE = 1.02
MAX_ORIGINAL_TARGET_SHIFT_Z = 0.75
MAX_NEAR_DUPLICATE_FRACTION = 0.60
NEAR_DUPLICATE_DISTANCE = 0.05


def _load_module(filename: str, module_name: str):
    script = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase130 = _load_module("build_phase130_matbench_log_kvrh_baseline_gate.py", "phase130_helpers")
phase127 = _load_module("build_phase127_matbench_phonons_focused_review.py", "phase127_helpers")

PROFILE_COLUMNS: dict[str, dict[str, Any]] = {
    "composition_descriptors": {
        **phase130.PROFILE_COLUMNS["composition_descriptors"],
        "role": "admissible",
    },
    "common_element_fractions": {
        **phase130.PROFILE_COLUMNS["common_element_fractions"],
        "role": "admissible",
    },
    "lattice_descriptors": {
        **phase130.PROFILE_COLUMNS["lattice_descriptors"],
        "role": "admissible",
    },
    "composition_lattice_descriptors": {
        **phase130.PROFILE_COLUMNS["composition_lattice_descriptors"],
        "role": "admissible",
    },
    "composition_hash_shortcut": {
        **phase130.PROFILE_COLUMNS["composition_hash_shortcut"],
        "role": "negative_control",
    },
    "chemistry_family_shortcut": {
        **phase130.PROFILE_COLUMNS["chemistry_family_shortcut"],
        "role": "negative_control",
    },
    "dominant_element_shortcut": {
        "role": "negative_control",
        "numeric": (),
        "categorical": ("dominant_element",),
    },
}
MODEL_METHODS = phase130.MODEL_METHODS
PROFILE_METHODS = {
    "composition_hash_shortcut": ("knn",),
    "chemistry_family_shortcut": ("knn",),
    "dominant_element_shortcut": ("knn", "extra_trees"),
}
NN_FEATURE_COLUMNS = tuple(
    column
    for column in (
        *(f"frac_{element}" for element in phase130.ELEMENTS),
        "n_sites",
        "lattice_a",
        "lattice_b",
        "lattice_c",
        "lattice_alpha",
        "lattice_beta",
        "lattice_gamma",
        "lattice_volume",
        "volume_per_site",
        "abc_anisotropy",
        "angle_deviation",
        "density_z_proxy",
        "site_z_mean",
        "site_z_std",
    )
)
SPLIT_PLAN = (
    ("phase130_registered_split", "phase130_manifest", "phase130"),
    ("chemistry_family_hash_0", "group:chemistry_family_key", "phase131_family_0"),
    ("chemistry_family_hash_1", "group:chemistry_family_key", "phase131_family_1"),
    ("chemistry_family_hash_2", "group:chemistry_family_key", "phase131_family_2"),
    ("dominant_element_hash", "group:dominant_element", "phase131_dominant"),
    ("lattice_volume_bins", "bins:lattice_volume", "phase131_lattice_volume"),
    ("volume_per_site_bins", "bins:volume_per_site", "phase131_volume_per_site"),
    ("element_count_bins", "bins:element_count", "phase131_element_count"),
    ("density_anisotropy_bins", "bins:density_z_proxy,abc_anisotropy", "phase131_density_aniso"),
)


def _configure_phase127_helpers() -> None:
    phase127.PROFILE_COLUMNS = PROFILE_COLUMNS
    phase127.MODEL_METHODS = MODEL_METHODS
    phase127.PROFILE_METHODS = PROFILE_METHODS
    phase127.NN_FEATURE_COLUMNS = NN_FEATURE_COLUMNS
    phase127.MIN_SPLIT_ROWS = MIN_SPLIT_ROWS
    phase127.MIN_RELATIVE_VAL_GAIN = MIN_RELATIVE_VAL_GAIN
    phase127.NEGATIVE_DOMINANCE_TOLERANCE = NEGATIVE_DOMINANCE_TOLERANCE
    phase127.NEAR_DUPLICATE_DISTANCE = NEAR_DUPLICATE_DISTANCE


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


def _registered_split(split_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "split_strategy": split_manifest.get("split_strategy", "phase130_registered_split"),
        "split_salt": "phase130",
        "split_viable": True,
        "reason": "ok",
        "splits": split_manifest["splits"],
        "group_splits": split_manifest.get("group_splits", {}),
        "n_groups": split_manifest.get("n_groups"),
        "leakage_safe": split_manifest.get("leakage_safe"),
    }


def build_split_reviews(df: pd.DataFrame, split_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    reviews: dict[str, dict[str, Any]] = {}
    for split_id, split_spec, salt in SPLIT_PLAN:
        if split_spec == "phase130_manifest":
            reviews[split_id] = _registered_split(split_manifest)
        elif split_spec.startswith("group:"):
            reviews[split_id] = phase127._group_split(df, split_spec.split(":", 1)[1], salt)
        elif split_spec.startswith("bins:"):
            columns = tuple(split_spec.split(":", 1)[1].split(","))
            reviews[split_id] = phase127._bin_split(df, columns, salt)
        else:
            raise ValueError(f"Unsupported split spec: {split_spec}")
    return reviews


def _audit_row(
    *,
    audit: str,
    status: str,
    severity: str,
    value: Any,
    threshold: Any,
    reason: str,
) -> dict[str, Any]:
    return {
        "audit": audit,
        "status": status,
        "severity": severity,
        "value": value,
        "threshold": threshold,
        "reason": reason,
    }


def build_audit_rows(*, phase130_gate: dict[str, Any], split_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    phase130_ready = phase130_gate.get("status") == "phase130_matbench_log_kvrh_ready_focused_review"
    rows.append(
        _audit_row(
            audit="phase130_gate_status",
            status="pass" if phase130_ready else "block",
            severity="blocking" if not phase130_ready else "info",
            value=phase130_gate.get("status"),
            threshold="phase130_matbench_log_kvrh_ready_focused_review",
            reason="focused review requires a Phase 130 focused-review gate",
        )
    )
    original = next((row for row in split_rows if row.get("split_id") == "phase130_registered_split"), None)
    if original and original.get("split_viable"):
        rows.append(
            _audit_row(
                audit="original_split_admissible_gain",
                status="pass" if phase127._is_true(original.get("split_pass")) else "block",
                severity="blocking" if not phase127._is_true(original.get("split_pass")) else "info",
                value=original.get("best_admissible_relative_val_gain"),
                threshold=MIN_RELATIVE_VAL_GAIN,
                reason="Phase 130 selected KVRH target must preserve validation and test gain",
            )
        )
        rows.append(
            _audit_row(
                audit="original_split_shortcut_dominance",
                status="block" if phase127._is_true(original.get("negative_control_dominates")) else "pass",
                severity="blocking" if phase127._is_true(original.get("negative_control_dominates")) else "info",
                value=original.get("best_negative_profile"),
                threshold=f"negative val RMSE > admissible * {NEGATIVE_DOMINANCE_TOLERANCE}",
                reason="composition, chemistry-family, or dominant-element shortcuts must not dominate",
            )
        )
        rows.append(
            _audit_row(
                audit="original_split_nearest_neighbor_dominance",
                status="block" if phase127._is_true(original.get("nearest_neighbor_dominates")) else "pass",
                severity="blocking" if phase127._is_true(original.get("nearest_neighbor_dominates")) else "info",
                value=original.get("nearest_neighbor_val_rmse"),
                threshold=f"nearest val RMSE > admissible * {NEGATIVE_DOMINANCE_TOLERANCE}",
                reason="nearest-neighbor composition/lattice identity control must not dominate",
            )
        )
        distribution_shift = float(original.get("target_distribution_shift_z") or 0.0)
        rows.append(
            _audit_row(
                audit="original_split_target_distribution_balance",
                status="block" if distribution_shift > MAX_ORIGINAL_TARGET_SHIFT_Z else "pass",
                severity="blocking" if distribution_shift > MAX_ORIGINAL_TARGET_SHIFT_Z else "info",
                value=original.get("target_distribution_shift_z"),
                threshold=MAX_ORIGINAL_TARGET_SHIFT_Z,
                reason="registered split target mean/median/q90 shift must not dominate interpretation",
            )
        )
        near_duplicate_fraction = max(
            float(original.get("val_near_duplicate_fraction") or 0.0),
            float(original.get("test_near_duplicate_fraction") or 0.0),
        )
        rows.append(
            _audit_row(
                audit="original_split_near_duplicate_fraction",
                status="block" if near_duplicate_fraction > MAX_NEAR_DUPLICATE_FRACTION else "pass",
                severity="blocking" if near_duplicate_fraction > MAX_NEAR_DUPLICATE_FRACTION else "info",
                value=near_duplicate_fraction,
                threshold=MAX_NEAR_DUPLICATE_FRACTION,
                reason="registered split must not contain too many near-duplicate composition/lattice rows relative to train",
            )
        )
    else:
        rows.append(
            _audit_row(
                audit="original_split_viability",
                status="block",
                severity="blocking",
                value=original.get("split_reason") if original else "missing",
                threshold=f"all splits >= {MIN_SPLIT_ROWS} rows",
                reason="Phase 130 registered split must be reviewable",
            )
        )
    viable = [row for row in split_rows if phase127._is_true(row.get("split_viable"))]
    passed = [row for row in viable if phase127._is_true(row.get("split_pass"))]
    stable_rate = len(passed) / len(viable) if viable else 0.0
    rows.append(
        _audit_row(
            audit="split_sensitivity_pass_rate",
            status="pass" if stable_rate >= MIN_STABLE_SPLIT_PASS_RATE else "block",
            severity="blocking" if stable_rate < MIN_STABLE_SPLIT_PASS_RATE else "info",
            value=stable_rate,
            threshold=MIN_STABLE_SPLIT_PASS_RATE,
            reason="KVRH target gain must survive deterministic chemistry, dominant-element, and lattice split perturbations",
        )
    )
    shortcut_dominant = [row for row in viable if phase127._is_true(row.get("negative_control_dominates"))]
    rows.append(
        _audit_row(
            audit="shortcut_dominant_split_count",
            status="block" if shortcut_dominant else "pass",
            severity="blocking" if shortcut_dominant else "info",
            value=len(shortcut_dominant),
            threshold=0,
            reason="no viable split may be dominated by composition, chemistry-family, or dominant-element shortcuts",
        )
    )
    nearest_dominant = [row for row in viable if phase127._is_true(row.get("nearest_neighbor_dominates"))]
    rows.append(
        _audit_row(
            audit="nearest_neighbor_dominant_split_count",
            status="block" if nearest_dominant else "pass",
            severity="blocking" if nearest_dominant else "info",
            value=len(nearest_dominant),
            threshold=0,
            reason="no viable split may be dominated by nearest-neighbor composition/lattice identity control",
        )
    )
    target_imbalanced = [
        row
        for row in viable
        if float(row.get("target_distribution_shift_z") or 0.0) > MAX_ORIGINAL_TARGET_SHIFT_Z
    ]
    rows.append(
        _audit_row(
            audit="target_distribution_imbalanced_split_count",
            status="block" if target_imbalanced else "pass",
            severity="blocking" if target_imbalanced else "info",
            value=len(target_imbalanced),
            threshold=0,
            reason="no viable split may have severe train/validation/test KVRH target mean, median, or q90 imbalance",
        )
    )
    return rows


def build_gate(*, phase130_gate: dict[str, Any], split_rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    blockers = [row for row in audit_rows if row["status"] == "block"]
    viable = [row for row in split_rows if phase127._is_true(row.get("split_viable"))]
    passed = [row for row in viable if phase127._is_true(row.get("split_pass"))]
    shortcut_dominant = [row for row in viable if phase127._is_true(row.get("negative_control_dominates"))]
    nearest_dominant = [row for row in viable if phase127._is_true(row.get("nearest_neighbor_dominates"))]
    target_imbalanced = [
        row
        for row in viable
        if float(row.get("target_distribution_shift_z") or 0.0) > MAX_ORIGINAL_TARGET_SHIFT_Z
    ]
    original = next((row for row in split_rows if row.get("split_id") == "phase130_registered_split"), {})
    if phase130_gate.get("status") != "phase130_matbench_log_kvrh_ready_focused_review":
        status = "phase131_matbench_log_kvrh_review_blocked_by_phase130"
        mechanism_allowed = False
        next_action = "complete or close Phase 130 before focused review"
    elif blockers:
        status = "phase131_matbench_log_kvrh_focused_review_closed_split_sensitivity_or_shortcut"
        mechanism_allowed = False
        next_action = "close the Phase 130 KVRH target as diagnostic; do not train"
    else:
        status = "phase131_matbench_log_kvrh_focused_review_ready_low_capacity_mechanism_gate"
        mechanism_allowed = True
        next_action = "design a separate no-training low-capacity KVRH mechanism gate; keep model training closed"
    return {
        "status": status,
        "phase130_status": phase130_gate.get("status"),
        "selected_target": phase130_gate.get("selected_target", "log10_k_vrh"),
        "phase130_selected_profile": phase130_gate.get("selected_profile"),
        "phase130_selected_method": phase130_gate.get("selected_method"),
        "viable_split_reviews": len(viable),
        "passed_split_reviews": len(passed),
        "split_pass_rate": len(passed) / len(viable) if viable else 0.0,
        "shortcut_dominant_splits": len(shortcut_dominant),
        "nearest_neighbor_dominant_splits": len(nearest_dominant),
        "target_distribution_imbalanced_splits": len(target_imbalanced),
        "blocking_audit_rows": len(blockers),
        "blocking_audits": [row["audit"] for row in blockers],
        "original_best_admissible_profile": original.get("best_admissible_profile"),
        "original_best_admissible_method": original.get("best_admissible_method"),
        "original_best_admissible_val_rmse": original.get("best_admissible_val_rmse"),
        "original_best_admissible_test_rmse": original.get("best_admissible_test_rmse"),
        "original_best_negative_profile": original.get("best_negative_profile"),
        "original_best_negative_method": original.get("best_negative_method"),
        "original_best_negative_val_rmse": original.get("best_negative_val_rmse"),
        "original_best_negative_test_rmse": original.get("best_negative_test_rmse"),
        "original_nearest_neighbor_val_rmse": original.get("nearest_neighbor_val_rmse"),
        "original_nearest_neighbor_test_rmse": original.get("nearest_neighbor_test_rmse"),
        "original_target_distribution_shift_z": original.get("target_distribution_shift_z"),
        "phase131_model_mechanism_allowed": mechanism_allowed,
        "phase131_low_capacity_mechanism_design_allowed": mechanism_allowed,
        "phase131_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": next_action,
    }


def build_markdown(gate: dict[str, Any], audit_rows: list[dict[str, Any]], split_rows: list[dict[str, Any]]) -> str:
    blocking = [row for row in audit_rows if row["status"] == "block"]
    viable = [row for row in split_rows if phase127._is_true(row.get("split_viable"))]
    lines = [
        "# Phase 131 Matbench Log KVRH Focused Review",
        "",
        f"- Status: `{gate['status']}`",
        f"- Split pass rate: `{gate['split_pass_rate']:.6g}`",
        f"- Blocking audits: `{', '.join(gate['blocking_audits']) or 'none'}`",
        f"- Low-capacity mechanism design allowed: `{gate['phase131_low_capacity_mechanism_design_allowed']}`",
        f"- Model training allowed: `{gate['phase131_model_training_allowed']}`",
        "",
        "## Blocking Audits",
        "",
        phase127._markdown_table(blocking, [("Audit", "audit"), ("Value", "value"), ("Threshold", "threshold"), ("Reason", "reason")]),
        "",
        "## Split Reviews",
        "",
        phase127._markdown_table(
            viable,
            [
                ("Split", "split_id"),
                ("Pass", "split_pass"),
                ("Best profile", "best_admissible_profile"),
                ("Val RMSE", "best_admissible_val_rmse"),
                ("Test RMSE", "best_admissible_test_rmse"),
                ("Shortcut", "negative_control_dominates"),
                ("NN", "nearest_neighbor_dominates"),
                ("Target shift", "target_distribution_shift_z"),
            ],
        ),
    ]
    return "\n".join(lines) + "\n"


def build_package(*, root: Path, phase130_dir: Path, output_dir: Path) -> dict[str, Any]:
    _configure_phase127_helpers()
    field_path = phase130_dir / "phase130_matbench_log_kvrh_field_table.csv"
    split_path = phase130_dir / "phase130_matbench_log_kvrh_split_manifest.json"
    gate_path = phase130_dir / "phase130_matbench_log_kvrh_gate.json"
    df = pd.read_csv(field_path)
    split_manifest = _read_json(split_path)
    phase130_gate = _read_json(gate_path)
    target = str(phase130_gate.get("selected_target") or "log10_k_vrh")

    split_infos = build_split_reviews(df, split_manifest)
    profile_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for split_id, split_info in split_infos.items():
        rows, summary = phase127.evaluate_split(df, target=target, split_id=split_id, split_info=split_info)
        profile_rows.extend(rows)
        split_rows.append(summary)
    audit_rows = build_audit_rows(phase130_gate=phase130_gate, split_rows=split_rows)
    gate = build_gate(phase130_gate=phase130_gate, split_rows=split_rows, audit_rows=audit_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "phase131_matbench_log_kvrh_focused_profile_table.csv"
    split_review_path = output_dir / "phase131_matbench_log_kvrh_split_sensitivity_table.csv"
    audit_path = output_dir / "phase131_matbench_log_kvrh_shortcut_audit_table.csv"
    split_manifest_path = output_dir / "phase131_matbench_log_kvrh_split_review_manifest.json"
    gate_out = output_dir / "phase131_matbench_log_kvrh_focused_review_gate.json"
    markdown_path = output_dir / "phase131_matbench_log_kvrh_focused_review.md"
    manifest_path = output_dir / "phase131_matbench_log_kvrh_focused_review_manifest.json"

    phase127._write_csv(profile_path, profile_rows, phase127.PROFILE_FIELDS)
    phase127._write_csv(split_review_path, split_rows, phase127.SPLIT_FIELDS)
    phase127._write_csv(audit_path, audit_rows, phase127.AUDIT_FIELDS)
    phase127._write_json(split_manifest_path, split_infos)
    phase127._write_json(gate_out, gate)
    markdown_path.write_text(build_markdown(gate, audit_rows, split_rows), encoding="utf-8")

    manifest = {
        "phase": 131,
        "objective": "matbench_log_kvrh_focused_split_shortcut_review_no_training",
        "inputs": {
            "phase130_dir": phase127._display_path(phase130_dir, root),
            "field_table": phase127._display_path(field_path, root),
            "split_manifest": phase127._display_path(split_path, root),
            "phase130_gate": phase127._display_path(gate_path, root),
        },
        "outputs": {
            "profile_table": phase127._display_path(profile_path, root),
            "split_sensitivity_table": phase127._display_path(split_review_path, root),
            "shortcut_audit_table": phase127._display_path(audit_path, root),
            "split_review_manifest": phase127._display_path(split_manifest_path, root),
            "gate_json": phase127._display_path(gate_out, root),
            "markdown": phase127._display_path(markdown_path, root),
            "manifest": phase127._display_path(manifest_path, root),
        },
        "counts": {
            "profile_rows": len(profile_rows),
            "split_reviews": len(split_rows),
            "viable_split_reviews": gate["viable_split_reviews"],
            "audit_rows": len(audit_rows),
            "blocking_audit_rows": gate["blocking_audit_rows"],
        },
        "gate": gate,
    }
    phase127._write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--phase130-dir", type=Path, default=DEFAULT_PHASE130_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    phase130_dir = args.phase130_dir if args.phase130_dir.is_absolute() else root / args.phase130_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = build_package(root=root, phase130_dir=phase130_dir, output_dir=output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
