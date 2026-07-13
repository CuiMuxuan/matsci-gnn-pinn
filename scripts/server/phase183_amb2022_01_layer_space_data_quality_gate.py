#!/usr/bin/env python3
"""Audit Phase 182 AMB2022-01 layer-space data before baseline design."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATASET = Path(
    os.environ.get(
        "AMB2022_01_PHASE182_DATASET",
        "/root/matsci-gnn-pinn-data/derived/ambench/2022_3d_build/AMB2022-01/"
        "phase182/phase182_layer_space_dataset.h5",
    )
)
DEFAULT_MANIFEST = Path(
    os.environ.get(
        "AMB2022_01_PHASE182_MANIFEST",
        "/root/matsci-gnn-pinn-ops/phase182_layer_space_dataset_manifest.json",
    )
)
EXPECTED_BUILD_TO_SPLIT = {"B6": "train", "B7": "val", "B8": "test"}
EXPECTED_SPLIT_CODES = {0: "train", 1: "val", 2: "test"}
MIN_ROWS_PER_SPLIT = 50_000
MIN_VALID_BOTH_FRACTION = 0.60
MIN_ACTIVE_LASER_FRACTION = 0.05
MAX_ACTIVE_LASER_FRACTION = 0.95


def _h5py() -> Any:
    try:
        import h5py
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("h5py is required for Phase 183 quality checks") from exc
    return h5py


def _decode_json_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def inspect_dataset(dataset_path: Path, manifest_path: Path) -> dict[str, Any]:
    h5py = _h5py()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with h5py.File(dataset_path, "r") as handle:
        features = handle["features"][...]
        tam = handle["target_tam_s"][...]
        scr = handle["target_scr_C_per_s"][...]
        valid = handle["target_valid_mask"][...].astype(bool)
        build = handle["build_index"][...]
        split = handle["primary_split"][...]
        feature_names = _decode_json_attr(handle.attrs["feature_names"])
        build_map = {int(key): value for key, value in _decode_json_attr(handle.attrs["build_index_map"]).items()}
        split_map = {int(key): value for key, value in _decode_json_attr(handle.attrs["primary_split_map"]).items()}
        scr_units = str(handle["target_scr_C_per_s"].attrs.get("units", ""))

    if len(features) != len(tam) or len(features) != len(scr) or len(features) != len(valid):
        raise ValueError("Dataset arrays do not share a row count")
    per_split: dict[str, dict[str, Any]] = {}
    for code, name in sorted(split_map.items()):
        rows = np.flatnonzero(split == code)
        if len(rows) == 0:
            per_split[name] = {"rows": 0, "build_ids": []}
            continue
        mask_tam = valid[rows, 0]
        mask_scr = valid[rows, 1]
        mask_both = mask_tam & mask_scr
        per_split[name] = {
            "rows": int(len(rows)),
            "build_ids": sorted({build_map[int(value)] for value in np.unique(build[rows])}),
            "valid_tam_fraction": float(np.mean(mask_tam)),
            "valid_scr_fraction": float(np.mean(mask_scr)),
            "valid_both_fraction": float(np.mean(mask_both)),
            "active_laser_fraction": float(np.mean(features[rows, feature_names.index("laser_active")])),
            "tam_std_valid": float(np.nanstd(tam[rows][mask_tam])) if np.any(mask_tam) else 0.0,
            "scr_std_valid": float(np.nanstd(scr[rows][mask_scr])) if np.any(mask_scr) else 0.0,
        }
    return {
        "dataset": str(dataset_path),
        "dataset_byte_size": dataset_path.stat().st_size,
        "manifest": str(manifest_path),
        "manifest_row_count": int(manifest["row_count"]),
        "dataset_row_count": int(len(features)),
        "feature_count": int(features.shape[1]),
        "feature_names": feature_names,
        "feature_matrix_finite": bool(np.isfinite(features).all()),
        "contains_build_identity_feature": any("build" in str(name).lower() for name in feature_names),
        "scr_units": scr_units,
        "build_map": build_map,
        "split_map": split_map,
        "manifest_primary_split": manifest.get("primary_split", {}),
        "per_split": per_split,
    }


def build_gate(inspected: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if inspected["dataset_row_count"] != inspected["manifest_row_count"]:
        blockers.append("manifest_row_count_mismatch")
    if not inspected["feature_matrix_finite"]:
        blockers.append("nonfinite_model_features")
    if inspected["contains_build_identity_feature"]:
        blockers.append("build_identity_feature_present")
    if inspected["scr_units"] != "C/s":
        blockers.append("scr_canonical_units_missing")
    manifest_split = inspected.get("manifest_primary_split", {})
    if manifest_split.get("strategy") != "build_holdout_replication":
        blockers.append("unexpected_primary_split_strategy")
    if manifest_split.get("build_identity_in_feature_matrix") is not False:
        blockers.append("manifest_does_not_exclude_build_identity")
    for build_id, split_name in EXPECTED_BUILD_TO_SPLIT.items():
        summary = inspected["per_split"].get(split_name, {})
        if int(summary.get("rows", 0)) < MIN_ROWS_PER_SPLIT:
            blockers.append(f"{split_name}_insufficient_rows")
            continue
        if summary.get("build_ids") != [build_id]:
            blockers.append(f"{split_name}_build_holdout_not_disjoint")
        if float(summary.get("valid_both_fraction", 0.0)) < MIN_VALID_BOTH_FRACTION:
            blockers.append(f"{split_name}_insufficient_dual_target_coverage")
        active_fraction = float(summary.get("active_laser_fraction", 0.0))
        if not MIN_ACTIVE_LASER_FRACTION <= active_fraction <= MAX_ACTIVE_LASER_FRACTION:
            blockers.append(f"{split_name}_scan_feature_support_invalid")
        if float(summary.get("tam_std_valid", 0.0)) <= 1e-6:
            blockers.append(f"{split_name}_tam_target_degenerate")
        if float(summary.get("scr_std_valid", 0.0)) <= 1.0:
            blockers.append(f"{split_name}_scr_target_degenerate")
    ready = not blockers
    return {
        "status": (
            "phase183_layer_space_data_quality_ready_phase184_baseline_design"
            if ready
            else "phase183_layer_space_data_quality_blocked"
        ),
        "data_quality_ready": ready,
        "phase184_baseline_design_allowed": ready,
        "model_training_allowed": False,
        "a800_training_allowed_now": False,
        "blocking_audits": blockers,
        "next_action": (
            "define leakage-controlled baseline models and an ablation contract before GPU fitting"
            if ready
            else "repair the dataset or its provenance before any model design"
        ),
    }


def run(dataset_path: Path, manifest_path: Path) -> dict[str, Any]:
    inspected = inspect_dataset(dataset_path, manifest_path)
    return {
        "phase": 183,
        "objective": "layer_space_dataset_quality_and_baseline_readiness",
        "inspection": inspected,
        "gate": build_gate(inspected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.dataset, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
