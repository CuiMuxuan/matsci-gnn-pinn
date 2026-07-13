#!/usr/bin/env python3
"""Produce a post-B8 spatial error atlas from the frozen AMB2022-01 checkpoints."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PHASE189 = Path(
    os.environ.get(
        "AMB2022_01_PHASE189_REVIEW",
        "/root/matsci-gnn-pinn-ops/phase189_replicate_stability_review.json",
    )
)
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
DEFAULT_DATASET = Path(
    os.environ.get(
        "AMB2022_01_PHASE182_DATASET",
        "/root/matsci-gnn-pinn-data/derived/ambench/2022_3d_build/AMB2022-01/"
        "phase182/phase182_layer_space_dataset.h5",
    )
)
DEFAULT_KERNEL = Path(
    os.environ.get(
        "AMB2022_01_PHASE186_KERNEL",
        "/root/matsci-gnn-pinn-data/derived/ambench/2022_3d_build/AMB2022-01/"
        "phase186/causal_heat_kernel_features.h5",
    )
)
DEFAULT_CHECKPOINT_DIR = Path(
    os.environ.get(
        "AMB2022_01_PHASE188_CHECKPOINT_DIR",
        "/root/matsci-gnn-pinn-data/derived/ambench/2022_3d_build/AMB2022-01/phase188/checkpoints",
    )
)
DEFAULT_PHASE188_SCRIPT = Path(__file__).with_name("phase188_amb2022_01_bounded_gpu_training.py")

TARGETS = ("tam_s", "scr_C_per_s")
SEEDS = (1871, 1872, 1873)
DATA_ONLY_VARIANT = "small_data_only_mlp"
CANDIDATE_VARIANT = "physics_regularized_history_mlp"
VARIANTS = (DATA_ONLY_VARIANT, CANDIDATE_VARIANT)
ERROR_FIELDS = (
    "target",
    "stratum_family",
    "stratum_value",
    "n_valid",
    "data_only_ensemble_rmse",
    "candidate_ensemble_rmse",
    "candidate_gain_vs_data_only_rmse",
    "candidate_better",
)
LAYER_FIELDS = (
    "target",
    "layer_index",
    "n_valid",
    "data_only_ensemble_rmse",
    "candidate_ensemble_rmse",
    "candidate_gain_vs_data_only_rmse",
    "candidate_better",
)
CELL_FIELDS = (
    "target",
    "block_row",
    "block_col",
    "n_valid",
    "data_only_ensemble_rmse",
    "candidate_ensemble_rmse",
    "candidate_gain_vs_data_only_rmse",
    "candidate_better",
)
VERIFICATION_FIELDS = (
    "variant_id",
    "seed",
    "target",
    "n_valid",
    "phase188_reported_rmse",
    "recomputed_rmse",
    "absolute_difference",
    "matches_phase188_rmse",
)


def _h5py() -> Any:
    try:
        import h5py
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("h5py is required for Phase 190") from exc
    return h5py


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_phase188_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("phase188_for_phase190", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Phase 188 trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rmse(target: np.ndarray, prediction: np.ndarray, valid: np.ndarray) -> float | None:
    if not np.any(valid):
        return None
    residual = target[valid].astype(np.float64) - prediction[valid].astype(np.float64)
    return float(math.sqrt(np.mean(residual**2)))


def _error_values(
    *,
    target: np.ndarray,
    data_only_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any] | None:
    data_only_rmse = _rmse(target, data_only_prediction, valid)
    candidate_rmse = _rmse(target, candidate_prediction, valid)
    if data_only_rmse is None or candidate_rmse is None:
        return None
    gain = data_only_rmse - candidate_rmse
    return {
        "n_valid": int(np.sum(valid)),
        "data_only_ensemble_rmse": data_only_rmse,
        "candidate_ensemble_rmse": candidate_rmse,
        "candidate_gain_vs_data_only_rmse": gain,
        "candidate_better": gain > 0.0,
    }


def _layer_quartiles(layer_index: np.ndarray) -> np.ndarray:
    lower = int(np.min(layer_index))
    span = max(1, int(np.max(layer_index)) - lower + 1)
    quartile_index = np.minimum(3, ((layer_index.astype(np.int64) - lower) * 4) // span)
    labels = np.asarray(("Q1_lower", "Q2_lower_mid", "Q3_upper_mid", "Q4_upper"), dtype=object)
    return labels[quartile_index]


def build_spatial_rows(
    *,
    targets: np.ndarray,
    valid_mask: np.ndarray,
    data_only_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    layer_index: np.ndarray,
    block_row: np.ndarray,
    block_col: np.ndarray,
    laser_active: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if targets.shape != valid_mask.shape or targets.shape != data_only_prediction.shape or targets.shape != candidate_prediction.shape:
        raise ValueError("Targets, masks, and predictions must have identical shapes")
    n_rows = len(targets)
    if any(len(values) != n_rows for values in (layer_index, block_row, block_col, laser_active)):
        raise ValueError("Spatial metadata must align with the prediction rows")
    stratum_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    layer_labels = _layer_quartiles(layer_index)
    laser_labels = np.where(laser_active > 0.5, "laser_active", "laser_inactive")
    edge = (
        (block_row == np.min(block_row))
        | (block_row == np.max(block_row))
        | (block_col == np.min(block_col))
        | (block_col == np.max(block_col))
    )
    spatial_labels = np.where(edge, "edge", "interior")
    strata = (
        ("layer_quartile", layer_labels),
        ("laser_state", laser_labels),
        ("spatial_region", spatial_labels),
    )
    for target_index, target_name in enumerate(TARGETS):
        target = targets[:, target_index]
        valid = valid_mask[:, target_index]
        baseline = data_only_prediction[:, target_index]
        candidate = candidate_prediction[:, target_index]
        for family, labels in strata:
            for label in sorted(set(labels.tolist())):
                values = _error_values(
                    target=target,
                    data_only_prediction=baseline,
                    candidate_prediction=candidate,
                    valid=valid & (labels == label),
                )
                if values is not None:
                    stratum_rows.append(
                        {
                            "target": target_name,
                            "stratum_family": family,
                            "stratum_value": str(label),
                            **values,
                        }
                    )
        for layer in sorted(int(value) for value in np.unique(layer_index)):
            values = _error_values(
                target=target,
                data_only_prediction=baseline,
                candidate_prediction=candidate,
                valid=valid & (layer_index == layer),
            )
            if values is not None:
                layer_rows.append({"target": target_name, "layer_index": layer, **values})
        cells = sorted({(int(row), int(col)) for row, col in zip(block_row, block_col, strict=True)})
        for row, col in cells:
            values = _error_values(
                target=target,
                data_only_prediction=baseline,
                candidate_prediction=candidate,
                valid=valid & (block_row == row) & (block_col == col),
            )
            if values is not None:
                cell_rows.append({"target": target_name, "block_row": row, "block_col": col, **values})
    return stratum_rows, layer_rows, cell_rows


def _phase188_metric(
    metric_rows: list[dict[str, str]], *, variant_id: str, seed: int, target: str
) -> float | None:
    subset = [
        row
        for row in metric_rows
        if row.get("variant_id") == variant_id
        and row.get("seed") == str(seed)
        and row.get("target") == target
        and row.get("split") == "test"
    ]
    if len(subset) != 1:
        return None
    try:
        return float(subset[0]["rmse"])
    except (KeyError, TypeError, ValueError):
        return None


def _load_test_metadata(dataset_path: Path) -> dict[str, np.ndarray]:
    h5py = _h5py()
    with h5py.File(dataset_path, "r") as handle:
        split_codes = handle["primary_split"][...]
        test_rows = split_codes == 2
        feature_names_value = handle.attrs["feature_names"]
        if isinstance(feature_names_value, bytes):
            feature_names_value = feature_names_value.decode("utf-8")
        feature_names = json.loads(str(feature_names_value))
        feature_matrix = handle["features"][test_rows]
        return {
            "layer_index": handle["layer_index"][...][test_rows].astype(np.int64),
            "block_row": handle["block_row"][...][test_rows].astype(np.int64),
            "block_col": handle["block_col"][...][test_rows].astype(np.int64),
            "laser_active": feature_matrix[:, feature_names.index("laser_active")].astype(np.float32),
        }


def reconstruct_ensembles(
    *,
    phase188_script: Path,
    dataset_path: Path,
    kernel_path: Path,
    checkpoint_dir: Path,
    metric_rows: list[dict[str, str]],
    device_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    phase188 = _load_phase188_module(phase188_script)
    torch, _, _, _ = phase188._torch()
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    prepared_data: dict[str, Any] = {}
    for variant_id, use_kernel in ((DATA_ONLY_VARIANT, False), (CANDIDATE_VARIANT, True)):
        prepared_data[variant_id] = phase188.prepare_data(dataset_path, kernel_path, use_kernel=use_kernel)
    baseline_data = prepared_data[DATA_ONLY_VARIANT]
    test_rows = baseline_data.split_codes == phase188.SPLITS["test"]
    if not np.any(test_rows):
        raise ValueError("Phase 182 dataset has no B8 test rows")
    target_values = baseline_data.targets[test_rows] * baseline_data.target_scales + baseline_data.target_means
    valid_mask = baseline_data.masks[test_rows]
    verification_rows: list[dict[str, Any]] = []
    ensembles: dict[str, np.ndarray] = {}
    for variant_id in VARIANTS:
        data = prepared_data[variant_id]
        if not np.array_equal(data.split_codes, baseline_data.split_codes) or not np.array_equal(data.masks, baseline_data.masks):
            raise ValueError("Variant preprocessing changed the registered split or target mask")
        seed_predictions: list[np.ndarray] = []
        for seed in SEEDS:
            checkpoint_path = checkpoint_dir / f"{variant_id}_seed{seed}.pt"
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            if checkpoint.get("variant_id") != variant_id or int(checkpoint.get("seed")) != seed:
                raise ValueError(f"Checkpoint identity mismatch: {checkpoint_path}")
            if tuple(checkpoint.get("feature_names", [])) != tuple(data.feature_names):
                raise ValueError(f"Checkpoint feature contract mismatch: {checkpoint_path}")
            model = phase188._make_model(data.features.shape[1]).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            standardized_prediction = phase188._predict(model, data.features[test_rows], device)
            prediction = standardized_prediction * data.target_scales + data.target_means
            seed_predictions.append(prediction.astype(np.float32))
            for target_index, target_name in enumerate(TARGETS):
                recomputed = _rmse(target_values[:, target_index], prediction[:, target_index], valid_mask[:, target_index])
                reported = _phase188_metric(
                    metric_rows,
                    variant_id=variant_id,
                    seed=seed,
                    target=target_name,
                )
                difference = abs(recomputed - reported) if recomputed is not None and reported is not None else None
                verification_rows.append(
                    {
                        "variant_id": variant_id,
                        "seed": seed,
                        "target": target_name,
                        "n_valid": int(np.sum(valid_mask[:, target_index])),
                        "phase188_reported_rmse": reported,
                        "recomputed_rmse": recomputed,
                        "absolute_difference": difference,
                        "matches_phase188_rmse": bool(
                            recomputed is not None
                            and reported is not None
                            and math.isclose(recomputed, reported, rel_tol=1e-5, abs_tol=1e-8)
                        ),
                    }
                )
            del model
        ensembles[variant_id] = np.mean(np.stack(seed_predictions, axis=0), axis=0).astype(np.float32)
    return target_values, valid_mask, ensembles[DATA_ONLY_VARIANT], ensembles[CANDIDATE_VARIANT], verification_rows


def _phase189_ready(phase189: dict[str, Any]) -> bool:
    gate = phase189.get("gate", {})
    return (
        gate.get("status") == "phase189_replicate_stability_review_ready_phase190_spatial_failure_analysis"
        and bool(gate.get("phase190_spatial_failure_analysis_allowed"))
        and gate.get("model_training_allowed") is False
        and gate.get("post_b8_model_reselection_allowed") is False
    )


def _candidate_better_fraction(rows: list[dict[str, Any]], target: str) -> float | None:
    subset = [row for row in rows if row["target"] == target]
    if not subset:
        return None
    return float(sum(bool(row["candidate_better"]) for row in subset) / len(subset))


def build_gate(
    *,
    phase189: dict[str, Any],
    verification_rows: list[dict[str, Any]],
    stratum_rows: list[dict[str, Any]],
    layer_rows: list[dict[str, Any]],
    cell_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    blockers: list[str] = []
    if not _phase189_ready(phase189):
        blockers.append("phase189_stability_gate_not_ready")
    expected_verification = {(variant, seed, target) for variant in VARIANTS for seed in SEEDS for target in TARGETS}
    observed_verification = {
        (str(row.get("variant_id")), int(row.get("seed")), str(row.get("target")))
        for row in verification_rows
    }
    if len(verification_rows) != len(expected_verification) or observed_verification != expected_verification:
        blockers.append("checkpoint_prediction_verification_incomplete")
    if not all(bool(row.get("matches_phase188_rmse")) for row in verification_rows):
        blockers.append("checkpoint_prediction_rmse_mismatch")
    for target in TARGETS:
        for family in ("layer_quartile", "laser_state", "spatial_region"):
            if not any(row["target"] == target and row["stratum_family"] == family for row in stratum_rows):
                blockers.append(f"{target}_{family}_summary_missing")
        if not any(row["target"] == target for row in layer_rows):
            blockers.append(f"{target}_layer_summary_missing")
        if not any(row["target"] == target for row in cell_rows):
            blockers.append(f"{target}_cell_summary_missing")
    blockers = sorted(set(blockers))
    ready = not blockers
    return {
        "status": (
            "phase190_spatial_failure_analysis_ready_phase191_external_confirmation_design"
            if ready
            else "phase190_spatial_failure_analysis_incomplete_or_unverified"
        ),
        "phase191_external_confirmation_design_allowed": ready,
        "model_training_allowed": False,
        "hyperparameter_search_allowed": False,
        "post_b8_model_reselection_allowed": False,
        "a800_training_allowed_now": False,
        "candidate_better_layer_fraction": {
            target: _candidate_better_fraction(layer_rows, target) for target in TARGETS
        },
        "candidate_better_cell_fraction": {
            target: _candidate_better_fraction(cell_rows, target) for target in TARGETS
        },
        "spatial_error_stratification_descriptive_only": True,
        "component_attribution_claim_allowed": False,
        "external_generalization_claim_allowed": False,
        "absolute_wall_clock_trajectory_claim_allowed": False,
        "raw_frame_event_causal_training_allowed": False,
        "blocking_audits": blockers,
        "next_action": (
            "freeze the error atlas and design an independent-build confirmation protocol; do not tune after B8"
            if ready
            else "repair missing prediction verification or spatial summaries without model reselection"
        ),
    }


def build_analysis(
    *,
    phase189: dict[str, Any],
    phase188: dict[str, Any],
    target_values: np.ndarray,
    valid_mask: np.ndarray,
    data_only_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    metadata: dict[str, np.ndarray],
    verification_rows: list[dict[str, Any]],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    stratum_rows, layer_rows, cell_rows = build_spatial_rows(
        targets=target_values,
        valid_mask=valid_mask,
        data_only_prediction=data_only_prediction,
        candidate_prediction=candidate_prediction,
        layer_index=metadata["layer_index"],
        block_row=metadata["block_row"],
        block_col=metadata["block_col"],
        laser_active=metadata["laser_active"],
    )
    gate = build_gate(
        phase189=phase189,
        verification_rows=verification_rows,
        stratum_rows=stratum_rows,
        layer_rows=layer_rows,
        cell_rows=cell_rows,
    )
    return {
        "phase": 190,
        "objective": "post_b8_registered_layer_space_spatial_failure_analysis",
        "ensemble_policy": "Unweighted mean of the three frozen-seed checkpoint predictions; descriptive only.",
        "phase188_training_status": phase188.get("gate", {}).get("status"),
        "spatial_scope": "B8 registered per-layer block grid only; no raw-frame or absolute-clock inference.",
        "prediction_verification": verification_rows,
        "stratum_summary": stratum_rows,
        "layer_summary_count": len(layer_rows),
        "cell_summary_count": len(cell_rows),
        "gate": gate,
    }, stratum_rows, layer_rows, cell_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase189", type=Path, default=DEFAULT_PHASE189)
    parser.add_argument("--phase188", type=Path, default=DEFAULT_PHASE188)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--kernel", type=Path, default=DEFAULT_KERNEL)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--phase188-script", type=Path, default=DEFAULT_PHASE188_SCRIPT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    phase189 = _read_json(args.phase189)
    phase188 = _read_json(args.phase188)
    metric_rows = _read_csv(args.metrics)
    target_values, valid_mask, data_only_prediction, candidate_prediction, verification_rows = reconstruct_ensembles(
        phase188_script=args.phase188_script,
        dataset_path=args.dataset,
        kernel_path=args.kernel,
        checkpoint_dir=args.checkpoint_dir,
        metric_rows=metric_rows,
        device_name=args.device,
    )
    metadata = _load_test_metadata(args.dataset)
    if len(metadata["layer_index"]) != len(target_values):
        raise ValueError("B8 spatial metadata does not align with reconstructed predictions")
    payload, stratum_rows, layer_rows, cell_rows = build_analysis(
        phase189=phase189,
        phase188=phase188,
        target_values=target_values,
        valid_mask=valid_mask,
        data_only_prediction=data_only_prediction,
        candidate_prediction=candidate_prediction,
        metadata=metadata,
        verification_rows=verification_rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(args.output_dir / "phase190_prediction_verification.csv", verification_rows, VERIFICATION_FIELDS)
    _write_csv(args.output_dir / "phase190_stratum_summary.csv", stratum_rows, ERROR_FIELDS)
    _write_csv(args.output_dir / "phase190_layer_summary.csv", layer_rows, LAYER_FIELDS)
    _write_csv(args.output_dir / "phase190_cell_summary.csv", cell_rows, CELL_FIELDS)
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
