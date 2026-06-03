#!/usr/bin/env python3
"""Phase 105 NIST AMMT source/path feature gate.

Tests deterministic source/path and heat-kernel proxy features on the Phase 104
registered numeric table before any neural model training. This is a
low-capacity mechanism gate: it writes small CSV/JSON/Markdown artifacts and
keeps A100 training disabled.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from gnnpinn.eval.field_baseline import evaluate_table


TARGET = "target_intensity_std"
METHODS = ("knn", "extra_trees", "hist_gradient_boosting")
BASE_FEATURES = (
    "x",
    "y",
    "t",
    "source_p_mean",
    "source_p_nonzero_fraction",
    "source_x_range",
    "source_y_range",
    "target_camera_code",
)
PHYSICS_FEATURES = (
    "source_path_area_proxy",
    "source_path_diag_proxy",
    "source_energy_proxy",
    "source_energy_density_proxy",
    "source_time_density_proxy",
    "source_power_variation_proxy",
    "source_green_log_proxy",
    "source_heat_decay_layer_proxy",
    "source_kernel_compactness_proxy",
)
FEATURE_PROFILES: dict[str, tuple[str, ...]] = {
    "base_guard_replay": BASE_FEATURES,
    "base_energy": BASE_FEATURES + ("source_energy_proxy",),
    "base_density": BASE_FEATURES
    + ("source_energy_density_proxy", "source_time_density_proxy"),
    "base_green": BASE_FEATURES
    + ("source_green_log_proxy", "source_heat_decay_layer_proxy"),
    "base_all_physics": BASE_FEATURES + PHYSICS_FEATURES,
    "physics_only": PHYSICS_FEATURES + ("target_camera_code", "t"),
}
METRIC_FIELDS = (
    "feature_profile",
    "method",
    "baseline",
    "split",
    "n_points",
    "rmse",
    "mae",
    "relative_l2",
    "normalized_rmse",
    "hot_q90_rmse",
    "gradient_q90_rmse",
)
REVIEW_FIELDS = (
    "feature_profile",
    "selected_validation_method",
    "selected_validation_rmse",
    "selected_test_rmse",
    "selected_validation_hot_q90_rmse",
    "selected_test_hot_q90_rmse",
    "selected_validation_gradient_q90_rmse",
    "selected_test_gradient_q90_rmse",
    "guard_validation_rmse",
    "guard_test_rmse",
    "validation_improvement_over_guard",
    "test_improvement_over_guard",
    "validation_improves_guard",
    "test_improves_guard",
    "status",
    "phase105_candidate",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.6f}"
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _display_path(path: Path, root: Path | None = None) -> str:
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def build_augmented_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    for row in rows:
        x_range = _float(row, "source_x_range")
        y_range = _float(row, "source_y_range")
        p_mean = _float(row, "source_p_mean")
        p_fraction = _float(row, "source_p_nonzero_fraction")
        p_range = _float(row, "source_p_range")
        t_range = _float(row, "source_t_range")
        layer = max(_float(row, "source_layer_index"), 1.0)
        area = max(x_range * y_range, 1e-12)
        diag = max(math.hypot(x_range, y_range), 1e-12)
        active_time = max(t_range, 1e-12)
        active_power = p_mean * p_fraction
        energy = active_power * active_time
        augmented.append(
            {
                **row,
                "source_path_area_proxy": area,
                "source_path_diag_proxy": diag,
                "source_energy_proxy": energy,
                "source_energy_density_proxy": energy / area,
                "source_time_density_proxy": active_time / area,
                "source_power_variation_proxy": p_range * max(p_fraction, 1e-12),
                "source_green_log_proxy": energy * math.log1p(diag) / area,
                "source_heat_decay_layer_proxy": energy / math.sqrt(layer),
                "source_kernel_compactness_proxy": area / (diag * diag),
            }
        )
    return augmented


def _region_rmse(split_payload: dict[str, Any], name: str) -> float | None:
    region = (split_payload.get("region_metrics") or {}).get(name) or {}
    metrics = region.get("metrics") or {}
    value = metrics.get("rmse")
    return float(value) if isinstance(value, (int, float)) else None


def _metric_rows(payloads: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile, profile_payloads in payloads.items():
        for method, payload in profile_payloads.items():
            result = payload["results"][0]
            for split, split_payload in result["split_metrics"].items():
                metrics = split_payload["metrics"]
                rows.append(
                    {
                        "feature_profile": profile,
                        "method": method,
                        "baseline": result["baseline"],
                        "split": split,
                        "n_points": split_payload["n_points"],
                        "rmse": metrics.get("rmse"),
                        "mae": metrics.get("mae"),
                        "relative_l2": metrics.get("relative_l2"),
                        "normalized_rmse": metrics.get("normalized_rmse"),
                        "hot_q90_rmse": _region_rmse(split_payload, "hot_q90"),
                        "gradient_q90_rmse": _region_rmse(split_payload, "gradient_q90"),
                    }
                )
    return rows


def _metric_index(metric_rows: list[dict[str, Any]], metric: str) -> dict[tuple[str, str, str], float]:
    index: dict[tuple[str, str, str], float] = {}
    for row in metric_rows:
        value = row.get(metric)
        if isinstance(value, (int, float)):
            index[(str(row["feature_profile"]), str(row["method"]), str(row["split"]))] = float(value)
    return index


def _review_rows(
    *,
    metric_rows: list[dict[str, Any]],
    guard_validation_rmse: float,
    guard_test_rmse: float,
    min_validation_improvement: float,
) -> list[dict[str, Any]]:
    rmse = _metric_index(metric_rows, "rmse")
    hot = _metric_index(metric_rows, "hot_q90_rmse")
    grad = _metric_index(metric_rows, "gradient_q90_rmse")
    rows: list[dict[str, Any]] = []
    for profile in FEATURE_PROFILES:
        candidates = [
            (rmse[(profile, method, "val")], method)
            for method in METHODS
            if (profile, method, "val") in rmse
        ]
        if not candidates:
            raise ValueError(f"No validation metrics found for feature profile {profile}")
        selected_val, selected_method = min(candidates)
        selected_test = rmse[(profile, selected_method, "test")]
        validation_delta = guard_validation_rmse - selected_val
        test_delta = guard_test_rmse - selected_test
        validation_improves = validation_delta >= min_validation_improvement
        test_improves = test_delta >= 0.0
        if profile == "base_guard_replay":
            status = "guard_replay_reference"
            phase105_candidate = False
        elif validation_improves and test_improves:
            status = "candidate_feature_profile_ready_for_phase105_smoke"
            phase105_candidate = True
        elif not validation_improves:
            status = "blocked_no_validation_gain_over_hgb_guard"
            phase105_candidate = False
        else:
            status = "blocked_test_reversal_against_hgb_guard"
            phase105_candidate = False
        rows.append(
            {
                "feature_profile": profile,
                "selected_validation_method": selected_method,
                "selected_validation_rmse": selected_val,
                "selected_test_rmse": selected_test,
                "selected_validation_hot_q90_rmse": hot.get((profile, selected_method, "val")),
                "selected_test_hot_q90_rmse": hot.get((profile, selected_method, "test")),
                "selected_validation_gradient_q90_rmse": grad.get((profile, selected_method, "val")),
                "selected_test_gradient_q90_rmse": grad.get((profile, selected_method, "test")),
                "guard_validation_rmse": guard_validation_rmse,
                "guard_test_rmse": guard_test_rmse,
                "validation_improvement_over_guard": validation_delta,
                "test_improvement_over_guard": test_delta,
                "validation_improves_guard": validation_improves,
                "test_improves_guard": test_improves,
                "status": status,
                "phase105_candidate": phase105_candidate,
            }
        )
    return rows


def _build_gate(
    *,
    target_hardness_gate: dict[str, Any],
    review_rows: list[dict[str, Any]],
    min_validation_improvement: float,
) -> dict[str, Any]:
    candidates = [row for row in review_rows if row["phase105_candidate"]]
    candidates.sort(key=lambda row: float(row["selected_validation_rmse"]))
    selected = candidates[0] if candidates else None
    target_ready = bool(target_hardness_gate.get("phase105_model_mechanism_allowed"))
    if not target_ready:
        status = "phase105_source_path_gate_blocked_by_phase104_target_review"
        next_action = "complete Phase 104 target-hardness review first"
    elif selected:
        status = "phase105_source_path_feature_gate_ready_for_cpu_smoke"
        next_action = (
            "run CPU-small low-capacity mechanism smoke using "
            f"{selected['feature_profile']} on {TARGET}"
        )
    else:
        status = "phase105_source_path_feature_gate_blocked_no_hgb_gain"
        next_action = "close deterministic source/path proxy features as diagnostic or refine registered targets"
    return {
        "status": status,
        "target": TARGET,
        "phase104_selected_target": target_hardness_gate.get("selected_target"),
        "phase104_selected_validation_method": target_hardness_gate.get(
            "selected_validation_method"
        ),
        "phase104_selected_validation_rmse": target_hardness_gate.get(
            "selected_validation_rmse"
        ),
        "phase104_selected_test_rmse": target_hardness_gate.get("selected_test_rmse"),
        "candidate_feature_profiles": [row["feature_profile"] for row in candidates],
        "selected_feature_profile": selected["feature_profile"] if selected else None,
        "selected_validation_method": selected["selected_validation_method"] if selected else None,
        "selected_validation_rmse": selected["selected_validation_rmse"] if selected else None,
        "selected_test_rmse": selected["selected_test_rmse"] if selected else None,
        "min_validation_improvement": min_validation_improvement,
        "phase105_cpu_smoke_allowed": bool(selected and target_ready),
        "phase105_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": next_action,
    }


def _write_markdown(path: Path, gate: dict[str, Any], review_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase 105 NIST AMMT Source/Path Feature Gate",
        "",
        f"- Status: `{gate['status']}`",
        f"- Target: `{gate['target']}`",
        f"- Phase 104 guard: `{gate['phase104_selected_validation_method']}` val RMSE `{gate['phase104_selected_validation_rmse']}`",
        f"- Selected feature profile: `{gate['selected_feature_profile']}`",
        f"- CPU smoke allowed: `{gate['phase105_cpu_smoke_allowed']}`",
        "- A100 training allowed now: `false`",
        "",
        "| Feature profile | Status | Method | Val RMSE | Test RMSE | Val gain vs guard | Test gain vs guard |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in review_rows:
        lines.append(
            "| {feature_profile} | {status} | {selected_validation_method} | {selected_validation_rmse} | {selected_test_rmse} | {validation_improvement_over_guard} | {test_improvement_over_guard} |".format(
                **{key: _csv_value(value) for key, value in row.items()}
            )
        )
    lines.extend(["", f"Next action: {gate['next_action']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_package(
    *,
    root: Path,
    field_table: Path,
    split_manifest: Path,
    target_hardness_gate_path: Path,
    output_dir: Path,
    min_validation_improvement: float,
    n_neighbors: int,
    n_estimators: int,
) -> dict[str, Any]:
    target_hardness_gate = _read_json(target_hardness_gate_path)
    rows = build_augmented_rows(_read_csv(field_table))

    output_dir.mkdir(parents=True, exist_ok=True)
    augmented_path = output_dir / "phase105_nist_ammt_source_path_augmented_field_table.csv"
    fieldnames = tuple(rows[0].keys())
    _write_csv(augmented_path, rows, fieldnames)

    payloads: dict[str, dict[str, dict[str, Any]]] = {}
    for profile, features in FEATURE_PROFILES.items():
        payloads[profile] = {}
        for method in METHODS:
            payloads[profile][method] = {
                "target": TARGET,
                "strategy": method,
                "feature_profile": profile,
                "feature_columns": list(features),
                "split_manifest": str(split_manifest),
                "fit_split": "train",
                "results": [
                    evaluate_table(
                        table_path=augmented_path,
                        target=TARGET,
                        strategy=method,
                        split_manifest_path=split_manifest,
                        fit_split="train",
                        feature_columns=list(features),
                        n_neighbors=n_neighbors,
                        n_estimators=n_estimators,
                        random_state=7,
                        hot_quantiles=[0.9],
                        gradient_quantiles=[0.9],
                    )
                ],
            }
    metric_rows = _metric_rows(payloads)
    guard_validation = float(target_hardness_gate["selected_validation_rmse"])
    guard_test = float(target_hardness_gate["selected_test_rmse"])
    review_rows = _review_rows(
        metric_rows=metric_rows,
        guard_validation_rmse=guard_validation,
        guard_test_rmse=guard_test,
        min_validation_improvement=min_validation_improvement,
    )
    gate = _build_gate(
        target_hardness_gate=target_hardness_gate,
        review_rows=review_rows,
        min_validation_improvement=min_validation_improvement,
    )

    metrics_path = output_dir / "phase105_nist_ammt_source_path_metric_table.csv"
    review_path = output_dir / "phase105_nist_ammt_source_path_feature_review_table.csv"
    gate_path = output_dir / "phase105_nist_ammt_source_path_feature_gate.json"
    markdown_path = output_dir / "phase105_nist_ammt_source_path_feature_summary.md"
    manifest_path = output_dir / "phase105_nist_ammt_source_path_feature_manifest.json"
    payload_path = output_dir / "phase105_nist_ammt_source_path_feature_payloads.json"
    _write_csv(metrics_path, metric_rows, METRIC_FIELDS)
    _write_csv(review_path, review_rows, REVIEW_FIELDS)
    _write_json(gate_path, gate)
    _write_json(payload_path, payloads)
    _write_markdown(markdown_path, gate, review_rows)
    manifest = {
        "phase": 105,
        "objective": "nist_ammt_source_path_heat_kernel_proxy_feature_gate_no_training",
        "inputs": {
            "field_table": _display_path(field_table, root),
            "split_manifest": _display_path(split_manifest, root),
            "target_hardness_gate": _display_path(target_hardness_gate_path, root),
        },
        "outputs": {
            "augmented_field_table": _display_path(augmented_path, root),
            "metric_table": _display_path(metrics_path, root),
            "review_table": _display_path(review_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown_summary": _display_path(markdown_path, root),
            "payloads": _display_path(payload_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "rows": len(rows),
            "feature_profiles": len(FEATURE_PROFILES),
            "metric_rows": len(metric_rows),
            "candidate_feature_profiles": len(gate["candidate_feature_profiles"]),
        },
        "gate": gate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--field-table",
        type=Path,
        default=Path(
            "docs/results/phase104_nist_ammt_baseline_smoke/"
            "phase104_nist_ammt_tiny_numeric_field_table.csv"
        ),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path(
            "docs/results/phase104_nist_ammt_baseline_smoke/"
            "phase104_nist_ammt_tiny_numeric_split_manifest.json"
        ),
    )
    parser.add_argument(
        "--target-hardness-gate",
        type=Path,
        default=Path(
            "docs/results/phase104_nist_ammt_target_hardness_review/"
            "phase104_nist_ammt_target_hardness_review_gate.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase105_nist_ammt_source_path_feature_gate"),
    )
    parser.add_argument("--min-validation-improvement", type=float, default=0.005)
    parser.add_argument("--n-neighbors", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    field_table = args.field_table if args.field_table.is_absolute() else root / args.field_table
    split_manifest = (
        args.split_manifest if args.split_manifest.is_absolute() else root / args.split_manifest
    )
    target_hardness_gate = (
        args.target_hardness_gate
        if args.target_hardness_gate.is_absolute()
        else root / args.target_hardness_gate
    )
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = build_package(
        root=root,
        field_table=field_table,
        split_manifest=split_manifest,
        target_hardness_gate_path=target_hardness_gate,
        output_dir=output_dir,
        min_validation_improvement=args.min_validation_improvement,
        n_neighbors=args.n_neighbors,
        n_estimators=args.n_estimators,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
