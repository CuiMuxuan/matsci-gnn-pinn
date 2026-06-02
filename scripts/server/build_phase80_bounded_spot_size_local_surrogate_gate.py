#!/usr/bin/env python3
"""Build the Phase 80 bounded spot-size local surrogate gate.

Phase 80 is the local/no-training follow-up required by Phase 79. It evaluates
low-capacity bounded surrogate variants already present in the Phase 59
no-test-leakage upper-bound summary. Variant selection uses validation RMSE
only; test metrics are reported after selection as gate evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CANDIDATE_VARIANTS = (
    "broad_process_v1:train_global_bias",
    "broad_process_v1:train_group_bias:spot_size_um",
    "broad_process_v1:train_group_bias:process_tuple",
)
IDENTITY_VARIANT = "broad_process_v1:identity"
METRICS = ("rmse", "hot_q90_rmse", "gradient_q90_rmse")
MIN_VALIDATION_RMSE_GAIN = 1.0

SURROGATE_FIELDS = (
    "variant",
    "role",
    "selection_split",
    "analysis_split",
    "val_rmse",
    "val_gain_vs_identity",
    "val_gain_vs_reference",
    "test_rmse",
    "test_hot_q90_rmse",
    "test_gradient_q90_rmse",
    "test_gain_vs_identity_rmse",
    "test_gain_vs_reference_rmse",
    "test_preserves_identity_regions",
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
        "phase59_upper": root
        / "docs/results/phase59_residual_anatomy/phase59_broad21_density_residual_upper_bound.json",
        "phase79_manifest": root
        / "docs/results/phase79_bounded_spot_size_parameterization_gate/phase79_bounded_spot_size_parameterization_gate_manifest.json",
    }


def _variant_map(upper: dict[str, Any]) -> dict[str, dict[str, Any]]:
    variants: dict[str, dict[str, Any]] = {}
    for variant in upper.get("variants", []):
        name = str(variant.get("name") or "")
        if name:
            variants[name] = variant
    return variants


def _metrics(variant: dict[str, Any], split: str) -> dict[str, float]:
    payload = ((variant.get("metrics") or {}).get(split) or {})
    return {metric: float(payload[metric]) for metric in METRICS if metric in payload}


def _baseline_variant(name: str, metrics_by_split: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "metrics": {
            split: dict(payload)
            for split, payload in metrics_by_split.items()
            if isinstance(payload, dict)
        },
    }


def _row_for_variant(
    variant: dict[str, Any],
    *,
    role: str,
    identity: dict[str, Any],
    reference: dict[str, Any],
    selection_split: str,
    analysis_split: str,
) -> dict[str, Any]:
    val = _metrics(variant, selection_split)
    test = _metrics(variant, analysis_split)
    identity_val = _metrics(identity, selection_split)
    identity_test = _metrics(identity, analysis_split)
    reference_val = _metrics(reference, selection_split)
    reference_test = _metrics(reference, analysis_split)
    preserves_identity_regions = all(
        test.get(metric, float("inf")) <= identity_test.get(metric, float("-inf"))
        for metric in ("hot_q90_rmse", "gradient_q90_rmse")
    )
    val_gain = identity_val["rmse"] - val["rmse"]
    reference_val_gain = reference_val["rmse"] - val["rmse"]
    test_gain = identity_test["rmse"] - test["rmse"]
    reference_test_gain = reference_test["rmse"] - test["rmse"]
    if role == "candidate" and val_gain < MIN_VALIDATION_RMSE_GAIN:
        status = "insufficient_validation_gain"
        interpretation = "validation gain over identity is below the pre-declared minimum"
    elif role == "candidate" and not preserves_identity_regions:
        status = "region_regression_vs_identity"
        interpretation = "candidate worsens hot or gradient q90 regions versus identity"
    elif role == "candidate" and reference_test_gain < 0.0:
        status = "fails_strong_reference_on_analysis"
        interpretation = "candidate improves identity but remains worse than the mean reference"
    elif role == "candidate":
        status = "passes_local_surrogate"
        interpretation = "candidate clears validation gain, region preservation, and reference checks"
    else:
        status = "reference"
        interpretation = "control row"
    return {
        "variant": str(variant.get("name")),
        "role": role,
        "selection_split": selection_split,
        "analysis_split": analysis_split,
        "val_rmse": val.get("rmse"),
        "val_gain_vs_identity": val_gain,
        "val_gain_vs_reference": reference_val_gain,
        "test_rmse": test.get("rmse"),
        "test_hot_q90_rmse": test.get("hot_q90_rmse"),
        "test_gradient_q90_rmse": test.get("gradient_q90_rmse"),
        "test_gain_vs_identity_rmse": test_gain,
        "test_gain_vs_reference_rmse": reference_test_gain,
        "test_preserves_identity_regions": preserves_identity_regions,
        "status": status,
        "interpretation": interpretation,
    }


def build_surrogate_rows(upper: dict[str, Any]) -> list[dict[str, Any]]:
    selection_split = str(upper.get("selection_split") or "val")
    analysis_split = str(upper.get("analysis_split") or "test")
    variants = _variant_map(upper)
    identity = variants.get(IDENTITY_VARIANT) or _baseline_variant(
        IDENTITY_VARIANT,
        upper.get("baseline_metrics", {}).get("broad_process_v1", {}),
    )
    reference_name = str(upper.get("reference") or "mean")
    reference = _baseline_variant(
        reference_name,
        upper.get("baseline_metrics", {}).get(reference_name, {}),
    )
    rows = [
        _row_for_variant(
            identity,
            role="identity",
            identity=identity,
            reference=reference,
            selection_split=selection_split,
            analysis_split=analysis_split,
        ),
        _row_for_variant(
            reference,
            role="strong_reference",
            identity=identity,
            reference=reference,
            selection_split=selection_split,
            analysis_split=analysis_split,
        ),
    ]
    for name in CANDIDATE_VARIANTS:
        if name in variants:
            rows.append(
                _row_for_variant(
                    variants[name],
                    role="candidate",
                    identity=identity,
                    reference=reference,
                    selection_split=selection_split,
                    analysis_split=analysis_split,
                )
            )
    return rows


def _select_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [row for row in rows if row["role"] == "candidate"]
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row["val_rmse"]))


def build_gate(rows: list[dict[str, Any]], phase79_manifest: dict[str, Any]) -> dict[str, Any]:
    phase79_gate = phase79_manifest.get("gate") or {}
    phase79_allows_local = bool(phase79_gate.get("local_surrogate_allowed"))
    selected = _select_candidate(rows)
    if selected is None:
        return {
            "candidate": "bounded_spot_size_parameterization_v1",
            "status": "blocked_no_candidate_variants",
            "local_surrogate_passed": False,
            "a100_seed7_allowed": False,
            "a100_80gb_request_now": False,
            "reason": "no bounded surrogate candidate variants were found in the Phase 59 upper-bound summary",
            "next_action": "redesign the local surrogate gate or regenerate Phase 59 predictions",
        }
    selected_passed = selected["status"] == "passes_local_surrogate"
    local_passed = phase79_allows_local and selected_passed
    if not phase79_allows_local:
        status = "blocked_by_phase79"
        reason = "Phase 79 did not allow a local surrogate gate for this candidate"
    elif selected_passed:
        status = "opened_for_phase76_seed7"
        reason = "validation-selected bounded surrogate passed all local checks"
    else:
        status = "blocked_by_local_surrogate_gate"
        reason = str(selected["interpretation"])
    return {
        "candidate": "bounded_spot_size_parameterization_v1",
        "status": status,
        "selected_variant": selected["variant"],
        "selected_variant_status": selected["status"],
        "phase79_local_surrogate_allowed": phase79_allows_local,
        "local_surrogate_passed": local_passed,
        "a100_seed7_allowed": local_passed,
        "a100_80gb_request_now": False,
        "minimum_validation_rmse_gain_required": MIN_VALIDATION_RMSE_GAIN,
        "selected_val_gain_vs_identity": selected["val_gain_vs_identity"],
        "selected_test_gain_vs_identity_rmse": selected["test_gain_vs_identity_rmse"],
        "selected_test_gain_vs_reference_rmse": selected["test_gain_vs_reference_rmse"],
        "selected_preserves_identity_regions": selected["test_preserves_identity_regions"],
        "reason": reason,
        "next_action": (
            "run Phase 76 seed-7 A100 validation for bounded_spot_size_parameterization_v1"
            if local_passed
            else "do not run broad12/broad21 A100 training; close this surrogate or redesign with a stronger validation-visible signal"
        ),
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


def build_markdown(gate: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "# Phase 80 Bounded Spot-Size Local Surrogate Gate",
            "",
            "## Gate Decision",
            "",
            f"Status: `{gate['status']}`.",
            f"Selected variant: `{gate.get('selected_variant', '')}`.",
            f"Local surrogate passed: `{str(gate['local_surrogate_passed']).lower()}`.",
            f"A100 seed-7 allowed: `{str(gate['a100_seed7_allowed']).lower()}`.",
            f"A100-SXM4-80GB request now: `{str(gate['a100_80gb_request_now']).lower()}`.",
            "",
            gate["reason"],
            "",
            "## Surrogate Rows",
            "",
            _markdown_table(
                rows,
                [
                    ("variant", "Variant"),
                    ("role", "Role"),
                    ("val_rmse", "Val RMSE"),
                    ("val_gain_vs_identity", "Val gain vs identity"),
                    ("test_rmse", "Test RMSE"),
                    ("test_gain_vs_reference_rmse", "Test gain vs reference"),
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
    upper = _read_json(resolved["phase59_upper"])
    phase79_manifest = _read_json(resolved["phase79_manifest"])
    rows = build_surrogate_rows(upper)
    gate = build_gate(rows, phase79_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)

    table_path = output_dir / "phase80_bounded_spot_size_local_surrogate_gate_table.csv"
    gate_path = output_dir / "phase80_bounded_spot_size_local_surrogate_gate.json"
    markdown_path = output_dir / "phase80_bounded_spot_size_local_surrogate_gate.md"
    manifest_path = output_dir / "phase80_bounded_spot_size_local_surrogate_gate_manifest.json"

    _write_csv(table_path, rows, SURROGATE_FIELDS)
    _write_json(gate_path, gate)
    markdown_path.write_text(build_markdown(gate, rows), encoding="utf-8")
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[str(row["status"])] = status_counts.get(str(row["status"]), 0) + 1
    manifest = {
        "phase": 80,
        "objective": "bounded_spot_size_local_surrogate_gate",
        "inputs": {key: _display_path(path, root) for key, path in sorted(resolved.items())},
        "outputs": {
            "surrogate_table": _display_path(table_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "surrogate_rows": len(rows),
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
        default=Path("docs/results/phase80_bounded_spot_size_local_surrogate_gate"),
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
