"""CLI for dependency-light field-table baseline evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gnnpinn.data.loaders import load_field_table
from gnnpinn.data.manifest import candidate_files_from_audit, load_audit_report
from gnnpinn.data.splits import load_split_manifest, split_indices, subset_sequence
from gnnpinn.eval.baselines import constant_predictions, regression_metric_table


def evaluate_table(
    table_path: Path,
    target: str,
    strategy: str,
    prediction_column: str | None = None,
    split_manifest_path: Path | None = None,
    fit_split: str = "train",
) -> dict[str, Any]:
    sample = load_field_table(table_path)
    y_true = sample.require_observation(target)
    split_manifest = load_split_manifest(split_manifest_path) if split_manifest_path else None
    if split_manifest:
        fit_indices = split_indices(split_manifest, fit_split)
        fit_values = subset_sequence(y_true, fit_indices)
        split_metrics = {}
        for split_name in split_manifest["splits"]:
            indices = split_indices(split_manifest, split_name)
            y_split = subset_sequence(y_true, indices)
            if prediction_column:
                pred_values = sample.require_observation(prediction_column)
                y_pred = subset_sequence(pred_values, indices)
                baseline_name = f"column:{prediction_column}"
            else:
                y_pred = _constant_predictions_from_fit(y_split, fit_values, strategy)
                baseline_name = f"constant:{strategy}:fit={fit_split}"
            split_metrics[split_name] = {
                "n_points": len(indices),
                "metrics": regression_metric_table(y_split, y_pred),
            }
        return {
            "sample_id": sample.sample_id,
            "source_path": str(table_path),
            "target": target,
            "baseline": baseline_name,
            "n_points": sample.n_points,
            "split_manifest": str(split_manifest_path),
            "fit_split": fit_split,
            "split_metrics": split_metrics,
            "metadata": sample.metadata,
        }
    if prediction_column:
        y_pred = sample.require_observation(prediction_column)
        baseline_name = f"column:{prediction_column}"
    else:
        y_pred = constant_predictions(y_true, strategy=strategy)
        baseline_name = f"constant:{strategy}"
    return {
        "sample_id": sample.sample_id,
        "source_path": str(table_path),
        "target": target,
        "baseline": baseline_name,
        "n_points": sample.n_points,
        "metrics": regression_metric_table(y_true, y_pred),
        "metadata": sample.metadata,
    }


def _constant_predictions_from_fit(values: list[float], fit_values: list[float], strategy: str) -> list[float]:
    if strategy == "mean":
        value = sum(float(item) for item in fit_values) / len(fit_values)
    elif strategy == "first":
        value = float(fit_values[0])
    elif strategy == "zero":
        value = 0.0
    else:
        raise ValueError(f"Unsupported baseline strategy: {strategy}")
    return [value for _ in values]


def discover_tables(args: argparse.Namespace) -> list[Path]:
    tables = [Path(item) for item in args.table]
    if args.audit_report:
        audit = load_audit_report(args.audit_report)
        tables.extend(
            candidate_files_from_audit(
                audit,
                modality=args.modality,
                project_root=args.project_root,
            )
        )
    unique = sorted({path.resolve() for path in tables})
    if not unique:
        raise ValueError("No field tables were provided or discovered from the audit report")
    return unique


def run(args: argparse.Namespace) -> dict[str, Any]:
    results = [
        evaluate_table(
            table_path=path,
            target=args.target,
            strategy=args.strategy,
            prediction_column=args.prediction_column,
            split_manifest_path=args.split_manifest,
            fit_split=args.fit_split,
        )
        for path in discover_tables(args)
    ]
    return {
        "target": args.target,
        "strategy": args.strategy,
        "prediction_column": args.prediction_column,
        "split_manifest": str(args.split_manifest) if args.split_manifest else None,
        "fit_split": args.fit_split,
        "results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", action="append", default=[], help="CSV/JSON field table path. Can repeat.")
    parser.add_argument("--audit-report", type=Path, help="Audit JSON report to discover matched local files.")
    parser.add_argument("--modality", help="Optional audit modality filter, e.g. thermal.")
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Project root for relative audit paths.")
    parser.add_argument("--target", required=True, help="Observation column to score.")
    parser.add_argument("--prediction-column", help="Optional prediction column to score against target.")
    parser.add_argument("--split-manifest", type=Path, help="Optional JSON split manifest.")
    parser.add_argument("--fit-split", default="train", help="Split used to fit constant baselines.")
    parser.add_argument(
        "--strategy",
        default="mean",
        choices=["mean", "first", "zero"],
        help="Constant baseline strategy when no prediction column is provided.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run(args)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote: {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
