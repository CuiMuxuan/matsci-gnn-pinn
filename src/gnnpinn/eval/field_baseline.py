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
from gnnpinn.eval.regions import region_metric_tables


MODEL_BASELINES = {"knn", "random_forest", "extra_trees"}


def evaluate_table(
    table_path: Path,
    target: str,
    strategy: str,
    prediction_column: str | None = None,
    split_manifest_path: Path | None = None,
    fit_split: str = "train",
    feature_columns: list[str] | None = None,
    n_neighbors: int = 8,
    n_estimators: int = 200,
    random_state: int = 7,
    hot_quantiles: list[float] | None = None,
    gradient_quantiles: list[float] | None = None,
) -> dict[str, Any]:
    sample = load_field_table(table_path)
    y_true = sample.require_observation(target)
    split_manifest = load_split_manifest(split_manifest_path) if split_manifest_path else None
    feature_matrix: list[list[float]] | None = None
    model_predictions: list[float] | None = None
    if strategy in MODEL_BASELINES:
        if prediction_column:
            raise ValueError("prediction_column cannot be combined with model baselines")
        feature_matrix, feature_columns = _feature_matrix(sample, feature_columns)
        fit_indices = (
            split_indices(split_manifest, fit_split)
            if split_manifest
            else list(range(sample.n_points))
        )
        model_predictions = _fit_predict_model_baseline(
            strategy=strategy,
            features=feature_matrix,
            target=y_true,
            fit_indices=fit_indices,
            n_neighbors=n_neighbors,
            n_estimators=n_estimators,
            random_state=random_state,
        )
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
            elif model_predictions is not None:
                y_pred = subset_sequence(model_predictions, indices)
                baseline_name = f"model:{strategy}:fit={fit_split}"
            else:
                y_pred = _constant_predictions_from_fit(y_split, fit_values, strategy)
                baseline_name = f"constant:{strategy}:fit={fit_split}"
            split_metrics[split_name] = {
                "n_points": len(indices),
                "metrics": regression_metric_table(y_split, y_pred),
            }
            regions = region_metric_tables(
                sample,
                target=target,
                y_pred=model_predictions if model_predictions is not None else pred_values if prediction_column else _constant_predictions_from_fit(y_true, fit_values, strategy),
                indices=indices,
                hot_quantiles=hot_quantiles,
                gradient_quantiles=gradient_quantiles,
            )
            if regions:
                split_metrics[split_name]["region_metrics"] = regions
        return {
            "sample_id": sample.sample_id,
            "source_path": str(table_path),
            "target": target,
            "baseline": baseline_name,
            "n_points": sample.n_points,
            "split_manifest": str(split_manifest_path),
            "fit_split": fit_split,
            "split_metrics": split_metrics,
            "baseline_parameters": _baseline_parameters(
                strategy=strategy,
                feature_columns=feature_columns,
                n_neighbors=n_neighbors,
                n_estimators=n_estimators,
                random_state=random_state,
            ),
            "metadata": sample.metadata,
        }
    if prediction_column:
        y_pred = sample.require_observation(prediction_column)
        baseline_name = f"column:{prediction_column}"
    elif model_predictions is not None:
        y_pred = model_predictions
        baseline_name = f"model:{strategy}:fit=all"
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
        "baseline_parameters": _baseline_parameters(
            strategy=strategy,
            feature_columns=feature_columns,
            n_neighbors=n_neighbors,
            n_estimators=n_estimators,
            random_state=random_state,
        ),
        "region_metrics": region_metric_tables(
            sample,
            target=target,
            y_pred=y_pred,
            hot_quantiles=hot_quantiles,
            gradient_quantiles=gradient_quantiles,
        ),
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


def _feature_matrix(sample: Any, feature_columns: list[str] | None) -> tuple[list[list[float]], list[str]]:
    default_columns = list(sample.metadata.get("coordinate_columns") or [])
    time_column = sample.metadata.get("time_column")
    if time_column:
        default_columns.append(time_column)
    columns = feature_columns or default_columns
    if not columns:
        raise ValueError("Model baselines require at least one feature column")

    rows: list[list[float]] = []
    coord_columns = list(sample.metadata.get("coordinate_columns") or [])
    time_column = sample.metadata.get("time_column")
    for row_idx in range(sample.n_points):
        row: list[float] = []
        for column in columns:
            if column in coord_columns:
                row.append(float(sample.coordinates[row_idx][coord_columns.index(column)]))
            elif time_column and column == time_column:
                row.append(float(sample.time[row_idx]))
            elif column in sample.observations:
                row.append(float(sample.observations[column][row_idx]))
            else:
                raise ValueError(f"Feature column not found: {column}")
        rows.append(row)
    return rows, columns


def _fit_predict_model_baseline(
    strategy: str,
    features: list[list[float]],
    target: list[float],
    fit_indices: list[int],
    n_neighbors: int,
    n_estimators: int,
    random_state: int,
) -> list[float]:
    if strategy == "knn":
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=max(1, min(n_neighbors, len(fit_indices)))),
        )
    elif strategy == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
    elif strategy == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model baseline: {strategy}")

    x_fit = [features[index] for index in fit_indices]
    y_fit = [target[index] for index in fit_indices]
    model.fit(x_fit, y_fit)
    return [float(item) for item in model.predict(features)]


def _baseline_parameters(
    strategy: str,
    feature_columns: list[str] | None,
    n_neighbors: int,
    n_estimators: int,
    random_state: int,
) -> dict[str, Any]:
    if strategy not in MODEL_BASELINES:
        return {}
    return {
        "feature_columns": feature_columns,
        "n_neighbors": n_neighbors if strategy == "knn" else None,
        "n_estimators": n_estimators if strategy in {"random_forest", "extra_trees"} else None,
        "random_state": random_state,
    }


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
            feature_columns=args.feature_columns,
            n_neighbors=args.n_neighbors,
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            hot_quantiles=args.hot_quantiles,
            gradient_quantiles=args.gradient_quantiles,
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
        # Keep choices close to the legacy constant baselines and the first
        # stronger model baselines needed for server-stage AM-Bench runs.
        choices=["mean", "first", "zero", "knn", "random_forest", "extra_trees"],
        help="Baseline strategy when no prediction column is provided.",
    )
    parser.add_argument(
        "--feature-column",
        action="append",
        dest="feature_columns",
        help="Feature column for model baselines. Defaults to coordinate columns plus time.",
    )
    parser.add_argument("--n-neighbors", type=int, default=8, help="k for knn baseline.")
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of trees for tree baselines.")
    parser.add_argument("--random-state", type=int, default=7, help="Random state for model baselines.")
    parser.add_argument(
        "--hot-quantile",
        action="append",
        type=float,
        dest="hot_quantiles",
        help="Report metrics on target values above this split-local quantile, e.g. 0.9.",
    )
    parser.add_argument(
        "--gradient-quantile",
        action="append",
        type=float,
        dest="gradient_quantiles",
        help="Report metrics on spatial-gradient scores above this split-local quantile, e.g. 0.9.",
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
