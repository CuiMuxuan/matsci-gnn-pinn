"""Fit a no-test-leakage convex stack over row-aligned prediction CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from gnnpinn.data.loaders import load_field_table
from gnnpinn.data.splits import load_split_manifest, split_indices
from gnnpinn.eval.baselines import regression_metric_table
from gnnpinn.eval.regions import region_metric_tables


def _prediction_label(value: str) -> str:
    path = Path(value)
    return path.stem.replace("_predictions", "").replace("_prediction", "")


def _read_prediction_csv(path: Path, label: str | None = None) -> dict[str, Any]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")
    predictions: dict[int, float] = {}
    target_values: dict[int, float] = {}
    for row in rows:
        index = int(row["row_index"])
        predictions[index] = float(row["prediction"])
        target_columns = [
            key
            for key in row
            if key not in {"row_index", "split", "sample_id", "method", "prediction", "error", "abs_error"}
            and key not in {"x", "y", "z", "t", "time"}
        ]
        target_column = next((key for key in target_columns if key in row and _is_float_like(row[key])), None)
        if target_column is not None:
            target_values[index] = float(row[target_column])
    return {
        "label": label or _prediction_label(str(path)),
        "path": str(path),
        "predictions": predictions,
        "target_values": target_values,
    }


def _is_float_like(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _aligned_matrix(prediction_sets: list[dict[str, Any]], n_points: int) -> list[list[float]]:
    matrix: list[list[float]] = []
    for row_index in range(n_points):
        row: list[float] = []
        for prediction_set in prediction_sets:
            predictions = prediction_set["predictions"]
            if row_index not in predictions:
                raise ValueError(
                    f"Prediction set {prediction_set['label']!r} is missing row_index {row_index}"
                )
            row.append(float(predictions[row_index]))
        matrix.append(row)
    return matrix


def _simplex_weights(n_experts: int, step: float) -> list[list[float]]:
    if n_experts < 1:
        raise ValueError("At least one expert prediction is required")
    if n_experts == 1:
        return [[1.0]]
    if step <= 0.0 or step > 1.0:
        raise ValueError("--weight-step must be in (0, 1]")
    units = int(round(1.0 / step))
    if abs(units * step - 1.0) > 1e-9:
        raise ValueError("--weight-step must divide 1.0 exactly, e.g. 0.1 or 0.05")
    weights: list[list[float]] = []
    for counts in _integer_simplex(n_experts, units):
        weights.append([count / units for count in counts])
    return weights


def _integer_simplex(length: int, total: int) -> list[tuple[int, ...]]:
    if length == 1:
        return [(total,)]
    output: list[tuple[int, ...]] = []
    for value in range(total + 1):
        for suffix in _integer_simplex(length - 1, total - value):
            output.append((value, *suffix))
    return output


def _stack_predictions(matrix: list[list[float]], weights: list[float]) -> list[float]:
    return [sum(value * weight for value, weight in zip(row, weights)) for row in matrix]


def _rmse_score(y_true: list[float], y_pred: list[float]) -> float:
    return float(regression_metric_table(y_true, y_pred)["rmse"])


def _score_weights(
    matrix: list[list[float]],
    y_true: list[float],
    indices: list[int],
    weights: list[float],
) -> float:
    predictions = _stack_predictions([matrix[index] for index in indices], weights)
    truth = [y_true[index] for index in indices]
    return _rmse_score(truth, predictions)


def fit_simplex_stack(
    matrix: list[list[float]],
    y_true: list[float],
    fit_indices: list[int],
    weight_step: float,
) -> dict[str, Any]:
    n_experts = len(matrix[0]) if matrix else 0
    candidates = _simplex_weights(n_experts, weight_step)
    best_weights = min(
        candidates,
        key=lambda weights: (
            _score_weights(matrix, y_true, fit_indices, weights),
            tuple(weights),
        ),
    )
    return {
        "weights": best_weights,
        "candidate_count": len(candidates),
        "fit_rmse": _score_weights(matrix, y_true, fit_indices, best_weights),
    }


def _split_metrics(
    sample: Any,
    target: str,
    split_manifest: dict[str, Any],
    y_true: list[float],
    y_pred: list[float],
    hot_quantiles: list[float] | None,
    gradient_quantiles: list[float] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for split_name in split_manifest["splits"]:
        indices = split_indices(split_manifest, split_name)
        payload[split_name] = {
            "n_points": len(indices),
            "metrics": regression_metric_table(
                [y_true[index] for index in indices],
                [y_pred[index] for index in indices],
            ),
        }
        regions = region_metric_tables(
            sample,
            target=target,
            y_pred=y_pred,
            indices=indices,
            hot_quantiles=hot_quantiles,
            gradient_quantiles=gradient_quantiles,
        )
        if regions:
            payload[split_name]["region_metrics"] = regions
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    labels = list(args.label or [])
    if labels and len(labels) != len(args.prediction):
        raise ValueError("--label must be provided once per --prediction")
    sample = load_field_table(args.table, observation_columns=[args.target])
    split_manifest = load_split_manifest(args.split_manifest)
    y_true = sample.require_observation(args.target)
    prediction_sets = [
        _read_prediction_csv(path, labels[index] if labels else None)
        for index, path in enumerate(args.prediction)
    ]
    matrix = _aligned_matrix(prediction_sets, sample.n_points)
    fit_indices = []
    fit_splits = args.fit_split or ["train", "val"]
    for split_name in fit_splits:
        fit_indices.extend(split_indices(split_manifest, split_name))
    if not fit_indices:
        raise ValueError("Stack fit splits are empty")
    fit_payload = fit_simplex_stack(
        matrix=matrix,
        y_true=y_true,
        fit_indices=fit_indices,
        weight_step=args.weight_step,
    )
    stacked = _stack_predictions(matrix, fit_payload["weights"])
    expert_metrics = {}
    for expert_index, prediction_set in enumerate(prediction_sets):
        expert_pred = [row[expert_index] for row in matrix]
        expert_metrics[prediction_set["label"]] = _split_metrics(
            sample,
            args.target,
            split_manifest,
            y_true,
            expert_pred,
            args.hot_quantiles,
            args.gradient_quantiles,
        )
    return {
        "target": args.target,
        "table": str(args.table),
        "split_manifest": str(args.split_manifest),
        "fit_splits": fit_splits,
        "selection_metric": "rmse",
        "uses_test_for_selection": False,
        "weight_step": args.weight_step,
        "experts": [
            {"label": item["label"], "path": item["path"]}
            for item in prediction_sets
        ],
        "weights": dict(
            zip([item["label"] for item in prediction_sets], fit_payload["weights"])
        ),
        "fit": {
            "candidate_count": fit_payload["candidate_count"],
            "rmse": fit_payload["fit_rmse"],
            "n_points": len(fit_indices),
        },
        "stack_metrics": _split_metrics(
            sample,
            args.target,
            split_manifest,
            y_true,
            stacked,
            args.hot_quantiles,
            args.gradient_quantiles,
        ),
        "expert_metrics": expert_metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--prediction", required=True, action="append", type=Path)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--fit-split", action="append", default=[])
    parser.add_argument("--weight-step", type=float, default=0.1)
    parser.add_argument("--hot-quantile", action="append", type=float, dest="hot_quantiles")
    parser.add_argument("--gradient-quantile", action="append", type=float, dest="gradient_quantiles")
    parser.add_argument("--json-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text, encoding="utf-8")
        print(f"Wrote: {args.json_output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
