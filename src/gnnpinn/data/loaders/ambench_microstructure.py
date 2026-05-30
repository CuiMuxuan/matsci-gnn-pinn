"""Inspect AM-Bench optical microscopy TIFF files and build coarse micro graphs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from gnnpinn.data.graph_schema import MicrostructureGraph
from gnnpinn.data.transforms import knn_edges_2d


def _numpy() -> Any:
    import numpy as np

    return np


def _read_image(path: Path) -> Any:
    try:
        from skimage.io import imread
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scikit-image is required for AM-Bench microstructure TIFF inspection. "
            "Install it with `python -m pip install scikit-image>=0.23` or the project science extra."
        ) from exc
    return imread(path)


def inspect_microstructure_image(
    image_path: str | Path,
    *,
    sample_id: str | None = None,
    threshold_quantile: float = 0.9,
    grid_rows: int = 4,
    grid_cols: int = 4,
    graph_k: int = 2,
) -> dict[str, Any]:
    """Return image statistics and a coarse grid graph from a microscopy image."""

    path = Path(image_path)
    if not (0.0 < threshold_quantile < 1.0):
        raise ValueError("threshold_quantile must be in (0, 1)")
    if grid_rows <= 0 or grid_cols <= 0:
        raise ValueError("grid_rows and grid_cols must be positive")
    if graph_k <= 0:
        raise ValueError("graph_k must be positive")

    np = _numpy()
    image = np.asarray(_read_image(path))
    gray = _to_grayscale(image)
    finite = gray[np.isfinite(gray)]
    if finite.size == 0:
        raise ValueError(f"Image contains no finite pixel values: {path}")

    threshold = float(np.quantile(finite, threshold_quantile))
    mask = gray >= threshold
    graph = _build_grid_micro_graph(gray, mask, grid_rows=grid_rows, grid_cols=grid_cols, k=graph_k)
    derived_features = {
        **_intensity_distribution_features(finite),
        **_mask_geometry_features(mask),
        **_texture_features(gray, mask),
    }
    stats = {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "threshold_quantile": threshold_quantile,
        "threshold_value": threshold,
        "mask_fraction": float(np.mean(mask)),
    }
    return {
        "dataset_id": "mds2-2718",
        "feature_schema_version": "micrograph_features_v2",
        "sample_id": sample_id or path.stem,
        "source": str(path),
        "sample_metadata": parse_ambench_microstructure_filename(path.name),
        "image": {
            "shape": [int(value) for value in image.shape],
            "gray_shape": [int(value) for value in gray.shape],
            "dtype": str(image.dtype),
            "channels": _channel_count(image),
            "statistics": stats,
            "derived_features": derived_features,
        },
        "graph": {
            "schema": "MicrostructureGraph",
            "num_nodes": graph.num_nodes,
            "num_edges": graph.num_edges,
            "node_feature_names": [
                "center_row_norm",
                "center_col_norm",
                "mean_intensity_norm",
                "std_intensity_norm",
                "mask_fraction",
            ],
            "edge_feature_names": [],
            "node_features": graph.node_features,
            "edge_index": graph.edge_index,
            "global_features": graph.global_features,
            "target_statistics": graph.target_statistics,
        },
    }


def graph_record_from_inspection(inspection: dict[str, Any]) -> dict[str, Any]:
    """Flatten one inspection payload into sample-level graph features."""

    graph = inspection["graph"]
    image = inspection["image"]
    stats = image["statistics"]
    node_features = graph.get("node_features", [])
    np = _numpy()
    node_array = np.asarray(node_features, dtype=float)
    if node_array.ndim != 2 or node_array.shape[0] == 0:
        raise ValueError("Inspection graph must contain a non-empty 2D node_features array")
    node_names = graph.get("node_feature_names") or [f"node_feature_{index}" for index in range(node_array.shape[1])]

    features: dict[str, float] = {
        "image_mean_intensity": float(stats["mean"]),
        "image_std_intensity": float(stats["std"]),
        "image_mask_fraction": float(stats["mask_fraction"]),
        "graph_num_nodes": float(graph["num_nodes"]),
        "graph_num_edges": float(graph["num_edges"]),
    }
    for name, value in image.get("derived_features", {}).items():
        features[name] = float(value)
    for index, name in enumerate(node_names):
        values = node_array[:, index]
        features[f"node_{name}_mean"] = float(np.mean(values))
        features[f"node_{name}_std"] = float(np.std(values))
        features[f"node_{name}_min"] = float(np.min(values))
        features[f"node_{name}_max"] = float(np.max(values))

    return {
        "dataset_id": inspection.get("dataset_id"),
        "sample_id": inspection.get("sample_id"),
        "source": inspection.get("source"),
        "sample_metadata": inspection.get("sample_metadata", {}),
        "feature_schema_version": inspection.get("feature_schema_version", "micrograph_features_v1"),
        "feature_names": list(features.keys()),
        "features": features,
        "graph_summary": {
            "num_nodes": graph.get("num_nodes"),
            "num_edges": graph.get("num_edges"),
            "node_feature_names": node_names,
            "image_shape": image.get("shape"),
            "gray_shape": image.get("gray_shape"),
        },
    }


def build_graph_feature_table(
    inspections: list[str | Path],
    *,
    jsonl_output: str | Path | None = None,
    csv_output: str | Path | None = None,
) -> dict[str, Any]:
    """Aggregate one or more inspection JSON files into graph-feature records."""

    records = []
    for inspection_path in inspections:
        path = Path(inspection_path)
        inspection = json.loads(path.read_text(encoding="utf-8"))
        record = graph_record_from_inspection(inspection)
        record["inspection"] = str(path)
        records.append(record)
    if not records:
        raise ValueError("At least one inspection JSON is required")

    feature_names = records[0]["feature_names"]
    for record in records[1:]:
        if record["feature_names"] != feature_names:
            raise ValueError("All inspection records must produce the same feature names")

    if jsonl_output:
        jsonl_path = Path(jsonl_output)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_path.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
            encoding="utf-8",
        )
    if csv_output:
        csv_path = Path(csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = ["sample_id", "source", *feature_names]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "sample_id": record["sample_id"],
                        "source": record["source"],
                        **record["features"],
                    }
                )
    return {
        "dataset_id": records[0].get("dataset_id"),
        "n_records": len(records),
        "feature_names": feature_names,
        "jsonl_output": str(jsonl_output) if jsonl_output else None,
        "csv_output": str(csv_output) if csv_output else None,
        "records": records,
    }


def parse_ambench_microstructure_filename(filename: str) -> dict[str, Any]:
    """Parse AMB2022 optical microscopy filename tokens when present."""

    stem = Path(filename).stem
    masked = stem.endswith("_m")
    if masked:
        stem = stem[:-2]
    result: dict[str, Any] = {
        "filename": filename,
        "masked": masked,
    }
    pattern = re.compile(
        r"^(?P<campaign>AMB2022)-(?P<alloy>\d+)-(?P<section>SH\d+)-"
        r"(?P<build_plate>BP\d+)(?:-(?P<process>P\d+))?"
        r"(?:-(?P<line>L[0-9.]+))?(?:-(?P<replicate>\d+))?"
        r"(?:-(?P<view>BEFORE|CUTS))?$"
    )
    match = pattern.match(stem)
    if not match:
        result["parsed"] = False
        return result
    result.update({key: value for key, value in match.groupdict().items() if value is not None})
    result["parsed"] = True
    if "process" in result:
        result["process_index"] = int(str(result["process"])[1:])
    if "build_plate" in result:
        result["build_plate_index"] = int(str(result["build_plate"])[2:])
    if "line" in result:
        result["line_value"] = float(str(result["line"])[1:])
    if "replicate" in result:
        result["replicate_index"] = int(str(result["replicate"]))
    return result


def _to_grayscale(image: Any) -> Any:
    np = _numpy()
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(float)
    if array.ndim == 3:
        if array.shape[-1] in {3, 4}:
            rgb = array[..., :3].astype(float)
            return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        return array.mean(axis=-1).astype(float)
    raise ValueError(f"Expected a 2D or 3D image, got shape {array.shape}")


def _channel_count(image: Any) -> int:
    if image.ndim == 2:
        return 1
    return int(image.shape[-1])


def _intensity_distribution_features(finite: Any) -> dict[str, float]:
    np = _numpy()
    intensity_min = float(np.min(finite))
    intensity_max = float(np.max(finite))
    intensity_range = intensity_max - intensity_min
    if intensity_range <= 0.0:
        intensity_range = 1.0
    q10, q25, q50, q75, q90 = [float(value) for value in np.quantile(finite, [0.1, 0.25, 0.5, 0.75, 0.9])]
    histogram_max = intensity_max if intensity_max > intensity_min else intensity_min + 1.0
    histogram, _ = np.histogram(finite, bins=32, range=(intensity_min, histogram_max))
    probabilities = histogram.astype(float)
    probability_sum = float(np.sum(probabilities))
    if probability_sum > 0.0:
        probabilities = probabilities / probability_sum
        nonzero = probabilities[probabilities > 0.0]
        entropy = float(-np.sum(nonzero * np.log(nonzero)) / np.log(32.0))
    else:
        entropy = 0.0
    return {
        "image_intensity_q10": q10,
        "image_intensity_q25": q25,
        "image_intensity_q50": q50,
        "image_intensity_q75": q75,
        "image_intensity_q90": q90,
        "image_intensity_iqr_norm": float((q75 - q25) / intensity_range),
        "image_intensity_p90_p10_norm": float((q90 - q10) / intensity_range),
        "image_entropy_32bin": entropy,
    }


def _mask_geometry_features(mask: Any) -> dict[str, float]:
    np = _numpy()
    height, width = mask.shape
    mask_count = int(np.sum(mask))
    if mask_count == 0:
        return {
            "mask_centroid_row_norm": 0.0,
            "mask_centroid_col_norm": 0.0,
            "mask_row_std_norm": 0.0,
            "mask_col_std_norm": 0.0,
            "mask_span_row_norm": 0.0,
            "mask_span_col_norm": 0.0,
            "mask_bbox_area_fraction": 0.0,
            "mask_fill_fraction": 0.0,
            "mask_perimeter_fraction": 0.0,
            "mask_border_touch_fraction": 0.0,
            "mask_top_half_fraction": 0.0,
            "mask_bottom_half_fraction": 0.0,
            "mask_left_half_fraction": 0.0,
            "mask_right_half_fraction": 0.0,
            "mask_anisotropy": 0.0,
        }

    rows, cols = np.nonzero(mask)
    row_min, row_max = int(np.min(rows)), int(np.max(rows))
    col_min, col_max = int(np.min(cols)), int(np.max(cols))
    span_rows = row_max - row_min + 1
    span_cols = col_max - col_min + 1
    bbox_area = span_rows * span_cols
    boundary = _mask_boundary(mask)
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = mask[0, :]
    border[-1, :] = mask[-1, :]
    border[:, 0] = np.logical_or(border[:, 0], mask[:, 0])
    border[:, -1] = np.logical_or(border[:, -1], mask[:, -1])
    row_centered = rows.astype(float) - float(np.mean(rows))
    col_centered = cols.astype(float) - float(np.mean(cols))
    if mask_count > 1:
        covariance = np.cov(np.stack([row_centered, col_centered], axis=0))
        eigenvalues = np.linalg.eigvalsh(covariance)
        anisotropy = float((eigenvalues[-1] - eigenvalues[0]) / max(eigenvalues[-1] + eigenvalues[0], 1e-12))
    else:
        anisotropy = 0.0
    return {
        "mask_centroid_row_norm": float(np.mean(rows) / max(height - 1, 1)),
        "mask_centroid_col_norm": float(np.mean(cols) / max(width - 1, 1)),
        "mask_row_std_norm": float(np.std(rows) / max(height - 1, 1)),
        "mask_col_std_norm": float(np.std(cols) / max(width - 1, 1)),
        "mask_span_row_norm": float(span_rows / height),
        "mask_span_col_norm": float(span_cols / width),
        "mask_bbox_area_fraction": float(bbox_area / mask.size),
        "mask_fill_fraction": float(mask_count / max(bbox_area, 1)),
        "mask_perimeter_fraction": float(np.sum(boundary) / mask.size),
        "mask_border_touch_fraction": float(np.sum(border) / mask_count),
        "mask_top_half_fraction": float(np.sum(mask[: height // 2, :]) / mask_count),
        "mask_bottom_half_fraction": float(np.sum(mask[height // 2 :, :]) / mask_count),
        "mask_left_half_fraction": float(np.sum(mask[:, : width // 2]) / mask_count),
        "mask_right_half_fraction": float(np.sum(mask[:, width // 2 :]) / mask_count),
        "mask_anisotropy": anisotropy,
    }


def _texture_features(gray: Any, mask: Any) -> dict[str, float]:
    np = _numpy()
    finite = gray[np.isfinite(gray)]
    intensity_min = float(np.min(finite))
    intensity_max = float(np.max(finite))
    intensity_scale = intensity_max - intensity_min
    if intensity_scale <= 0.0:
        intensity_scale = 1.0
    normalized = np.nan_to_num((gray.astype(float) - intensity_min) / intensity_scale)
    grad_row, grad_col = np.gradient(normalized)
    gradient_magnitude = np.sqrt(grad_row**2 + grad_col**2)
    laplacian = np.gradient(grad_row, axis=0) + np.gradient(grad_col, axis=1)
    boundary = _mask_boundary(mask)
    boundary_gradients = gradient_magnitude[boundary]
    if boundary_gradients.size == 0:
        boundary_gradient_mean = 0.0
        boundary_gradient_q90 = 0.0
    else:
        boundary_gradient_mean = float(np.mean(boundary_gradients))
        boundary_gradient_q90 = float(np.quantile(boundary_gradients, 0.9))
    return {
        "gradient_magnitude_mean_norm": float(np.mean(gradient_magnitude)),
        "gradient_magnitude_std_norm": float(np.std(gradient_magnitude)),
        "gradient_magnitude_q90_norm": float(np.quantile(gradient_magnitude, 0.9)),
        "gradient_magnitude_max_norm": float(np.max(gradient_magnitude)),
        "laplacian_abs_mean_norm": float(np.mean(np.abs(laplacian))),
        "laplacian_abs_q90_norm": float(np.quantile(np.abs(laplacian), 0.9)),
        "mask_boundary_gradient_mean_norm": boundary_gradient_mean,
        "mask_boundary_gradient_q90_norm": boundary_gradient_q90,
    }


def _mask_boundary(mask: Any) -> Any:
    np = _numpy()
    padded = np.pad(mask.astype(bool), 1, constant_values=False)
    center = padded[1:-1, 1:-1]
    eroded = (
        center
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return center & ~eroded


def _build_grid_micro_graph(gray: Any, mask: Any, *, grid_rows: int, grid_cols: int, k: int) -> MicrostructureGraph:
    np = _numpy()
    height, width = gray.shape
    intensity_min = float(np.min(gray))
    intensity_max = float(np.max(gray))
    intensity_scale = intensity_max - intensity_min
    if intensity_scale <= 0.0:
        intensity_scale = 1.0

    node_features: list[list[float]] = []
    points: list[tuple[float, float]] = []
    row_edges = np.linspace(0, height, grid_rows + 1, dtype=int)
    col_edges = np.linspace(0, width, grid_cols + 1, dtype=int)
    for row_index in range(grid_rows):
        row_start, row_stop = int(row_edges[row_index]), int(row_edges[row_index + 1])
        for col_index in range(grid_cols):
            col_start, col_stop = int(col_edges[col_index]), int(col_edges[col_index + 1])
            patch = gray[row_start:row_stop, col_start:col_stop]
            patch_mask = mask[row_start:row_stop, col_start:col_stop]
            center_row = (row_start + row_stop - 1) / 2.0
            center_col = (col_start + col_stop - 1) / 2.0
            mean_intensity = (float(np.mean(patch)) - intensity_min) / intensity_scale
            std_intensity = float(np.std(patch)) / intensity_scale
            mask_fraction = float(np.mean(patch_mask))
            row_norm = center_row / max(height - 1, 1)
            col_norm = center_col / max(width - 1, 1)
            node_features.append([row_norm, col_norm, mean_intensity, std_intensity, mask_fraction])
            points.append((row_norm, col_norm))

    edge_index = knn_edges_2d(points, k=min(k, max(len(points) - 1, 1)), bidirectional=True)
    return MicrostructureGraph(
        node_features=node_features,
        edge_index=edge_index,
        global_features={
            "height": int(height),
            "width": int(width),
            "grid_rows": int(grid_rows),
            "grid_cols": int(grid_cols),
            "threshold_mask_fraction": float(np.mean(mask)),
        },
        target_statistics={
            "mean_intensity": float(np.mean(gray)),
            "std_intensity": float(np.std(gray)),
            "threshold_mask_fraction": float(np.mean(mask)),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["inspect", "aggregate"],
        default="inspect",
        help="inspect reads one TIFF; aggregate flattens inspection JSON files into graph-feature records.",
    )
    parser.add_argument("--image", type=Path, help="Input AM-Bench optical microscopy TIFF.")
    parser.add_argument("--sample-id", help="Sample id recorded in the output manifest.")
    parser.add_argument("--threshold-quantile", type=float, default=0.9)
    parser.add_argument("--grid-rows", type=int, default=4)
    parser.add_argument("--grid-cols", type=int, default=4)
    parser.add_argument("--graph-k", type=int, default=2)
    parser.add_argument(
        "--inspection",
        action="append",
        dest="inspections",
        default=[],
        type=Path,
        help="Inspection JSON to aggregate. Can be repeated.",
    )
    parser.add_argument("--jsonl-output", type=Path, help="JSONL output path for aggregate mode.")
    parser.add_argument("--csv-output", type=Path, help="CSV output path for aggregate mode.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "inspect":
        if args.image is None:
            raise ValueError("--image is required in inspect mode")
        report = inspect_microstructure_image(
            args.image,
            sample_id=args.sample_id,
            threshold_quantile=args.threshold_quantile,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            graph_k=args.graph_k,
        )
    else:
        report = build_graph_feature_table(
            args.inspections,
            jsonl_output=args.jsonl_output,
            csv_output=args.csv_output,
        )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote: {args.output}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
