"""Inspect AM-Bench optical microscopy TIFF files and build coarse micro graphs."""

from __future__ import annotations

import argparse
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
        "sample_id": sample_id or path.stem,
        "source": str(path),
        "sample_metadata": parse_ambench_microstructure_filename(path.name),
        "image": {
            "shape": [int(value) for value in image.shape],
            "gray_shape": [int(value) for value in gray.shape],
            "dtype": str(image.dtype),
            "channels": _channel_count(image),
            "statistics": stats,
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
    parser.add_argument("--image", required=True, type=Path, help="Input AM-Bench optical microscopy TIFF.")
    parser.add_argument("--sample-id", help="Sample id recorded in the output manifest.")
    parser.add_argument("--threshold-quantile", type=float, default=0.9)
    parser.add_argument("--grid-rows", type=int, default=4)
    parser.add_argument("--grid-cols", type=int, default=4)
    parser.add_argument("--graph-k", type=int, default=2)
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = inspect_microstructure_image(
        args.image,
        sample_id=args.sample_id,
        threshold_quantile=args.threshold_quantile,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        graph_k=args.graph_k,
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
