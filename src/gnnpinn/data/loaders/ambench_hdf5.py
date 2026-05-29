"""Convert AM-Bench thermography HDF5 subsets into field-table CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from gnnpinn.data.loaders.ambench import build_split_manifest


DEFAULT_THERMAL_HDF5 = Path(
    "data/raw/ambench/2022_single_track/AMB2022-03/mds2-2716/"
    "Thermography/AMB2022-03-718-AMMT-StaringCamera_Signal.h5"
)
DEFAULT_DATASET = "ThermalData/Line_0_1/Signal"


def calibrate_signal_to_temperature_c(signal: Any, coeff_a: float, coeff_b: float, coeff_c: float) -> Any:
    """Apply the AM-Bench thermal calibration equation to raw signal values."""

    import numpy as np

    signal_array = np.asarray(signal, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = coeff_a * np.log((coeff_c * np.e / signal_array + 1.0) - coeff_b / coeff_a)
        return 14388.0 / denominator


def _h5py() -> Any:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "h5py is required for AM-Bench HDF5 conversion. "
            "Install it with `python -m pip install h5py>=3.10` or the project science extra."
        ) from exc
    return h5py


def convert_thermography_hdf5(args: argparse.Namespace) -> dict[str, Any]:
    h5py = _h5py()
    source = Path(args.thermal_hdf5)
    output = Path(args.output)
    dataset_path = args.dataset
    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(source, "r") as handle:
        dataset = handle[dataset_path]
        if len(dataset.shape) != 3:
            raise ValueError(f"Expected a 3D thermography signal dataset, got shape {dataset.shape}")
        n_frames, n_rows, n_cols = (int(value) for value in dataset.shape)
        frame_indices = _sample_indices(
            start=args.frame_start,
            step=args.frame_step,
            stop=n_frames,
            max_count=args.max_frames,
        )
        row_indices = _sample_indices(
            start=args.row_start,
            step=args.row_step,
            stop=n_rows,
            max_count=args.max_rows,
        )
        col_indices = _sample_indices(
            start=args.col_start,
            step=args.col_step,
            stop=n_cols,
            max_count=args.max_cols,
        )
        parent = handle[_parent_path(dataset_path)]
        thermal_data = handle["ThermalData"]
        thermal_cal = handle.get("Calibration/ThermalCal")
        frame_rate_hz = _float_attr(thermal_data.attrs.get("frame_rate"), default=1.0)
        calibration = _read_calibration(thermal_cal) if thermal_cal is not None else None
        process = {
            "laser_power_W": _float_attr(parent.attrs.get("laser_power")),
            "scan_speed_mm_s": _float_attr(parent.attrs.get("scan_speed")),
            "spot_size_um": _float_attr(parent.attrs.get("spot_size")),
        }
        fieldnames = [
            "x",
            "y",
            "z",
            "t",
            "signal",
            "frame_index",
            "row_index",
            "col_index",
            "laser_power_W",
            "scan_speed_mm_s",
            "spot_size_um",
        ]
        target = "signal"
        if args.calibrate_temperature:
            if calibration is None:
                raise ValueError("Calibration/ThermalCal group is missing; cannot calibrate temperature")
            fieldnames.insert(5, "temperature_C")
            target = "temperature_C"
        n_written, rows_by_frame, sampling_frames = _write_sampled_signal_csv(
            output=output,
            dataset=dataset,
            fieldnames=fieldnames,
            frame_indices=frame_indices,
            row_indices=row_indices,
            col_indices=col_indices,
            frame_rate_hz=frame_rate_hz,
            process=process,
            calibration=calibration if args.calibrate_temperature else None,
            min_signal=args.min_signal,
            sampling_mode=args.sampling_mode,
            hot_quantile=args.hot_quantile,
            gradient_quantile=args.gradient_quantile,
            background_fraction=args.background_fraction,
            max_points_per_frame=args.max_points_per_frame,
        )
        manifest = {
            "dataset_id": "mds2-2716",
            "sample_id": args.sample_id,
            "source": str(source),
            "output": str(output),
            "dataset_path": dataset_path,
            "source_shape": [n_frames, n_rows, n_cols],
            "source_dtype": str(dataset.dtype),
            "n_rows": n_written,
            "frame_indices": frame_indices,
            "row_indices": row_indices,
            "col_indices": col_indices,
            "frame_rate_hz": frame_rate_hz,
            "target": target,
            "coordinate_system": "camera_pixel_index",
            "process_parameters": process,
            "metadata": {
                "source_units": str(dataset.attrs.get("units", "digital levels")),
                "calibration": calibration if args.calibrate_temperature else None,
                "sampling": {
                    "frame_start": args.frame_start,
                    "frame_step": args.frame_step,
                    "max_frames": args.max_frames,
                    "row_start": args.row_start,
                    "row_step": args.row_step,
                    "max_rows": args.max_rows,
                    "col_start": args.col_start,
                    "col_step": args.col_step,
                    "max_cols": args.max_cols,
                    "min_signal": args.min_signal,
                    "selection": {
                        "mode": args.sampling_mode,
                        "active_target": target,
                        "hot_quantile": args.hot_quantile,
                        "gradient_quantile": args.gradient_quantile,
                        "background_fraction": args.background_fraction,
                        "max_points_per_frame": args.max_points_per_frame,
                        "frames": sampling_frames,
                    },
                },
            },
        }

    if args.split_manifest:
        if args.split_strategy == "frame":
            split = build_frame_split_manifest(
                rows_by_frame=rows_by_frame,
                sample_id=args.sample_id,
                train_fraction=args.train_fraction,
                val_fraction=args.val_fraction,
                test_fraction=args.test_fraction,
            )
        else:
            split = build_split_manifest(
                n_rows=n_written,
                sample_id=args.sample_id,
                split_config={
                    "train_fraction": args.train_fraction,
                    "val_fraction": args.val_fraction,
                    "test_fraction": args.test_fraction,
                    "seed": args.seed,
                },
            )
            split["strategy"] = "random_row"
        args.split_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.split_manifest.write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest["split_manifest"] = str(args.split_manifest)
    return manifest


def _write_sampled_signal_csv(
    output: Path,
    dataset: Any,
    fieldnames: list[str],
    frame_indices: list[int],
    row_indices: list[int],
    col_indices: list[int],
    frame_rate_hz: float,
    process: dict[str, float | None],
    calibration: dict[str, Any] | None = None,
    min_signal: float | None = None,
    sampling_mode: str = "uniform",
    hot_quantile: float = 0.9,
    gradient_quantile: float = 0.9,
    background_fraction: float = 0.1,
    max_points_per_frame: int | None = None,
) -> tuple[int, dict[int, list[int]], list[dict[str, Any]]]:
    import numpy as np

    _validate_sampling_config(
        sampling_mode=sampling_mode,
        hot_quantile=hot_quantile,
        gradient_quantile=gradient_quantile,
        background_fraction=background_fraction,
        max_points_per_frame=max_points_per_frame,
    )
    n_written = 0
    rows_by_frame: dict[int, list[int]] = {frame_index: [] for frame_index in frame_indices}
    sampling_frames: list[dict[str, Any]] = []
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame_index in frame_indices:
            frame = dataset[frame_index, :, :]
            t = frame_index / frame_rate_hz if frame_rate_hz else float(frame_index)
            signal_grid = np.asarray(frame[row_indices, :][:, col_indices], dtype=float)
            temperature_grid = None
            if calibration is not None:
                temperature_grid = calibrate_signal_to_temperature_c(
                    signal_grid,
                    coeff_a=float(calibration["coeff_a"]),
                    coeff_b=float(calibration["coeff_b"]),
                    coeff_c=float(calibration["coeff_c"]),
                )
            target_grid = temperature_grid if temperature_grid is not None else signal_grid
            selected_points, frame_sampling = _select_sampled_grid_points(
                signal_grid=signal_grid,
                target_grid=target_grid,
                row_indices=row_indices,
                col_indices=col_indices,
                min_signal=min_signal,
                sampling_mode=sampling_mode,
                hot_quantile=hot_quantile,
                gradient_quantile=gradient_quantile,
                background_fraction=background_fraction,
                max_points_per_frame=max_points_per_frame,
            )
            frame_sampling["frame_index"] = frame_index
            sampling_frames.append(frame_sampling)
            for row_position, col_position in selected_points:
                row_index = row_indices[row_position]
                col_index = col_indices[col_position]
                signal = signal_grid[row_position, col_position]
                if min_signal is not None and float(signal) < min_signal:
                    continue
                row = {
                    "x": float(col_index),
                    "y": float(row_index),
                    "z": 0.0,
                    "t": t,
                    "signal": float(signal),
                    "frame_index": frame_index,
                    "row_index": row_index,
                    "col_index": col_index,
                    **process,
                }
                if temperature_grid is not None:
                    row["temperature_C"] = float(temperature_grid[row_position, col_position])
                writer.writerow(row)
                rows_by_frame[frame_index].append(n_written)
                n_written += 1
    return n_written, rows_by_frame, sampling_frames


def _validate_sampling_config(
    sampling_mode: str,
    hot_quantile: float,
    gradient_quantile: float,
    background_fraction: float,
    max_points_per_frame: int | None,
) -> None:
    if sampling_mode not in {"uniform", "hot", "gradient", "hot_gradient", "balanced_hot_gradient"}:
        raise ValueError(f"Unsupported sampling mode: {sampling_mode}")
    for name, value in {"hot_quantile": hot_quantile, "gradient_quantile": gradient_quantile}.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    if not 0.0 <= background_fraction <= 1.0:
        raise ValueError(f"background_fraction must be in [0, 1], got {background_fraction}")
    if max_points_per_frame is not None and max_points_per_frame <= 0:
        raise ValueError("max_points_per_frame must be positive when provided")


def _select_sampled_grid_points(
    signal_grid: Any,
    target_grid: Any,
    row_indices: list[int],
    col_indices: list[int],
    min_signal: float | None,
    sampling_mode: str,
    hot_quantile: float,
    gradient_quantile: float,
    background_fraction: float,
    max_points_per_frame: int | None,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    import numpy as np

    signals = np.asarray(signal_grid, dtype=float)
    targets = np.asarray(target_grid, dtype=float)
    signal_mask = np.ones(signals.shape, dtype=bool)
    if min_signal is not None:
        signal_mask &= signals >= float(min_signal)

    if sampling_mode == "uniform":
        valid_mask = signal_mask
        selected_mask = valid_mask.copy()
        hot_mask = np.zeros(signals.shape, dtype=bool)
        gradient_mask = np.zeros(signals.shape, dtype=bool)
        background_mask = np.zeros(signals.shape, dtype=bool)
        hot_threshold = None
        gradient_threshold = None
    else:
        valid_mask = signal_mask & np.isfinite(signals) & np.isfinite(targets)
        hot_mask = np.zeros(signals.shape, dtype=bool)
        gradient_mask = np.zeros(signals.shape, dtype=bool)
        hot_threshold = None
        gradient_threshold = None
        if np.any(valid_mask) and sampling_mode in {"hot", "hot_gradient", "balanced_hot_gradient"}:
            hot_threshold = float(np.quantile(targets[valid_mask], hot_quantile))
            hot_mask = valid_mask & (targets >= hot_threshold)
        if np.any(valid_mask) and sampling_mode in {"gradient", "hot_gradient", "balanced_hot_gradient"}:
            gradient_scores = _sampled_grid_gradient_scores(
                targets=targets,
                row_indices=row_indices,
                col_indices=col_indices,
            )
            gradient_threshold = float(np.quantile(gradient_scores[valid_mask], gradient_quantile))
            gradient_mask = valid_mask & (gradient_scores >= gradient_threshold)
        selected_mask = hot_mask | gradient_mask
        background_mask = _background_anchor_mask(
            valid_mask=valid_mask,
            selected_mask=selected_mask,
            background_fraction=background_fraction,
        )
        selected_mask |= background_mask

    if max_points_per_frame is not None and int(np.count_nonzero(selected_mask)) > max_points_per_frame:
        selected_mask = _cap_selected_mask(selected_mask, max_points_per_frame)

    selected_flat = np.flatnonzero(selected_mask.ravel())
    selected_points = [
        (int(row_position), int(col_position))
        for row_position, col_position in zip(*np.unravel_index(selected_flat, signals.shape), strict=True)
    ]
    frame_sampling = {
        "mode": sampling_mode,
        "grid_points": int(signals.size),
        "valid_points": int(np.count_nonzero(valid_mask)),
        "written_points": len(selected_points),
        "hot_points": int(np.count_nonzero(hot_mask)),
        "gradient_points": int(np.count_nonzero(gradient_mask)),
        "background_points": int(np.count_nonzero(background_mask)),
        "hot_threshold": hot_threshold,
        "gradient_threshold": gradient_threshold,
    }
    return selected_points, frame_sampling


def _sampled_grid_gradient_scores(targets: Any, row_indices: list[int], col_indices: list[int]) -> Any:
    import numpy as np

    values = np.asarray(targets, dtype=float)
    scores = np.zeros(values.shape, dtype=float)
    for row_position in range(values.shape[0]):
        for neighbor_position in (row_position - 1, row_position + 1):
            if neighbor_position < 0 or neighbor_position >= values.shape[0]:
                continue
            distance = abs(row_indices[row_position] - row_indices[neighbor_position]) or 1
            scores[row_position, :] = np.maximum(
                scores[row_position, :],
                np.abs(values[row_position, :] - values[neighbor_position, :]) / distance,
            )
    for col_position in range(values.shape[1]):
        for neighbor_position in (col_position - 1, col_position + 1):
            if neighbor_position < 0 or neighbor_position >= values.shape[1]:
                continue
            distance = abs(col_indices[col_position] - col_indices[neighbor_position]) or 1
            scores[:, col_position] = np.maximum(
                scores[:, col_position],
                np.abs(values[:, col_position] - values[:, neighbor_position]) / distance,
            )
    return scores


def _background_anchor_mask(valid_mask: Any, selected_mask: Any, background_fraction: float) -> Any:
    import math
    import numpy as np

    background_mask = np.zeros(valid_mask.shape, dtype=bool)
    if background_fraction <= 0.0:
        return background_mask
    candidates = np.flatnonzero((valid_mask & ~selected_mask).ravel())
    if len(candidates) == 0:
        return background_mask
    n_background = min(len(candidates), int(math.ceil(background_fraction * int(np.count_nonzero(valid_mask)))))
    if n_background <= 0:
        return background_mask
    chosen = _spread_flat_indices(candidates, n_background)
    background_mask.ravel()[chosen] = True
    return background_mask


def _cap_selected_mask(selected_mask: Any, max_points: int) -> Any:
    import numpy as np

    capped = np.zeros(selected_mask.shape, dtype=bool)
    selected = np.flatnonzero(selected_mask.ravel())
    chosen = _spread_flat_indices(selected, max_points)
    capped.ravel()[chosen] = True
    return capped


def _spread_flat_indices(candidates: Any, count: int) -> list[int]:
    import numpy as np

    if count >= len(candidates):
        return [int(item) for item in candidates]
    positions = np.linspace(0, len(candidates) - 1, count)
    return [int(candidates[int(round(position))]) for position in positions]


def build_frame_split_manifest(
    rows_by_frame: dict[int, list[int]],
    sample_id: str,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, Any]:
    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Split fractions must sum to 1.0")
    frame_indices = [frame_index for frame_index, rows in rows_by_frame.items() if rows]
    if not frame_indices:
        raise ValueError("Frame split cannot be built because no rows were written")
    n_frames = len(frame_indices)
    train_end = int(round(train_fraction * n_frames))
    val_end = train_end + int(round(val_fraction * n_frames))
    frame_splits = {
        "train": frame_indices[:train_end],
        "val": frame_indices[train_end:val_end],
        "test": frame_indices[val_end:],
    }
    row_splits: dict[str, list[int]] = {}
    for split_name, split_frames in frame_splits.items():
        indices: list[int] = []
        for frame_index in frame_indices:
            if frame_index not in split_frames:
                continue
            indices.extend(rows_by_frame[frame_index])
        row_splits[split_name] = indices
    return {
        "sample_id": sample_id,
        "n_rows": sum(len(rows) for rows in rows_by_frame.values()),
        "strategy": "frame_order",
        "frame_splits": frame_splits,
        "rows_per_frame": {str(frame_index): len(rows) for frame_index, rows in rows_by_frame.items()},
        "splits": row_splits,
    }


def _sample_indices(start: int, step: int, stop: int, max_count: int | None) -> list[int]:
    if step <= 0:
        raise ValueError("Sampling step must be positive")
    if start < 0 or start >= stop:
        raise ValueError(f"Sampling start {start} is outside [0, {stop})")
    indices = list(range(start, stop, step))
    if max_count is not None:
        indices = indices[:max_count]
    if not indices:
        raise ValueError("Sampling configuration produced no indices")
    return indices


def _parent_path(dataset_path: str) -> str:
    if "/" not in dataset_path:
        raise ValueError(f"Dataset path has no parent group: {dataset_path}")
    return dataset_path.rsplit("/", 1)[0]


def _float_attr(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        value = value[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
        if not match:
            return default
        value = match.group(0)
    return float(value)


def _read_calibration(group: Any) -> dict[str, Any]:
    return {
        "coeff_a": _float_attr(group.attrs.get("Coeff_a")),
        "coeff_b": _float_attr(group.attrs.get("Coeff_b")),
        "coeff_c": _float_attr(group.attrs.get("Coeff_c")),
        "model": _decode_attr(group.attrs.get("Model")),
        "model_input": _decode_attr(group.attrs.get("Model_input")),
        "model_output": _decode_attr(group.attrs.get("Model_output")),
        "method": _decode_attr(group.attrs.get("Cal_Method")),
    }


def _decode_attr(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thermal-hdf5", type=Path, default=DEFAULT_THERMAL_HDF5)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--sample-id", default="amb2022_03_line_0_1_signal_subset")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=100)
    parser.add_argument("--max-frames", type=int, default=5)
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--row-step", type=int, default=80)
    parser.add_argument("--max-rows", type=int, default=8)
    parser.add_argument("--col-start", type=int, default=0)
    parser.add_argument("--col-step", type=int, default=38)
    parser.add_argument("--max-cols", type=int, default=8)
    parser.add_argument(
        "--calibrate-temperature",
        action="store_true",
        help="Add temperature_C using Calibration/ThermalCal attributes.",
    )
    parser.add_argument(
        "--min-signal",
        type=float,
        help="Drop sampled points with raw signal below this threshold before writing rows.",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=["uniform", "hot", "gradient", "hot_gradient", "balanced_hot_gradient"],
        default="uniform",
        help="Select points from the sampled frame grid. uniform preserves the legacy behavior.",
    )
    parser.add_argument(
        "--hot-quantile",
        type=float,
        default=0.9,
        help="Within-frame target quantile used by hot and hot_gradient sampling modes.",
    )
    parser.add_argument(
        "--gradient-quantile",
        type=float,
        default=0.9,
        help="Within-frame spatial-gradient quantile used by gradient and hot_gradient sampling modes.",
    )
    parser.add_argument(
        "--background-fraction",
        type=float,
        default=0.1,
        help="Fraction of valid non-active grid points retained as background anchors in active modes.",
    )
    parser.add_argument(
        "--max-points-per-frame",
        type=int,
        help="Optional deterministic cap after active/background selection.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["random_row", "frame"],
        default="random_row",
        help="Split strategy used when --split-manifest is provided.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = convert_thermography_hdf5(args)
    text = json.dumps(manifest, indent=2, ensure_ascii=False)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(text, encoding="utf-8")
        print(f"Wrote: {args.manifest}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
