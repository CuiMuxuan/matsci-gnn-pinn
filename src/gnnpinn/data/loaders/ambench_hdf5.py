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
        n_written, rows_by_frame = _write_sampled_signal_csv(
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
) -> tuple[int, dict[int, list[int]]]:
    n_written = 0
    rows_by_frame: dict[int, list[int]] = {frame_index: [] for frame_index in frame_indices}
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame_index in frame_indices:
            frame = dataset[frame_index, :, :]
            t = frame_index / frame_rate_hz if frame_rate_hz else float(frame_index)
            for row_index in row_indices:
                values = frame[row_index, col_indices]
                temperatures = None
                if calibration is not None:
                    temperatures = calibrate_signal_to_temperature_c(
                        values,
                        coeff_a=float(calibration["coeff_a"]),
                        coeff_b=float(calibration["coeff_b"]),
                        coeff_c=float(calibration["coeff_c"]),
                    )
                for value_index, (col_index, signal) in enumerate(zip(col_indices, values, strict=True)):
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
                    if temperatures is not None:
                        row["temperature_C"] = float(temperatures[value_index])
                    writer.writerow(row)
                    rows_by_frame[frame_index].append(n_written)
                    n_written += 1
    return n_written, rows_by_frame


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
