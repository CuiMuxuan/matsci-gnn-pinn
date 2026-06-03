#!/usr/bin/env python3
"""Phase 104 NIST AMMT tiny numeric field-table builder.

This no-training builder turns the Phase 103 member-level registered table into
a small numeric field table suitable for baseline smoke tests. It only reads
registered ZIP members, writes small CSV/JSON artifacts, and does not train.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import zipfile
from pathlib import Path
from typing import Any, BinaryIO


FIELD_COLUMNS = (
    "x",
    "y",
    "t",
    "target_intensity_mean",
    "target_intensity_std",
    "target_intensity_min",
    "target_intensity_max",
    "target_intensity_q90",
    "source_layer_index",
    "target_layer_index",
    "target_camera_code",
    "source_x_mean",
    "source_y_mean",
    "source_p_mean",
    "source_t_mean",
    "source_x_range",
    "source_y_range",
    "source_p_range",
    "source_t_range",
    "source_p_nonzero_fraction",
    "source_rows_read",
    "source_rows_truncated",
    "target_pixel_count",
    "target_width",
    "target_height",
    "target_bits_per_pixel",
    "split_name",
    "row_id",
    "source_member_name",
    "target_member_name",
)

BASELINE_SPLITS = ("train", "val", "test")


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
        return f"{value:.6f}"
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


def _zip_path(data_root: Path, file_name: str) -> Path:
    return data_root / file_name


def _source_stats(handle: BinaryIO, max_source_rows: int | None) -> dict[str, Any]:
    count = 0
    sums = [0.0, 0.0, 0.0, 0.0]
    mins = [math.inf, math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf, -math.inf]
    p_nonzero = 0
    truncated = False
    text_handle = (line.decode("utf-8", errors="replace") for line in handle)
    for line in text_handle:
        stripped = line.strip()
        if not stripped:
            continue
        parts = [item.strip() for item in stripped.split(",")]
        if len(parts) < 4:
            continue
        try:
            values = [float(parts[index]) for index in range(4)]
        except ValueError:
            continue
        count += 1
        for index, value in enumerate(values):
            sums[index] += value
            mins[index] = min(mins[index], value)
            maxs[index] = max(maxs[index], value)
        if abs(values[2]) > 0.0:
            p_nonzero += 1
        if max_source_rows is not None and count >= max_source_rows:
            truncated = True
            break
    if count == 0:
        raise ValueError("No numeric XYPT rows were read")
    means = [value / count for value in sums]
    return {
        "source_x_mean": means[0],
        "source_y_mean": means[1],
        "source_p_mean": means[2],
        "source_t_mean": means[3],
        "source_x_range": maxs[0] - mins[0],
        "source_y_range": maxs[1] - mins[1],
        "source_p_range": maxs[2] - mins[2],
        "source_t_range": maxs[3] - mins[3],
        "source_p_nonzero_fraction": p_nonzero / count,
        "source_rows_read": count,
        "source_rows_truncated": truncated,
    }


def _bmp_stats(payload: bytes) -> dict[str, Any]:
    if payload[:2] != b"BM":
        raise ValueError("Expected BMP target member")
    pixel_offset = struct.unpack_from("<I", payload, 10)[0]
    width = struct.unpack_from("<i", payload, 18)[0]
    height_raw = struct.unpack_from("<i", payload, 22)[0]
    bits_per_pixel = struct.unpack_from("<H", payload, 28)[0]
    if bits_per_pixel != 8:
        raise ValueError(f"Only 8-bit BMP targets are supported, got {bits_per_pixel}")
    height = abs(height_raw)
    row_stride = ((width * bits_per_pixel + 31) // 32) * 4
    pixel_count = width * height
    hist = [0 for _ in range(256)]
    for row_index in range(height):
        start = pixel_offset + row_index * row_stride
        row = payload[start : start + width]
        for value in row:
            hist[value] += 1
    observed = sum(hist)
    if observed != pixel_count:
        raise ValueError(f"BMP pixel count mismatch: observed {observed}, expected {pixel_count}")
    total = sum(index * count for index, count in enumerate(hist))
    mean = total / pixel_count
    variance = sum(((index - mean) ** 2) * count for index, count in enumerate(hist)) / pixel_count
    return {
        "target_intensity_mean": mean,
        "target_intensity_std": math.sqrt(variance),
        "target_intensity_min": next(index for index, count in enumerate(hist) if count),
        "target_intensity_max": next(index for index in range(255, -1, -1) if hist[index]),
        "target_intensity_q90": _hist_quantile(hist, 0.9),
        "target_pixel_count": pixel_count,
        "target_width": width,
        "target_height": height,
        "target_bits_per_pixel": bits_per_pixel,
    }


def _hist_quantile(hist: list[int], quantile: float) -> float:
    threshold = quantile * (sum(hist) - 1)
    cumulative = 0
    for index, count in enumerate(hist):
        cumulative += count
        if cumulative - 1 >= threshold:
            return float(index)
    return 255.0


def _camera_code(member_name: str) -> int:
    name = Path(member_name).name
    if name.startswith("A"):
        return 0
    if name.startswith("B"):
        return 1
    return -1


def build_numeric_rows(
    *,
    tiny_rows: list[dict[str, str]],
    data_root: Path,
    max_source_rows: int | None,
) -> list[dict[str, Any]]:
    source_cache: dict[tuple[str, str], dict[str, Any]] = {}
    target_cache: dict[tuple[str, str], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for tiny_row in tiny_rows:
        source_key = (tiny_row["source_file_name"], tiny_row["source_member_name"])
        target_key = (tiny_row["target_file_name"], tiny_row["target_member_name"])
        if source_key not in source_cache:
            with zipfile.ZipFile(_zip_path(data_root, source_key[0])) as archive:
                with archive.open(source_key[1]) as handle:
                    source_cache[source_key] = _source_stats(handle, max_source_rows)
        if target_key not in target_cache:
            with zipfile.ZipFile(_zip_path(data_root, target_key[0])) as archive:
                target_cache[target_key] = _bmp_stats(archive.read(target_key[1]))
        source = source_cache[source_key]
        target = target_cache[target_key]
        rows.append(
            {
                "x": source["source_x_mean"],
                "y": source["source_y_mean"],
                "t": int(tiny_row["source_layer_index"]),
                **target,
                "source_layer_index": int(tiny_row["source_layer_index"]),
                "target_layer_index": int(tiny_row["target_layer_index"]),
                "target_camera_code": _camera_code(tiny_row["target_member_name"]),
                **source,
                "split_name": tiny_row["split_name"],
                "row_id": tiny_row["row_id"],
                "source_member_name": tiny_row["source_member_name"],
                "target_member_name": tiny_row["target_member_name"],
            }
        )
    return rows


def _split_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    splits = {
        split: [index for index, row in enumerate(rows) if row["split_name"] == split]
        for split in BASELINE_SPLITS
    }
    by_source: dict[int, set[str]] = {}
    for row in rows:
        by_source.setdefault(int(row["source_layer_index"]), set()).add(str(row["split_name"]))
    return {
        "splits": splits,
        "split_counts": {split: len(indices) for split, indices in splits.items()},
        "leakage_group": "source_layer_index",
        "leakage_safe": all(len(split_names) == 1 for split_names in by_source.values()),
        "row_count": len(rows),
        "group_count": len(by_source),
    }


def _build_gate(rows: list[dict[str, Any]], split_manifest: dict[str, Any]) -> dict[str, Any]:
    ready = bool(rows) and bool(split_manifest.get("leakage_safe"))
    status = "phase104_numeric_field_table_ready_baseline_pending" if ready else "phase104_numeric_field_table_blocked"
    return {
        "status": status,
        "numeric_field_table_ready": ready,
        "baseline_smoke_ready": ready,
        "baseline_smoke_completed": False,
        "phase105_model_mechanism_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "row_count": len(rows),
        "split_counts": split_manifest.get("split_counts", {}),
        "leakage_safe_split_manifest_ready": bool(split_manifest.get("leakage_safe")),
        "target": "target_intensity_mean",
        "next_action": "run mean/kNN/ExtraTrees baseline smoke on the numeric tiny field table",
    }


def _write_markdown(path: Path, gate: dict[str, Any]) -> None:
    split_counts = gate.get("split_counts", {})
    lines = [
        "# Phase 104 NIST AMMT Tiny Numeric Field Table",
        "",
        f"- Status: `{gate['status']}`",
        f"- Numeric field table ready: `{gate['numeric_field_table_ready']}`",
        f"- Row count: `{gate['row_count']}`",
        f"- Split counts: `train={split_counts.get('train', 0)}, val={split_counts.get('val', 0)}, test={split_counts.get('test', 0)}`",
        f"- Leakage-safe split manifest ready: `{gate['leakage_safe_split_manifest_ready']}`",
        "- Phase 105 model mechanisms allowed: `false`",
        "- A100 training allowed now: `false`",
        "",
        "This package converts only the Phase 103 registered member rows into numeric source and target summaries. It does not run model mechanisms or training.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_package(
    *,
    root: Path,
    data_root: Path,
    tiny_table_csv: Path,
    output_dir: Path,
    max_source_rows: int | None,
) -> dict[str, Any]:
    tiny_rows = _read_csv(tiny_table_csv)
    rows = build_numeric_rows(
        tiny_rows=tiny_rows,
        data_root=data_root,
        max_source_rows=max_source_rows,
    )
    split_manifest = _split_manifest(rows)
    gate = _build_gate(rows, split_manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "phase104_nist_ammt_tiny_numeric_field_table.csv"
    split_path = output_dir / "phase104_nist_ammt_tiny_numeric_split_manifest.json"
    gate_path = output_dir / "phase104_nist_ammt_tiny_numeric_field_gate.json"
    markdown_path = output_dir / "phase104_nist_ammt_tiny_numeric_field_summary.md"
    manifest_path = output_dir / "phase104_nist_ammt_tiny_numeric_field_manifest.json"
    _write_csv(table_path, rows, FIELD_COLUMNS)
    _write_json(split_path, split_manifest)
    _write_json(gate_path, gate)
    _write_markdown(markdown_path, gate)
    manifest = {
        "phase": 104,
        "objective": "nist_ammt_tiny_numeric_field_table_no_training",
        "inputs": {
            "tiny_registered_table": _display_path(tiny_table_csv, root),
            "data_root": _display_path(data_root, root),
        },
        "outputs": {
            "field_table": _display_path(table_path, root),
            "split_manifest": _display_path(split_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown_summary": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "limits": {"max_source_rows": max_source_rows},
        "counts": {
            "row_count": len(rows),
            "source_layers": len({row["source_layer_index"] for row in rows}),
            "target_members": len({row["target_member_name"] for row in rows}),
        },
        "gate": gate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/nist_ammt/mds2_2044"))
    parser.add_argument(
        "--tiny-table",
        type=Path,
        default=Path(
            "docs/results/phase103_nist_ammt_registered_intake/"
            "phase103_nist_ammt_tiny_registered_source_target_table.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase104_nist_ammt_baseline_smoke"),
    )
    parser.add_argument(
        "--max-source-rows",
        type=int,
        default=0,
        help="Maximum XYPT rows read per source member; 0 means read all rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    data_root = args.data_root if args.data_root.is_absolute() else root / args.data_root
    tiny_table = args.tiny_table if args.tiny_table.is_absolute() else root / args.tiny_table
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    max_source_rows = args.max_source_rows if args.max_source_rows > 0 else None
    manifest = build_package(
        root=root,
        data_root=data_root,
        tiny_table_csv=tiny_table,
        output_dir=output_dir,
        max_source_rows=max_source_rows,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
