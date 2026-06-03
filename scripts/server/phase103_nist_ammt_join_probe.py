#!/usr/bin/env python3
"""Phase 103 NIST AMMT source-target join probe.

This no-training probe consumes the deep registration probe sequence-group CSV
and searches for auditable integer-index joins between source/path layer command
files and target image sequences. It does not read raw image pixels, build sample
tables, run baselines, or open training gates.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


JOIN_FIELDS = (
    "source_group_key",
    "target_group_key",
    "target_type",
    "source_first_index",
    "source_last_index",
    "source_count",
    "target_first_index",
    "target_last_index",
    "target_count",
    "best_source_minus_target_offset",
    "matched_pairs",
    "source_coverage",
    "target_coverage",
    "first_pair",
    "last_pair",
    "join_evidence_status",
)


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


def _int(row: dict[str, str], key: str) -> int:
    return int(str(row[key]).strip())


def _source_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("file_name") == "Build Command Data.zip"
        and "XYPT Commands" in row.get("group_key", "")
        and "layer{index}" in row.get("group_key", "")
    ]


def _layer_camera_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("file_name") == "In-situ Meas Data.zip"
        and "Layer Camera" in row.get("group_key", "")
        and row.get("extension") == ".bmp"
    ]


def _melt_pool_layer_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    for row in rows:
        key = row.get("group_key", "")
        match = re.search(r"Melt Pool Camera/MIA_L(\d+)/frame", key)
        if not match:
            continue
        grouped.append(
            {
                "layer_index": int(match.group(1)),
                "group_key": key,
                "frame_count": _int(row, "count"),
            }
        )
    if not grouped:
        return []
    grouped.sort(key=lambda item: item["layer_index"])
    return [
        {
            "group_key": "In-situ Meas Data/Melt Pool Camera/MIA_L{layer}/frame{index}.bmp",
            "first_index": grouped[0]["layer_index"],
            "last_index": grouped[-1]["layer_index"],
            "count": len(grouped),
            "frame_count_min": min(item["frame_count"] for item in grouped),
            "frame_count_max": max(item["frame_count"] for item in grouped),
        }
    ]


def _best_offset(
    *,
    source_first: int,
    source_last: int,
    target_first: int,
    target_last: int,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for offset in range(source_first - target_last, source_last - target_first + 1):
        pairs = [
            (target_index + offset, target_index)
            for target_index in range(target_first, target_last + 1)
            if source_first <= target_index + offset <= source_last
        ]
        if not pairs:
            continue
        candidate = {
            "best_source_minus_target_offset": offset,
            "matched_pairs": len(pairs),
            "first_pair": f"source_layer={pairs[0][0]};target_index={pairs[0][1]}",
            "last_pair": f"source_layer={pairs[-1][0]};target_index={pairs[-1][1]}",
        }
        if best is None:
            best = candidate
            continue
        if candidate["matched_pairs"] > best["matched_pairs"]:
            best = candidate
        elif candidate["matched_pairs"] == best["matched_pairs"] and abs(offset) < abs(
            int(best["best_source_minus_target_offset"])
        ):
            best = candidate
    return best or {
        "best_source_minus_target_offset": "",
        "matched_pairs": 0,
        "first_pair": "",
        "last_pair": "",
    }


def _join_row(
    *,
    source: dict[str, str],
    target_group_key: str,
    target_type: str,
    target_first: int,
    target_last: int,
    target_count: int,
    min_target_coverage: float,
    min_pairs: int,
) -> dict[str, Any]:
    source_first = _int(source, "first_index")
    source_last = _int(source, "last_index")
    source_count = _int(source, "count")
    best = _best_offset(
        source_first=source_first,
        source_last=source_last,
        target_first=target_first,
        target_last=target_last,
    )
    matched_pairs = int(best["matched_pairs"])
    source_coverage = matched_pairs / max(source_count, 1)
    target_coverage = matched_pairs / max(target_count, 1)
    ready = target_coverage >= min_target_coverage and matched_pairs >= min_pairs
    return {
        "source_group_key": source["group_key"],
        "target_group_key": target_group_key,
        "target_type": target_type,
        "source_first_index": source_first,
        "source_last_index": source_last,
        "source_count": source_count,
        "target_first_index": target_first,
        "target_last_index": target_last,
        "target_count": target_count,
        **best,
        "source_coverage": source_coverage,
        "target_coverage": target_coverage,
        "join_evidence_status": "source_target_layer_join_ready" if ready else "partial_join_only",
    }


def _build_join_rows(
    *,
    sequence_rows: list[dict[str, str]],
    min_target_coverage: float,
    min_layer_pairs: int,
    min_melt_pool_pairs: int,
) -> list[dict[str, Any]]:
    sources = _source_rows(sequence_rows)
    rows: list[dict[str, Any]] = []
    for source in sources:
        for target in _layer_camera_rows(sequence_rows):
            rows.append(
                _join_row(
                    source=source,
                    target_group_key=target["group_key"],
                    target_type="layer_camera",
                    target_first=_int(target, "first_index"),
                    target_last=_int(target, "last_index"),
                    target_count=_int(target, "count"),
                    min_target_coverage=min_target_coverage,
                    min_pairs=min_layer_pairs,
                )
            )
        for target in _melt_pool_layer_rows(sequence_rows):
            rows.append(
                _join_row(
                    source=source,
                    target_group_key=str(target["group_key"]),
                    target_type="melt_pool_camera_layer_directory",
                    target_first=int(target["first_index"]),
                    target_last=int(target["last_index"]),
                    target_count=int(target["count"]),
                    min_target_coverage=min_target_coverage,
                    min_pairs=min_melt_pool_pairs,
                )
            )
    return rows


def _build_gate(join_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ready_rows = [
        row
        for row in join_rows
        if row["join_evidence_status"] == "source_target_layer_join_ready"
    ]
    layer_ready = any(row["target_type"] == "layer_camera" for row in ready_rows)
    melt_pool_ready = any(
        row["target_type"] == "melt_pool_camera_layer_directory" for row in ready_rows
    )
    source_target_join_ready = bool(ready_rows)
    if source_target_join_ready:
        status = "source_target_layer_join_ready_timing_not_absolute"
        next_action = (
            "construct a tiny layer-index registered table candidate; keep baseline/training locked"
        )
    else:
        status = "source_target_join_incomplete"
        next_action = "inspect additional metadata for source-target timing or layer joins"
    return {
        "status": status,
        "source_target_join_ready": source_target_join_ready,
        "explicit_absolute_timing_ready": False,
        "layer_camera_join_ready": layer_ready,
        "melt_pool_layer_join_ready": melt_pool_ready,
        "ready_join_rows": len(ready_rows),
        "join_candidate_rows": len(join_rows),
        "phase104_baseline_smoke_allowed": False,
        "phase105_model_mechanism_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": next_action,
        "required_before_baseline_smoke": [
            "tiny registered source/path-to-target sample table generated",
            "coordinate transform values applied to sample rows",
            "source-target join applied to sample rows",
            "leakage-safe split manifest generated",
            "baseline smoke plan validated without test-label route selection",
        ],
    }


def build_package(
    *,
    root: Path,
    sequence_groups_csv: Path,
    output_dir: Path,
    min_target_coverage: float = 0.95,
    min_layer_pairs: int = 20,
    min_melt_pool_pairs: int = 5,
) -> dict[str, Any]:
    sequence_rows = _read_csv(sequence_groups_csv) if sequence_groups_csv.exists() else []
    join_rows = _build_join_rows(
        sequence_rows=sequence_rows,
        min_target_coverage=min_target_coverage,
        min_layer_pairs=min_layer_pairs,
        min_melt_pool_pairs=min_melt_pool_pairs,
    )
    gate = _build_gate(join_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    join_path = output_dir / "phase103_nist_ammt_source_target_join_candidates.csv"
    gate_path = output_dir / "phase103_nist_ammt_join_probe_gate.json"
    manifest_path = output_dir / "phase103_nist_ammt_join_probe_manifest.json"
    _write_csv(join_path, join_rows, JOIN_FIELDS)
    _write_json(gate_path, gate)
    manifest = {
        "phase": 103,
        "objective": "nist_ammt_source_target_join_probe_no_training",
        "inputs": {"sequence_groups": _display_path(sequence_groups_csv, root)},
        "outputs": {
            "join_candidates": _display_path(join_path, root),
            "gate_json": _display_path(gate_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "limits": {
            "min_target_coverage": min_target_coverage,
            "min_layer_pairs": min_layer_pairs,
            "min_melt_pool_pairs": min_melt_pool_pairs,
        },
        "counts": {
            "sequence_group_rows": len(sequence_rows),
            "join_candidate_rows": len(join_rows),
        },
        "gate": gate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--sequence-groups-csv",
        type=Path,
        default=Path(
            "docs/results/phase103_nist_ammt_registered_intake/"
            "phase103_nist_ammt_deep_sequence_groups.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase103_nist_ammt_registered_intake"),
    )
    parser.add_argument("--min-target-coverage", type=float, default=0.95)
    parser.add_argument("--min-layer-pairs", type=int, default=20)
    parser.add_argument("--min-melt-pool-pairs", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    sequence_groups_csv = args.sequence_groups_csv
    output_dir = args.output_dir
    if not sequence_groups_csv.is_absolute():
        sequence_groups_csv = root / sequence_groups_csv
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    manifest = build_package(
        root=root,
        sequence_groups_csv=sequence_groups_csv,
        output_dir=output_dir,
        min_target_coverage=args.min_target_coverage,
        min_layer_pairs=args.min_layer_pairs,
        min_melt_pool_pairs=args.min_melt_pool_pairs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
