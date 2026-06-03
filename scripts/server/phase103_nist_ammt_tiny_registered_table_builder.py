#!/usr/bin/env python3
"""Phase 103 NIST AMMT tiny registered table builder.

This no-training builder consumes the deep/join Phase 103 artifacts and creates
a tiny registered source/path-to-target table plus a leakage-safe split manifest.
It does not read raw ZIPs, run baselines, or open training gates.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


TABLE_FIELDS = (
    "row_id",
    "source_layer_index",
    "split_name",
    "source_file_name",
    "source_group_key",
    "source_member_name",
    "target_file_name",
    "target_group_key",
    "target_member_name",
    "target_type",
    "target_layer_index",
    "join_offset",
    "source_coverage",
    "target_coverage",
    "target_width",
    "target_height",
    "target_bits_per_pixel",
    "target_channels",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


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


def _group_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["group_key"]: row for row in rows}


def _indexed_member_name(group_key: str, index: int, zero_padded_width: int) -> str:
    return group_key.replace("{index}", f"{index:0{zero_padded_width}d}")


def _target_sample_for_group(
    target_group_key: str,
    rows: list[dict[str, str]],
) -> dict[str, str] | None:
    if "{index}" not in target_group_key:
        return None
    prefix, suffix = target_group_key.split("{index}", 1)
    for row in rows:
        member_name = row.get("member_name", "")
        if member_name.startswith(prefix) and member_name.endswith(suffix):
            return row
    return None


def _build_rows(
    *,
    sequence_groups: list[dict[str, str]],
    join_rows: list[dict[str, str]],
    target_samples: list[dict[str, str]],
    rows_per_target_type: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sequence_by_group = _group_lookup(sequence_groups)
    rows: list[dict[str, Any]] = []
    split_groups: list[int] = []

    for join_row in join_rows:
        if join_row["target_type"] != "layer_camera":
            continue
        if join_row.get("join_evidence_status") != "source_target_layer_join_ready":
            continue
        source_group_key = join_row["source_group_key"]
        source_row = sequence_by_group[source_group_key]
        source_first = int(source_row["first_index"])
        source_last = int(source_row["last_index"])
        source_zero_pad = int(source_row["zero_padded_width"])
        source_file_name = source_row["file_name"]
        target_group_key = join_row["target_group_key"]
        target_group_row = sequence_by_group[target_group_key]
        target_first = int(target_group_row["first_index"])
        target_last = int(target_group_row["last_index"])
        target_file_name = target_group_row["file_name"]
        target_zero_pad = int(target_group_row["zero_padded_width"])
        offset = int(join_row["best_source_minus_target_offset"])
        target_sample = _target_sample_for_group(target_group_key, target_samples)
        if target_sample is None:
            continue
        for target_index in range(target_first, min(target_last, target_first + rows_per_target_type - 1) + 1):
            source_index = target_index + offset
            if source_index < source_first or source_index > source_last:
                continue
            source_member_name = _indexed_member_name(
                source_group_key,
                source_index,
                source_zero_pad,
            )
            target_member_name = _indexed_member_name(
                target_group_key,
                target_index,
                target_zero_pad,
            )
            rows.append(
                {
                    "row_id": f"{target_group_key}::{target_index:04d}",
                    "source_layer_index": source_index,
                    "split_name": "",
                    "source_file_name": source_file_name,
                    "source_group_key": source_group_key,
                    "source_member_name": source_member_name,
                    "target_file_name": target_file_name,
                    "target_group_key": target_group_key,
                    "target_member_name": target_member_name,
                    "target_type": join_row["target_type"],
                    "target_layer_index": target_index,
                    "join_offset": offset,
                    "source_coverage": float(join_row["source_coverage"]),
                    "target_coverage": float(join_row["target_coverage"]),
                    "target_width": int(target_sample["width"]),
                    "target_height": int(target_sample["height"]),
                    "target_bits_per_pixel": int(target_sample["bits_per_pixel"]),
                    "target_channels": int(target_sample["channels"]),
                }
            )
            split_groups.append(source_index)

    split_groups_sorted = sorted(set(split_groups))
    split_map: dict[int, str] = {}
    if split_groups_sorted:
        train_cut = max(1, int(len(split_groups_sorted) * 0.6))
        val_cut = max(train_cut + 1, int(len(split_groups_sorted) * 0.8))
        for idx, group in enumerate(split_groups_sorted):
            if idx < train_cut:
                split_map[group] = "train"
            elif idx < val_cut:
                split_map[group] = "val"
            else:
                split_map[group] = "test"
    for row in rows:
        row["split_name"] = split_map.get(int(row["source_layer_index"]), "train")

    group_to_splits: dict[int, set[str]] = {}
    for row in rows:
        group_to_splits.setdefault(int(row["source_layer_index"]), set()).add(
            str(row["split_name"])
        )
    leakage_safe = all(len(splits) == 1 for splits in group_to_splits.values())
    split_manifest = {
        "groups": [
            {
                "source_layer_index": group,
                "split_name": split_map[group],
            }
            for group in split_groups_sorted
        ],
        "split_counts": {
            split: sum(1 for row in rows if row["split_name"] == split)
            for split in ("train", "val", "test")
        },
        "group_count": len(split_groups_sorted),
        "row_count": len(rows),
        "leakage_group": "source_layer_index",
        "leakage_safe": leakage_safe,
    }
    return rows, split_manifest


def _build_gate(rows: list[dict[str, Any]], split_manifest: dict[str, Any], join_gate: dict[str, Any]) -> dict[str, Any]:
    ready = (
        bool(rows)
        and bool(split_manifest.get("leakage_safe"))
        and bool(join_gate.get("source_target_join_ready"))
    )
    status = (
        "tiny_registered_table_ready_manual_baseline_pending"
        if ready
        else "tiny_registered_table_build_failed"
    )
    return {
        "status": status,
        "tiny_registered_table_ready": ready,
        "leakage_safe_split_manifest_ready": bool(split_manifest.get("leakage_safe")),
        "phase104_baseline_smoke_allowed": False,
        "phase105_model_mechanism_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "source_target_join_ready": bool(join_gate.get("source_target_join_ready")),
        "explicit_absolute_timing_ready": bool(join_gate.get("explicit_absolute_timing_ready")),
        "row_count": len(rows),
        "split_group_count": int(split_manifest.get("group_count") or 0),
        "split_counts": split_manifest.get("split_counts", {}),
        "table_scope": "layer_member_source_target_join",
        "next_action": (
            "validate the tiny layer-member table and split manifest before any separate "
            "Phase 104 baseline-smoke decision"
        ),
    }


def _write_markdown(
    path: Path,
    gate: dict[str, Any],
    split_manifest: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    split_counts = split_manifest.get("split_counts", {})
    lines = [
        "# Phase 103 Tiny Registered Source-Target Table",
        "",
        f"- Status: `{gate['status']}`",
        f"- Tiny registered table ready: `{gate['tiny_registered_table_ready']}`",
        f"- Leakage-safe split manifest ready: `{gate['leakage_safe_split_manifest_ready']}`",
        f"- Row count: `{gate['row_count']}`",
        f"- Split group count: `{gate['split_group_count']}`",
        f"- Split counts: `train={split_counts.get('train', 0)}, val={split_counts.get('val', 0)}, test={split_counts.get('test', 0)}`",
        "- Phase 104 baseline smoke allowed: `false`",
        "- A100 training allowed now: `false`",
        f"- Next action: {gate['next_action']}",
        "",
        "This package is no-training evidence. It joins source command layer members to "
        "layer-camera target members by audited integer layer offsets and writes a split "
        "manifest grouped by `source_layer_index`; it does not read raw ZIP payloads, run "
        "baselines, train models, or open an A100 training gate.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_package(
    *,
    root: Path,
    sequence_groups_csv: Path,
    join_probe_gate_path: Path,
    target_binary_samples_csv: Path,
    output_dir: Path,
    rows_per_target_type: int = 12,
) -> dict[str, Any]:
    sequence_groups = _read_csv(sequence_groups_csv)
    join_gate = _read_json(join_probe_gate_path)
    target_samples = _read_csv(target_binary_samples_csv)
    join_rows = _read_csv(
        output_dir / "phase103_nist_ammt_source_target_join_candidates.csv"
    )
    rows, split_manifest = _build_rows(
        sequence_groups=sequence_groups,
        join_rows=join_rows,
        target_samples=target_samples,
        rows_per_target_type=rows_per_target_type,
    )
    gate = _build_gate(rows, split_manifest, join_gate)

    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "phase103_nist_ammt_tiny_registered_source_target_table.csv"
    split_path = output_dir / "phase103_nist_ammt_tiny_registered_split_manifest.json"
    gate_path = output_dir / "phase103_nist_ammt_tiny_registered_table_gate.json"
    markdown_path = output_dir / "phase103_nist_ammt_tiny_registered_table_summary.md"
    manifest_path = output_dir / "phase103_nist_ammt_tiny_registered_table_manifest.json"
    _write_csv(table_path, rows, TABLE_FIELDS)
    _write_json(split_path, split_manifest)
    _write_json(gate_path, gate)
    _write_markdown(markdown_path, gate, split_manifest)
    manifest = {
        "phase": 103,
        "objective": "nist_ammt_tiny_registered_source_target_table_build_no_training",
        "inputs": {
            "sequence_groups": _display_path(sequence_groups_csv, root),
            "join_probe_gate": _display_path(join_probe_gate_path, root),
            "target_binary_samples": _display_path(target_binary_samples_csv, root),
        },
        "outputs": {
            "tiny_table": _display_path(table_path, root),
            "split_manifest": _display_path(split_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown_summary": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "row_count": len(rows),
            "split_groups": split_manifest["group_count"],
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
        "--join-probe-gate",
        type=Path,
        default=Path(
            "docs/results/phase103_nist_ammt_registered_intake/"
            "phase103_nist_ammt_join_probe_gate.json"
        ),
    )
    parser.add_argument(
        "--target-binary-samples",
        type=Path,
        default=Path(
            "docs/results/phase103_nist_ammt_registered_intake/"
            "phase103_nist_ammt_deep_target_binary_samples.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase103_nist_ammt_registered_intake"),
    )
    parser.add_argument("--rows-per-target-type", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    sequence_groups_csv = args.sequence_groups_csv
    join_probe_gate = args.join_probe_gate
    target_binary_samples = args.target_binary_samples
    output_dir = args.output_dir
    if not sequence_groups_csv.is_absolute():
        sequence_groups_csv = root / sequence_groups_csv
    if not join_probe_gate.is_absolute():
        join_probe_gate = root / join_probe_gate
    if not target_binary_samples.is_absolute():
        target_binary_samples = root / target_binary_samples
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    manifest = build_package(
        root=root,
        sequence_groups_csv=sequence_groups_csv,
        join_probe_gate_path=join_probe_gate,
        target_binary_samples_csv=target_binary_samples,
        output_dir=output_dir,
        rows_per_target_type=args.rows_per_target_type,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
