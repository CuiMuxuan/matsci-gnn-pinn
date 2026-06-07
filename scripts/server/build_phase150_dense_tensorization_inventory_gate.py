#!/usr/bin/env python3
"""Build Phase 150 dense tensorization inventory gate.

This phase audits whether existing server-local dense sources can support a
future neural-operator branch. It reads only file metadata, ZIP central
directories, small BMP headers, existing small artifacts, and bounded CSV
previews. It does not tensorize full fields, train models, or request larger
GPUs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import struct
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("docs/results/phase150_dense_tensorization_inventory_gate")

PHASE_INPUTS = {
    "phase149_gate": Path(
        "docs/results/phase149_neural_operator_readiness_gate/"
        "phase149_neural_operator_readiness_gate.json"
    ),
    "phase116_gate": Path(
        "docs/results/phase116_paper_evidence_consolidation/"
        "phase116_paper_evidence_consolidation_gate.json"
    ),
    "phase116_positive_floor": Path(
        "docs/results/phase116_paper_evidence_consolidation/phase116_positive_floor_table.csv"
    ),
    "phase106_gate": Path(
        "docs/results/phase106_nist_ammt_spatial_target_representation_gate/"
        "phase106_nist_ammt_spatial_target_gate.json"
    ),
    "phase148_gate": Path(
        "docs/results/phase148_nist_ammt_path_contact_graph_audit/"
        "phase148_nist_ammt_path_contact_graph_audit_gate.json"
    ),
    "phase53_pivot": Path("docs/results/ambench_source_path_data_pivot_gate_v1.md"),
}

DEFAULT_CANDIDATE_SOURCES: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "nist_ammt_layer_camera_bmp_zip",
        "source_kind": "zip_bmp_members",
        "path": Path("data/raw/nist_ammt/mds2_2044/In-situ Meas Data.zip"),
        "member_needles": ("layer camera", "layer_camera", "layercamera"),
        "target_family": "NIST AMMT registered Layer Camera BMP targets",
    },
    {
        "candidate_id": "nist_ammt_melt_pool_bmp_zip",
        "source_kind": "zip_bmp_members",
        "path": Path("data/raw/nist_ammt/mds2_2044/In-situ Meas Data.zip"),
        "member_needles": ("melt pool", "melt_pool", "meltpool"),
        "target_family": "NIST AMMT Melt Pool Camera image sequence targets",
    },
    {
        "candidate_id": "ambench_mds2_2716_raw_thermography_hdf5",
        "source_kind": "hdf5_dataset",
        "path": Path(
            "data/raw/ambench/2022_single_track/AMB2022-03/mds2-2716/"
            "Thermography/AMB2022-03-718-AMMT-StaringCamera_Signal.h5"
        ),
        "dataset_regex": r"ThermalData/.*/Signal$",
        "target_family": "AMB2022-03 raw thermography dense fields",
    },
    {
        "candidate_id": "ambench_line0_1_phase51_dense_csv",
        "source_kind": "dense_csv",
        "path": Path(
            "data/interim/ambench/2022_single_track/AMB2022-03/"
            "line_0_1_temperature_phase51_dense.csv"
        ),
        "target_column": "temperature_C",
        "target_family": "AMB2022-03 Line_0_1 temperature indexed dense CSV",
    },
    {
        "candidate_id": "ambench_line0_1_dense_a800_csv",
        "source_kind": "dense_csv",
        "path": Path(
            "data/interim/ambench/2022_single_track/AMB2022-03/"
            "line_0_1_temperature_dense_a100_sxm4_40gb_v1.csv"
        ),
        "target_column": "temperature_C",
        "target_family": "AMB2022-03 Line_0_1 A800 indexed dense CSV",
    },
    {
        "candidate_id": "ambench_multiline_process_dense_csv",
        "source_kind": "dense_csv",
        "path": Path(
            "data/interim/ambench/2022_single_track/AMB2022-03/"
            "ambench_multiline_process_temperature_a100_sxm4_40gb_v1.csv"
        ),
        "target_column": "temperature_C",
        "target_family": "AMB2022-03 multiline process indexed dense CSV",
    },
)

INVENTORY_FIELDS = (
    "candidate_id",
    "source_kind",
    "target_family",
    "path",
    "present",
    "bytes",
    "metadata_count",
    "shape_evidence",
    "target_column",
    "grid_index_columns",
    "split_contract_status",
    "registration_status",
    "baseline_gap_status",
    "tensorization_status",
    "operator_route_status",
    "blocker",
    "next_action",
)
BASELINE_FIELDS = (
    "audit_id",
    "evidence_source",
    "target_family",
    "evidence_status",
    "baseline_gap_status",
    "operator_gap_ready",
    "rationale",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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
        if math.isfinite(value):
            return f"{value:.6f}"
        return ""
    if isinstance(value, (dict, list, tuple)):
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
    try:
        if root is not None:
            return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        pass
    return path.as_posix()


def _is_false(value: Any) -> bool:
    if isinstance(value, bool):
        return value is False
    if isinstance(value, str):
        return value.strip().lower() in {"", "0", "false", "no"}
    return not bool(value)


def _resolve_sources(root: Path, candidate_sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    for source in candidate_sources:
        item = dict(source)
        path = item["path"]
        path = path if isinstance(path, Path) else Path(path)
        item["path"] = path if path.is_absolute() else root / path
        resolved.append(item)
    return resolved


def _bmp_header_from_zip(zf: zipfile.ZipFile, member: str) -> dict[str, Any]:
    with zf.open(member) as handle:
        payload = handle.read(64)
    if len(payload) < 30 or payload[:2] != b"BM":
        return {"bmp_header_status": "not_bmp_header"}
    width = struct.unpack_from("<i", payload, 18)[0]
    height = abs(struct.unpack_from("<i", payload, 22)[0])
    bits_per_pixel = struct.unpack_from("<H", payload, 28)[0]
    return {
        "bmp_header_status": "ok",
        "width": width,
        "height": height,
        "bits_per_pixel": bits_per_pixel,
    }


def _member_matches(name: str, needles: tuple[str, ...]) -> bool:
    lower = name.lower()
    return any(needle in lower for needle in needles)


def _inspect_zip_bmp_source(source: dict[str, Any], root: Path) -> dict[str, Any]:
    path: Path = source["path"]
    row = _base_inventory_row(source, root)
    if not path.exists():
        return _missing_row(row, "missing_raw_zip", "verify server-local ZIP availability")

    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
            needles = tuple(str(value).lower() for value in source.get("member_needles", ()))
            bmp_infos = [
                info for info in infos if info.filename.lower().endswith(".bmp")
            ]
            matched = [
                info for info in bmp_infos if not needles or _member_matches(info.filename, needles)
            ]
            sample_header: dict[str, Any] = {}
            if matched:
                sample_header = _bmp_header_from_zip(zf, matched[0].filename)
            shape = {
                "zip_members": len(infos),
                "bmp_members": len(bmp_infos),
                "matched_bmp_members": len(matched),
                "sample_member": matched[0].filename if matched else "",
                "sample_header": sample_header,
            }
    except Exception as exc:  # pragma: no cover - defensive for server corrupt members
        row.update(
            {
                "present": True,
                "bytes": path.stat().st_size if path.exists() else 0,
                "metadata_count": 0,
                "shape_evidence": f"zip_inspection_error:{exc.__class__.__name__}",
                "tensorization_status": "blocked_zip_inspection_error",
                "operator_route_status": "blocked",
                "blocker": str(exc),
                "next_action": "do not redownload unless a required member cannot be read",
            }
        )
        return row

    matched_count = int(shape["matched_bmp_members"])
    row.update(
        {
            "present": True,
            "bytes": path.stat().st_size,
            "metadata_count": matched_count,
            "shape_evidence": shape,
            "grid_index_columns": "not_applicable_zip_members",
            "split_contract_status": "no_operator_split_manifest",
            "registration_status": "registered_for_phase106_aggregate_targets_only",
            "baseline_gap_status": "phase148_closed_no_guarded_graph_gap",
            "tensorization_status": (
                "candidate_image_sequence_metadata_only"
                if matched_count
                else "blocked_no_matching_bmp_members"
            ),
            "operator_route_status": "not_ready",
            "blocker": (
                "image members exist but no fixed dense tensor manifest or operator baseline gap exists"
                if matched_count
                else "no matching BMP members found"
            ),
            "next_action": (
                "build only a no-training fixed-grid tensor manifest if this route is reopened"
                if matched_count
                else "inspect member naming before using this source"
            ),
        }
    )
    return row


def _h5py_module() -> Any | None:
    try:
        import h5py  # type: ignore
    except Exception:
        return None
    return h5py


def _inspect_hdf5_source(source: dict[str, Any], root: Path) -> dict[str, Any]:
    path: Path = source["path"]
    row = _base_inventory_row(source, root)
    if not path.exists():
        return _missing_row(row, "missing_hdf5", "keep route blocked unless raw HDF5 is restored")
    h5py = _h5py_module()
    if h5py is None:
        row.update(
            {
                "present": True,
                "bytes": path.stat().st_size,
                "metadata_count": 0,
                "shape_evidence": "h5py_unavailable",
                "split_contract_status": "unknown",
                "registration_status": "unknown",
                "baseline_gap_status": "unknown",
                "tensorization_status": "blocked_h5py_unavailable",
                "operator_route_status": "blocked",
                "blocker": "h5py is required for HDF5 metadata inspection",
                "next_action": "run inventory in the gnnpinn conda env with h5py",
            }
        )
        return row

    dataset_regex = re.compile(str(source.get("dataset_regex", ".*")))
    datasets: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        def visit(name: str, obj: Any) -> None:
            if hasattr(obj, "shape") and dataset_regex.search(name):
                datasets.append(
                    {
                        "path": name,
                        "shape": list(obj.shape),
                        "dtype": str(obj.dtype),
                    }
                )

        handle.visititems(visit)
    dense_like = [item for item in datasets if len(item["shape"]) >= 2]
    row.update(
        {
            "present": True,
            "bytes": path.stat().st_size,
            "metadata_count": len(dense_like),
            "shape_evidence": {"dataset_count": len(datasets), "examples": dense_like[:5]},
            "grid_index_columns": "raw_hdf5_arrays",
            "split_contract_status": "no_operator_split_manifest",
            "registration_status": "phase53_reports_missing_scan_path_registration",
            "baseline_gap_status": "no_operator_specific_baseline_gap",
            "tensorization_status": (
                "raw_dense_arrays_present_no_operator_manifest"
                if dense_like
                else "blocked_no_dense_datasets"
            ),
            "operator_route_status": "not_ready",
            "blocker": "raw arrays need leakage-safe tensor split and baseline-gap audit",
            "next_action": "build no-training fixed-grid tensor manifest before any FNO training",
        }
    )
    return row


def _inspect_dense_csv_source(
    source: dict[str, Any],
    root: Path,
    *,
    max_preview_rows: int,
) -> dict[str, Any]:
    path: Path = source["path"]
    row = _base_inventory_row(source, root)
    if not path.exists():
        return _missing_row(row, "missing_dense_csv", "skip or regenerate this ignored interim source")

    target_column = str(source.get("target_column", "temperature_C"))
    row_count = 0
    preview_count = 0
    unique_frames: set[str] = set()
    unique_rows: set[str] = set()
    unique_cols: set[str] = set()
    unique_lines: set[str] = set()
    frame_counts: Counter[str] = Counter()
    header: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        for record in reader:
            row_count += 1
            if preview_count >= max_preview_rows:
                continue
            preview_count += 1
            frame = record.get("frame_index", "")
            row_index = record.get("row_index", record.get("y", ""))
            col_index = record.get("col_index", record.get("x", ""))
            line_id = record.get("line_id", record.get("dataset_path", ""))
            if frame:
                unique_frames.add(frame)
                frame_counts[frame] += 1
            if row_index:
                unique_rows.add(row_index)
            if col_index:
                unique_cols.add(col_index)
            if line_id:
                unique_lines.add(line_id)

    grid_cols = {"frame_index", "row_index", "col_index"}.issubset(set(header))
    target_present = target_column in header
    split_present = bool({"split_name", "split"}.intersection(header))
    shape = {
        "row_count": row_count,
        "preview_rows": preview_count,
        "header": header,
        "unique_frame_count_preview": len(unique_frames),
        "unique_row_count_preview": len(unique_rows),
        "unique_col_count_preview": len(unique_cols),
        "unique_line_count_preview": len(unique_lines),
        "frame_count_min_preview": min(frame_counts.values()) if frame_counts else 0,
        "frame_count_max_preview": max(frame_counts.values()) if frame_counts else 0,
    }
    tensorizable = grid_cols and target_present and row_count > 0
    row.update(
        {
            "present": True,
            "bytes": path.stat().st_size,
            "metadata_count": row_count,
            "shape_evidence": shape,
            "target_column": target_column if target_present else "",
            "grid_index_columns": "frame_index,row_index,col_index" if grid_cols else "missing_grid_index_columns",
            "split_contract_status": (
                "inline_split_column_present"
                if split_present
                else "missing_operator_split_contract"
            ),
            "registration_status": "camera_pixel_index_or_derived_table",
            "baseline_gap_status": "no_operator_specific_dense_baseline_gap",
            "tensorization_status": (
                "candidate_indexed_dense_csv_needs_split_and_operator_baseline"
                if tensorizable
                else "blocked_not_tensorizable_csv"
            ),
            "operator_route_status": "not_ready",
            "blocker": (
                "indexed dense CSV exists but lacks operator-specific split and dense baseline gate"
                if tensorizable
                else "CSV lacks grid indices or target column"
            ),
            "next_action": (
                "Phase 151 may build no-training fixed-grid tensor/split and baseline review"
                if tensorizable
                else "do not use this CSV for operator route"
            ),
        }
    )
    return row


def _base_inventory_row(source: dict[str, Any], root: Path) -> dict[str, Any]:
    return {
        "candidate_id": source["candidate_id"],
        "source_kind": source["source_kind"],
        "target_family": source.get("target_family", ""),
        "path": _display_path(source["path"], root),
        "present": False,
        "bytes": 0,
        "metadata_count": 0,
        "shape_evidence": "",
        "target_column": source.get("target_column", ""),
        "grid_index_columns": "",
        "split_contract_status": "",
        "registration_status": "",
        "baseline_gap_status": "",
        "tensorization_status": "",
        "operator_route_status": "",
        "blocker": "",
        "next_action": "",
    }


def _missing_row(row: dict[str, Any], blocker: str, next_action: str) -> dict[str, Any]:
    row.update(
        {
            "present": False,
            "metadata_count": 0,
            "shape_evidence": "missing",
            "split_contract_status": "missing",
            "registration_status": "missing",
            "baseline_gap_status": "missing",
            "tensorization_status": f"blocked_{blocker}",
            "operator_route_status": "blocked",
            "blocker": blocker,
            "next_action": next_action,
        }
    )
    return row


def build_inventory_rows(
    *,
    root: Path,
    candidate_sources: list[dict[str, Any]],
    max_preview_rows: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in _resolve_sources(root, candidate_sources):
        kind = source["source_kind"]
        if kind == "zip_bmp_members":
            rows.append(_inspect_zip_bmp_source(source, root))
        elif kind == "hdf5_dataset":
            rows.append(_inspect_hdf5_source(source, root))
        elif kind == "dense_csv":
            rows.append(_inspect_dense_csv_source(source, root, max_preview_rows=max_preview_rows))
        else:
            row = _base_inventory_row(source, root)
            row.update(
                {
                    "tensorization_status": "blocked_unknown_source_kind",
                    "operator_route_status": "blocked",
                    "blocker": f"unknown source_kind {kind}",
                    "next_action": "add an explicit Phase 150 inspector before using this source",
                }
            )
            rows.append(row)
    return rows


def build_baseline_gap_rows(
    *,
    phase149_gate: dict[str, Any],
    phase116_gate: dict[str, Any],
    positive_floor_rows: list[dict[str, str]],
    phase106_gate: dict[str, Any],
    phase148_gate: dict[str, Any],
    phase53_text: str,
    phase_inputs: dict[str, Path],
    root: Path,
) -> list[dict[str, Any]]:
    positive_floor_count = len(positive_floor_rows)
    phase149_allows_inventory = bool(phase149_gate.get("phase150_dense_tensorization_inventory_allowed"))
    nist_closed = phase148_gate.get("status") == "phase148_path_contact_graph_audit_closed_no_guarded_graph_gap"
    phase53_registration_blocked = "no HDF5 camera-pixel to galvo-mm registration metadata" in phase53_text
    return [
        {
            "audit_id": "P150-BASE-001",
            "evidence_source": _display_path(phase_inputs["phase149_gate"], root),
            "target_family": "neural_operator_route",
            "evidence_status": "inventory_allowed_only" if phase149_allows_inventory else "inventory_not_allowed",
            "baseline_gap_status": "no_operator_gap_from_phase149",
            "operator_gap_ready": False,
            "rationale": "Phase 149 explicitly blocks FNO/operator training and permits only inventory.",
        },
        {
            "audit_id": "P150-BASE-002",
            "evidence_source": _display_path(phase_inputs["phase116_positive_floor"], root),
            "target_family": "paper_one_spot_size_floor",
            "evidence_status": f"positive_floor_rows={positive_floor_count}",
            "baseline_gap_status": "route_guard_floor_not_dense_operator_gap",
            "operator_gap_ready": False,
            "rationale": "The paper floor is useful but is not a fixed-grid operator-learning target.",
        },
        {
            "audit_id": "P150-BASE-003",
            "evidence_source": _display_path(phase_inputs["phase106_gate"], root),
            "target_family": "nist_ammt_layer_camera_spatial_targets",
            "evidence_status": phase106_gate.get("status", "unknown"),
            "baseline_gap_status": (
                "closed_by_phase148_route_guard" if nist_closed else "requires_route_guard_review"
            ),
            "operator_gap_ready": False,
            "rationale": "Layer-camera spatial summaries are aggregate targets and later path-contact review did not leave guarded graph/operator space.",
        },
        {
            "audit_id": "P150-BASE-004",
            "evidence_source": _display_path(phase_inputs["phase53_pivot"], root),
            "target_family": "ambench_registered_source_path",
            "evidence_status": (
                "registration_blocked" if phase53_registration_blocked else "registration_evidence_incomplete"
            ),
            "baseline_gap_status": "no_registered_source_path_for_operator_baseline",
            "operator_gap_ready": False,
            "rationale": "Operator route needs leakage-safe source/target registration, not only sampled field rows.",
        },
    ]


def build_gate(
    *,
    inventory_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    phase149_gate: dict[str, Any],
    phase116_gate: dict[str, Any],
    phase148_gate: dict[str, Any],
) -> dict[str, Any]:
    tensorizable_rows = [
        row
        for row in inventory_rows
        if str(row.get("tensorization_status", "")).startswith("candidate_")
        or str(row.get("tensorization_status", "")).startswith("raw_dense_arrays_present")
    ]
    operator_gap_ready_rows = [row for row in baseline_rows if row.get("operator_gap_ready")]
    all_prior_locks_false = (
        _is_false(phase149_gate.get("operator_training_allowed_now"))
        and _is_false(phase149_gate.get("phase149_model_training_allowed"))
        and _is_false(phase149_gate.get("a100_training_allowed_now"))
        and _is_false(phase149_gate.get("a100_80gb_request_now"))
        and _is_false(phase116_gate.get("a100_training_allowed_now"))
        and _is_false(phase148_gate.get("a100_training_allowed_now"))
    )
    phase151_allowed = bool(tensorizable_rows) and not operator_gap_ready_rows
    if operator_gap_ready_rows:
        status = "phase150_dense_tensorization_inventory_ready_operator_baseline_gap_review"
        next_action = "run a focused no-training dense baseline review before any operator training"
    elif phase151_allowed:
        status = "phase150_dense_tensorization_inventory_ready_phase151_fixed_grid_baseline_review"
        next_action = (
            "implement Phase 151 as a no-training fixed-grid tensor/split manifest and dense baseline review"
        )
    else:
        status = "phase150_dense_tensorization_inventory_closed_no_tensor_source"
        next_action = "close neural-operator route until a leakage-safe dense tensor source is available"
    return {
        "status": status,
        "inventory_rows": len(inventory_rows),
        "present_source_rows": len([row for row in inventory_rows if row.get("present")]),
        "tensorizable_candidate_rows": len(tensorizable_rows),
        "operator_gap_ready_rows": len(operator_gap_ready_rows),
        "phase149_gate_status": phase149_gate.get("status"),
        "phase148_gate_status": phase148_gate.get("status"),
        "phase151_fixed_grid_baseline_review_allowed": phase151_allowed,
        "operator_training_allowed_now": False,
        "phase150_model_mechanism_allowed": False,
        "phase150_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "prior_training_locks_verified_false": all_prior_locks_false,
        "next_action": next_action,
    }


def _markdown_table(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[str]:
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    body = ["| " + " | ".join(_csv_value(row.get(field)) for field in fields) + " |" for row in rows]
    return [header, sep, *body]


def _write_markdown(
    path: Path,
    *,
    gate: dict[str, Any],
    inventory_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Phase 150 Dense Tensorization Inventory Gate",
        "",
        f"- Status: `{gate['status']}`",
        f"- Tensorizable candidate rows: `{gate['tensorizable_candidate_rows']}`",
        f"- Operator gap ready rows: `{gate['operator_gap_ready_rows']}`",
        "- Phase 151 fixed-grid baseline review allowed: "
        f"`{_csv_value(gate['phase151_fixed_grid_baseline_review_allowed'])}`",
        "- Operator training allowed now: `false`",
        "- A100 training allowed now: `false`",
        "",
        "## Source Inventory",
        "",
        *_markdown_table(inventory_rows, INVENTORY_FIELDS),
        "",
        "## Baseline-Gap Audit",
        "",
        *_markdown_table(baseline_rows, BASELINE_FIELDS),
        "",
        f"Next action: {gate['next_action']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_package(
    *,
    root: Path,
    output_dir: Path,
    phase_inputs: dict[str, Path] | None = None,
    candidate_sources: list[dict[str, Any]] | None = None,
    max_preview_rows: int = 20000,
) -> dict[str, Any]:
    phase_inputs = dict(phase_inputs or PHASE_INPUTS)
    resolved_inputs = {
        key: path if path.is_absolute() else root / path
        for key, path in phase_inputs.items()
    }
    sources = list(candidate_sources or DEFAULT_CANDIDATE_SOURCES)
    phase149_gate = _read_json(resolved_inputs["phase149_gate"])
    phase116_gate = _read_json(resolved_inputs["phase116_gate"])
    positive_floor_rows = _read_csv(resolved_inputs["phase116_positive_floor"])
    phase106_gate = _read_json(resolved_inputs["phase106_gate"])
    phase148_gate = _read_json(resolved_inputs["phase148_gate"])
    phase53_text = _read_text(resolved_inputs["phase53_pivot"])

    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_rows = build_inventory_rows(
        root=root,
        candidate_sources=sources,
        max_preview_rows=max_preview_rows,
    )
    baseline_rows = build_baseline_gap_rows(
        phase149_gate=phase149_gate,
        phase116_gate=phase116_gate,
        positive_floor_rows=positive_floor_rows,
        phase106_gate=phase106_gate,
        phase148_gate=phase148_gate,
        phase53_text=phase53_text,
        phase_inputs=resolved_inputs,
        root=root,
    )
    gate = build_gate(
        inventory_rows=inventory_rows,
        baseline_rows=baseline_rows,
        phase149_gate=phase149_gate,
        phase116_gate=phase116_gate,
        phase148_gate=phase148_gate,
    )

    inventory_path = output_dir / "phase150_dense_source_inventory_table.csv"
    baseline_path = output_dir / "phase150_dense_baseline_gap_audit_table.csv"
    gate_path = output_dir / "phase150_dense_tensorization_inventory_gate.json"
    markdown_path = output_dir / "phase150_dense_tensorization_inventory_gate.md"
    manifest_path = output_dir / "phase150_dense_tensorization_inventory_manifest.json"

    _write_csv(inventory_path, inventory_rows, INVENTORY_FIELDS)
    _write_csv(baseline_path, baseline_rows, BASELINE_FIELDS)
    _write_json(gate_path, gate)
    _write_markdown(
        markdown_path,
        gate=gate,
        inventory_rows=inventory_rows,
        baseline_rows=baseline_rows,
    )
    manifest = {
        "phase": 150,
        "objective": "dense_tensorization_inventory_and_baseline_gap_audit_no_training",
        "inputs": {key: _display_path(path, root) for key, path in resolved_inputs.items()},
        "candidate_sources": [
            {
                key: _display_path(value, root) if isinstance(value, Path) else value
                for key, value in source.items()
            }
            for source in _resolve_sources(root, sources)
        ],
        "outputs": {
            "inventory_table": _display_path(inventory_path, root),
            "baseline_gap_audit_table": _display_path(baseline_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown_summary": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "inventory_rows": len(inventory_rows),
            "baseline_rows": len(baseline_rows),
            "tensorizable_candidate_rows": gate["tensorizable_candidate_rows"],
            "operator_gap_ready_rows": gate["operator_gap_ready_rows"],
        },
        "limits": {"max_preview_rows": max_preview_rows},
        "gate": gate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-preview-rows", type=int, default=20000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = build_package(
        root=root,
        output_dir=output_dir,
        max_preview_rows=args.max_preview_rows,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
