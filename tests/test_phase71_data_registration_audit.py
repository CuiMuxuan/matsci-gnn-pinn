from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase71_data_registration_audit.py")
    spec = importlib.util.spec_from_file_location("phase71_data_registration", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _phase60_rows(status: str = "blocked by registration data") -> list[dict[str, object]]:
    return [
        {
            "branch": "Candidate C: heat-kernel or Green's-function source features",
            "status": status,
            "entry_condition": "Requires aligned single-track scan-path metadata or a defensible pad thermography target.",
            "focused_validation": "Start with no-training-change feature gates before Macro PINN integration.",
            "seed_expansion_gate": "Seed-expand only if seed 7 is non-worse than broad_process_v1.",
            "manuscript_rule": "Do not claim source-path physics on broad12/broad21 until registration is resolved.",
        }
    ]


def _phase68_rows(status: str = "blocked_by_registration_data") -> list[dict[str, object]]:
    return [
        {
            "candidate_id": "C",
            "candidate_family": "data-aligned heat-kernel or Green's-function features",
            "status": status,
            "decision": "data_audit_before_training",
            "validation_visible_signal": "missing registration",
            "broad12_signal": "no direct broad12 signal",
            "broad21_signal": "no direct broad21 signal",
            "entry_evidence": "Requires aligned scan path.",
            "blocking_evidence": "Phase 52/53 registration blocker.",
            "required_first_action": "inventory aligned scan-path/pad-thermography data",
            "a100_40gb_action": "feature audits only",
            "a100_80gb_trigger": "only if dense aligned target exceeds 40GB",
            "seed7_gate": "seed 7 must preserve broad12/broad21",
            "seed_expansion_gate": "expand after seed 7 passes",
            "manuscript_use": "not allowed as current source-path claim",
            "evidence_locator": "phase52; phase53",
        }
    ]


def _paths(tmp_path: Path, *, open_registration: bool = False) -> dict[str, Path]:
    if open_registration:
        phase52 = """
# Gate
Formal decision is positive.
coordinate.compatible=true.
single-track scan-path source aligned to ThermalData/Line_0_1.
"""
        phase53 = """
# Inventory
Formal inventory decision:

```text
positive: single-track scan-path groups and registration metadata are available
```

| Item | Value |
| --- | --- |
| Single-track scan-path groups | Line_0_1 |
| HDF5 registration metadata keys | camera_to_galvo_affine |
"""
        phase60_status = "open after registration audit"
        phase68_status = "open_for_data_registration_gate"
    else:
        phase52 = """
# Gate
The formal decision is negative:
scan strategy file contains pad XYPT groups but table is a single-track Line_* dataset.
The table uses camera pixel indices and XYPT uses galvo millimeters.
"""
        phase53 = """
# Inventory
Formal inventory decision:

```text
negative: scan strategy exposes pad XYPT only; thermography has pad tables, but no HDF5 camera-pixel to galvo-mm registration metadata was found
```

| Item | Value |
| --- | --- |
| Single-track scan-path groups | none |
| HDF5 registration metadata keys | none found |

Pad thermography groups include X_pad1 and Y_pad1. X_pad1 improves hot/gradient but worsens global RMSE. Y_pad1 worsens global, hot, gradient, and coverage.
"""
        phase60_status = "blocked by registration data"
        phase68_status = "blocked_by_registration_data"

    return {
        "phase52_doc": _write_text(tmp_path / "phase52.md", phase52),
        "phase53_doc": _write_text(tmp_path / "phase53.md", phase53),
        "phase60_next_gate": _write_csv(tmp_path / "phase60.csv", _phase60_rows(phase60_status)),
        "phase68_scorecard": _write_csv(tmp_path / "phase68.csv", _phase68_rows(phase68_status)),
        "phase68_manifest": _write_json(
            tmp_path / "phase68_manifest.json",
            {"current_decision": {"trainable_model_opened": False}},
        ),
    }


def test_phase71_blocks_candidate_c_without_registered_target(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(tmp_path, tmp_path / "out", paths=_paths(tmp_path))

    gate = manifest["candidate_c_gate"]
    assert manifest["phase"] == 71
    assert gate["status"] == "blocked_by_registration_data"
    assert gate["open_aligned_feature_gate"] is False
    assert gate["fixed_feature_gate_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert gate["aligned_target_count"] == 0
    assert gate["blocking_row_count"] >= 3
    with (tmp_path / manifest["outputs"]["audit_table"]).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert any(row["status"] == "blocked_single_track_source_path" for row in rows)
    assert any(row["status"] == "blocked_broad_source_path" for row in rows)
    assert any(row["status"] == "phase68_blocks_candidate_c" for row in rows)


def test_phase71_opens_fixed_feature_gate_with_registered_target(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        tmp_path,
        tmp_path / "out",
        paths=_paths(tmp_path, open_registration=True),
    )

    gate = manifest["candidate_c_gate"]
    assert gate["status"] == "opened_for_aligned_feature_gate"
    assert gate["open_aligned_feature_gate"] is True
    assert gate["fixed_feature_gate_allowed"] is True
    assert gate["a100_training_allowed_now"] is False
    assert gate["aligned_target_count"] >= 2
    with (tmp_path / manifest["outputs"]["audit_table"]).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert any(row["status"] == "aligned_single_track_source_path" for row in rows)
    assert any(row["status"] == "registered_feature_gate_ready" for row in rows)
