from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase146_paper_evidence_refresh.py")
    spec = importlib.util.spec_from_file_location("phase146_refresh", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _paths(tmp_path: Path, *, unlock_training: bool = False) -> dict[str, Path]:
    phase145_gate = {
        "status": "phase145_mpea_mechanical_focused_review_closed_split_sensitivity_or_shortcut",
        "selected_target": "hardness_hv",
        "blocking_audits": [
            "stable_split_pass_rate",
            "shortcut_or_process_control_dominant_split_count",
        ],
        "phase145_model_training_allowed": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
    }
    if unlock_training:
        phase145_gate["phase145_model_training_allowed"] = True
    return {
        "phase143_gate": _write_json(
            tmp_path / "phase143_gate.json",
            {
                "status": "phase143_paper_evidence_refresh_ready_first_paper_narrow_claims",
                "first_paper_draft_allowed_now": True,
                "main_paper_floor": "fixed-sampling broad12/broad21 spot_size under broad_process_v1",
            },
        ),
        "phase145_mpea_terminal": _write_json(tmp_path / "phase145_gate.json", phase145_gate),
    }


def test_phase146_refreshes_mpea_diagnostic_and_keeps_training_locked(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path),
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 146
    assert gate["status"] == "phase146_paper_evidence_refresh_ready_first_paper_narrow_claims"
    assert gate["first_paper_draft_allowed_now"] is True
    assert gate["new_external_model_claim_ready"] is False
    assert gate["phase146_model_mechanism_allowed"] is False
    assert gate["phase146_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert manifest["counts"]["latest_external_diagnostic_rows"] == 1
    assert manifest["counts"]["training_allowed_external_rows"] == 0

    markdown = (tmp_path / "out/phase146_paper_evidence_refresh.md").read_text(encoding="utf-8")
    assert "mpea_mechanical" in markdown
    assert "P146-CLAIM-003" in markdown
    assert "MPEA hardness model success" in markdown
    assert "|  |  |  |  |" not in markdown


def test_phase146_incomplete_if_mpea_gate_unlocks_training(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path, unlock_training=True),
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase146_paper_evidence_refresh_incomplete"
    assert gate["latest_external_training_locks_verified"] is False
    assert gate["phase146_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert manifest["counts"]["training_allowed_external_rows"] == 1
