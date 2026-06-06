from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase137_paper_evidence_refresh.py")
    spec = importlib.util.spec_from_file_location("phase137_refresh", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _terminal_gate(status: str, training_key: str, *, target: str = "target") -> dict[str, object]:
    return {
        "status": status,
        "selected_target": target,
        "blocking_audits": ["diagnostic_blocker"],
        training_key: False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "next_action": "do not train",
    }


def _paths(tmp_path: Path, *, unlock_training: bool = False) -> dict[str, Path]:
    module = _load_module()
    paths = {
        "phase116_gate": _write_json(
            tmp_path / "phase116_gate.json",
            {
                "status": "phase116_paper_evidence_consolidation_ready_venue_unresolved",
                "paper_evidence_consolidated": True,
                "main_paper_floor": "fixed-sampling broad12/broad21 spot_size under broad_process_v1",
            },
        ),
        "phase116_positive_floor": _write_csv(
            tmp_path / "phase116_positive.csv",
            [{"dataset": "broad12", "metric": "Test RMSE"}],
        ),
        "phase116_claim_status": _write_csv(
            tmp_path / "phase116_claims.csv",
            [{"claim_id": "C1", "status": "supported"}],
        ),
        "phase116_blockers": _write_csv(
            tmp_path / "phase116_blockers.csv",
            [{"blocker_id": "B1", "blocks_submission": "true"}],
        ),
    }
    for spec in module.TERMINAL_BRANCHES:
        gate = _terminal_gate(
            f"{spec['branch_id']}_closed_diagnostic",
            spec["training_lock_key"],
            target=spec["target"],
        )
        if unlock_training and spec["branch_id"] == "matbench_perovskites":
            gate[spec["training_lock_key"]] = True
        paths[spec["input_key"]] = _write_json(tmp_path / f"{spec['input_key']}.json", gate)
    return paths


def test_phase137_refreshes_evidence_and_keeps_training_locked(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path),
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 137
    assert gate["status"] == "phase137_paper_evidence_refresh_ready_first_paper_narrow_claims"
    assert gate["first_paper_draft_allowed_now"] is True
    assert gate["new_external_model_claim_ready"] is False
    assert gate["phase137_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert manifest["counts"]["external_diagnostic_rows"] == len(module.TERMINAL_BRANCHES)
    assert manifest["counts"]["training_allowed_external_rows"] == 0

    with (tmp_path / "out/phase137_claim_boundary_refresh_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        claims = list(csv.DictReader(handle))
    blocked = next(row for row in claims if row["claim_id"] == "P137-CLAIM-003")
    assert "complete GNN-PINN" in blocked["wording_guard"]
    assert "source-path/Green" in blocked["wording_guard"]


def test_phase137_incomplete_if_external_terminal_gate_unlocks_training(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path, unlock_training=True),
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase137_paper_evidence_refresh_incomplete"
    assert gate["external_training_locks_verified"] is False
    assert gate["phase137_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert manifest["counts"]["training_allowed_external_rows"] == 1
