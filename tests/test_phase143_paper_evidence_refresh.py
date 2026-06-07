from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase143_paper_evidence_refresh.py")
    spec = importlib.util.spec_from_file_location("phase143_refresh", script)
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
        "phase137_gate": _write_json(
            tmp_path / "phase137_gate.json",
            {
                "status": "phase137_paper_evidence_refresh_ready_first_paper_narrow_claims",
                "first_paper_draft_allowed_now": True,
            },
        ),
        "phase116_gate": _write_json(
            tmp_path / "phase116_gate.json",
            {
                "status": "phase116_paper_evidence_consolidation_ready_venue_unresolved",
                "paper_evidence_consolidated": True,
                "main_paper_floor": "fixed-sampling broad12/broad21 spot_size under broad_process_v1",
            },
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
        if unlock_training and spec["branch_id"] == "matbench_expt_is_metal":
            gate[spec["training_lock_key"]] = True
        paths[spec["input_key"]] = _write_json(tmp_path / f"{spec['input_key']}.json", gate)
    return paths


def _paths_without_optional_phase140(tmp_path: Path) -> dict[str, Path]:
    paths = _paths(tmp_path)
    paths.pop("phase140_mp_is_metal_blocker")
    return paths


def test_phase143_refreshes_latest_external_evidence_and_keeps_training_locked(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path),
    )

    gate = manifest["gate"]
    assert manifest["phase"] == 143
    assert gate["status"] == "phase143_paper_evidence_refresh_ready_first_paper_narrow_claims"
    assert gate["first_paper_draft_allowed_now"] is True
    assert gate["new_external_model_claim_ready"] is False
    assert gate["phase143_model_mechanism_allowed"] is False
    assert gate["phase143_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert manifest["counts"]["latest_external_diagnostic_rows"] == len(module.TERMINAL_BRANCHES)
    assert manifest["counts"]["training_allowed_external_rows"] == 0

    with (tmp_path / "out/phase143_claim_boundary_refresh_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        claims = list(csv.DictReader(handle))
    blocked = next(row for row in claims if row["claim_id"] == "P143-CLAIM-003")
    assert "complete GNN-PINN" in blocked["wording_guard"]
    assert "Matbench glass/is-metal model success" in blocked["wording_guard"]
    markdown = (tmp_path / "out/phase143_paper_evidence_refresh.md").read_text(encoding="utf-8")
    assert "matbench_glass" in markdown
    assert "P143-CLAIM-003" in markdown
    assert "P143-DECISION-002" in markdown
    assert "|  |  |  |  |" not in markdown


def test_phase143_incomplete_if_latest_external_gate_unlocks_training(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        phase_inputs=_paths(tmp_path, unlock_training=True),
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase143_paper_evidence_refresh_incomplete"
    assert gate["latest_external_training_locks_verified"] is False
    assert gate["phase143_model_training_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert manifest["counts"]["training_allowed_external_rows"] == 1


def test_phase143_allows_missing_optional_phase140_real_gate(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        root=Path(".").resolve(),
        output_dir=tmp_path / "out",
        phase_inputs=_paths_without_optional_phase140(tmp_path),
    )

    gate = manifest["gate"]
    assert gate["status"] == "phase143_paper_evidence_refresh_ready_first_paper_narrow_claims"
    assert gate["latest_external_training_locks_verified"] is True
    with (tmp_path / "out/phase143_external_diagnostic_refresh_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    phase140 = next(row for row in rows if row["branch_id"] == "matbench_mp_is_metal_large_source")
    assert phase140["terminal_status"] == "phase140_matbench_mp_is_metal_real_gate_missing_source_acquisition_blocked"
    assert phase140["model_training_allowed"] == "false"
