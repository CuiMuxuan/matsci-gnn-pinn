from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase75_bayesian_inverse_closure_gate.py")
    spec = importlib.util.spec_from_file_location("phase75_gate", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _phase74_manifest(path: Path) -> Path:
    return _write_json(
        path,
        {
            "writing_stage_gate": {
                "status": "ready_for_internal_manuscript_review",
                "trainable_model_opened_now": False,
                "main_claim_locked": True,
            }
        },
    )


def _line0_like_table(path: Path) -> Path:
    rows = ["x,y,t,temperature_C,frame_index,row_index,col_index"]
    for frame in range(8):
        for row in range(10):
            for col in range(10):
                x = col / 9.0
                y = row / 9.0
                t = frame / 7.0
                moving = 180.0 * (x - (0.18 + 0.62 * t)) ** 2
                ridge = 45.0 * abs(y - 0.5)
                temp = 980.0 + 30.0 * x - 12.0 * y + 25.0 * t + moving + ridge
                rows.append(f"{x},{y},{t},{temp},{frame},{row},{col}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _split_manifest(path: Path) -> Path:
    train = list(range(0, 480))
    val = list(range(480, 640))
    test = list(range(640, 800))
    return _write_json(path, {"splits": {"train": train, "val": val, "test": test}})


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "phase74_manifest": _phase74_manifest(tmp_path / "phase74_manifest.json"),
        "local_table": _line0_like_table(tmp_path / "line0_like.csv"),
        "local_split": _split_manifest(tmp_path / "line0_like_split.json"),
    }


def test_phase75_blocks_phase76_when_local_gate_fails(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        Path(".").resolve(),
        tmp_path / "out",
        paths=_paths(tmp_path),
        seed=75,
        repeats=1,
    )

    gate = manifest["gate_status"]
    assert manifest["phase"] == 75
    assert manifest["candidate"] == "bayesian_inverse_closure_v1"
    assert gate["synthetic_gate_passed"] is True
    assert gate["local_gate_passed"] is False
    assert gate["phase76_seed7_allowed"] is False
    assert gate["a100_training_allowed_now"] is False
    assert gate["a100_80gb_request_now"] is False
    assert manifest["counts"]["gate_rows"] == 2
    assert manifest["counts"]["positive_gate_rows"] == 1
    assert manifest["counts"]["negative_gate_rows"] == 1

    with (Path(".").resolve() / manifest["outputs"]["gate_table"]).open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    local = next(row for row in rows if row["gate_id"] == "P75-LOCAL")
    assert local["status"] == "negative"
    assert local["a100_training_allowed"] == "false"
    assert float(local["rmse_gain"]) < 0.0


def test_phase75_candidate_design_records_compute_governance(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        Path(".").resolve(),
        tmp_path / "out",
        paths=_paths(tmp_path),
        seed=75,
        repeats=1,
    )

    design_path = Path(".").resolve() / manifest["outputs"]["candidate_design"]
    design = json.loads(design_path.read_text(encoding="utf-8"))
    assert design["candidate_id"] == "bayesian_inverse_closure_v1"
    assert "Synthetic known-parameter recovery" in design["go_no_go_rule"]
    assert "A100-SXM4-40GB" in design["a100_40gb_policy"]
    assert "A100-SXM4-80GB" in design["a100_80gb_policy"]


def test_phase75_probe_artifacts_omit_raw_runs(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        Path(".").resolve(),
        tmp_path / "out",
        paths=_paths(tmp_path),
        seed=75,
        repeats=1,
    )

    for key in ("synthetic_probe", "local_ambench_probe"):
        probe_path = Path(".").resolve() / manifest["outputs"][key]
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        assert "runs" not in probe
        assert probe["raw_runs_persisted"] is False
        assert probe["raw_run_count"] > 0
        assert "summary" in probe
        assert "decision" in probe
