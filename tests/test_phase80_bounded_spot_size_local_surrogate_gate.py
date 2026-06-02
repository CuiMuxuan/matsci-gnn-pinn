from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase80_bounded_spot_size_local_surrogate_gate.py")
    spec = importlib.util.spec_from_file_location("phase80_gate", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _metrics(rmse: float, hot: float = 100.0, grad: float = 100.0) -> dict:
    return {
        "rmse": rmse,
        "hot_q90_rmse": hot,
        "gradient_q90_rmse": grad,
    }


def _variant(name: str, *, val_rmse: float, test_rmse: float, hot: float = 100.0, grad: float = 100.0) -> dict:
    return {
        "name": name,
        "metrics": {
            "train": _metrics(80.0, hot=90.0, grad=90.0),
            "val": _metrics(val_rmse, hot=hot, grad=grad),
            "test": _metrics(test_rmse, hot=hot, grad=grad),
        },
    }


def _upper_payload(*, passing_candidate: bool = False, weak_candidate: bool = True) -> dict:
    if passing_candidate:
        spot = _variant(
            "broad_process_v1:train_group_bias:spot_size_um",
            val_rmse=158.0,
            test_rmse=135.0,
            hot=98.0,
            grad=99.0,
        )
    elif weak_candidate:
        spot = _variant(
            "broad_process_v1:train_group_bias:spot_size_um",
            val_rmse=161.8,
            test_rmse=152.0,
            hot=99.0,
            grad=99.0,
        )
    else:
        spot = _variant(
            "broad_process_v1:train_group_bias:spot_size_um",
            val_rmse=158.0,
            test_rmse=135.0,
            hot=111.0,
            grad=99.0,
        )
    return {
        "selection_split": "val",
        "analysis_split": "test",
        "reference": "mean",
        "baseline_metrics": {
            "broad_process_v1": {
                "train": _metrics(83.0, hot=95.0, grad=95.0),
                "val": _metrics(162.0, hot=110.0, grad=110.0),
                "test": _metrics(153.0, hot=110.0, grad=110.0),
            },
            "mean": {
                "train": _metrics(118.0, hot=120.0, grad=120.0),
                "val": _metrics(138.0, hot=120.0, grad=120.0),
                "test": _metrics(140.0, hot=120.0, grad=120.0),
            },
        },
        "variants": [
            _variant("broad_process_v1:identity", val_rmse=162.0, test_rmse=153.0, hot=110.0, grad=110.0),
            _variant("broad_process_v1:train_global_bias", val_rmse=161.7, test_rmse=151.0, hot=109.0, grad=109.0),
            spot,
            _variant("broad_process_v1:train_group_bias:process_tuple", val_rmse=161.9, test_rmse=152.5, hot=109.0, grad=109.0),
        ],
    }


def _paths(
    tmp_path: Path,
    *,
    local_allowed: bool = True,
    passing_candidate: bool = False,
    weak_candidate: bool = True,
) -> dict[str, Path]:
    return {
        "phase59_upper": _write_json(
            tmp_path / "phase59_upper.json",
            _upper_payload(passing_candidate=passing_candidate, weak_candidate=weak_candidate),
        ),
        "phase79_manifest": _write_json(
            tmp_path / "phase79_manifest.json",
            {"gate": {"local_surrogate_allowed": local_allowed}},
        ),
    }


def test_phase80_blocks_current_weak_validation_surrogate(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(tmp_path, tmp_path / "out", paths=_paths(tmp_path))

    gate = manifest["gate"]
    assert manifest["phase"] == 80
    assert gate["status"] == "blocked_by_local_surrogate_gate"
    assert gate["a100_seed7_allowed"] is False
    assert gate["local_surrogate_passed"] is False
    assert gate["selected_variant"] == "broad_process_v1:train_global_bias"
    assert gate["selected_variant_status"] == "insufficient_validation_gain"
    assert gate["a100_80gb_request_now"] is False

    table_path = tmp_path / manifest["outputs"]["surrogate_table"]
    with table_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert any(row["role"] == "strong_reference" for row in rows)


def test_phase80_opens_a100_when_local_surrogate_passes(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        tmp_path,
        tmp_path / "out",
        paths=_paths(tmp_path, passing_candidate=True, weak_candidate=False),
    )

    gate = manifest["gate"]
    assert gate["status"] == "opened_for_phase76_seed7"
    assert gate["selected_variant"] == "broad_process_v1:train_group_bias:spot_size_um"
    assert gate["local_surrogate_passed"] is True
    assert gate["a100_seed7_allowed"] is True
    assert gate["selected_preserves_identity_regions"] is True


def test_phase80_respects_phase79_local_surrogate_block(tmp_path: Path):
    module = _load_module()

    manifest = module.build_package(
        tmp_path,
        tmp_path / "out",
        paths=_paths(tmp_path, local_allowed=False, passing_candidate=True, weak_candidate=False),
    )

    gate = manifest["gate"]
    assert gate["status"] == "blocked_by_phase79"
    assert gate["phase79_local_surrogate_allowed"] is False
    assert gate["a100_seed7_allowed"] is False
