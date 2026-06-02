from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/build_phase57_claim_governance.py")
    spec = importlib.util.spec_from_file_location("phase57_governance", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _route(selected: str = "film", norm: str = "global_standard") -> dict:
    return {
        "selected_conditioning_mode": selected,
        "selected_feature_normalization": norm,
        "effective_conditioning_mode": selected,
        "effective_feature_columns": ["laser_power_W", "scan_speed_mm_s", "spot_size_um"]
        if selected != "none"
        else [],
    }


def _metric(candidate: float, baseline: float, no_process: float) -> dict:
    return {
        "candidate": candidate,
        "best_strong_baseline": {"method": "mean", "value": baseline},
        "no_process": no_process,
        "candidate_beats_best_strong_baseline": candidate < baseline,
        "candidate_beats_no_process": candidate < no_process,
    }


def _phase54_payload() -> dict:
    return {
        "phase": 54,
        "classification_counts": {"paper_claim_positive": 2, "route_guard_positive": 2},
        "datasets": [
            {
                "label": "broad12",
                "splits": {
                    "spot_size": {
                        "classification": "paper_claim_positive",
                        "route": _route(),
                        "metrics": {
                            "rmse": _metric(136.0, 151.0, 238.0),
                            "hot_q90_rmse": _metric(162.0, 252.0, 424.0),
                            "gradient_q90_rmse": _metric(165.0, 233.0, 382.0),
                        },
                        "notes": ["seed check is handled by Phase 55"],
                    },
                    "laser_power": {
                        "classification": "route_guard_positive",
                        "route": _route("concat"),
                        "metrics": {},
                        "notes": ["strong baseline remains better"],
                    },
                    "line": {
                        "classification": "paper_claim_positive",
                        "route": _route("none", "none"),
                        "metrics": {},
                        "notes": ["The selected route is the no-process fallback."],
                    },
                },
            },
            {
                "label": "broad21",
                "splits": {
                    "spot_size": {
                        "classification": "paper_claim_positive",
                        "route": _route(),
                        "metrics": {
                            "rmse": _metric(146.0, 149.0, 217.0),
                            "hot_q90_rmse": _metric(164.0, 251.0, 401.0),
                            "gradient_q90_rmse": _metric(174.0, 231.0, 360.0),
                        },
                        "notes": ["seed check is handled by Phase 55"],
                    },
                    "scan_speed": {
                        "classification": "route_guard_positive",
                        "route": _route("none", "none"),
                        "metrics": {},
                        "notes": ["route guard only"],
                    },
                },
            },
        ],
    }


def _aggregate(rmse: float, hot: float, gradient: float) -> dict:
    return {
        "n": 3,
        "complete": True,
        "rmse": {"mean": rmse, "pstdev": 1.0},
        "hot_q90_rmse": {"mean": hot, "pstdev": 1.0},
        "gradient_q90_rmse": {"mean": gradient, "pstdev": 1.0},
    }


def _gate_metrics(rmse: float, hot: float, gradient: float) -> dict:
    return {
        "rmse": {
            "candidate": rmse,
            "best_strong_baseline": 151.0,
            "no_process": 238.0,
            "beats_best_strong_baseline": True,
            "beats_no_process": True,
        },
        "hot_q90_rmse": {
            "candidate": hot,
            "best_strong_baseline": 252.0,
            "no_process": 424.0,
            "beats_best_strong_baseline": True,
            "beats_no_process": True,
        },
        "gradient_q90_rmse": {
            "candidate": gradient,
            "best_strong_baseline": 233.0,
            "no_process": 382.0,
            "beats_best_strong_baseline": True,
            "beats_no_process": True,
        },
    }


def _phase55_payload() -> dict:
    datasets = []
    for label, values in {
        "broad12": (136.384782, 162.125337, 165.282182),
        "broad21": (146.002303, 164.313888, 174.735839),
    }.items():
        rmse, hot, gradient = values
        datasets.append(
            {
                "label": label,
                "split": "spot_size",
                "status": "seed_robust_transfer_positive",
                "route": _route(),
                "aggregates": {
                    "broad_process_v1": _aggregate(rmse, hot, gradient),
                    "no_process": _aggregate(220.0, 400.0, 360.0),
                },
                "aggregate_gate": {"complete": True, "pass": True, "metrics": _gate_metrics(rmse, hot, gradient)},
                "paired_seed_gate": {"pass": True},
            }
        )
    return {
        "phase": 55,
        "split": "spot_size",
        "seeds": [7, 1, 2],
        "transfer_gate": {"paper_claim_status": "seed_robust_transfer_positive"},
        "datasets": datasets,
    }


def test_phase57_builds_contract_and_ledger_from_phase54_55(tmp_path: Path):
    module = _load_module()
    phase54 = tmp_path / "phase54.json"
    phase55 = tmp_path / "phase55.json"
    negative = tmp_path / "negative.csv"
    _write_json(phase54, _phase54_payload())
    _write_json(phase55, _phase55_payload())
    negative.write_text(
        "phase,branch,target,result,paper_use,evidence\n"
        "33,Fourier,broad12,Negative,Appendix diagnostic,docs/results/x.md\n",
        encoding="utf-8",
    )

    manifest = module.build_governance_package(
        root=tmp_path,
        phase54_path=phase54,
        phase55_path=phase55,
        negative_table_path=negative,
        output_dir=tmp_path / "governance",
    )

    assert manifest["ledger_counts"]["paper_positive_seed_robust"] == 2
    assert manifest["ledger_counts"]["route_guard_only"] == 2
    assert manifest["ledger_counts"]["route_guard_no_process_positive"] == 1
    assert manifest["seed_expansion_self_check"]["pass"] is True
    contract = json.loads(Path(manifest["outputs"]["contract"]).read_text(encoding="utf-8"))
    assert contract["frozen_floor"]["datasets"]["broad12"]["metrics"]["rmse"]["mean"] == 136.384782
    ledger = Path(manifest["outputs"]["ledger"]).read_text(encoding="utf-8")
    assert "paper_positive_seed_robust" in ledger
    assert "route_guard_no_process_positive" in ledger
    markdown = Path(manifest["outputs"]["markdown"]).read_text(encoding="utf-8")
    assert "Future branch contract" in markdown
    assert "spot_size" in markdown


def test_phase57_candidate_gate_blocks_any_metric_regression():
    module = _load_module()
    floor = {
        "broad12": {"rmse": 100.0, "hot_q90_rmse": 110.0, "gradient_q90_rmse": 120.0},
        "broad21": {"rmse": 101.0, "hot_q90_rmse": 111.0, "gradient_q90_rmse": 121.0},
    }
    candidate = {
        "broad12": {"rmse": 99.0, "hot_q90_rmse": 110.0, "gradient_q90_rmse": 119.0},
        "broad21": {"rmse": 101.0, "hot_q90_rmse": 112.0, "gradient_q90_rmse": 120.0},
    }

    result = module.evaluate_candidate_against_floor(candidate, floor)

    assert result["pass"] is False
    assert result["decision"] == "block_seed_expansion"
    failed = [row for row in result["checks"] if not row["pass"]]
    assert failed == [
        {
            "dataset": "broad21",
            "metric": "hot_q90_rmse",
            "candidate": 112.0,
            "frozen_floor": 111.0,
            "delta_vs_floor": 1.0,
            "pass": False,
        }
    ]


def test_phase57_candidate_gate_allows_non_worse_candidate():
    module = _load_module()
    floor = {
        "broad12": {"rmse": 100.0, "hot_q90_rmse": 110.0, "gradient_q90_rmse": 120.0},
        "broad21": {"rmse": 101.0, "hot_q90_rmse": 111.0, "gradient_q90_rmse": 121.0},
    }
    candidate = {
        "broad12": {"rmse": 100.0, "hot_q90_rmse": 109.0, "gradient_q90_rmse": 120.0},
        "broad21": {"rmse": 100.0, "hot_q90_rmse": 111.0, "gradient_q90_rmse": 120.5},
    }

    result = module.evaluate_candidate_against_floor(candidate, floor)

    assert result["pass"] is True
    assert result["decision"] == "allow_seed_expansion"
