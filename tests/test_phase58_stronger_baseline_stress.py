from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path("scripts/server/summarize_phase58_stronger_baseline_stress.py")
    spec = importlib.util.spec_from_file_location("phase58_stress", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _baseline_payload(rmse: float, hot: float, gradient: float) -> dict:
    return {
        "results": [
            {
                "split_metrics": {
                    "test": {
                        "metrics": {"rmse": rmse},
                        "region_metrics": {
                            "hot_q90": {"metrics": {"rmse": hot}},
                            "gradient_q90": {"metrics": {"rmse": gradient}},
                        },
                    }
                }
            }
        ]
    }


def _phase55_dataset(label: str, base_run_id: str, rmse: float, hot: float, gradient: float) -> dict:
    return {
        "label": label,
        "base_run_id": base_run_id,
        "aggregates": {
            "broad_process_v1": {
                "rmse": {"mean": rmse},
                "hot_q90_rmse": {"mean": hot},
                "gradient_q90_rmse": {"mean": gradient},
            }
        },
        "aggregate_gate": {
            "metrics": {
                "rmse": {"best_strong_baseline": rmse + 10.0, "best_strong_baseline_method": "mean"},
                "hot_q90_rmse": {"best_strong_baseline": hot + 10.0, "best_strong_baseline_method": "mean"},
                "gradient_q90_rmse": {
                    "best_strong_baseline": gradient + 10.0,
                    "best_strong_baseline_method": "mean",
                },
            }
        },
    }


def test_phase58_stress_gate_passes_when_new_baselines_are_weaker(tmp_path: Path):
    module = _load_module()
    phase55 = tmp_path / "phase55.json"
    _write_json(
        phase55,
        {
            "datasets": [
                _phase55_dataset("broad12", "run12", 100.0, 110.0, 120.0),
                _phase55_dataset("broad21", "run21", 101.0, 111.0, 121.0),
            ]
        },
    )
    for run_id, base in (("run12", 130.0), ("run21", 131.0)):
        for method in ("random_forest", "hist_gradient_boosting"):
            for tag in ("coords", "process"):
                _write_json(
                    tmp_path / "outputs" / "baselines" / f"{run_id}_{method}_{tag}_regions_q90.json",
                    _baseline_payload(base, base + 10.0, base + 20.0),
                )

    summary = module.collect_summary(tmp_path, phase55)

    assert summary["stress_gate"]["status"] == "claim_survives_stronger_baselines"
    assert all(dataset["pass"] for dataset in summary["datasets"])


def test_phase58_stress_gate_fails_when_new_baseline_beats_frozen_floor(tmp_path: Path):
    module = _load_module()
    phase55 = tmp_path / "phase55.json"
    _write_json(
        phase55,
        {"datasets": [_phase55_dataset("broad12", "run12", 100.0, 110.0, 120.0)]},
    )
    for method in ("random_forest", "hist_gradient_boosting"):
        for tag in ("coords", "process"):
            value = 90.0 if method == "hist_gradient_boosting" and tag == "process" else 130.0
            _write_json(
                tmp_path / "outputs" / "baselines" / f"run12_{method}_{tag}_regions_q90.json",
                _baseline_payload(value, 140.0, 150.0),
            )

    summary = module.collect_summary(tmp_path, phase55)

    assert summary["stress_gate"]["status"] == "claim_challenged_by_stronger_baseline"
    assert summary["datasets"][0]["metrics"]["rmse"]["frozen_beats_best_after_stress"] is False
    best = summary["datasets"][0]["metrics"]["rmse"]["best_baseline_after_stress"]
    assert best["method"] == "hist_gradient_boosting_process"


def test_phase58_runner_uses_temp_python_file_instead_of_stdin():
    text = Path("scripts/server/run_phase58_stronger_baseline_stress_a100.sh").read_text(encoding="utf-8")

    assert "hist_gradient_boosting" in text
    assert "artifact_index_py=" in text
    assert "python - <<'PY'" not in text
    assert "phase58_stronger_baseline_stress_summary.json" in text
    assert '[[ -z "${dataset_label}${base_run_id}${table}${split_manifest}" ]]' in text
    assert "incomplete artifact index row" in text


def test_phase58_sampling_panel_runner_uses_isolated_stress_profiles():
    text = Path("scripts/server/run_phase58_sampling_panel_stress_a100.sh").read_text(encoding="utf-8")

    assert "phase58_density_profile" in text
    assert "phase58_panel_profile" in text
    assert "phase58_sampling_density_stress_summary" in text
    assert "phase58_process_panel_stress_summary" in text
    assert 'PANEL_DATASET_LIMITS="${PANEL_DATASET_LIMITS:-15}"' in text
    assert "--profile-tag \"$profile_tag\"" in text
    assert "--seed \"$SEED\"" in text
    assert "--require-complete" in text
    assert "--require-pass" not in text
