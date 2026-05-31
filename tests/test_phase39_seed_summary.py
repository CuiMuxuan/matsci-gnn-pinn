import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metrics(rmse: float, hot: float, gradient: float) -> dict:
    return {
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


def test_phase39_seed_summary_collects_seed7_and_seed_specific_artifacts(tmp_path: Path):
    import importlib.util

    path = Path("scripts/server/summarize_phase39_output_affine_seed_check.py")
    spec = importlib.util.spec_from_file_location("phase39_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for seed, base, affine in ((7, 140.0, 139.0), (1, 142.0, 141.0), (2, 138.0, 137.0)):
        tags = module._tags_for_seed(seed)
        for method, value in (("broad_process_v1", base), ("broad_output_affine", affine)):
            run_tag, metric_tag = tags[method]
            run_id = module._run_id("lp", 12, "rr", run_tag)
            _write_json(
                tmp_path / "outputs/runs" / f"{run_id}_macro_pinn_minmax_{metric_tag}_v1" / "metrics.json",
                _metrics(value, value + 10, value + 20),
            )

    payload = module.collect_summary(tmp_path, "lp", 12, "rr", [7, 1, 2])

    assert payload["aggregates"]["broad_process_v1"]["n"] == 3
    assert payload["aggregates"]["broad_output_affine"]["n"] == 3
    assert payload["aggregates"]["broad_process_v1"]["rmse"]["mean"] == 140.0
    assert payload["aggregates"]["broad_output_affine"]["rmse"]["mean"] == 139.0
    assert payload["delta_affine_minus_baseline"]["rmse"] == -1.0
    assert payload["missing"] == []
