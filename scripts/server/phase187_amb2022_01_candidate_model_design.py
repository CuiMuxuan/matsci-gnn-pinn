#!/usr/bin/env python3
"""Freeze bounded candidate-model settings after the Phase 186 mechanism gate."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_PHASE186 = Path(
    os.environ.get(
        "AMB2022_01_PHASE186_ABLATION",
        "/root/matsci-gnn-pinn-ops/phase186_feature_ablation.json",
    )
)
MODEL_FIELDS = (
    "variant_id",
    "input_feature_source",
    "hidden_widths",
    "uses_heat_kernel",
    "uses_tam_monotonic_prior",
    "monotonic_feature",
    "monotonic_weight",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "max_epochs",
    "early_stopping_patience",
    "seeds",
    "fit_split",
    "selection_split",
    "report_split",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_model_rows() -> list[dict[str, Any]]:
    common = {
        "hidden_widths": "64,32",
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 4096,
        "max_epochs": 100,
        "early_stopping_patience": 12,
        "seeds": "1871,1872,1873",
        "fit_split": "B6 train only",
        "selection_split": "B7 validation only",
        "report_split": "B8 held build only",
    }
    return [
        {
            "variant_id": "small_data_only_mlp",
            "input_feature_source": "phase182 features",
            "uses_heat_kernel": False,
            "uses_tam_monotonic_prior": False,
            "monotonic_feature": "",
            "monotonic_weight": 0.0,
            **common,
        },
        {
            "variant_id": "physics_regularized_history_mlp",
            "input_feature_source": "phase182 features + phase186 causal_heat_kernel_features",
            "uses_heat_kernel": True,
            "uses_tam_monotonic_prior": True,
            "monotonic_feature": "laser_energy_density_J_mm2",
            "monotonic_weight": 0.01,
            **common,
        },
    ]


def build_gate(phase186: dict[str, Any]) -> dict[str, Any]:
    upstream = phase186.get("gate", {})
    expected = "phase186_feature_ablation_ready_phase187_candidate_model_design"
    ready = upstream.get("status") == expected and bool(upstream.get("phase187_candidate_model_design_allowed"))
    return {
        "status": (
            "phase187_candidate_model_design_ready_phase188_bounded_gpu_training"
            if ready
            else "phase187_candidate_model_design_blocked"
        ),
        "phase188_bounded_gpu_training_allowed": ready,
        "model_training_allowed": ready,
        "a800_training_allowed_now": ready,
        "compute_budget": {
            "variant_count": 2,
            "seed_count": 3,
            "max_epochs_per_run": 100,
            "max_total_runs": 6,
            "hyperparameter_search_allowed": False,
        },
        "blocking_audits": [] if ready else ["phase186_feature_ablation_not_ready"],
        "next_action": (
            "run the fixed six-run B6/B7/B8 comparison once; do not change hyperparameters after B8 is read"
            if ready
            else "do not start neural training until Phase 186 mechanism controls pass"
        ),
    }


def run(phase186_path: Path) -> dict[str, Any]:
    phase186 = _read_json(phase186_path)
    return {
        "phase": 187,
        "objective": "bounded_replicate_aware_candidate_model_design",
        "phase186_ablation": str(phase186_path),
        "model_contract": build_model_rows(),
        "gate": build_gate(phase186),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MODEL_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase186", type=Path, default=DEFAULT_PHASE186)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract-csv", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.phase186)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(args.contract_csv, payload["model_contract"])
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
