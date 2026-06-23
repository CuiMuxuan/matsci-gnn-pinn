#!/usr/bin/env python3
"""Build Phase 163 Bayesian/adaptive/hybrid PINN roadmap package.

This phase is deliberately a no-training route-planning gate. It turns verified
literature and the current repository boundary into executable next steps, while
keeping all neural-training and A100-80GB locks closed until a later baseline or
identifiability gate creates a measurable modeling space.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("docs/results/phase163_pinn_bayesian_hybrid_roadmap")

PHASE_INPUTS = {
    "phase154_gate": Path(
        "docs/results/phase154_route_coverage_and_remaining_scheme_audit/"
        "phase154_route_coverage_gate.json"
    ),
    "phase154_route_table": Path(
        "docs/results/phase154_route_coverage_and_remaining_scheme_audit/"
        "phase154_route_coverage_table.csv"
    ),
    "phase162_gate": Path(
        "docs/results/phase162_uci_steel_industry_energy_baseline_gate/"
        "phase162_uci_steel_industry_energy_baseline_gate.json"
    ),
}

LITERATURE_FIELDS = (
    "source_id",
    "title",
    "authors",
    "year",
    "venue",
    "doi",
    "url",
    "trust_state",
    "route_signal",
    "project_relevance",
    "limitation",
)

ROUTE_FIELDS = (
    "route_id",
    "route_name",
    "literature_anchor_ids",
    "physical_pain_point",
    "first_executable_gate",
    "baseline_or_control_guard",
    "promotion_condition",
    "training_allowed_now",
    "a100_80gb_request_now",
    "priority",
    "why_or_why_not_now",
)

QUEUE_FIELDS = (
    "step_id",
    "phase",
    "action",
    "implementation_scope",
    "expected_artifacts",
    "entry_condition",
    "exit_condition",
    "training_allowed_now",
    "a100_80gb_request_now",
)

CLAIM_GUARD_FIELDS = (
    "guard_id",
    "unsafe_claim",
    "safe_project_language",
    "reason",
    "evidence_anchor",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _csv_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def _display_path(path: Path, root: Path | None = None) -> str:
    if root is None:
        return str(path).replace("\\", "/")
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _is_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def _is_false(value: Any) -> bool:
    if value is False or value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "0", "false", "none", "no"}
    return False


def build_literature_rows() -> list[dict[str, Any]]:
    return [
        {
            "source_id": "P163-LIT-001",
            "title": (
                "Physics-informed neural networks: A deep learning framework for "
                "solving forward and inverse problems involving nonlinear partial "
                "differential equations"
            ),
            "authors": "Maziar Raissi; Paris Perdikaris; George Em Karniadakis",
            "year": 2019,
            "venue": "Journal of Computational Physics",
            "doi": "10.1016/j.jcp.2018.10.045",
            "url": "https://doi.org/10.1016/j.jcp.2018.10.045",
            "trust_state": "verified",
            "route_signal": "PINN foundation for forward/inverse PDE learning",
            "project_relevance": (
                "Supports keeping hidden-parameter discovery as a legitimate PINN use, "
                "but not as evidence that the current AM-Bench branch is solved."
            ),
            "limitation": "Generic PINN foundation; project-specific baselines remain required.",
        },
        {
            "source_id": "P163-LIT-002",
            "title": (
                "B-PINNs: Bayesian Physics-Informed Neural Networks for Forward and "
                "Inverse PDE Problems with Noisy Data"
            ),
            "authors": "Liu Yang; Xuhui Meng; George Em Karniadakis",
            "year": 2021,
            "venue": "Journal of Computational Physics",
            "doi": "10.1016/j.jcp.2020.109913",
            "url": "https://arxiv.org/abs/2003.06097",
            "trust_state": "verified",
            "route_signal": "Bayesian posterior estimation for noisy forward/inverse PDEs",
            "project_relevance": (
                "Directly matches Bayesian hidden-parameter inference and uncertainty "
                "calibration, suggesting a lightweight last-layer or parameter-posterior "
                "gate before full Bayesian networks."
            ),
            "limitation": "Full HMC/BNN can be expensive; start with synthetic identifiability.",
        },
        {
            "source_id": "P163-LIT-003",
            "title": (
                "A comprehensive study of non-adaptive and residual-based adaptive "
                "sampling for physics-informed neural networks"
            ),
            "authors": "Chenxi Wu; Min Zhu; Qinyang Tan; Yadhu Kartha; Lu Lu",
            "year": 2023,
            "venue": "Computer Methods in Applied Mechanics and Engineering",
            "doi": "10.1016/j.cma.2022.115671",
            "url": "https://arxiv.org/abs/2207.10289",
            "trust_state": "verified",
            "route_signal": "RAD/RAR-D adaptive residual sampling for more efficient PINNs",
            "project_relevance": (
                "Supports a no-training sampler audit and later low-budget synthetic heat "
                "equation gate instead of larger AM training."
            ),
            "limitation": "Sampler gains on benchmark PDEs do not imply AM target gain.",
        },
        {
            "source_id": "P163-LIT-004",
            "title": "Failure-Informed Adaptive Sampling for PINNs",
            "authors": "Zhiwei Gao; Liang Yan; Tao Zhou",
            "year": 2023,
            "venue": "SIAM Journal on Scientific Computing",
            "doi": "10.1137/22M1527763",
            "url": "https://epubs.siam.org/doi/10.1137/22M1527763",
            "trust_state": "verified",
            "route_signal": "Failure probability as residual-based enrichment indicator",
            "project_relevance": (
                "Useful for an adaptive-collocation design where the sampler must target "
                "sharp thermal gradients rather than uniform field points."
            ),
            "limitation": "Requires a PDE residual whose error indicator is meaningful.",
        },
        {
            "source_id": "P163-LIT-005",
            "title": (
                "Machine learning for metal additive manufacturing: predicting "
                "temperature and melt pool fluid dynamics using physics-informed "
                "neural networks"
            ),
            "authors": "Qiming Zhu; Zeliang Liu; Jinhui Yan",
            "year": 2021,
            "venue": "Computational Mechanics",
            "doi": "10.1007/s00466-020-01952-9",
            "url": "https://link.springer.com/article/10.1007/s00466-020-01952-9",
            "trust_state": "verified",
            "route_signal": "AM thermal/melt-pool PINN with conservation-law losses",
            "project_relevance": (
                "Confirms AM thermal PINN is a mature context; the project needs a more "
                "specific pain point than simply applying PINNs to AM."
            ),
            "limitation": "Not a novel route by itself for this project.",
        },
        {
            "source_id": "P163-LIT-006",
            "title": (
                "Hybrid thermal modeling of additive manufacturing processes using "
                "physics-informed neural networks for temperature prediction and "
                "parameter identification"
            ),
            "authors": (
                "Shuheng Liao; Tianju Xue; Jihoon Jeong; Samantha Webster; "
                "Kornel Ehmann; Jian Cao"
            ),
            "year": 2023,
            "venue": "Computational Mechanics",
            "doi": "10.1007/s00466-022-02257-9",
            "url": "https://link.springer.com/article/10.1007/s00466-022-02257-9",
            "trust_state": "verified",
            "route_signal": "AM PINN for temperature prediction plus parameter identification",
            "project_relevance": (
                "Closest match to a hidden thermal-parameter discovery route; suggests "
                "emissivity/effective diffusivity/source-width inference from sparse data."
            ),
            "limitation": "Still needs project-specific leakage-safe target and baselines.",
        },
        {
            "source_id": "P163-LIT-007",
            "title": (
                "PhyCRNet: Physics-informed convolutional-recurrent network for solving "
                "spatiotemporal PDEs"
            ),
            "authors": "Pu Ren; Chengping Rao; Yang Liu; Jian-Xun Wang; Hao Sun",
            "year": 2022,
            "venue": "Computer Methods in Applied Mechanics and Engineering",
            "doi": "10.1016/j.cma.2021.114399",
            "url": "https://arxiv.org/abs/2106.14103",
            "trust_state": "verified",
            "route_signal": "CNN/recurrent architecture with physics residuals for grid PDEs",
            "project_relevance": (
                "Supports CNN-style residual completion only for leakage-safe fixed-grid "
                "targets; current Phase 151 dense review closed that path."
            ),
            "limitation": "Current project dense targets are strong-baseline solved.",
        },
        {
            "source_id": "P163-LIT-008",
            "title": (
                "Physics-informed graph neural Galerkin networks: A unified framework "
                "for solving PDE-governed forward and inverse problems"
            ),
            "authors": "Han Gao; Matthew J. Zahr; Jian-Xun Wang",
            "year": 2022,
            "venue": "Computer Methods in Applied Mechanics and Engineering",
            "doi": "10.1016/j.cma.2021.114502",
            "url": "https://doi.org/10.1016/j.cma.2021.114502",
            "trust_state": "verified",
            "route_signal": "GCN/Galerkin PINN for unstructured meshes and inverse PDEs",
            "project_relevance": (
                "Supports replacing CNNs with graph operators for non-grid scan paths, "
                "but only after a non-scalar graph source clears a baseline gate."
            ),
            "limitation": "Phase 148 path-contact graph audit did not clear the project guard.",
        },
        {
            "source_id": "P163-LIT-009",
            "title": (
                "PhyGNNet: Solving Spatiotemporal PDEs with Physics-Informed Graph "
                "Neural Network"
            ),
            "authors": "Longxiang Jiang; Liyuan Wang; Xinkun Chu; Yonghao Xiao; Hao Zhang",
            "year": 2023,
            "venue": "ACM Asia Conference on Algorithms, Computing and Machine Learning",
            "doi": "10.1145/3590003.3590029",
            "url": "https://github.com/echowve/phygnnet",
            "trust_state": "verified",
            "route_signal": "Physics-informed GNN for spatiotemporal PDEs",
            "project_relevance": (
                "Secondary support for graph-PDE architectures on non-grid states; use "
                "only after project graph data pass no-training controls."
            ),
            "limitation": "Conference-scale support; not enough to override closed Phase 148.",
        },
        {
            "source_id": "P163-LIT-010",
            "title": "Self-Adaptive Physics-Informed Neural Networks using a Soft Attention Mechanism",
            "authors": "Levi McClenny; Ulisses Braga-Neto",
            "year": 2020,
            "venue": "arXiv",
            "doi": "10.48550/arXiv.2009.04544",
            "url": "https://arxiv.org/abs/2009.04544",
            "trust_state": "verified",
            "route_signal": "Trainable pointwise attention weights for difficult PDE regions",
            "project_relevance": (
                "Maps to a future residual/collocation focus mechanism, but it should not "
                "be trained until a sampler gate identifies hard regions."
            ),
            "limitation": "Attention weights can overfit without split-safe validation.",
        },
        {
            "source_id": "P163-LIT-011",
            "title": "Understanding and mitigating gradient pathologies in physics-informed neural networks",
            "authors": "Sifan Wang; Yujun Teng; Paris Perdikaris",
            "year": 2021,
            "venue": "SIAM Journal on Scientific Computing",
            "doi": "10.1137/20M1318043",
            "url": "https://arxiv.org/abs/2001.04536",
            "trust_state": "verified",
            "route_signal": "Gradient-statistics loss balancing for stiff composite PINN losses",
            "project_relevance": (
                "Supports logging gradient imbalance and using adaptive loss balancing in "
                "a later synthetic gate; it is not a data-source substitute."
            ),
            "limitation": "Training-strategy improvement only, not an independent pain point.",
        },
        {
            "source_id": "P163-LIT-012",
            "title": "Meta-learning PINN loss functions",
            "authors": "Apostolos F. Psaros; Kenji Kawaguchi; George Em Karniadakis",
            "year": 2022,
            "venue": "Journal of Computational Physics",
            "doi": "10.1016/j.jcp.2022.111121",
            "url": "https://arxiv.org/abs/2107.05544",
            "trust_state": "verified",
            "route_signal": "Meta-learned loss functions for families of PINN tasks",
            "project_relevance": (
                "Useful only after there are multiple related validated tasks; not the next "
                "single-branch implementation."
            ),
            "limitation": "Requires a task family and should remain behind Phase 164/165 gates.",
        },
    ]


def build_route_rows(
    *,
    phase154_gate: dict[str, Any],
    phase162_gate: dict[str, Any],
    route_table_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    current_routes_verified = _is_true(
        phase154_gate.get("currently_executable_model_routes_verified")
    )
    future_not_exhausted = not _is_true(
        phase154_gate.get("all_possible_future_schemes_exhausted")
    )
    phase162_closed = (
        phase162_gate.get("status")
        == "phase162_uci_steel_industry_energy_closed_no_stable_guarded_gap"
    )
    closed_graph_route = any(
        row.get("route_family") == "registered_source_path_green_capl"
        and "closed" in row.get("current_status", "")
        for row in route_table_rows
    )
    return [
        {
            "route_id": "P163-ROUTE-001",
            "route_name": "bayesian_inverse_heat_parameter_synthetic_gate",
            "literature_anchor_ids": "P163-LIT-001;P163-LIT-002;P163-LIT-006",
            "physical_pain_point": (
                "Sparse/noisy AM thermal observations can leave effective diffusivity, "
                "absorptivity/emissivity, or heat-source width underdetermined."
            ),
            "first_executable_gate": (
                "Phase 164 local synthetic inverse heat benchmark with known hidden "
                "parameter and noisy sparse sensors."
            ),
            "baseline_or_control_guard": (
                "grid-search least squares, ridge, ExtraTrees on engineered sensor "
                "features, posterior calibration control, wrong-physics control"
            ),
            "promotion_condition": (
                "Bayesian or ensemble posterior reduces validation parameter error and "
                "calibration error against all non-neural controls without test reversal."
            ),
            "training_allowed_now": False,
            "a100_80gb_request_now": False,
            "priority": 1,
            "why_or_why_not_now": (
                "Most aligned with the user's Bayesian/scientific-discovery request and "
                "does not depend on closed AMMT dense or graph gates."
            ),
        },
        {
            "route_id": "P163-ROUTE-002",
            "route_name": "adaptive_residual_sampler_heat_gate",
            "literature_anchor_ids": "P163-LIT-003;P163-LIT-004;P163-LIT-010;P163-LIT-011",
            "physical_pain_point": (
                "Uniform collocation under-samples sharp gradients near moving heat "
                "sources and can waste residual evaluations in easy regions."
            ),
            "first_executable_gate": (
                "Phase 165 no-training sampler diagnostic on analytic heat solutions: "
                "compare uniform/Sobol/RAD/RAR-D/FI point sets before training."
            ),
            "baseline_or_control_guard": (
                "residual coverage, hot-zone coverage, boundary coverage, seed stability, "
                "and equal-budget collocation count"
            ),
            "promotion_condition": (
                "Adaptive sampler improves held-out residual coverage and hotspot coverage "
                "at fixed point budget across seeds."
            ),
            "training_allowed_now": False,
            "a100_80gb_request_now": False,
            "priority": 2,
            "why_or_why_not_now": (
                "Good next no-training implementation after Phase 164 or in parallel only "
                "after the active roadmap branch is closed."
            ),
        },
        {
            "route_id": "P163-ROUTE-003",
            "route_name": "lightweight_bayesian_macro_pinn_uq",
            "literature_anchor_ids": "P163-LIT-002;P163-LIT-011",
            "physical_pain_point": (
                "Prediction intervals are missing from the current Macro PINN floor, but "
                "full Bayesian neural PINNs are too costly before a gap is proven."
            ),
            "first_executable_gate": (
                "After Phase 164/165 pass: last-layer Laplace or small ensemble UQ on a "
                "validated low-capacity PINN task."
            ),
            "baseline_or_control_guard": (
                "conformal residual baseline, bootstrap ExtraTrees, calibration and sharpness"
            ),
            "promotion_condition": (
                "Improves calibration/sharpness while preserving RMSE against the selected "
                "validated task guard."
            ),
            "training_allowed_now": False,
            "a100_80gb_request_now": False,
            "priority": 3,
            "why_or_why_not_now": (
                "Deferred because current project lacks a second-paper neural task with a "
                "stable baseline-visible gap."
            ),
        },
        {
            "route_id": "P163-ROUTE-004",
            "route_name": "gcn_or_path_graph_pinn_residual",
            "literature_anchor_ids": "P163-LIT-008;P163-LIT-009",
            "physical_pain_point": (
                "Scan paths and melt-pool histories are non-grid objects where graph "
                "operators are more natural than CNNs."
            ),
            "first_executable_gate": (
                "Requires a new non-scalar path/camera registration source; current Phase "
                "148 path-contact graph gate is closed."
            ),
            "baseline_or_control_guard": (
                "scalar source/path proxy, shuffled graph, layer/time/camera controls, "
                "strong HGB/ExtraTrees"
            ),
            "promotion_condition": (
                "New graph representation clears all controls on validation and does not "
                "reverse on test."
            ),
            "training_allowed_now": False,
            "a100_80gb_request_now": False,
            "priority": 4,
            "why_or_why_not_now": (
                "Not the next implementation because Phase 148 already closed the current "
                f"path graph source; closed_graph_route={closed_graph_route}."
            ),
        },
        {
            "route_id": "P163-ROUTE-005",
            "route_name": "cnn_or_fixed_grid_pinn_residual_completion",
            "literature_anchor_ids": "P163-LIT-005;P163-LIT-007",
            "physical_pain_point": (
                "CNN/ConvLSTM can learn spatial residuals on dense grids, but only if the "
                "target is not solved by non-neural fixed-grid baselines."
            ),
            "first_executable_gate": (
                "Requires new leakage-safe dense target/split; current Phase 151 dense "
                "operator route is closed."
            ),
            "baseline_or_control_guard": "mean/kNN/ExtraTrees/HGB dense baselines and split-contract audit",
            "promotion_condition": (
                "A dense target remains unsolved by strong baselines and has a defensible "
                "leakage-safe split."
            ),
            "training_allowed_now": False,
            "a100_80gb_request_now": False,
            "priority": 5,
            "why_or_why_not_now": (
                "Deferred because Phase 151 found zero low-capacity dense design candidates."
            ),
        },
        {
            "route_id": "P163-ROUTE-006",
            "route_name": "meta_learned_pinn_loss_family",
            "literature_anchor_ids": "P163-LIT-012",
            "physical_pain_point": (
                "Fast adaptation across multiple related PDE/process tasks matters only "
                "after several validated tasks exist."
            ),
            "first_executable_gate": (
                "Requires a family of at least three validated Phase 164/165-style tasks."
            ),
            "baseline_or_control_guard": "per-task tuned loss, gradient-statistic annealing, no-meta initialization",
            "promotion_condition": (
                "Meta-learned loss improves adaptation speed and final validation metric "
                "across held-out tasks."
            ),
            "training_allowed_now": False,
            "a100_80gb_request_now": False,
            "priority": 6,
            "why_or_why_not_now": (
                "Not enough validated tasks yet; future scheme space remains open="
                f"{future_not_exhausted}, current routes verified={current_routes_verified}, "
                f"Phase162 closed={phase162_closed}."
            ),
        },
    ]


def build_queue_rows() -> list[dict[str, Any]]:
    return [
        {
            "step_id": "P163-QUEUE-001",
            "phase": "Phase 164",
            "action": "synthetic_bayesian_inverse_heat_identifiability_gate",
            "implementation_scope": (
                "Generate analytic/noisy 1D or 2D heat-equation sensor data with hidden "
                "diffusivity/source-width; evaluate non-neural inverse controls and a "
                "lightweight Bayesian posterior diagnostic without AM training."
            ),
            "expected_artifacts": (
                "gate JSON, manifest, metric CSV, calibration CSV, parameter posterior "
                "summary, markdown, tests, runner"
            ),
            "entry_condition": "Phase 163 roadmap gate ready and all training locks false.",
            "exit_condition": (
                "Either close as baseline-solved/unidentifiable or open a low-capacity "
                "PINN mechanism smoke gate only."
            ),
            "training_allowed_now": False,
            "a100_80gb_request_now": False,
        },
        {
            "step_id": "P163-QUEUE-002",
            "phase": "Phase 165",
            "action": "adaptive_residual_sampler_no_training_gate",
            "implementation_scope": (
                "Compare uniform/Sobol/RAD/RAR-D/FI collocation sets against analytic heat "
                "residual and hot-gradient coverage metrics before model training."
            ),
            "expected_artifacts": (
                "sampler table, residual coverage table, seed stability table, gate JSON, "
                "markdown, tests, runner"
            ),
            "entry_condition": "Phase 164 closes or opens sampler-specific need.",
            "exit_condition": "Open low-budget PINN training only if fixed-budget sampler gain is stable.",
            "training_allowed_now": False,
            "a100_80gb_request_now": False,
        },
        {
            "step_id": "P163-QUEUE-003",
            "phase": "Phase 166",
            "action": "low_capacity_pinn_or_bayesian_last_layer_smoke",
            "implementation_scope": (
                "Only if Phase 164/165 pass: compare tiny PINN, last-layer Laplace, "
                "small ensemble, and non-neural UQ controls."
            ),
            "expected_artifacts": "training metrics, calibration metrics, gate JSON, small checkpoints only",
            "entry_condition": "A previous no-training gate explicitly opens low-capacity training.",
            "exit_condition": "Promote only with validation-selected seed-stable gain.",
            "training_allowed_now": False,
            "a100_80gb_request_now": False,
        },
    ]


def build_claim_guard_rows() -> list[dict[str, Any]]:
    return [
        {
            "guard_id": "P163-GUARD-001",
            "unsafe_claim": "Bayesian PINN has succeeded for this project",
            "safe_project_language": (
                "Bayesian PINN is a verified next-route candidate; Phase 164 must first "
                "prove identifiability and baseline-visible value."
            ),
            "reason": "Phase 163 is a roadmap package, not a training or performance result.",
            "evidence_anchor": "phase163_route_candidate_table.csv",
        },
        {
            "guard_id": "P163-GUARD-002",
            "unsafe_claim": "CNN/GCN-PINN residual completion is now the main contribution",
            "safe_project_language": (
                "CNN and graph-PINN routes remain conditional on new leakage-safe dense or "
                "non-grid source gates."
            ),
            "reason": "Phase 148 and Phase 151 closed current graph/dense candidates.",
            "evidence_anchor": "phase154_route_coverage_table.csv",
        },
        {
            "guard_id": "P163-GUARD-003",
            "unsafe_claim": "Adaptive sampling or attention can replace source gating",
            "safe_project_language": (
                "Adaptive sampling is a training-efficiency mechanism and must follow a "
                "valid physical task and residual definition."
            ),
            "reason": "Sampler papers solve collocation efficiency, not data leakage or target choice.",
            "evidence_anchor": "phase163_literature_evidence_table.csv",
        },
        {
            "guard_id": "P163-GUARD-004",
            "unsafe_claim": "A100-SXM4-80GB is needed for the next phase",
            "safe_project_language": (
                "A800/A100-40GB remains sufficient; 80GB is requested only after a "
                "seed-positive branch hits a measured 40GB bottleneck."
            ),
            "reason": "Phase 163 opens only local/no-training gates.",
            "evidence_anchor": "phase163_pinn_bayesian_hybrid_roadmap_gate.json",
        },
    ]


def build_gate(
    *,
    phase154_gate: dict[str, Any],
    phase162_gate: dict[str, Any],
    literature_rows: list[dict[str, Any]],
    route_rows: list[dict[str, Any]],
    queue_rows: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    phase154_ready = (
        phase154_gate.get("status")
        == "phase154_route_coverage_audit_ready_current_routes_verified_future_not_exhausted"
    )
    phase162_closed = (
        phase162_gate.get("status")
        == "phase162_uci_steel_industry_energy_closed_no_stable_guarded_gap"
    )
    verified_sources = sum(1 for row in literature_rows if row["trust_state"] == "verified")
    all_locks_false = (
        all(_is_false(row["training_allowed_now"]) for row in route_rows)
        and all(_is_false(row["a100_80gb_request_now"]) for row in route_rows)
        and all(_is_false(row["training_allowed_now"]) for row in queue_rows)
        and all(_is_false(row["a100_80gb_request_now"]) for row in queue_rows)
        and _is_false(phase154_gate.get("phase154_model_training_allowed"))
        and _is_false(phase154_gate.get("a100_training_allowed_now"))
        and _is_false(phase154_gate.get("a100_80gb_request_now"))
        and _is_false(phase162_gate.get("phase162_model_training_allowed"))
        and _is_false(phase162_gate.get("a100_training_allowed_now"))
        and _is_false(phase162_gate.get("a100_80gb_request_now"))
    )
    ready = phase154_ready and phase162_closed and verified_sources >= 10 and all_locks_false
    return {
        "status": (
            "phase163_pinn_bayesian_hybrid_roadmap_ready_phase164_synthetic_inverse_gate"
            if ready
            else "phase163_pinn_bayesian_hybrid_roadmap_incomplete"
        ),
        "verified_literature_rows": verified_sources,
        "route_candidate_rows": len(route_rows),
        "execution_queue_rows": len(queue_rows),
        "claim_guard_rows": len(claim_rows),
        "recommended_next_phase": "phase164_synthetic_bayesian_inverse_heat_identifiability_gate",
        "recommended_next_route": "bayesian_inverse_heat_parameter_synthetic_gate",
        "phase164_no_training_design_allowed": bool(ready),
        "phase164_low_capacity_training_allowed": False,
        "phase163_model_mechanism_allowed": False,
        "phase163_model_training_allowed": False,
        "bayesian_pinn_training_allowed_now": False,
        "adaptive_sampling_training_allowed_now": False,
        "gcn_pinn_training_allowed_now": False,
        "cnn_pinn_training_allowed_now": False,
        "operator_training_allowed_now": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "phase154_status": phase154_gate.get("status"),
        "phase162_status": phase162_gate.get("status"),
        "next_action": (
            "enter Phase 164 as a local no-training synthetic inverse-heat "
            "identifiability gate; do not train Bayesian/CNN/GCN PINNs or request 80GB "
            "until a later gate explicitly opens that path"
        ),
    }


def _markdown_table(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[str]:
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    body = [
        "| " + " | ".join(_csv_value(row.get(field, "")) for field in fields) + " |"
        for row in rows
    ]
    return [header, sep, *body]


def build_markdown(
    *,
    gate: dict[str, Any],
    literature_rows: list[dict[str, Any]],
    route_rows: list[dict[str, Any]],
    queue_rows: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
) -> str:
    lines: list[str] = [
        "# Phase 163 Bayesian/adaptive/hybrid PINN Roadmap",
        "",
        "## Gate",
        f"- Status: `{gate['status']}`",
        f"- Verified literature rows: `{gate['verified_literature_rows']}`",
        f"- Recommended next phase: `{gate['recommended_next_phase']}`",
        f"- Phase 164 no-training design allowed: `{_csv_value(gate['phase164_no_training_design_allowed'])}`",
        f"- Phase 163 model training allowed: `{_csv_value(gate['phase163_model_training_allowed'])}`",
        f"- Bayesian PINN training allowed now: `{_csv_value(gate['bayesian_pinn_training_allowed_now'])}`",
        f"- GCN PINN training allowed now: `{_csv_value(gate['gcn_pinn_training_allowed_now'])}`",
        f"- A100 training allowed now: `{_csv_value(gate['a100_training_allowed_now'])}`",
        f"- A100-SXM4-80GB request now: `{_csv_value(gate['a100_80gb_request_now'])}`",
        "",
        "## Interpretation",
        (
            "The literature supports Bayesian inverse PINNs, residual/adaptive sampling, "
            "CNN/grid PINNs, and graph PDE networks as plausible mechanisms. In this "
            "repository, however, Phase 148 closed the current path-graph source, Phase "
            "151 closed current dense-grid operator candidates, and Phase 162 closed the "
            "latest external source. The only immediate executable path is therefore a "
            "local no-training synthetic inverse-heat identifiability gate."
        ),
        "",
        "## Literature Evidence",
        *_markdown_table(literature_rows, LITERATURE_FIELDS),
        "",
        "## Candidate Routes",
        *_markdown_table(route_rows, ROUTE_FIELDS),
        "",
        "## Execution Queue",
        *_markdown_table(queue_rows, QUEUE_FIELDS),
        "",
        "## Claim Guards",
        *_markdown_table(claim_rows, CLAIM_GUARD_FIELDS),
        "",
    ]
    return "\n".join(lines)


def build_package(*, root: Path, output_dir: Path, phase_inputs: dict[str, Path]) -> dict[str, Any]:
    root = root.resolve()
    output_dir = output_dir if output_dir.is_absolute() else root / output_dir
    resolved = {
        name: path if path.is_absolute() else root / path for name, path in phase_inputs.items()
    }

    phase154_gate = _read_json(resolved["phase154_gate"])
    phase162_gate = _read_json(resolved["phase162_gate"])
    route_table_rows = _read_csv(resolved["phase154_route_table"])

    literature_rows = build_literature_rows()
    route_rows = build_route_rows(
        phase154_gate=phase154_gate,
        phase162_gate=phase162_gate,
        route_table_rows=route_table_rows,
    )
    queue_rows = build_queue_rows()
    claim_rows = build_claim_guard_rows()
    gate = build_gate(
        phase154_gate=phase154_gate,
        phase162_gate=phase162_gate,
        literature_rows=literature_rows,
        route_rows=route_rows,
        queue_rows=queue_rows,
        claim_rows=claim_rows,
    )

    literature_path = output_dir / "phase163_literature_evidence_table.csv"
    route_path = output_dir / "phase163_route_candidate_table.csv"
    queue_path = output_dir / "phase163_execution_queue_table.csv"
    claim_path = output_dir / "phase163_claim_guard_table.csv"
    gate_path = output_dir / "phase163_pinn_bayesian_hybrid_roadmap_gate.json"
    markdown_path = output_dir / "phase163_pinn_bayesian_hybrid_roadmap.md"
    manifest_path = output_dir / "phase163_pinn_bayesian_hybrid_roadmap_manifest.json"

    _write_csv(literature_path, literature_rows, LITERATURE_FIELDS)
    _write_csv(route_path, route_rows, ROUTE_FIELDS)
    _write_csv(queue_path, queue_rows, QUEUE_FIELDS)
    _write_csv(claim_path, claim_rows, CLAIM_GUARD_FIELDS)
    _write_json(gate_path, gate)
    with markdown_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            build_markdown(
                gate=gate,
                literature_rows=literature_rows,
                route_rows=route_rows,
                queue_rows=queue_rows,
                claim_rows=claim_rows,
            )
        )

    manifest = {
        "phase": 163,
        "description": "Bayesian/adaptive/hybrid PINN literature-grounded no-training roadmap",
        "inputs": {name: _display_path(path, root) for name, path in resolved.items()},
        "outputs": {
            "literature_evidence_table": _display_path(literature_path, root),
            "route_candidate_table": _display_path(route_path, root),
            "execution_queue_table": _display_path(queue_path, root),
            "claim_guard_table": _display_path(claim_path, root),
            "gate": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "phase154_route_rows": len(route_table_rows),
            "verified_literature_rows": gate["verified_literature_rows"],
            "route_candidate_rows": gate["route_candidate_rows"],
            "execution_queue_rows": gate["execution_queue_rows"],
            "claim_guard_rows": gate["claim_guard_rows"],
            "training_allowed_route_rows": sum(
                1 for row in route_rows if not _is_false(row["training_allowed_now"])
            ),
            "a100_80gb_request_route_rows": sum(
                1 for row in route_rows if not _is_false(row["a100_80gb_request_now"])
            ),
        },
        "gate": gate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    for name, default in PHASE_INPUTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=default)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase_inputs = {name: getattr(args, name) for name in PHASE_INPUTS}
    manifest = build_package(root=args.root, output_dir=args.output_dir, phase_inputs=phase_inputs)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
