#!/usr/bin/env python3
"""Build Phase 176 evidence refresh for the hidden source/closure branch.

This phase consumes only existing small Phase 169-175 artifacts. It refreshes
the paper-facing boundary for the synthetic inverse-heat hidden-closure route
after the Phase 175 low-capacity expansion closed without incremental gain.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("docs/results/phase176_hidden_closure_evidence_refresh")

PHASE_INPUTS = {
    "phase169_gate": Path(
        "docs/results/phase169_hidden_source_closure_identifiability_gate/"
        "phase169_hidden_source_closure_identifiability_gate.json"
    ),
    "phase170_gate": Path(
        "docs/results/phase170_hidden_closure_mechanism_smoke_design_gate/"
        "phase170_hidden_closure_mechanism_smoke_design_gate.json"
    ),
    "phase171_gate": Path(
        "docs/results/phase171_hidden_closure_low_budget_smoke/"
        "phase171_hidden_closure_low_budget_smoke_gate.json"
    ),
    "phase172_gate": Path(
        "docs/results/phase172_trainable_hidden_closure_smoke_design_gate/"
        "phase172_trainable_hidden_closure_smoke_design_gate.json"
    ),
    "phase173_gate": Path(
        "docs/results/phase173_trainable_hidden_closure_low_budget_smoke/"
        "phase173_trainable_hidden_closure_low_budget_smoke_gate.json"
    ),
    "phase174_gate": Path(
        "docs/results/phase174_low_capacity_hidden_closure_design_gate/"
        "phase174_low_capacity_hidden_closure_design_gate.json"
    ),
    "phase175_gate": Path(
        "docs/results/phase175_low_capacity_hidden_closure_smoke/"
        "phase175_low_capacity_hidden_closure_smoke_gate.json"
    ),
}

ROUTE_FIELDS = (
    "route_id",
    "phase",
    "artifact",
    "route_status",
    "evidence_type",
    "positive_signal",
    "limitation_or_closure",
    "paper_use",
    "model_training_allowed",
    "a100_training_allowed_now",
    "a100_80gb_request_now",
    "next_action",
)

CLAIM_FIELDS = (
    "claim_id",
    "claim_area",
    "claim_status",
    "paper_boundary",
    "evidence_anchor",
    "allowed_final_use",
)

DECISION_FIELDS = (
    "decision_id",
    "decision",
    "status",
    "rationale",
    "evidence_anchor",
    "next_action",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.10f}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _display_path(path: Path, root: Path | None = None) -> str:
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def _is_false(value: Any) -> bool:
    if value is False or value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "0", "false", "none", "no"}
    return False


def _is_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def _blocking_text(gate: dict[str, Any]) -> str:
    audits = gate.get("blocking_audits")
    if isinstance(audits, list):
        return ";".join(str(item) for item in audits)
    if audits:
        return str(audits)
    return ""


def _training_lock(gate: dict[str, Any], key: str) -> bool:
    return bool(gate.get(key, False))


def build_route_rows(
    *, gates: dict[str, dict[str, Any]], input_paths: dict[str, Path], root: Path
) -> list[dict[str, Any]]:
    phase169 = gates["phase169_gate"]
    phase170 = gates["phase170_gate"]
    phase171 = gates["phase171_gate"]
    phase172 = gates["phase172_gate"]
    phase173 = gates["phase173_gate"]
    phase174 = gates["phase174_gate"]
    phase175 = gates["phase175_gate"]
    return [
        {
            "route_id": "P176-ROUTE-001",
            "phase": "phase169",
            "artifact": _display_path(input_paths["phase169_gate"], root),
            "route_status": phase169.get("status"),
            "evidence_type": "no_training_synthetic_identifiability_positive",
            "positive_signal": (
                "calibrated Bayesian hidden source/closure posterior beat "
                f"{phase169.get('best_control_method')} by validation score gain "
                f"{phase169.get('validation_score_gain_vs_best_control')}"
            ),
            "limitation_or_closure": (
                "synthetic inverse-heat only; no neural PINN training or AM-Bench evidence"
            ),
            "paper_use": "second_paper_concept_positive",
            "model_training_allowed": phase169.get("phase169_model_training_allowed", False),
            "a100_training_allowed_now": phase169.get("a100_training_allowed_now", False),
            "a100_80gb_request_now": phase169.get("a100_80gb_request_now", False),
            "next_action": "use as calibrated identifiability evidence, not as trained PINN evidence",
        },
        {
            "route_id": "P176-ROUTE-002",
            "phase": "phase170",
            "artifact": _display_path(input_paths["phase170_gate"], root),
            "route_status": phase170.get("status"),
            "evidence_type": "design_only_mechanism_protocol",
            "positive_signal": (
                f"bounded mechanism package with {phase170.get('control_rows')} controls "
                f"and {phase170.get('promotion_rows')} promotion rules"
            ),
            "limitation_or_closure": "protocol only; no training executed in the phase",
            "paper_use": "methods_guard_or_appendix_protocol",
            "model_training_allowed": phase170.get("phase170_model_training_allowed", False),
            "a100_training_allowed_now": phase170.get("a100_training_allowed_now", False),
            "a100_80gb_request_now": phase170.get("a100_80gb_request_now", False),
            "next_action": "cite only as the gate that constrained Phase 171",
        },
        {
            "route_id": "P176-ROUTE-003",
            "phase": "phase171",
            "artifact": _display_path(input_paths["phase171_gate"], root),
            "route_status": phase171.get("status"),
            "evidence_type": "numpy_low_budget_closure_positive",
            "positive_signal": (
                f"selected {phase171.get('selected_variant')} over "
                f"{phase171.get('best_control_variant')} with validation gain "
                f"{phase171.get('validation_score_gain_vs_best_control')} and seed stability "
                f"{phase171.get('seed_stability_pass_rate')}"
            ),
            "limitation_or_closure": (
                "small NumPy synthetic closure head; not Bayesian neural inference or AM data"
            ),
            "paper_use": "second_paper_supporting_positive",
            "model_training_allowed": phase171.get("phase171_model_training_allowed", False),
            "a100_training_allowed_now": phase171.get("a100_training_allowed_now", False),
            "a100_80gb_request_now": phase171.get("a100_80gb_request_now", False),
            "next_action": "write as bounded explicit closure recovery evidence",
        },
        {
            "route_id": "P176-ROUTE-004",
            "phase": "phase172",
            "artifact": _display_path(input_paths["phase172_gate"], root),
            "route_status": phase172.get("status"),
            "evidence_type": "design_only_trainable_smoke_protocol",
            "positive_signal": (
                f"candidate route {phase172.get('candidate_trainable_route')} opened "
                "only a bounded Phase 173 smoke"
            ),
            "limitation_or_closure": "design-only; no model result",
            "paper_use": "appendix_protocol",
            "model_training_allowed": phase172.get("phase172_model_training_allowed", False),
            "a100_training_allowed_now": phase172.get("a100_training_allowed_now", False),
            "a100_80gb_request_now": phase172.get("a100_80gb_request_now", False),
            "next_action": "cite only as route-control evidence",
        },
        {
            "route_id": "P176-ROUTE-005",
            "phase": "phase173",
            "artifact": _display_path(input_paths["phase173_gate"], root),
            "route_status": phase173.get("status"),
            "evidence_type": "tiny_explicit_latent_synthetic_positive",
            "positive_signal": (
                f"selected {phase173.get('selected_variant')} over "
                f"{phase173.get('best_control_variant')} with validation gain "
                f"{phase173.get('validation_score_gain_vs_best_control')}, test reversal ratio "
                f"{phase173.get('test_reversal_ratio_vs_best_control')}, and seed stability "
                f"{phase173.get('seed_stability_pass_rate')}"
            ),
            "limitation_or_closure": (
                "tiny synthetic trainable-latent smoke; not full PINN, not AM-Bench, not GNN/CNN/operator"
            ),
            "paper_use": "second_paper_candidate_core_synthetic_positive",
            "model_training_allowed": phase173.get("phase173_model_training_allowed", False),
            "a100_training_allowed_now": phase173.get("a100_training_allowed_now", False),
            "a100_80gb_request_now": phase173.get("a100_80gb_request_now", False),
            "next_action": "preserve as the strongest branch result before redesign",
        },
        {
            "route_id": "P176-ROUTE-006",
            "phase": "phase174",
            "artifact": _display_path(input_paths["phase174_gate"], root),
            "route_status": phase174.get("status"),
            "evidence_type": "design_only_low_capacity_expansion_protocol",
            "positive_signal": (
                f"candidate route {phase174.get('candidate_low_capacity_route')} was "
                "properly controlled before Phase 175"
            ),
            "limitation_or_closure": "design-only; later Phase 175 did not validate the expansion",
            "paper_use": "appendix_protocol",
            "model_training_allowed": phase174.get("phase174_model_training_allowed", False),
            "a100_training_allowed_now": phase174.get("a100_training_allowed_now", False),
            "a100_80gb_request_now": phase174.get("a100_80gb_request_now", False),
            "next_action": "do not treat as model success",
        },
        {
            "route_id": "P176-ROUTE-007",
            "phase": "phase175",
            "artifact": _display_path(input_paths["phase175_gate"], root),
            "route_status": phase175.get("status"),
            "evidence_type": "low_capacity_expansion_closure_negative",
            "positive_signal": (
                "none; validation selected "
                f"{phase175.get('selected_variant')} instead of "
                f"{phase175.get('candidate_variant')}"
            ),
            "limitation_or_closure": (
                f"closed by {phase175.get('validation_score_gain_vs_best_control')} validation gain, "
                f"test reversal {phase175.get('test_reversal_ratio_vs_best_control')}, "
                f"seed stability {phase175.get('seed_stability_pass_rate')}; blockers "
                f"{_blocking_text(phase175)}"
            ),
            "paper_use": "route_closure_or_limitations",
            "model_training_allowed": phase175.get("phase175_model_training_allowed", False),
            "a100_training_allowed_now": phase175.get("a100_training_allowed_now", False),
            "a100_80gb_request_now": phase175.get("a100_80gb_request_now", False),
            "next_action": "do not retune the same low-capacity head",
        },
    ]


def build_claim_boundary_rows(*, root: Path, input_paths: dict[str, Path]) -> list[dict[str, Any]]:
    phase173_anchor = _display_path(input_paths["phase173_gate"], root)
    phase175_anchor = _display_path(input_paths["phase175_gate"], root)
    return [
        {
            "claim_id": "P176-CLAIM-001",
            "claim_area": "synthetic_hidden_source_closure_identifiability",
            "claim_status": "allowed_narrow_positive",
            "paper_boundary": (
                "May claim calibrated hidden source/closure identifiability and explicit "
                "latent closure recovery on controlled synthetic inverse-heat tasks."
            ),
            "evidence_anchor": _display_path(input_paths["phase169_gate"], root),
            "allowed_final_use": "second_paper_concept_or_methods_result",
        },
        {
            "claim_id": "P176-CLAIM-002",
            "claim_area": "tiny_explicit_latent_trainable_smoke",
            "claim_status": "allowed_narrow_positive",
            "paper_boundary": (
                "May describe Phase 173 as a bounded tiny synthetic explicit-latent "
                "positive over posterior/grid/wrong-source/no-closure/data-only controls."
            ),
            "evidence_anchor": phase173_anchor,
            "allowed_final_use": "second_paper_candidate_core_synthetic_result",
        },
        {
            "claim_id": "P176-CLAIM-003",
            "claim_area": "low_capacity_head_expansion",
            "claim_status": "closed_negative",
            "paper_boundary": (
                "Do not claim the Phase 174/175 low-capacity head improves the route; "
                "Phase 175 selected the simpler Phase 173 control."
            ),
            "evidence_anchor": phase175_anchor,
            "allowed_final_use": "limitations_or_appendix",
        },
        {
            "claim_id": "P176-CLAIM-004",
            "claim_area": "full_bayesian_pinn_or_adaptive_pinn_training",
            "claim_status": "blocked_success_claim",
            "paper_boundary": (
                "Do not write Bayesian PINN training success, adaptive sampling training "
                "success, or full neural PINN success from Phase 169-175."
            ),
            "evidence_anchor": "docs/results/phase167_low_budget_pinn_smoke/phase167_low_budget_pinn_smoke_gate.json; "
            + phase175_anchor,
            "allowed_final_use": "claim_guardrail",
        },
        {
            "claim_id": "P176-CLAIM-005",
            "claim_area": "am_bench_nist_or_registered_camera_generalization",
            "claim_status": "blocked_success_claim",
            "paper_boundary": (
                "Do not transfer the synthetic hidden-closure result to AM-Bench, NIST "
                "AMMT, registered camera targets, scan-path/Green features, or "
                "general AM process modeling."
            ),
            "evidence_anchor": phase175_anchor,
            "allowed_final_use": "claim_guardrail",
        },
        {
            "claim_id": "P176-CLAIM-006",
            "claim_area": "gcn_cnn_operator_microstructure_or_path_graph",
            "claim_status": "blocked_success_claim",
            "paper_boundary": (
                "Do not write GCN/PINN, CNN/operator, microstructure GNN, path-contact "
                "graph, MAM-PhyGNN, CAPL, or FNO success from this branch."
            ),
            "evidence_anchor": "phase148/phase151 closures; " + phase175_anchor,
            "allowed_final_use": "claim_guardrail",
        },
        {
            "claim_id": "P176-CLAIM-007",
            "claim_area": "compute_need",
            "claim_status": "blocked_80gb_claim",
            "paper_boundary": (
                "Do not claim A100-SXM4-80GB is needed. Phase 169-175 used tiny "
                "synthetic or design-only evidence and all compute locks stayed false."
            ),
            "evidence_anchor": phase175_anchor,
            "allowed_final_use": "project_boundary_note",
        },
    ]


def build_gate(
    *, gates: dict[str, dict[str, Any]], route_rows: list[dict[str, Any]], claim_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    phase169_ready = (
        gates["phase169_gate"].get("status")
        == "phase169_hidden_source_closure_identifiability_ready_phase170_low_budget_mechanism_design"
    )
    phase171_ready = (
        gates["phase171_gate"].get("status")
        == "phase171_hidden_closure_low_budget_smoke_ready_phase172_trainable_design"
    )
    phase173_ready = (
        gates["phase173_gate"].get("status")
        == "phase173_trainable_hidden_closure_low_budget_smoke_ready_phase174_low_capacity_hidden_closure_design"
    )
    phase175_closed = (
        gates["phase175_gate"].get("status")
        == "phase175_low_capacity_hidden_closure_smoke_closed_no_incremental_gain"
        and _is_false(gates["phase175_gate"].get("phase176_focused_review_allowed"))
    )
    locks_ok = all(
        _is_false(row.get("model_training_allowed"))
        and _is_false(row.get("a100_training_allowed_now"))
        and _is_false(row.get("a100_80gb_request_now"))
        for row in route_rows
    )
    branch_refreshed = phase169_ready and phase171_ready and phase173_ready and phase175_closed and locks_ok
    status = (
        "phase176_hidden_closure_evidence_refresh_ready_synthetic_claims_low_capacity_closed"
        if branch_refreshed
        else "phase176_hidden_closure_evidence_refresh_incomplete"
    )
    return {
        "status": status,
        "phase169_identifiability_positive_preserved": bool(phase169_ready),
        "phase171_numpy_closure_positive_preserved": bool(phase171_ready),
        "phase173_tiny_trainable_positive_preserved": bool(phase173_ready),
        "phase175_low_capacity_route_closed": bool(phase175_closed),
        "hidden_closure_branch_refreshed": bool(branch_refreshed),
        "synthetic_hidden_closure_claim_allowed_now": bool(branch_refreshed),
        "second_paper_core_claim_ready": False,
        "low_capacity_head_claim_ready": False,
        "full_bayesian_pinn_claim_ready": False,
        "am_bench_hidden_closure_claim_ready": False,
        "gcn_cnn_operator_claim_ready": False,
        "phase177_materially_different_mechanism_design_allowed": bool(branch_refreshed),
        "phase176_model_mechanism_allowed": False,
        "phase176_model_training_allowed": False,
        "phase177_training_allowed_now": False,
        "bayesian_pinn_training_allowed_now": False,
        "adaptive_sampling_training_allowed_now": False,
        "gcn_pinn_training_allowed_now": False,
        "cnn_operator_training_allowed_now": False,
        "a100_training_allowed_now": False,
        "a100_80gb_request_now": False,
        "route_evidence_rows": len(route_rows),
        "claim_boundary_rows": len(claim_rows),
        "blocked_success_claim_rows": sum(
            1 for row in claim_rows if row["claim_status"] == "blocked_success_claim"
        ),
        "next_action": (
            "preserve the synthetic hidden-closure positives as bounded second-paper "
            "evidence, close the low-capacity head, and only design a materially "
            "different mechanism before any further training"
        ),
    }


def build_decision_rows(
    *, gate: dict[str, Any], root: Path, output_dir: Path
) -> list[dict[str, Any]]:
    route_path = _display_path(output_dir / "phase176_hidden_closure_route_evidence_table.csv", root)
    claim_path = _display_path(output_dir / "phase176_claim_boundary_refresh_table.csv", root)
    return [
        {
            "decision_id": "P176-DECISION-001",
            "decision": "preserve_synthetic_hidden_closure_positive",
            "status": "allowed_narrow" if gate["synthetic_hidden_closure_claim_allowed_now"] else "blocked",
            "rationale": (
                "Phase 169, 171, and 173 remain positive under their registered controls, "
                "but only on controlled synthetic inverse-heat tasks."
            ),
            "evidence_anchor": route_path,
            "next_action": "write as bounded synthetic mechanism evidence, not as AM or full PINN success",
        },
        {
            "decision_id": "P176-DECISION-002",
            "decision": "close_low_capacity_head_expansion",
            "status": "closed" if gate["phase175_low_capacity_route_closed"] else "incomplete",
            "rationale": "Phase 175 selected the Phase 173 control and failed validation/test/stability guards.",
            "evidence_anchor": route_path,
            "next_action": "do not retune the same low-capacity ridge/head route",
        },
        {
            "decision_id": "P176-DECISION-003",
            "decision": "second_paper_core_claim",
            "status": "not_ready",
            "rationale": (
                "The branch supports a concept/methods boundary but still lacks an AM-data "
                "or stronger model-mechanism result for a complete second-paper core."
            ),
            "evidence_anchor": claim_path,
            "next_action": "design a materially different mechanism gate before training",
        },
        {
            "decision_id": "P176-DECISION-004",
            "decision": "compute_escalation",
            "status": "blocked",
            "rationale": "All Phase 169-175 training/A100/80GB locks are false.",
            "evidence_anchor": claim_path,
            "next_action": "continue on local/A800 small gates; do not request A100-SXM4-80GB",
        },
    ]


def _markdown_table(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    body = [
        "| " + " | ".join(_csv_value(row.get(field)).replace("\n", " ") for field in fields) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def build_markdown(
    *,
    gate: dict[str, Any],
    route_rows: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "# Phase 176 Hidden-Closure Evidence Refresh",
            "",
            "## Gate Decision",
            "",
            f"Status: `{gate['status']}`.",
            f"Synthetic hidden-closure claim allowed now: `{_csv_value(gate['synthetic_hidden_closure_claim_allowed_now'])}`.",
            f"Second-paper core claim ready: `{_csv_value(gate['second_paper_core_claim_ready'])}`.",
            f"Low-capacity head claim ready: `{_csv_value(gate['low_capacity_head_claim_ready'])}`.",
            f"Phase 176 model training allowed: `{_csv_value(gate['phase176_model_training_allowed'])}`.",
            f"Phase 177 training allowed now: `{_csv_value(gate['phase177_training_allowed_now'])}`.",
            f"A100 training allowed now: `{_csv_value(gate['a100_training_allowed_now'])}`.",
            f"A100-SXM4-80GB request now: `{_csv_value(gate['a100_80gb_request_now'])}`.",
            "",
            "Phase 176 reads only existing small Phase 169-175 artifacts. It does not read raw data, run baselines, or train a model.",
            "",
            "## Interpretation",
            "",
            (
                "The synthetic inverse-heat hidden source/closure branch keeps useful "
                "bounded positives in Phase 169, Phase 171, and Phase 173. Phase 175 "
                "closes the low-capacity head expansion, so the next research move must "
                "be a materially different mechanism design rather than retuning the same head."
            ),
            "",
            "## Route Evidence",
            "",
            _markdown_table(route_rows, ROUTE_FIELDS),
            "",
            "## Claim Boundaries",
            "",
            _markdown_table(claim_rows, CLAIM_FIELDS),
            "",
            "## Next Decisions",
            "",
            _markdown_table(decision_rows, DECISION_FIELDS),
            "",
        ]
    )


def build_package(*, root: Path, output_dir: Path, phase_inputs: dict[str, Path]) -> dict[str, Any]:
    root = root.resolve()
    output_dir = output_dir if output_dir.is_absolute() else root / output_dir
    resolved = {
        name: path if path.is_absolute() else root / path for name, path in phase_inputs.items()
    }
    gates = {name: _read_json(path) for name, path in resolved.items()}
    route_rows = build_route_rows(gates=gates, input_paths=resolved, root=root)
    claim_rows = build_claim_boundary_rows(root=root, input_paths=resolved)
    gate = build_gate(gates=gates, route_rows=route_rows, claim_rows=claim_rows)
    decision_rows = build_decision_rows(gate=gate, root=root, output_dir=output_dir)

    route_path = output_dir / "phase176_hidden_closure_route_evidence_table.csv"
    claim_path = output_dir / "phase176_claim_boundary_refresh_table.csv"
    decision_path = output_dir / "phase176_next_decision_table.csv"
    gate_path = output_dir / "phase176_hidden_closure_evidence_refresh_gate.json"
    markdown_path = output_dir / "phase176_hidden_closure_evidence_refresh.md"
    manifest_path = output_dir / "phase176_hidden_closure_evidence_refresh_manifest.json"

    _write_csv(route_path, route_rows, ROUTE_FIELDS)
    _write_csv(claim_path, claim_rows, CLAIM_FIELDS)
    _write_csv(decision_path, decision_rows, DECISION_FIELDS)
    _write_json(gate_path, gate)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        build_markdown(
            gate=gate,
            route_rows=route_rows,
            claim_rows=claim_rows,
            decision_rows=decision_rows,
        ),
        encoding="utf-8",
    )

    manifest = {
        "phase": 176,
        "objective": "hidden_closure_evidence_refresh_after_phase175_low_capacity_closure",
        "inputs": {name: _display_path(path, root) for name, path in sorted(resolved.items())},
        "outputs": {
            "route_evidence_table": _display_path(route_path, root),
            "claim_boundary_table": _display_path(claim_path, root),
            "next_decision_table": _display_path(decision_path, root),
            "gate_json": _display_path(gate_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": {
            "route_evidence_rows": len(route_rows),
            "claim_boundary_rows": len(claim_rows),
            "next_decision_rows": len(decision_rows),
            "blocked_success_claim_rows": gate["blocked_success_claim_rows"],
            "model_training_allowed_route_rows": sum(
                1 for row in route_rows if not _is_false(row["model_training_allowed"])
            ),
            "a100_training_allowed_route_rows": sum(
                1 for row in route_rows if not _is_false(row["a100_training_allowed_now"])
            ),
            "a100_80gb_allowed_route_rows": sum(
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
    manifest = build_package(
        root=args.root,
        output_dir=args.output_dir,
        phase_inputs=phase_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
