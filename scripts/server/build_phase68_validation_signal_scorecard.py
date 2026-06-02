#!/usr/bin/env python3
"""Build the Phase 68 validation-visible signal mining scorecard.

Phase 68 is a gate package, not a training run. It converts the Phase 59-61
evidence boundary into machine-readable decisions about which model innovation
branches can be reopened, which require new data, and when a larger 80GB A100
server should be requested.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SCORECARD_FIELDS = (
    "candidate_id",
    "candidate_family",
    "status",
    "decision",
    "validation_visible_signal",
    "broad12_signal",
    "broad21_signal",
    "entry_evidence",
    "blocking_evidence",
    "required_first_action",
    "a100_40gb_action",
    "a100_80gb_trigger",
    "seed7_gate",
    "seed_expansion_gate",
    "manuscript_use",
    "evidence_locator",
)

ACTION_FIELDS = (
    "priority",
    "action_id",
    "action_type",
    "description",
    "inputs",
    "expected_output",
    "requires_a100",
    "requires_a100_80gb",
    "exit_gate",
    "status",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Expected at least one row in {path}")
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _display_path(path: Path, root: Path | None = None) -> str:
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def _default_paths(root: Path) -> dict[str, Path]:
    phase60 = root / "docs/results/phase60_manuscript_evidence_package"
    phase61 = root / "docs/results/phase61_manuscript_draft_package"
    phase59 = root / "docs/results/phase59_residual_anatomy"
    return {
        "phase60_manifest": phase60 / "phase60_manuscript_evidence_package_manifest.json",
        "phase60_next_gate": phase60 / "phase60_next_branch_gate_table.csv",
        "phase60_stress": phase60 / "phase60_stress_boundary_table.csv",
        "phase60_appendix": phase60 / "phase60_appendix_negative_diagnostic_table.csv",
        "phase61_crosswalk": phase61 / "phase61_claim_evidence_crosswalk.csv",
        "phase61_manifest": phase61 / "phase61_manuscript_draft_package_manifest.json",
        "phase59_upper": phase59 / "phase59_broad21_density_residual_upper_bound.json",
    }


def _rows_where(rows: list[dict[str, str]], key: str, value: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get(key) == value]


def _count_status(rows: list[dict[str, str]], status: str) -> int:
    return sum(1 for row in rows if row.get("status") == status)


def _gate_row(next_gate_rows: list[dict[str, str]], prefix: str) -> dict[str, str]:
    for row in next_gate_rows:
        if (row.get("branch") or "").startswith(prefix):
            return row
    raise KeyError(f"Missing next-branch gate row starting with {prefix!r}")


def _candidate(
    candidate_id: str,
    candidate_family: str,
    status: str,
    decision: str,
    validation_visible_signal: str,
    broad12_signal: str,
    broad21_signal: str,
    entry_evidence: str,
    blocking_evidence: str,
    required_first_action: str,
    a100_40gb_action: str,
    a100_80gb_trigger: str,
    seed7_gate: str,
    seed_expansion_gate: str,
    manuscript_use: str,
    evidence_locator: str,
) -> dict[str, str]:
    return {
        "candidate_id": candidate_id,
        "candidate_family": candidate_family,
        "status": status,
        "decision": decision,
        "validation_visible_signal": validation_visible_signal,
        "broad12_signal": broad12_signal,
        "broad21_signal": broad21_signal,
        "entry_evidence": entry_evidence,
        "blocking_evidence": blocking_evidence,
        "required_first_action": required_first_action,
        "a100_40gb_action": a100_40gb_action,
        "a100_80gb_trigger": a100_80gb_trigger,
        "seed7_gate": seed7_gate,
        "seed_expansion_gate": seed_expansion_gate,
        "manuscript_use": manuscript_use,
        "evidence_locator": evidence_locator,
    }


def build_candidate_scorecard(
    paths: dict[str, Path],
    phase60_manifest: dict[str, Any],
    phase61_manifest: dict[str, Any],
    next_gate_rows: list[dict[str, str]],
    stress_rows: list[dict[str, str]],
    appendix_rows: list[dict[str, str]],
    crosswalk_rows: list[dict[str, str]],
    upper: dict[str, Any],
    root: Path,
) -> list[dict[str, str]]:
    gate = phase60_manifest.get("model_expansion_gate") or {}
    upper_decision = upper.get("decision") or {}
    selected_variant = str(upper_decision.get("selected_variant") or gate.get("selected_variant") or "")
    uses_test_for_selection = bool(upper.get("uses_test_for_selection"))
    selected_beats_reference = bool(upper_decision.get("selected_beats_reference_rmse"))
    boundary_count = _count_status(stress_rows, "boundary")
    pass_count = _count_status(stress_rows, "pass")
    diagnostic_count = len(appendix_rows)
    c61_gate = next(
        (row for row in crosswalk_rows if row.get("claim_anchor_id") == "C61-GATE-001"),
        {},
    )

    common_seed7_gate = (
        "seed 7 must be non-worse than the frozen broad_process_v1 floor on broad12 "
        "and broad21 for Test RMSE, Hot q90 RMSE, and Gradient q90 RMSE"
    )
    common_seed_expansion = (
        "expand to seeds 7/1/2 only after seed 7 passes; expand to five seeds only "
        "after the three-seed aggregate remains positive"
    )
    gate_path = _display_path(paths["phase60_next_gate"], root)
    stress_path = _display_path(paths["phase60_stress"], root)
    appendix_path = _display_path(paths["phase60_appendix"], root)
    crosswalk_path = _display_path(paths["phase61_crosswalk"], root)
    upper_path = _display_path(paths["phase59_upper"], root)

    candidate_a = _gate_row(next_gate_rows, "Candidate A")
    candidate_b = _gate_row(next_gate_rows, "Candidate B")
    candidate_c = _gate_row(next_gate_rows, "Candidate C")
    external = _gate_row(next_gate_rows, "External")

    density_gate_blocks = (
        gate.get("decision") == "block_density_failure_driven_model_expansion"
        and not uses_test_for_selection
        and not selected_beats_reference
    )

    rows = [
        _candidate(
            "A",
            "bounded physical spot-size parameterization",
            "paused_no_training_signal",
            "do_not_train_from_density_failure",
            (
                "missing: Phase 59 selected a validation-visible mean fallback, not a "
                "learnable spot-size correction"
                if density_gate_blocks
                else "review required"
            ),
            "fixed-sampling and alternate-density broad12 support the current floor but do not prove a new parameterization",
            "fixed-sampling broad21 is positive, but alternate-density broad21 is a boundary and upper-bound selection falls back to mean",
            candidate_a.get("entry_condition", ""),
            f"Phase 59 selected {selected_variant}; C61 gate: {c61_gate.get('claim_summary', '')}",
            "build a no-training signal probe for spot_size physics features from train/validation artifacts before changing the model",
            "only after the probe opens Candidate A: run focused broad12/broad21 spot_size seed-7 validation on the current A100-SXM4-40GB",
            "request A100-SXM4-80GB only if the reopened model requires large multi-panel or multi-seed training that exceeds measured 40GB memory",
            common_seed7_gate,
            common_seed_expansion,
            "not manuscript-promotable now; keep Phase 55/60 as the main model floor",
            f"{gate_path}; {stress_path}; {upper_path}; {crosswalk_path}",
        ),
        _candidate(
            "B",
            "validation-auditable route policy",
            "blocked_by_phase59_validation_gate",
            "policy_audit_only",
            "missing: current no-test-leakage policy upper bound selects fallback to mean, not a transferable route improvement",
            "existing fixed-sampling floor is stable; route policy must not weaken spot_size",
            "density failure is validation-visible only as mean fallback, so learned or high-capacity route policy is not justified",
            candidate_b.get("entry_condition", ""),
            f"Phase 59 upper-bound decision uses_test_for_selection={uses_test_for_selection} and selected_beats_reference_rmse={selected_beats_reference}",
            "run a non-trainable validation-only policy audit among existing comparable routes before any trainable policy",
            "no A100 training until the non-trainable policy preserves spot_size and improves at least one route-boundary axis from validation evidence",
            "request A100-SXM4-80GB only for a later high-capacity policy after low-capacity policy evidence passes on 40GB",
            common_seed7_gate,
            common_seed_expansion,
            "can be discussed only as future work unless the policy passes the frozen-floor gate",
            f"{gate_path}; {upper_path}; {crosswalk_path}",
        ),
        _candidate(
            "C",
            "data-aligned heat-kernel or Green's-function features",
            "blocked_by_registration_data",
            "data_audit_before_training",
            "missing: current AM-Bench single-track target lacks aligned source-path registration for paper-facing features",
            "no direct broad12 signal; prior heat-kernel/source-path branch is diagnostic only under current registration",
            "no direct broad21 signal; source-path broad12/broad21 validation remains blocked by registration",
            candidate_c.get("entry_condition", ""),
            "Phase 52/53 registration blocker plus Phase 60 next-branch gate",
            "inventory or add aligned scan-path/pad-thermography data and pass coordinate/unit/coverage checks before model changes",
            "A100-SXM4-40GB is enough for feature audits and first aligned-target gate",
            "request A100-SXM4-80GB only if aligned dense pad or multi-target feature training exceeds measured 40GB memory",
            common_seed7_gate,
            common_seed_expansion,
            "not allowed as a current broad12/broad21 source-path claim",
            f"{gate_path}; {appendix_path}; {crosswalk_path}",
        ),
        _candidate(
            "D",
            "larger model architecture branch: Bayesian PINN, attention, GCN/CNN, or meta-learning",
            "deferred_requires_local_identifiability_gate",
            "synthetic_or_local_gate_first",
            "missing: current appendix records diagnostic negatives and no broad12/broad21 validation-visible signal for a larger architecture",
            "no broad12 training should start until a small local/synthetic gate preserves all three metrics",
            "no broad21 training should start until a small local/synthetic gate preserves all three metrics",
            "user allows additional model innovations, but the current evidence requires a local gate before A100 expansion",
            f"Phase 60 appendix contains {diagnostic_count} diagnostic or boundary rows",
            "define a small identifiability or region-preserving local gate for the architecture and compare against deterministic controls",
            "run on A100-SXM4-40GB only after local/synthetic gate passes; start with seed-7 focused validation",
            "request A100-SXM4-80GB before launching learned image encoders, large GCN/CNN backbones, large ensembles, or multi-dataset training expected to exceed 40GB",
            common_seed7_gate,
            common_seed_expansion,
            "future-work or second-branch only until it beats the frozen floor",
            f"{appendix_path}; {crosswalk_path}",
        ),
        _candidate(
            "E",
            "external robustness or second dataset branch",
            "open_for_data_planning_only",
            "prepare_data_card_and_split_manifest",
            "available as a planning branch after Phase 61, but not a substitute for current AM-Bench claim gates",
            "must define its own broad12-like baseline and split evidence",
            "must define its own broad21-like transfer or stress evidence",
            external.get("entry_condition", ""),
            "no external dataset package has a frozen floor yet",
            "prepare a data card, download/verification route, split manifest, baseline table, and local feasibility gate",
            "A100-SXM4-40GB is enough for first data audit, baseline, and small feasibility runs",
            "request A100-SXM4-80GB if the chosen external branch needs dense multi-process or image-backbone training beyond measured 40GB memory",
            "define a new seed-7 gate after the dataset has a frozen baseline floor",
            "define seed expansion only after the new dataset has a three-metric gate",
            "external robustness only unless it independently passes strong-baseline and route-guard gates",
            f"{gate_path}; {crosswalk_path}",
        ),
    ]

    # Keep the package self-auditing: if Phase 60/61 evidence changes, the summary
    # rows expose it without requiring a reader to open every upstream artifact.
    rows.append(
        _candidate(
            "SUMMARY",
            "phase68 evidence summary",
            "summary",
            "no_trainable_model_opened_by_current_evidence",
            f"stress pass rows={pass_count}; boundary rows={boundary_count}; phase61 gate={phase61_manifest.get('writing_stage_gate', {}).get('active_gate', '')}",
            "broad12 fixed-sampling and density stress are positive",
            "broad21 fixed-sampling is positive but density stress is a boundary",
            "Phase 60/61 packages are writing-ready for internal Results/Methods",
            f"model expansion gate={gate.get('decision')}",
            "complete Phase 68 scorecard, then run the first non-training signal probe or manuscript v0 audit",
            "no immediate training job",
            "no immediate 80GB request",
            common_seed7_gate,
            common_seed_expansion,
            "summarizes why the current manuscript remains Phase 55/60-first",
            f"{stress_path}; {crosswalk_path}",
        )
    )
    return rows


def build_action_queue(scorecard_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "priority": "1",
            "action_id": "P68-AUDIT",
            "action_type": "package_closeout",
            "description": "Commit and sync the Phase 68 scorecard after local/server verification.",
            "inputs": "Phase 59/60/61 artifacts",
            "expected_output": "phase68 scorecard manifest reproducible locally and on A100",
            "requires_a100": "no",
            "requires_a100_80gb": "no",
            "exit_gate": "local/GitHub/server same commit and manifest counts match",
            "status": "planned",
        },
        {
            "priority": "2",
            "action_id": "P68-SPOT-SIGNAL",
            "action_type": "non_training_signal_probe",
            "description": "Mine train/validation artifacts for a bounded spot_size physics signal before Candidate A training.",
            "inputs": "broad12/broad21 fixed-sampling spot_size artifacts plus density stress summaries",
            "expected_output": "candidate A status is either opened_for_seed7 or closed_as_no_signal",
            "requires_a100": "no",
            "requires_a100_80gb": "no",
            "exit_gate": "signal appears on both broad12 and broad21 without relying on test labels",
            "status": "planned",
        },
        {
            "priority": "3",
            "action_id": "P68-ROUTE-POLICY",
            "action_type": "non_training_policy_audit",
            "description": "Audit whether validation-only route choice among existing comparable routes can improve boundary axes without weakening spot_size.",
            "inputs": "Phase 30-60 route summaries and Phase 59 upper-bound report",
            "expected_output": "candidate B remains blocked or opens only a low-capacity route policy gate",
            "requires_a100": "no",
            "requires_a100_80gb": "no",
            "exit_gate": "validation policy preserves broad12/broad21 spot_size and improves at least one boundary axis",
            "status": "planned",
        },
        {
            "priority": "4",
            "action_id": "P68-DATA-REGISTRATION",
            "action_type": "data_audit",
            "description": "Search local/server data for aligned scan-path or pad-thermography targets before reopening heat-kernel or Green's-function features.",
            "inputs": "AM-Bench thermography, scan strategy, and available metadata",
            "expected_output": "candidate C remains blocked or receives a coordinate-compatible target",
            "requires_a100": "no",
            "requires_a100_80gb": "no",
            "exit_gate": "coordinate units, coverage, and target/source registration are compatible",
            "status": "planned",
        },
        {
            "priority": "5",
            "action_id": "P68-80GB-TRIGGER",
            "action_type": "resource_gate",
            "description": "Request a new A100-SXM4-80GB server only after a planned training branch has measured or clearly projected 40GB memory overflow.",
            "inputs": "nvidia-smi memory logs, batch/table size, model architecture memory estimate",
            "expected_output": "explicit user request for 80GB server only when justified",
            "requires_a100": "yes",
            "requires_a100_80gb": "conditional",
            "exit_gate": "current A100-SXM4-40GB cannot run the validated branch safely",
            "status": "standing_gate",
        },
    ]


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(key, "")).replace("\n", " ") for key, _ in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def build_markdown(scorecard_rows: list[dict[str, str]], actions: list[dict[str, str]]) -> str:
    active_training = [
        row
        for row in scorecard_rows
        if row["candidate_id"] != "SUMMARY" and "opened" in row["status"]
    ]
    lines = [
        "# Phase 68 Validation-Visible Signal Mining Scorecard",
        "",
        "## Purpose",
        "",
        "Phase 68 translates the Phase 59-61 evidence boundary into explicit next-step decisions for model innovation. It does not add training evidence.",
        "",
        "## Current Decision",
        "",
        f"Trainable model branches opened by current evidence: `{len(active_training)}`.",
        "No Candidate A/B/C training should start from the Phase 58/59 density failure alone. New model work must first pass a train/validation-visible signal probe.",
        "",
        "## Candidate Scorecard",
        "",
        _markdown_table(
            [row for row in scorecard_rows if row["candidate_id"] != "SUMMARY"],
            [
                ("candidate_id", "ID"),
                ("candidate_family", "Candidate"),
                ("status", "Status"),
                ("decision", "Decision"),
                ("required_first_action", "First action"),
                ("a100_80gb_trigger", "80GB trigger"),
            ],
        ),
        "",
        "## Action Queue",
        "",
        _markdown_table(
            actions,
            [
                ("priority", "Priority"),
                ("action_id", "Action"),
                ("action_type", "Type"),
                ("description", "Description"),
                ("requires_a100", "A100"),
                ("requires_a100_80gb", "80GB"),
                ("exit_gate", "Exit gate"),
            ],
        ),
        "",
        "## Interpretation",
        "",
        "The immediate path is manuscript-first plus non-training signal mining. Candidate A/B/C can be reopened, but only after a validation-visible signal appears on both broad12 and broad21 or after new registered data removes a data blocker. Larger model architectures and new datasets are allowed, but they must start from small identifiability, data-card, split-manifest, and baseline gates. A100-SXM4-80GB should be requested only when a validated branch demonstrably cannot run on the current 40GB server.",
        "",
    ]
    return "\n".join(lines)


def build_package(
    root: Path,
    output_dir: Path,
    paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    resolved = _default_paths(root)
    if paths:
        resolved.update(paths)

    phase60_manifest = _read_json(resolved["phase60_manifest"])
    phase61_manifest = _read_json(resolved["phase61_manifest"])
    next_gate_rows = _read_csv(resolved["phase60_next_gate"])
    stress_rows = _read_csv(resolved["phase60_stress"])
    appendix_rows = _read_csv(resolved["phase60_appendix"])
    crosswalk_rows = _read_csv(resolved["phase61_crosswalk"])
    upper = _read_json(resolved["phase59_upper"])

    output_dir.mkdir(parents=True, exist_ok=True)
    scorecard_path = output_dir / "phase68_candidate_signal_scorecard.csv"
    actions_path = output_dir / "phase68_next_action_queue.csv"
    markdown_path = output_dir / "phase68_validation_signal_scorecard.md"
    manifest_path = output_dir / "phase68_validation_signal_scorecard_manifest.json"

    scorecard_rows = build_candidate_scorecard(
        resolved,
        phase60_manifest,
        phase61_manifest,
        next_gate_rows,
        stress_rows,
        appendix_rows,
        crosswalk_rows,
        upper,
        root,
    )
    action_rows = build_action_queue(scorecard_rows)

    _write_csv(scorecard_path, scorecard_rows, SCORECARD_FIELDS)
    _write_csv(actions_path, action_rows, ACTION_FIELDS)
    markdown_path.write_text(build_markdown(scorecard_rows, action_rows), encoding="utf-8")

    counts = {
        "candidate_rows": len(scorecard_rows),
        "action_rows": len(action_rows),
        "opened_trainable_candidates": sum(1 for row in scorecard_rows if "opened" in row["status"]),
        "blocked_or_paused_candidates": sum(
            1
            for row in scorecard_rows
            if row["status"].startswith(("blocked", "paused", "deferred"))
        ),
    }
    manifest = {
        "phase": 68,
        "objective": "validation_visible_signal_mining_scorecard",
        "inputs": {key: _display_path(path, root) for key, path in sorted(resolved.items())},
        "outputs": {
            "candidate_signal_scorecard": _display_path(scorecard_path, root),
            "next_action_queue": _display_path(actions_path, root),
            "markdown": _display_path(markdown_path, root),
            "manifest": _display_path(manifest_path, root),
        },
        "counts": counts,
        "current_decision": {
            "trainable_model_opened": counts["opened_trainable_candidates"] > 0,
            "next_step": "non_training_signal_probe_or_manuscript_v0_claim_audit",
            "a100_80gb_request_now": False,
            "reason": "current evidence opens no trainable branch directly and current 40GB A100 is sufficient for package and signal-mining work",
        },
        "inherited_model_expansion_gate": phase60_manifest.get("model_expansion_gate"),
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/phase68_validation_signal_scorecard"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    manifest = build_package(root=root, output_dir=output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
