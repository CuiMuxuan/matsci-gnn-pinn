"""Helpers for connecting audit reports to local data loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_audit_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    data = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Audit report must be a JSON object: {report_path}")
    return data


def candidate_files_from_audit(
    audit_report: dict[str, Any],
    modality: str | None = None,
    suffixes: tuple[str, ...] = (".csv", ".json"),
    project_root: str | Path | None = None,
) -> list[Path]:
    """Return local candidate files recorded by an audit JSON report."""

    root = Path(project_root or ".").resolve()
    candidates: list[Path] = []
    for item in audit_report.get("modalities", []):
        if modality and item.get("name") != modality:
            continue
        for text in item.get("matched_files", []):
            path = Path(text)
            if not path.is_absolute():
                path = root / path
            if path.suffix.lower() in suffixes:
                candidates.append(path)
    return sorted(set(candidates))

