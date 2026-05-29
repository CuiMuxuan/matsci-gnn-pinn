"""Audit public-dataset configuration and local file readiness."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModalityAudit:
    name: str
    required: bool
    patterns: list[str]
    matched_files: list[str]

    @property
    def present(self) -> bool:
        return bool(self.matched_files)


@dataclass(frozen=True)
class DatasetAudit:
    dataset_id: str
    config_path: str
    local_root: str
    local_root_exists: bool
    file_count: int
    modalities: list[ModalityAudit]
    readiness: str
    generated_at: str
    source_pages: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "config_path": self.config_path,
            "local_root": self.local_root,
            "local_root_exists": self.local_root_exists,
            "file_count": self.file_count,
            "modalities": [
                {
                    "name": modality.name,
                    "required": modality.required,
                    "patterns": modality.patterns,
                    "present": modality.present,
                    "matched_files": modality.matched_files,
                }
                for modality in self.modalities
            ],
            "readiness": self.readiness,
            "generated_at": self.generated_at,
            "source_pages": self.source_pages,
            "notes": self.notes,
        }


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    if "dataset_id" not in data:
        raise ValueError(f"Config missing required key 'dataset_id': {config_path}")
    if "local_root" not in data:
        raise ValueError(f"Config missing required key 'local_root': {config_path}")
    return data


def _resolve_path(path_text: str, project_root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return project_root / path


def _relative(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def audit_dataset(config_path: Path, project_root: Path | None = None) -> DatasetAudit:
    project_root = (project_root or Path.cwd()).resolve()
    config_path = config_path.resolve()
    config = load_config(config_path)

    local_root = _resolve_path(str(config["local_root"]), project_root).resolve()
    local_root_exists = local_root.exists()
    all_files = [path for path in local_root.rglob("*") if path.is_file()] if local_root_exists else []

    modalities: list[ModalityAudit] = []
    for name, spec in (config.get("expected_modalities") or {}).items():
        if not isinstance(spec, dict):
            raise ValueError(f"Modality spec must be a mapping: {name}")
        patterns = [str(pattern) for pattern in spec.get("patterns", [])]
        matched: list[Path] = []
        if local_root_exists:
            for pattern in patterns:
                matched.extend(path for path in local_root.glob(pattern) if path.is_file())
        matched_unique = sorted({_relative(path.resolve(), project_root) for path in matched})
        modalities.append(
            ModalityAudit(
                name=name,
                required=bool(spec.get("required", False)),
                patterns=patterns,
                matched_files=matched_unique,
            )
        )

    readiness = determine_readiness(
        file_count=len(all_files),
        modalities=modalities,
        phase0_gates=config.get("phase0_gates") or {},
    )

    return DatasetAudit(
        dataset_id=str(config["dataset_id"]),
        config_path=_relative(config_path, project_root),
        local_root=_relative(local_root, project_root),
        local_root_exists=local_root_exists,
        file_count=len(all_files),
        modalities=modalities,
        readiness=readiness,
        generated_at=datetime.now(timezone.utc).isoformat(),
        source_pages=[str(url) for url in config.get("source_pages", [])],
        notes=[str(note) for note in config.get("notes", [])],
    )


def determine_readiness(
    file_count: int,
    modalities: list[ModalityAudit],
    phase0_gates: dict[str, Any],
) -> str:
    require_local_data = bool(phase0_gates.get("require_local_data", False))
    min_existing_files = int(phase0_gates.get("min_existing_files", 1))
    min_modalities_present = int(phase0_gates.get("min_modalities_present", 1))

    missing_required = [modality.name for modality in modalities if modality.required and not modality.present]
    present_modalities = sum(1 for modality in modalities if modality.present)

    if missing_required:
        return "blocked_by_missing_required_modalities"
    if file_count < min_existing_files:
        return "blocked_by_missing_local_files" if require_local_data else "source_registered_no_local_files"
    if present_modalities < min_modalities_present:
        return "blocked_by_unmatched_modalities"
    return "ready_for_phase1"


def write_reports(audit: DatasetAudit, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{audit.dataset_id}_audit.json"
    md_path = output_dir / f"{audit.dataset_id}_audit.md"

    audit_dict = audit.to_dict()
    json_path.write_text(json.dumps(audit_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(audit), encoding="utf-8")
    return json_path, md_path


def render_markdown(audit: DatasetAudit) -> str:
    modality_rows = [
        "| Modality | Required | Present | Matched files |",
        "|---|---:|---:|---:|",
    ]
    for modality in audit.modalities:
        modality_rows.append(
            f"| {modality.name} | {modality.required} | {modality.present} | {len(modality.matched_files)} |"
        )

    source_lines = "\n".join(f"- {url}" for url in audit.source_pages) or "- Not specified"
    notes = "\n".join(f"- {note}" for note in audit.notes) or "- None"

    return "\n".join(
        [
            f"# Dataset Audit: {audit.dataset_id}",
            "",
            f"- Generated at: `{audit.generated_at}`",
            f"- Config: `{audit.config_path}`",
            f"- Local root: `{audit.local_root}`",
            f"- Local root exists: `{audit.local_root_exists}`",
            f"- File count: `{audit.file_count}`",
            f"- Readiness: `{audit.readiness}`",
            "",
            "## Modalities",
            "",
            *modality_rows,
            "",
            "## Sources",
            "",
            source_lines,
            "",
            "## Notes",
            "",
            notes,
            "",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to dataset YAML config.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/data_audits"),
        help="Directory for JSON and Markdown audit reports.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    audit = audit_dataset(args.config)
    json_path, md_path = write_reports(audit, args.output_dir)
    print(f"Readiness: {audit.readiness}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

