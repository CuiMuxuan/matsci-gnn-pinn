"""Validate and download pinned AM-Bench dataset subsets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import urllib.request

import yaml

DEFAULT_SOURCE_MANIFEST = Path("configs/data/ambench_mds2_2716_sources.yaml")
DEFAULT_MDS2_2718_SOURCE_MANIFEST = Path("configs/data/ambench_mds2_2718_sources.yaml")

FALLBACK_MDS2_2716_SOURCES: dict[str, Any] = {
    "dataset_id": "mds2-2716",
    "dataset_name": "AMB2022-03",
    "record": {
        "doi": "https://doi.org/10.18434/mds2-2716",
        "pdr_landing_page": "https://data.nist.gov/od/id/mds2-2716",
    },
    "required_files": [
        {
            "id": "readme",
            "relative_path": "2716_README.txt",
            "size_bytes": 12573,
            "sha256": "ba44076ed51b69c0e4ca80ff0e2568eed2dc6459e85c9ad83b85860bee5760f2",
            "download_url": "https://data.nist.gov/od/ds/mds2-2716/2716_README.txt",
        },
        {
            "id": "thermography_signal_hdf5",
            "relative_path": "Thermography/AMB2022-03-718-AMMT-StaringCamera_Signal.h5",
            "size_bytes": 549979044,
            "sha256": "f6fe21ec911707f72e7efda2932c77eae2b75d84765848878fe5beb6b728cd43",
            "download_url": (
                "https://data.nist.gov/od/ds/ark:/88434/mds2-2716/"
                "Thermography/AMB2022-03-718-AMMT-StaringCamera_Signal.h5"
            ),
        },
        {
            "id": "scan_strategy_hdf5",
            "relative_path": "ScanStrategy/AMB2022-03-AMMT-718-Pad_XYPT.h5",
            "size_bytes": 406992,
            "sha256": "7b7004753e150bc26632e9ce356e0440429160fa92cbff8fc8559202fdce2103",
            "download_url": (
                "https://data.nist.gov/od/ds/ark:/88434/mds2-2716/"
                "ScanStrategy/AMB2022-03-AMMT-718-Pad_XYPT.h5"
            ),
        },
    ],
}


def load_mds2_2716_sources(source_manifest: str | Path | None = None) -> dict[str, Any]:
    """Load known NIST PDR file sources for AMB2022-03 / mds2-2716."""

    return load_source_manifest(source_manifest or DEFAULT_SOURCE_MANIFEST, fallback=FALLBACK_MDS2_2716_SOURCES)


def load_mds2_2718_sources(source_manifest: str | Path | None = None) -> dict[str, Any]:
    """Load known NIST PDR file sources for AMB2022-03 optical microscopy / mds2-2718."""

    return load_source_manifest(source_manifest or DEFAULT_MDS2_2718_SOURCE_MANIFEST)


def load_source_manifest(source_manifest: str | Path, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a pinned source manifest and annotate it with its path."""

    manifest_path = Path(source_manifest)
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        data["_source_manifest"] = str(manifest_path)
        return data
    if fallback is not None:
        fallback_copy = dict(fallback)
        fallback_copy["_source_manifest"] = None
        return fallback_copy
    raise FileNotFoundError(f"Source manifest not found: {manifest_path}")


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_url(url: str, destination: Path, overwrite: bool = False) -> dict[str, Any]:
    if destination.exists() and not overwrite:
        return {
            "path": str(destination),
            "url": url,
            "status": "skipped_existing",
            "bytes_written": destination.stat().st_size,
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_name(f"{destination.name}.part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "gnn-pinn-data-downloader/0.1"},
    )
    bytes_written = 0
    expected_length = None
    with urllib.request.urlopen(request, timeout=120) as response:
        content_length = response.headers.get("Content-Length")
        expected_length = int(content_length) if content_length and content_length.isdigit() else None
        with part_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                bytes_written += len(chunk)
    if expected_length is not None and bytes_written != expected_length:
        return {
            "path": str(destination),
            "partial_path": str(part_path),
            "url": url,
            "status": "incomplete",
            "bytes_written": bytes_written,
            "expected_content_length": expected_length,
        }
    part_path.replace(destination)
    return {
        "path": str(destination),
        "url": url,
        "status": "downloaded",
        "bytes_written": bytes_written,
        "expected_content_length": expected_length,
    }


def _check_expected_file(root: Path, source: dict[str, Any], verify_hashes: bool) -> dict[str, Any]:
    relative_path = source["relative_path"]
    path = root / relative_path
    expected_size = source.get("size_bytes")
    expected_sha256 = source.get("sha256")
    present = path.exists() and path.is_file()
    actual_size = path.stat().st_size if present else None
    size_ok = actual_size == expected_size if present and expected_size is not None else None
    actual_sha256 = None
    sha256_ok = None
    if present and verify_hashes and expected_sha256:
        actual_sha256 = _sha256_file(path)
        sha256_ok = actual_sha256.lower() == str(expected_sha256).lower()

    issues: list[str] = []
    if not present:
        issues.append("missing")
    if present and size_ok is False:
        issues.append("size_mismatch")
    if present and sha256_ok is False:
        issues.append("sha256_mismatch")

    output_path = str(path)
    parent = path.parent
    mkdir_command = f"New-Item -ItemType Directory -Force -Path \"{parent}\""
    curl_command = f"curl.exe -L \"{source.get('download_url')}\" -o \"{output_path}\""
    return {
        "id": source.get("id", relative_path),
        "relative_path": relative_path,
        "path": output_path,
        "present": present,
        "expected_size_bytes": expected_size,
        "actual_size_bytes": actual_size,
        "size_ok": size_ok,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "sha256_ok": sha256_ok,
        "download_url": source.get("download_url"),
        "purpose": source.get("purpose"),
        "issues": issues,
        "powershell_download": [mkdir_command, curl_command],
    }


def download_mds2_2716(
    root: str | Path,
    source_manifest: str | Path | None = None,
    *,
    file_ids: list[str] | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Download required AMB2022-03 / mds2-2716 files from the source manifest."""

    sources = load_mds2_2716_sources(source_manifest)
    return download_source_manifest(
        root,
        sources,
        file_ids=file_ids,
        overwrite=overwrite,
        dry_run=dry_run,
        verify_hashes=verify_hashes,
    )


def download_mds2_2718(
    root: str | Path,
    source_manifest: str | Path | None = None,
    *,
    file_ids: list[str] | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Download selected AMB2022-03 optical microscopy files from the source manifest."""

    sources = load_mds2_2718_sources(source_manifest)
    return download_source_manifest(
        root,
        sources,
        file_ids=file_ids,
        overwrite=overwrite,
        dry_run=dry_run,
        verify_hashes=verify_hashes,
    )


def download_source_manifest(
    root: str | Path,
    sources: dict[str, Any],
    *,
    file_ids: list[str] | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Download files listed in a loaded AM-Bench source manifest."""

    root = Path(root)
    selected_ids = set(file_ids or [])
    required_sources = [
        source
        for source in sources.get("required_files", [])
        if not selected_ids or source.get("id") in selected_ids
    ]

    actions: list[dict[str, Any]] = []
    for source in required_sources:
        destination = root / source["relative_path"]
        check = _check_expected_file(root, source, verify_hashes=False)
        should_download = overwrite or not check["present"] or "size_mismatch" in check["issues"]
        if not should_download:
            actions.append(
                {
                    "id": source.get("id"),
                    "relative_path": source["relative_path"],
                    "status": "skipped_existing",
                    "path": str(destination),
                    "download_url": source.get("download_url"),
                }
            )
            continue
        if dry_run:
            actions.append(
                {
                    "id": source.get("id"),
                    "relative_path": source["relative_path"],
                    "status": "planned",
                    "path": str(destination),
                    "download_url": source.get("download_url"),
                }
            )
            continue
        result = _download_url(str(source["download_url"]), destination, overwrite=True)
        actions.append(
            {
                "id": source.get("id"),
                "relative_path": source["relative_path"],
                "download_url": source.get("download_url"),
                **result,
            }
        )

    final_report = validate_source_manifest(root, sources, verify_hashes=verify_hashes)
    _add_dataset_ready_aliases(final_report, sources.get("dataset_id"))
    validation_failed = not final_report["ready"]
    return {
        "dataset_id": sources.get("dataset_id"),
        "dataset_name": sources.get("dataset_name"),
        "root": str(root),
        "source_manifest": sources.get("_source_manifest"),
        "dry_run": dry_run,
        "overwrite": overwrite,
        "file_ids": sorted(selected_ids),
        "actions": actions,
        "validation_failed": validation_failed,
        "validation": final_report,
    }


def _add_dataset_ready_aliases(report: dict[str, Any], dataset_id: str | None) -> None:
    if dataset_id == "mds2-2716":
        report["ready_for_hdf5_adapter"] = report["ready"]
        report["next_step"] = (
            "Run HDF5 adapter development against Thermography and ScanStrategy files."
            if report["ready"]
            else "Download missing AM-Bench files from suggested_downloads, then rerun this command."
        )
    elif dataset_id == "mds2-2718":
        report["ready_for_microstructure_adapter"] = report["ready"]
        report["next_step"] = (
            "Run microstructure TIFF inspection/preprocessing on selected optical microscopy files."
            if report["ready"]
            else "Download missing AM-Bench optical microscopy files from suggested_downloads, then rerun this command."
        )


def validate_mds2_2716(
    root: str | Path,
    source_manifest: str | Path | None = None,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    sources = load_mds2_2716_sources(source_manifest)
    report = validate_source_manifest(root, sources, verify_hashes=verify_hashes)
    _add_dataset_ready_aliases(report, sources.get("dataset_id"))
    return report


def validate_mds2_2718(
    root: str | Path,
    source_manifest: str | Path | None = None,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    sources = load_mds2_2718_sources(source_manifest)
    report = validate_source_manifest(root, sources, verify_hashes=verify_hashes)
    _add_dataset_ready_aliases(report, sources.get("dataset_id"))
    return report


def validate_source_manifest(
    root: str | Path,
    sources: dict[str, Any],
    verify_hashes: bool = False,
) -> dict[str, Any]:
    root = Path(root)
    required_sources = sources.get("required_files", [])
    checks = {
        source["id"]: _check_expected_file(root, source, verify_hashes)
        for source in required_sources
    }
    hdf5_files = sorted([*root.rglob("*.h5"), *root.rglob("*.hdf5"), *root.rglob("*.hdf")])
    tiff_files = sorted([*root.rglob("*.tif"), *root.rglob("*.tiff")])
    hdf5_check = {
        "present": bool(hdf5_files),
        "count": len(hdf5_files),
        "examples": [str(path) for path in hdf5_files[:10]],
    }
    tiff_check = {
        "present": bool(tiff_files),
        "count": len(tiff_files),
        "examples": [str(path) for path in tiff_files[:10]],
    }
    missing = [name for name, check in checks.items() if not check["present"]]
    mismatched = [
        name
        for name, check in checks.items()
        if check["present"] and ("size_mismatch" in check["issues"] or "sha256_mismatch" in check["issues"])
    ]
    ready = root.exists() and not missing and not mismatched
    suggested_downloads = [
        {
            "id": check["id"],
            "relative_path": check["relative_path"],
            "download_url": check["download_url"],
            "powershell_download": check["powershell_download"],
        }
        for check in checks.values()
        if not check["present"]
    ]
    return {
        "dataset_id": sources.get("dataset_id", "mds2-2716"),
        "dataset_name": sources.get("dataset_name", "AMB2022-03"),
        "root": str(root),
        "root_exists": root.exists(),
        "source_manifest": sources.get("_source_manifest"),
        "record": sources.get("record", {}),
        "ready": ready,
        "missing_required": missing,
        "mismatched_required": mismatched,
        "checks": {**checks, "hdf5_files": hdf5_check, "tiff_files": tiff_check},
        "suggested_downloads": suggested_downloads,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Root directory for selected AM-Bench files.")
    parser.add_argument(
        "--dataset-id",
        default="mds2-2716",
        choices=["mds2-2716", "mds2-2718"],
        help="Pinned AM-Bench source manifest to use when --source-manifest is not overridden.",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=None,
        help="YAML manifest containing known NIST PDR file URLs and checksums.",
    )
    parser.add_argument(
        "--verify-sha256",
        action="store_true",
        help="Compute SHA256 for present expected files. This can take time for large HDF5 files.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing or size-mismatched required files from the source manifest before reporting.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="When used with --download, re-download selected files even if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="When used with --download, only report planned downloads without writing files.",
    )
    parser.add_argument(
        "--file-id",
        action="append",
        dest="file_ids",
        help="Restrict --download to a required file id. Can be repeated.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_manifest = args.source_manifest or (
        DEFAULT_MDS2_2718_SOURCE_MANIFEST if args.dataset_id == "mds2-2718" else DEFAULT_SOURCE_MANIFEST
    )
    if args.download:
        sources = load_source_manifest(
            source_manifest,
            fallback=FALLBACK_MDS2_2716_SOURCES if args.dataset_id == "mds2-2716" else None,
        )
        report = download_source_manifest(
            args.root,
            sources,
            file_ids=args.file_ids,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            verify_hashes=args.verify_sha256,
        )
    else:
        sources = load_source_manifest(
            source_manifest,
            fallback=FALLBACK_MDS2_2716_SOURCES if args.dataset_id == "mds2-2716" else None,
        )
        report = validate_source_manifest(args.root, sources, verify_hashes=args.verify_sha256)
        _add_dataset_ready_aliases(report, sources.get("dataset_id"))
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote: {args.output}")
    print(text)
    if args.download and report.get("validation_failed"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
