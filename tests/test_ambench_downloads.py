from pathlib import Path
import urllib.error

import pytest
import yaml

from gnnpinn.data.ambench_downloads import (
    download_mds2_2716,
    download_mds2_2718,
    download_source_manifest,
    load_mds2_2716_sources,
    load_mds2_2718_sources,
    main,
    validate_mds2_2716,
    validate_mds2_2718,
)


def test_validate_mds2_2716_reports_missing(tmp_path: Path):
    report = validate_mds2_2716(tmp_path / "missing")

    assert report["root_exists"] is False
    assert report["ready_for_hdf5_adapter"] is False
    assert "readme" in report["missing_required"]
    assert report["suggested_downloads"]
    assert "download_url" in report["suggested_downloads"][0]


def test_validate_mds2_2716_ready(tmp_path: Path):
    root = tmp_path / "mds2-2716"
    manifest = tmp_path / "sources.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "dataset_id": "mds2-2716",
                "required_files": [
                        {
                            "id": "readme",
                            "relative_path": "2716_README.txt",
                            "size_bytes": 6,
                            "sha256": "711a6108ba2ce6ca93dd47d6817f2361db10d8ab6eec89460b2dfc2c325efabe",
                            "download_url": "https://example.test/2716_README.txt",
                        },
                        {
                            "id": "thermography_signal_hdf5",
                            "relative_path": "Thermography/toy.h5",
                            "size_bytes": 4,
                            "sha256": "3a6eb0790f39ac87c94f3856b2dd2c5d110e6811602261a9a923d3bb23adc8b7",
                            "download_url": "https://example.test/toy.h5",
                        },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (root / "Thermography").mkdir(parents=True)
    (root / "2716_README.txt").write_text("readme", encoding="utf-8")
    (root / "Thermography" / "toy.h5").write_text("data", encoding="utf-8")

    report = validate_mds2_2716(root, source_manifest=manifest, verify_hashes=True)

    assert report["root_exists"] is True
    assert report["ready_for_hdf5_adapter"] is True
    assert report["checks"]["hdf5_files"]["count"] == 1
    assert report["checks"]["readme"]["sha256_ok"] is True


def test_load_mds2_2716_sources_default_manifest():
    sources = load_mds2_2716_sources()

    file_ids = {entry["id"] for entry in sources["required_files"]}
    assert "thermography_signal_hdf5" in file_ids
    assert "scan_strategy_hdf5" in file_ids


def test_download_mds2_2716_from_manifest_file_urls(tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    readme_source = source_dir / "2716_README.txt"
    hdf5_source = source_dir / "toy.h5"
    readme_source.write_text("readme", encoding="utf-8")
    hdf5_source.write_text("data", encoding="utf-8")

    manifest = tmp_path / "sources.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "dataset_id": "mds2-2716",
                "required_files": [
                    {
                        "id": "readme",
                        "relative_path": "2716_README.txt",
                        "size_bytes": 6,
                        "sha256": "711a6108ba2ce6ca93dd47d6817f2361db10d8ab6eec89460b2dfc2c325efabe",
                        "download_url": readme_source.as_uri(),
                    },
                    {
                        "id": "thermography_signal_hdf5",
                        "relative_path": "Thermography/toy.h5",
                        "size_bytes": 4,
                        "sha256": "3a6eb0790f39ac87c94f3856b2dd2c5d110e6811602261a9a923d3bb23adc8b7",
                        "download_url": hdf5_source.as_uri(),
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    root = tmp_path / "downloaded"
    report = download_mds2_2716(root, source_manifest=manifest, verify_hashes=True)

    assert (root / "2716_README.txt").read_text(encoding="utf-8") == "readme"
    assert (root / "Thermography" / "toy.h5").read_text(encoding="utf-8") == "data"
    assert report["validation"]["ready_for_hdf5_adapter"] is True
    assert {action["status"] for action in report["actions"]} == {"downloaded"}


def test_download_mds2_2716_dry_run_does_not_write(tmp_path: Path):
    source = tmp_path / "source.txt"
    source.write_text("readme", encoding="utf-8")
    manifest = tmp_path / "sources.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "dataset_id": "mds2-2716",
                "required_files": [
                    {
                        "id": "readme",
                        "relative_path": "2716_README.txt",
                        "size_bytes": 6,
                        "sha256": "711a6108ba2ce6ca93dd47d6817f2361db10d8ab6eec89460b2dfc2c325efabe",
                        "download_url": source.as_uri(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    root = tmp_path / "downloaded"
    report = download_mds2_2716(root, source_manifest=manifest, dry_run=True)

    assert not (root / "2716_README.txt").exists()
    assert report["actions"][0]["status"] == "planned"
    assert report["validation_failed"] is True


def test_download_mds2_2716_marks_size_mismatch_after_download(tmp_path: Path):
    source = tmp_path / "source.txt"
    source.write_text("short", encoding="utf-8")
    manifest = tmp_path / "sources.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "dataset_id": "mds2-2716",
                "required_files": [
                    {
                        "id": "readme",
                        "relative_path": "2716_README.txt",
                        "size_bytes": 6,
                        "sha256": "711a6108ba2ce6ca93dd47d6817f2361db10d8ab6eec89460b2dfc2c325efabe",
                        "download_url": source.as_uri(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    root = tmp_path / "downloaded"
    report = download_mds2_2716(root, source_manifest=manifest, verify_hashes=True)

    assert report["actions"][0]["status"] == "downloaded"
    assert report["validation_failed"] is True
    assert report["validation"]["mismatched_required"] == ["readme"]


def test_load_mds2_2718_sources_default_manifest():
    sources = load_mds2_2718_sources()

    file_ids = {entry["id"] for entry in sources["required_files"]}
    optional_file_ids = {entry["id"] for entry in sources["optional_files"]}
    assert "melt_pool_measurements_xlsx" in file_ids
    assert "single_track_cross_section_representative_tif" in file_ids
    assert "single_track_cross_section_p4_l0_r2_masked_tif" in optional_file_ids
    assert "single_track_cross_section_p2_l2_1_r3_unmasked_tif" in optional_file_ids


def test_validate_mds2_2718_ready_for_microstructure_adapter(tmp_path: Path):
    root = tmp_path / "mds2-2718"
    manifest = tmp_path / "sources.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "dataset_id": "mds2-2718",
                "dataset_name": "AMB2022-03 optical microscopy",
                "required_files": [
                    {
                        "id": "readme",
                        "relative_path": "2718_README.txt",
                        "size_bytes": 6,
                        "sha256": "711a6108ba2ce6ca93dd47d6817f2361db10d8ab6eec89460b2dfc2c325efabe",
                        "download_url": "https://example.test/2718_README.txt",
                    },
                    {
                        "id": "micro_tif",
                        "relative_path": "Single_Track_Cross_Sections/toy.tif",
                        "size_bytes": 4,
                        "sha256": "3a6eb0790f39ac87c94f3856b2dd2c5d110e6811602261a9a923d3bb23adc8b7",
                        "download_url": "https://example.test/toy.tif",
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (root / "Single_Track_Cross_Sections").mkdir(parents=True)
    (root / "2718_README.txt").write_text("readme", encoding="utf-8")
    (root / "Single_Track_Cross_Sections" / "toy.tif").write_text("data", encoding="utf-8")

    report = validate_mds2_2718(root, source_manifest=manifest, verify_hashes=True)

    assert report["ready_for_microstructure_adapter"] is True
    assert report["checks"]["tiff_files"]["count"] == 1
    assert report["checks"]["readme"]["sha256_ok"] is True


def test_download_mds2_2718_from_manifest_file_urls(tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    readme_source = source_dir / "2718_README.txt"
    tif_source = source_dir / "toy.tif"
    readme_source.write_text("readme", encoding="utf-8")
    tif_source.write_text("data", encoding="utf-8")

    manifest = tmp_path / "sources.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "dataset_id": "mds2-2718",
                "dataset_name": "AMB2022-03 optical microscopy",
                "required_files": [
                    {
                        "id": "readme",
                        "relative_path": "2718_README.txt",
                        "size_bytes": 6,
                        "sha256": "711a6108ba2ce6ca93dd47d6817f2361db10d8ab6eec89460b2dfc2c325efabe",
                        "download_url": readme_source.as_uri(),
                    },
                    {
                        "id": "micro_tif",
                        "relative_path": "Single_Track_Cross_Sections/toy.tif",
                        "size_bytes": 4,
                        "sha256": "3a6eb0790f39ac87c94f3856b2dd2c5d110e6811602261a9a923d3bb23adc8b7",
                        "download_url": tif_source.as_uri(),
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    root = tmp_path / "downloaded"
    report = download_mds2_2718(root, source_manifest=manifest, verify_hashes=True)

    assert (root / "2718_README.txt").read_text(encoding="utf-8") == "readme"
    assert (root / "Single_Track_Cross_Sections" / "toy.tif").read_text(encoding="utf-8") == "data"
    assert report["validation"]["ready"] is True
    assert report["validation"]["checks"]["tiff_files"]["count"] == 1


def test_download_mds2_2718_file_id_can_select_optional_file(tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    readme_source = source_dir / "2718_README.txt"
    optional_tif_source = source_dir / "optional.tif"
    readme_source.write_text("readme", encoding="utf-8")
    optional_tif_source.write_text("data", encoding="utf-8")

    manifest = tmp_path / "sources.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "dataset_id": "mds2-2718",
                "required_files": [
                    {
                        "id": "readme",
                        "relative_path": "2718_README.txt",
                        "size_bytes": 6,
                        "download_url": readme_source.as_uri(),
                    },
                ],
                "optional_files": [
                    {
                        "id": "optional_tif",
                        "relative_path": "Single_Track_Cross_Sections/optional.tif",
                        "size_bytes": 4,
                        "sha256": "3a6eb0790f39ac87c94f3856b2dd2c5d110e6811602261a9a923d3bb23adc8b7",
                        "download_url": optional_tif_source.as_uri(),
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    root = tmp_path / "downloaded"
    report = download_mds2_2718(
        root,
        source_manifest=manifest,
        file_ids=["optional_tif"],
        verify_hashes=True,
    )

    assert not (root / "2718_README.txt").exists()
    assert (root / "Single_Track_Cross_Sections" / "optional.tif").read_text(encoding="utf-8") == "data"
    assert [action["id"] for action in report["actions"]] == ["optional_tif"]
    assert report["validation_failed"] is True
    assert report["validation"]["missing_required"] == ["readme"]
    assert report["mismatched_selected"] == []
    assert report["selected_checks"]["optional_tif"]["sha256_ok"] is True


def test_download_source_manifest_include_optional_downloads_panel(tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    readme_source = source_dir / "2718_README.txt"
    optional_tif_source = source_dir / "optional.tif"
    readme_source.write_text("readme", encoding="utf-8")
    optional_tif_source.write_text("data", encoding="utf-8")

    manifest = tmp_path / "sources.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "dataset_id": "mds2-2718",
                "required_files": [
                    {
                        "id": "readme",
                        "relative_path": "2718_README.txt",
                        "size_bytes": 6,
                        "sha256": "711a6108ba2ce6ca93dd47d6817f2361db10d8ab6eec89460b2dfc2c325efabe",
                        "download_url": readme_source.as_uri(),
                    },
                ],
                "optional_files": [
                    {
                        "id": "optional_tif",
                        "relative_path": "Single_Track_Cross_Sections/optional.tif",
                        "size_bytes": 4,
                        "sha256": "3a6eb0790f39ac87c94f3856b2dd2c5d110e6811602261a9a923d3bb23adc8b7",
                        "download_url": optional_tif_source.as_uri(),
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    root = tmp_path / "downloaded"
    report = download_mds2_2718(
        root,
        source_manifest=manifest,
        include_optional=True,
        verify_hashes=True,
    )

    assert (root / "2718_README.txt").read_text(encoding="utf-8") == "readme"
    assert (root / "Single_Track_Cross_Sections" / "optional.tif").read_text(encoding="utf-8") == "data"
    assert {action["id"] for action in report["actions"]} == {"readme", "optional_tif"}
    assert report["validation"]["ready"] is True
    assert report["validation_failed"] is False
    assert report["mismatched_selected"] == []


def test_download_source_manifest_records_download_error_and_retries(monkeypatch, tmp_path: Path):
    calls = {"count": 0}

    def flaky_urlopen(request, timeout):
        calls["count"] += 1
        raise urllib.error.URLError("timed out")

    import gnnpinn.data.ambench_downloads as module

    monkeypatch.setattr(module.urllib.request, "urlopen", flaky_urlopen)
    sources = {
        "dataset_id": "mds2-2718",
        "required_files": [
            {
                "id": "readme",
                "relative_path": "2718_README.txt",
                "size_bytes": 6,
                "download_url": "https://example.test/2718_README.txt",
            }
        ],
    }

    report = download_source_manifest(
        tmp_path / "downloaded",
        sources,
        retries=2,
        timeout_seconds=1,
    )

    assert calls["count"] == 3
    assert report["actions"][0]["status"] == "download_error"
    assert len(report["actions"][0]["attempts"]) == 3
    assert report["validation_failed"] is True


def test_download_source_manifest_can_resume_partial_file(monkeypatch, tmp_path: Path):
    class FakeHeaders(dict):
        def get(self, key, default=None):
            return super().get(key, default)

    class FakeResponse:
        status = 206

        def __init__(self, data: bytes):
            self._data = data
            self.headers = FakeHeaders(
                {
                    "Content-Length": str(len(data)),
                    "Content-Range": "bytes 3-5/6",
                }
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, size: int):
            data, self._data = self._data, b""
            return data

    captured_headers = {}

    def resume_urlopen(request, timeout):
        captured_headers.update(dict(request.header_items()))
        return FakeResponse(b"def")

    import gnnpinn.data.ambench_downloads as module

    monkeypatch.setattr(module.urllib.request, "urlopen", resume_urlopen)
    root = tmp_path / "downloaded"
    root.mkdir()
    (root / "toy.txt.part").write_bytes(b"abc")
    sources = {
        "dataset_id": "toy",
        "required_files": [
            {
                "id": "toy",
                "relative_path": "toy.txt",
                "size_bytes": 6,
                "sha256": "bef57ec7f53a6d40beb640a780a639c83bc29ac8a9816f1fc6c5c6dcd93c4721",
                "download_url": "https://example.test/toy.txt",
            }
        ],
    }

    report = download_source_manifest(
        root,
        sources,
        verify_hashes=True,
        resume_partial=True,
    )

    assert captured_headers["Range"] == "bytes=3-"
    assert (root / "toy.txt").read_bytes() == b"abcdef"
    assert report["actions"][0]["resume_from"] == 3
    assert report["validation_failed"] is False


def test_download_source_manifest_can_use_curl_backend(monkeypatch, tmp_path: Path):
    import gnnpinn.data.ambench_downloads as module

    def fake_which(name):
        return "curl"

    def fake_run(command, capture_output, text):
        part_path = Path(command[command.index("-o") + 1])
        part_path.write_bytes(b"data")

        class Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        return Completed()

    monkeypatch.setattr(module.shutil, "which", fake_which)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    sources = {
        "dataset_id": "toy",
        "required_files": [
            {
                "id": "toy",
                "relative_path": "toy.txt",
                "size_bytes": 4,
                "sha256": "3a6eb0790f39ac87c94f3856b2dd2c5d110e6811602261a9a923d3bb23adc8b7",
                "download_url": "https://example.test/toy.txt",
            }
        ],
    }

    report = download_source_manifest(
        tmp_path / "downloaded",
        sources,
        verify_hashes=True,
        download_backend="curl",
    )

    assert report["actions"][0]["backend"] == "curl"
    assert report["validation_failed"] is False


def test_download_cli_rejects_unknown_file_id(tmp_path: Path):
    manifest = tmp_path / "sources.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "dataset_id": "mds2-2718",
                "required_files": [
                    {
                        "id": "readme",
                        "relative_path": "2718_README.txt",
                        "size_bytes": 6,
                        "download_url": "file:///readme.txt",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown file id"):
        main(
            [
                "--dataset-id",
                "mds2-2718",
                "--root",
                str(tmp_path / "downloaded"),
                "--source-manifest",
                str(manifest),
                "--download",
                "--file-id",
                "does_not_exist",
            ]
        )


def test_download_cli_dry_run_returns_zero_when_files_missing(tmp_path: Path):
    manifest = tmp_path / "sources.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "dataset_id": "mds2-2718",
                "required_files": [
                    {
                        "id": "missing_tif",
                        "relative_path": "Single_Track_Cross_Sections/missing.tif",
                        "size_bytes": 4,
                        "download_url": "file:///missing.tif",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--dataset-id",
            "mds2-2718",
            "--root",
            str(tmp_path / "downloaded"),
            "--source-manifest",
            str(manifest),
            "--download",
            "--dry-run",
        ]
    )

    assert exit_code == 0
