from pathlib import Path

import yaml

from gnnpinn.data.ambench_downloads import (
    download_mds2_2716,
    download_mds2_2718,
    load_mds2_2716_sources,
    load_mds2_2718_sources,
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
    assert "melt_pool_measurements_xlsx" in file_ids
    assert "single_track_cross_section_representative_tif" in file_ids


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
