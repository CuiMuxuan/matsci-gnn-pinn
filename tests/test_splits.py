from pathlib import Path

from gnnpinn.data.splits import load_split_manifest, split_indices, subset_sequence


def test_split_manifest_helpers(tmp_path: Path):
    path = tmp_path / "split.json"
    path.write_text('{"splits":{"train":[2,0],"test":[1]}}', encoding="utf-8")

    manifest = load_split_manifest(path)

    assert split_indices(manifest, "train") == [2, 0]
    assert subset_sequence(["a", "b", "c"], [2, 0]) == ["c", "a"]
