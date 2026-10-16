"""The download cache, tested offline: any attempt to reach the network fails the test."""

import urllib.request

import pytest

from proteinmotion import download


@pytest.fixture
def offline(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("The cache test tried to download")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setattr(download, "_download", refuse)


def test_cache_directory_follows_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("PROTEINMOTION_CACHE", str(tmp_path / "cache"))
    assert download.cache_directory() == tmp_path / "cache" and (tmp_path / "cache").is_dir()
    assert download.cache_directory(tmp_path / "other") == tmp_path / "other"


def test_cached_alphafold_model_uses_highest_version(offline, tmp_path, monkeypatch):
    monkeypatch.setenv("PROTEINMOTION_CACHE", str(tmp_path))
    # As text, model_v10 sorts before model_v6.
    for name in ("model_v4", "model_v10", "model_v6"):
        (tmp_path / f"AF-P0DP23-F1-{name}.cif").write_text("data_test\n")
    # Another fragment of the same protein is a different entry.
    (tmp_path / "AF-P0DP23-F2-model_v12.cif").write_text("data_test\n")
    expected = tmp_path / "AF-P0DP23-F1-model_v10.cif"
    assert download.fetch_structure("AF-P0DP23-F1") == expected
    assert download.fetch_structure("af-p0dp23-f1", cache=tmp_path) == expected


def test_cached_pdb_entry_is_reused(offline, tmp_path):
    (tmp_path / "1ubq.cif").write_text("data_test\n")
    assert download.fetch_structure("1UBQ", cache=tmp_path) == tmp_path / "1ubq.cif"
