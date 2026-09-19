import sys
from pathlib import Path

import pytest

from proteinmotion.authoring import SKILL_NAME, init_movie, install_skill
from proteinmotion.cli import load_scene, main


def test_starter_builds_outside_checkout_and_keeps_local_data(tmp_path, monkeypatch):
    movie = init_movie(tmp_path / "a movie with spaces")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    scene = load_scene(movie / "film.py", "ProteinMovie", width=640, height=360).build()
    assert 15 < scene.duration < 30
    visible = scene.seek(5)
    assert any(hasattr(p, "topology") and len(p.topology.atoms) > 500 for p in visible)
    assert (movie / "1ubq.cif").stat().st_size > 10000


def test_init_refuses_overwrite_before_creating_other_files(tmp_path):
    movie = tmp_path / "movie"
    movie.mkdir()
    (movie / "film.py").write_text("my existing scene")
    with pytest.raises(FileExistsError):
        init_movie(movie)
    assert (movie / "film.py").read_text() == "my existing scene"
    assert not (movie / "1ubq.cif").exists()


def test_skill_default_location_and_idempotent_install(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    destination = install_skill()
    assert destination == tmp_path / "codex" / "skills" / SKILL_NAME
    before = {p.relative_to(destination): p.read_bytes() for p in destination.rglob("*") if p.is_file()}
    assert Path("SKILL.md") in before
    assert Path("agents/openai.yaml") in before
    assert Path("assets/film.py") in before
    assert Path("references/animation.md") in before
    assert install_skill() == destination
    assert before == {
        p.relative_to(destination): p.read_bytes() for p in destination.rglob("*") if p.is_file()
    }


def test_skill_update_preserves_custom_content_unless_force_requested(tmp_path):
    destination = install_skill(tmp_path / "skill")
    edited = destination / "SKILL.md"
    original = edited.read_text()
    edited.write_text("customized instructions")
    (destination / "my-notes.txt").write_text("keep this")
    with pytest.raises(FileExistsError):
        install_skill(destination)
    assert edited.read_text() == "customized instructions"
    install_skill(destination, force=True)
    assert edited.read_text() == original
    assert (destination / "my-notes.txt").read_text() == "keep this"


def test_bundle_rejects_symlinked_destinations(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "skill"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows symlinks require Developer Mode or administrator privileges")
        raise
    with pytest.raises(FileExistsError):
        install_skill(link, force=True)
    assert list(outside.iterdir()) == []


def test_cli_init_and_version(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["proteinmotion", "init", str(tmp_path / "movie")])
    main()
    assert (tmp_path / "movie" / "film.py").is_file()
    assert "ProteinMovie" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["proteinmotion", "--version"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 0
    assert "proteinmotion" in capsys.readouterr().out
