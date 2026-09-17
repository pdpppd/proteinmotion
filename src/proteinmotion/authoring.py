"""Installed authoring resources: starter movie and optional Codex skill."""

import os
import tempfile
from importlib.resources import files
from pathlib import Path

SKILL_NAME = "proteinmotion-movies"


def _resources():
    bundled = files("proteinmotion").joinpath("_skill")
    if bundled.joinpath("SKILL.md").is_file():
        return bundled
    # An editable checkout uses the same canonical files as the wheel builder.
    checkout = Path(__file__).resolve().parents[2] / "skills" / SKILL_NAME
    if checkout.joinpath("SKILL.md").is_file():
        return checkout
    raise FileNotFoundError("Authoring resources are missing; reinstall the complete ProteinMotion wheel")


def _walk(root, prefix=Path()):
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if entry.name.startswith(".") or entry.name == "__pycache__" or entry.name.endswith(".pyc"):
            continue
        relative = prefix / entry.name
        if entry.is_dir():
            yield from _walk(entry, relative)
        else:
            yield relative, entry.read_bytes()


def _write_bundle(destination, entries, *, force=False, identical_ok=False):
    destination = Path(destination).expanduser().absolute()
    entries = list(entries)
    # Preflight every path before making any changes to an existing project/skill.
    for relative, data in entries:
        target = destination / relative
        for parent in (target, *target.parents):
            if parent.is_symlink():
                raise FileExistsError(f"Refusing to write through a symlink: {parent}")
            if parent == destination:
                break
        if target.exists():
            if not target.is_file():
                raise FileExistsError(f"Expected a file: {target}")
            if identical_ok and target.read_bytes() == data:
                continue
            if not force:
                raise FileExistsError(f"File already exists: {target}")
        for parent in target.parents:
            if parent.exists() and not parent.is_dir():
                raise FileExistsError(f"Expected a directory: {parent}")
            if parent == destination:
                break
    for relative, data in entries:
        target = destination / relative
        if identical_ok and target.exists() and target.read_bytes() == data:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=".proteinmotion-", delete=False
            ) as out:
                temporary = Path(out.name)
                out.write(data)
            temporary.chmod(0o644)
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return destination


def init_movie(directory):
    """Create a runnable movie and input structure, refusing to overwrite either."""
    assets = _resources().joinpath("assets")
    entries = [(Path(name), assets.joinpath(name).read_bytes()) for name in ("film.py", "1ubq.cif")]
    return _write_bundle(directory, entries)


def install_skill(path=None, *, force=False):
    """Install the bundled skill only when explicitly invoked; preserve custom files."""
    if path is None:
        codex_dir = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser()
        path = codex_dir / "skills" / SKILL_NAME
    return _write_bundle(path, _walk(_resources()), force=force, identical_ok=True)
