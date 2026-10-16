"""Installed authoring resources: starter movie and optional Codex skill.

``proteinmotion init`` copies the starter film (``film.py``) and its input structure
(``1ubq.cif``) into a new folder. ``proteinmotion install-skill`` copies the whole
``proteinmotion-movies`` skill into Codex's skills folder. Both read the same files:
``skills/proteinmotion-movies`` in a source checkout, which the wheel bundles as
``proteinmotion/_skill`` (see pyproject.toml).
"""

import os
import tempfile
from importlib.resources import files
from pathlib import Path

SKILL_NAME = "proteinmotion-movies"


def _resources():
    """Folder of skill files: the wheel's bundled copy, else the source checkout's."""
    bundled = files("proteinmotion").joinpath("_skill")
    if bundled.joinpath("SKILL.md").is_file():
        return bundled
    # An editable checkout uses the same canonical files as the wheel builder.
    checkout = Path(__file__).resolve().parents[2] / "skills" / SKILL_NAME
    if checkout.joinpath("SKILL.md").is_file():
        return checkout
    raise FileNotFoundError("Authoring resources are missing; reinstall the complete ProteinMotion wheel")


def _walk(root, prefix=Path()):
    """Yield (relative path, bytes) per file under ``root``, sorted by name, minus hidden and cache files."""
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if entry.name.startswith(".") or entry.name == "__pycache__" or entry.name.endswith(".pyc"):
            continue
        relative = prefix / entry.name
        if entry.is_dir():
            yield from _walk(entry, relative)
        else:
            yield relative, entry.read_bytes()


def _write_bundle(destination, entries, *, force=False, identical_ok=False):
    """Write (relative path, bytes) entries under ``destination``; return its absolute path.

    Every path is checked before any file is written, so a conflict leaves the folder
    unchanged. Existing files raise FileExistsError unless ``force``; with
    ``identical_ok``, files that already hold the same bytes are skipped. Files in the
    folder that are not in ``entries`` are never touched.
    """
    destination = Path(destination).expanduser().absolute()
    entries = list(entries)
    # Preflight every path before making any changes to an existing project/skill.
    for relative, data in entries:
        target = destination / relative
        # A symlink in the target or any folder up to the destination could redirect the write.
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
        # A file where a folder must go would make mkdir fail midway.
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
        # Write a temporary file in the same folder, then rename it over the target, so an
        # interrupted write never leaves a half-written file. The temporary file is created
        # private (0600); 0644 gives the usual read permissions.
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
            # After a successful rename the temporary path is gone; otherwise remove it.
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
        # The Codex home folder: $CODEX_HOME, else ~/.codex.
        codex_dir = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser()
        path = codex_dir / "skills" / SKILL_NAME
    # Re-installing an unchanged skill is a no-op; only modified bundled files need force.
    return _write_bundle(path, _walk(_resources()), force=force, identical_ok=True)
