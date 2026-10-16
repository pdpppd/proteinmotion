"""The ``proteinmotion`` command line: doctor, init, install-skill, render, still and preview.

``main`` is the console script named in pyproject.toml and also runs for
``python -m proteinmotion``. It parses the arguments and hands each subcommand to its own
private function below. ``load_scene`` imports a user's scene file and creates its
``ProteinScene`` subclass; the scripts in ``scripts/`` use it as well.
"""

import argparse
import importlib.util
import json
import os
import shlex
import shutil
import sys
from pathlib import Path

from . import __version__
from .looks import EFFECTS_PRESETS, LIGHTING_PRESETS, MATERIAL_PRESETS, StudioLook
from .scene import ProteinScene


def _shell_quote(path):
    """Quote a path for pasting into the user's shell: PowerShell quoting on Windows, POSIX elsewhere."""
    value = str(path)
    return "'" + value.replace("'", "''") + "'" if sys.platform == "win32" else shlex.quote(value)


def load_scene(path, name=None, **kwargs):
    """Import a scene file and return an instance of its ProteinScene subclass ``name``.

    ``name`` may be omitted when the file defines exactly one scene class. Keyword arguments
    (width, height, fps, ...) go to the scene's constructor; None values are dropped, so the
    scene's own defaults apply.
    """
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("proteinmotion_user_scene", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import scene file: {path}")
    module = importlib.util.module_from_spec(spec)
    # The file's own folder is importable while it runs, so it can import helper modules
    # that sit next to it.
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    # Only scene classes defined in this file count, not ones it imports.
    candidates = {
        key: value
        for key, value in vars(module).items()
        if isinstance(value, type)
        and issubclass(value, ProteinScene)
        and value is not ProteinScene
        and value.__module__ == module.__name__
    }
    if name is None and len(candidates) == 1:
        name = next(iter(candidates))
    if name not in candidates:
        raise ValueError(f"Choose a scene class from {path.name}: {', '.join(candidates) or 'none found'}")
    cls = candidates[name]
    return cls(**{key: value for key, value in kwargs.items() if value is not None})


def _build_parser():
    """Argument parser for every subcommand, with its help text."""
    parser = argparse.ArgumentParser(
        prog="proteinmotion", description="Programmatic molecular films on native GPUs"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Report the native GPU and video encoder environment")
    doctor.add_argument(
        "--check-encoders", action="store_true", help="Test actual encoder initialization and encoding"
    )
    # Subcommands that use the native GPU; they get --gpu-backend and --gpu-adapter below.
    gpu_parsers = [doctor]
    starter = commands.add_parser("init", help="Create a starter film and bundled PDB input")
    starter.add_argument(
        "directory", type=Path, help="Movie folder; existing film/data are never overwritten"
    )
    skill = commands.add_parser("install-skill", help="Install the bundled Codex movie-making skill")
    skill.add_argument(
        "--path",
        type=Path,
        help="Destination skill folder (default: $CODEX_HOME/skills/proteinmotion-movies)",
    )
    skill.add_argument(
        "--force", action="store_true", help="Replace modified bundled files; preserve unrelated files"
    )
    # render, still and preview share the scene and renderer options.
    for mode in ("render", "still", "preview"):
        subparser = commands.add_parser(mode)
        gpu_parsers.append(subparser)
        subparser.add_argument("file", type=Path)
        subparser.add_argument("scene", nargs="?")
        subparser.add_argument("--width", type=int, default=None)
        subparser.add_argument("--height", type=int, default=None)
        subparser.add_argument("--fps", type=float, default=None)
        subparser.add_argument("--msaa", type=int, choices=(1, 4), default=None)
        if mode != "preview":
            subparser.add_argument("-o", "--output", type=Path, required=True)
        subparser.add_argument("--renderer", choices=("native", "studio", "eevee"), default=None)
        subparser.add_argument("--lighting", choices=tuple(LIGHTING_PRESETS), help="Studio lighting preset")
        subparser.add_argument("--material", choices=tuple(MATERIAL_PRESETS), help="Studio material preset")
        subparser.add_argument(
            "--effects", choices=tuple(EFFECTS_PRESETS), help="Studio global effects recipe"
        )
        for effect in ("grain", "bloom", "halation"):
            subparser.add_argument(
                f"--{effect}", help=f"Studio {effect}: preset name or numeric strength (0 disables)"
            )
        subparser.add_argument("--blender", help="Blender executable path (EEVEE only)")
        # None when not given, so a scene's own EEVEE options keep their sample count.
        subparser.add_argument(
            "--samples", type=int, default=None, help="EEVEE samples per pixel (default 64)"
        )
        subparser.add_argument(
            "--supersampling",
            type=float,
            default=None,
            help="Render at this multiple of the output size and average down: 1-4 for native and "
            "Studio (default 2 up to 1920x1080, else 1); any value of at least 1 for EEVEE (default 1.5)",
        )
        if mode == "render":
            subparser.add_argument("--codec", default="auto")
            subparser.add_argument("--bitrate", default="20M")
        if mode == "still":
            subparser.add_argument("--time", type=float, default=0)
    for subparser in gpu_parsers:
        subparser.add_argument(
            "--gpu-backend",
            choices=("auto", "Vulkan", "D3D12", "Metal", "OpenGL"),
            help="Native GPU backend (Windows/Linux prefer Vulkan; macOS prefers Metal)",
        )
        subparser.add_argument("--gpu-adapter", help="GPU name/vendor substring, e.g. NVIDIA or RTX 5070")
    return parser


def main():
    """Run the ``proteinmotion`` command line."""
    parser = _build_parser()
    args = parser.parse_args()
    # The GPU choice reaches the native renderer through these environment variables,
    # which _gpu.select_adapter reads.
    for flag, variable in (
        ("gpu_backend", "PROTEINMOTION_GPU_BACKEND"),
        ("gpu_adapter", "PROTEINMOTION_GPU_ADAPTER"),
    ):
        if getattr(args, flag, None) is not None:
            os.environ[variable] = getattr(args, flag)
    if args.command == "init":
        _init_command(parser, args)
    elif args.command == "install-skill":
        _install_skill_command(parser, args)
    elif args.command == "doctor":
        _doctor_command(args)
    else:
        # render, still and preview. Bad Studio options are reported before the scene file runs.
        look = _studio_look(parser, args)
        scene = _load_command_scene(parser, args)
        if args.command == "render":
            _render_command(args, scene, look, _eevee_options(parser, args, scene))
        elif args.command == "still":
            _still_command(args, scene, look, _eevee_options(parser, args, scene))
        else:
            _preview_command(args, scene, look)


def _init_command(parser, args):
    from .authoring import init_movie

    try:
        destination = init_movie(args.directory)
        print(f"Created {destination / 'film.py'} and {destination / '1ubq.cif'}")
        # A ready-to-paste command for the new film.
        print(
            f"proteinmotion render {_shell_quote(destination / 'film.py')} "
            f"ProteinMovie --fps 60 -o {_shell_quote(destination / 'film.mp4')}"
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


def _install_skill_command(parser, args):
    from .authoring import SKILL_NAME, install_skill

    try:
        destination = install_skill(args.path, force=args.force)
        print(f"Installed ${SKILL_NAME} at {destination}")
        print("If Codex is already open, start a new conversation to discover the skill.")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


def _doctor_command(args):
    """Print the GPU, encoder and Blender environment as JSON."""
    import platform

    import av
    import wgpu

    from ._gpu import select_adapter
    from .eevee import find_blender
    from .video import available_encoders, probe_encoders

    # Blender and a working GPU are reported as missing rather than raising.
    try:
        blender = find_blender()
    except RuntimeError:
        blender = None

    encoders = available_encoders()
    selected, gpu_error = None, None
    try:
        selected = dict(select_adapter().info)
    except (RuntimeError, ValueError) as exc:
        gpu_error = str(exc)
    print(
        json.dumps(
            {
                "proteinmotion": __version__,
                "python": sys.version,
                "machine": platform.machine(),
                "platform": platform.platform(),
                "wgpu": wgpu.__version__,
                "pyav": av.__version__,
                "adapters": [dict(a.info) for a in wgpu.gpu.enumerate_adapters_sync()],
                "selected_adapter": selected,
                "gpu_error": gpu_error,
                "ffmpeg_executable_optional": shutil.which("ffmpeg"),
                "encoders": encoders,
                **({"encoder_checks": probe_encoders()} if args.check_encoders else {}),
                "blender_executable_optional": blender,
            },
            indent=2,
        )
    )


def _studio_look(parser, args):
    """StudioLook from the --lighting, --material, --effects, --grain, --bloom and --halation flags.

    Returns None when none was given. Any of them selects the Studio renderer when no
    renderer was named, and is an error with another renderer.
    """
    look = None
    settings = {
        name: getattr(args, name)
        for name in ("lighting", "material", "effects", "grain", "bloom", "halation")
        if getattr(args, name) is not None
    }
    if settings:
        args.renderer = args.renderer or "studio"
        if args.renderer not in (None, "studio"):
            parser.error("Lighting, material and effects options require --renderer studio")
        # Grain, bloom and halation take a number or a preset name.
        for name in ("grain", "bloom", "halation"):
            if name in settings:
                try:
                    settings[name] = float(settings[name])
                except ValueError:
                    pass  # Preset name; StudioLook supplies a useful validation error.
        try:
            look = StudioLook(**settings)
        except (ValueError, TypeError) as error:
            parser.error(str(error))
    return look


def _load_command_scene(parser, args):
    """Load the scene named on the command line, with the size and sampling options given."""
    # Native and Studio supersample by a whole factor from 1 to 4; any other factor is
    # valid for EEVEE only.
    factor = getattr(args, "supersampling", None)
    whole = factor is not None and float(factor).is_integer() and 1 <= factor <= 4
    scene = load_scene(
        args.file,
        args.scene,
        width=args.width,
        height=args.height,
        fps=args.fps,
        msaa=args.msaa,
        supersampling=int(factor) if whole else None,
    )
    # Checked after loading, because the scene class may choose EEVEE itself.
    if factor is not None and not whole and (args.renderer or scene.renderer) != "eevee":
        parser.error("--supersampling must be 1, 2, 3, or 4 for the native and Studio renderers")
    return scene


def _eevee_options(parser, args, scene):
    """EEVEE options for render and still, or None when another renderer draws the scene.

    --renderer chooses the renderer, or else the scene's own setting does. The options are
    the scene's own (or the defaults), with --blender, --samples and --supersampling in
    place of their values when given. A bad value is a usage error, not a traceback.
    """
    from dataclasses import replace

    from .eevee import EEVEEOptions

    if (args.renderer or scene.renderer) != "eevee":
        return None
    flags = {"blender": args.blender, "samples": args.samples, "supersampling": args.supersampling}
    given = {name: value for name, value in flags.items() if value is not None}
    try:
        return replace(scene.eevee or EEVEEOptions(), **given)
    except ValueError as error:
        parser.error(str(error))


def _render_command(args, scene, look, eevee):
    scene.render(
        args.output,
        codec=args.codec,
        bitrate=args.bitrate,
        renderer=args.renderer,
        eevee=eevee,
        look=look,
    )


def _still_command(args, scene, look, eevee):
    scene.render_frame(
        args.time,
        output=args.output,
        renderer=args.renderer,
        eevee=eevee,
        look=look,
    )


def _preview_command(args, scene, look):
    scene.preview(renderer=args.renderer, look=look)


if __name__ == "__main__":
    main()
