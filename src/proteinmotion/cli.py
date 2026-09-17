import argparse
import importlib.util
import json
import shlex
import shutil
import sys
from pathlib import Path

from . import __version__
from .scene import ProteinScene


def load_scene(path, name, **kwargs):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("proteinmotion_user_scene", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import scene file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    cls = getattr(module, name)
    if not isinstance(cls, type) or not issubclass(cls, ProteinScene):
        raise TypeError(f"{name} must derive from ProteinScene")
    return cls(**kwargs)


def main():
    parser = argparse.ArgumentParser(
        prog="proteinmotion", description="Programmatic molecular films on native Metal"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Report the native GPU and video encoder environment")
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
    for mode in ("render", "still", "preview"):
        p = commands.add_parser(mode)
        p.add_argument("file", type=Path)
        p.add_argument("scene")
        p.add_argument("--width", type=int, default=1920 if mode != "preview" else 1280)
        p.add_argument("--height", type=int, default=1080 if mode != "preview" else 720)
        p.add_argument("--fps", type=float, default=30)
        p.add_argument("--msaa", type=int, choices=(1, 4), default=4)
        if mode != "preview":
            p.add_argument("-o", "--output", type=Path, required=True)
            p.add_argument("--renderer", choices=("native", "eevee"), default="native")
            p.add_argument("--blender", help="Blender executable path (EEVEE only)")
            p.add_argument("--samples", type=int, default=64, help="EEVEE samples per pixel")
            p.add_argument(
                "--supersampling", type=float, default=1.5, help="EEVEE spatial resolution multiplier"
            )
        if mode == "render":
            p.add_argument("--codec", default="auto")
            p.add_argument("--bitrate", default="20M")
        if mode == "still":
            p.add_argument("--time", type=float, default=0)
    args = parser.parse_args()
    if args.command in ("init", "install-skill"):
        from .authoring import SKILL_NAME, init_movie, install_skill

        try:
            if args.command == "init":
                destination = init_movie(args.directory)
                print(f"Created {destination / 'film.py'} and {destination / '1ubq.cif'}")
                print(
                    f"proteinmotion render {shlex.quote(str(destination / 'film.py'))} "
                    f"ProteinMovie --fps 60 -o {shlex.quote(str(destination / 'film.mp4'))}"
                )
            else:
                destination = install_skill(args.path, force=args.force)
                print(f"Installed ${SKILL_NAME} at {destination}")
                print("If Codex is already open, start a new conversation to discover the skill.")
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        return
    if args.command == "doctor":
        import platform

        import av
        import wgpu

        from .eevee import find_blender
        from .video import available_encoders

        try:
            blender = find_blender()
        except RuntimeError:
            blender = None

        encoders = available_encoders()
        print(
            json.dumps(
                {
                    "proteinmotion": __version__,
                    "python": sys.version,
                    "machine": platform.machine(),
                    "wgpu": wgpu.__version__,
                    "pyav": av.__version__,
                    "adapters": [dict(a.info) for a in wgpu.gpu.enumerate_adapters_sync()],
                    "ffmpeg_executable_optional": shutil.which("ffmpeg"),
                    "encoders": encoders,
                    "blender_executable_optional": blender,
                },
                indent=2,
            )
        )
        return
    scene = load_scene(
        args.file, args.scene, width=args.width, height=args.height, fps=args.fps, msaa=args.msaa
    )
    if args.command == "render":
        from .eevee import EEVEEOptions

        options = EEVEEOptions(blender=args.blender, samples=args.samples, supersampling=args.supersampling)
        scene.render(
            args.output,
            codec=args.codec,
            bitrate=args.bitrate,
            renderer=args.renderer,
            eevee=options if args.renderer == "eevee" else None,
        )
    elif args.command == "still":
        from .eevee import EEVEEOptions

        options = EEVEEOptions(blender=args.blender, samples=args.samples, supersampling=args.supersampling)
        scene.render_frame(
            args.time,
            output=args.output,
            renderer=args.renderer,
            eevee=options if args.renderer == "eevee" else None,
        )
    else:
        scene.preview()


if __name__ == "__main__":
    main()
