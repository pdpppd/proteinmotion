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
    value = str(path)
    return "'" + value.replace("'", "''") + "'" if sys.platform == "win32" else shlex.quote(value)


def load_scene(path, name=None, **kwargs):
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
    if not isinstance(cls, type) or not issubclass(cls, ProteinScene):
        raise TypeError(f"{name} must derive from ProteinScene")
    return cls(**{key: value for key, value in kwargs.items() if value is not None})


def main():
    parser = argparse.ArgumentParser(
        prog="proteinmotion", description="Programmatic molecular films on native GPUs"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Report the native GPU and video encoder environment")
    doctor.add_argument(
        "--check-encoders", action="store_true", help="Test actual encoder initialization and encoding"
    )
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
    for mode in ("render", "still", "preview"):
        p = commands.add_parser(mode)
        gpu_parsers.append(p)
        p.add_argument("file", type=Path)
        p.add_argument("scene", nargs="?")
        p.add_argument("--width", type=int, default=None)
        p.add_argument("--height", type=int, default=None)
        p.add_argument("--fps", type=float, default=None)
        p.add_argument("--msaa", type=int, choices=(1, 4), default=None)
        if mode != "preview":
            p.add_argument("-o", "--output", type=Path, required=True)
        p.add_argument("--renderer", choices=("native", "studio", "eevee"), default=None)
        p.add_argument("--lighting", choices=tuple(LIGHTING_PRESETS), help="Studio lighting preset")
        p.add_argument("--material", choices=tuple(MATERIAL_PRESETS), help="Studio material preset")
        p.add_argument("--effects", choices=tuple(EFFECTS_PRESETS), help="Studio global effects recipe")
        for effect in ("grain", "bloom", "halation"):
            p.add_argument(
                f"--{effect}", help=f"Studio {effect}: preset name or numeric strength (0 disables)"
            )
        p.add_argument("--blender", help="Blender executable path (EEVEE only)")
        p.add_argument("--samples", type=int, default=64, help="EEVEE samples per pixel")
        p.add_argument(
            "--supersampling",
            type=float,
            default=None,
            help="Render at this multiple of the output size and average down: 1-4 for native and "
            "Studio (default 2 up to 1920x1080, else 1); any value of at least 1 for EEVEE (default 1.5)",
        )
        if mode == "render":
            p.add_argument("--codec", default="auto")
            p.add_argument("--bitrate", default="20M")
        if mode == "still":
            p.add_argument("--time", type=float, default=0)
    for p in gpu_parsers:
        p.add_argument(
            "--gpu-backend",
            choices=("auto", "Vulkan", "D3D12", "Metal", "OpenGL"),
            help="Native GPU backend (Windows/Linux prefer Vulkan; macOS prefers Metal)",
        )
        p.add_argument("--gpu-adapter", help="GPU name/vendor substring, e.g. NVIDIA or RTX 5070")
    args = parser.parse_args()
    for flag, variable in (
        ("gpu_backend", "PROTEINMOTION_GPU_BACKEND"),
        ("gpu_adapter", "PROTEINMOTION_GPU_ADAPTER"),
    ):
        if getattr(args, flag, None) is not None:
            os.environ[variable] = getattr(args, flag)
    if args.command in ("init", "install-skill"):
        from .authoring import SKILL_NAME, init_movie, install_skill

        try:
            if args.command == "init":
                destination = init_movie(args.directory)
                print(f"Created {destination / 'film.py'} and {destination / '1ubq.cif'}")
                print(
                    f"proteinmotion render {_shell_quote(destination / 'film.py')} "
                    f"ProteinMovie --fps 60 -o {_shell_quote(destination / 'film.mp4')}"
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

        from ._gpu import select_adapter
        from .eevee import find_blender
        from .video import available_encoders, probe_encoders

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
        return
    look = None
    if args.command in ("render", "still", "preview"):
        settings = {
            name: getattr(args, name)
            for name in ("lighting", "material", "effects", "grain", "bloom", "halation")
            if getattr(args, name) is not None
        }
        if settings:
            args.renderer = args.renderer or "studio"
            if args.renderer not in (None, "studio"):
                parser.error("Lighting, material and effects options require --renderer studio")
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
    if factor is not None and not whole and (args.renderer or scene.renderer) != "eevee":
        parser.error("--supersampling must be 1, 2, 3, or 4 for the native and Studio renderers")
    eevee_supersampling = 1.5 if factor is None else factor
    if args.command == "render":
        from .eevee import EEVEEOptions

        options = EEVEEOptions(blender=args.blender, samples=args.samples, supersampling=eevee_supersampling)
        scene.render(
            args.output,
            codec=args.codec,
            bitrate=args.bitrate,
            renderer=args.renderer,
            eevee=options if args.renderer == "eevee" else None,
            look=look,
        )
    elif args.command == "still":
        from .eevee import EEVEEOptions

        options = EEVEEOptions(blender=args.blender, samples=args.samples, supersampling=eevee_supersampling)
        scene.render_frame(
            args.time,
            output=args.output,
            renderer=args.renderer,
            eevee=options if args.renderer == "eevee" else None,
            look=look,
        )
    else:
        scene.preview(renderer=args.renderer, look=look)


if __name__ == "__main__":
    main()
