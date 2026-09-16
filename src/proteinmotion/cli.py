import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

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
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Report the native GPU and video encoder environment")
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
        if mode == "render":
            p.add_argument("--codec", default="auto")
            p.add_argument("--bitrate", default="20M")
        if mode == "still":
            p.add_argument("--time", type=float, default=0)
    args = parser.parse_args()
    if args.command == "doctor":
        import platform

        import av
        import wgpu

        from .video import available_encoders

        encoders = available_encoders()
        print(
            json.dumps(
                {
                    "python": sys.version,
                    "machine": platform.machine(),
                    "wgpu": wgpu.__version__,
                    "pyav": av.__version__,
                    "adapters": [dict(a.info) for a in wgpu.gpu.enumerate_adapters_sync()],
                    "ffmpeg_executable_optional": shutil.which("ffmpeg"),
                    "encoders": encoders,
                },
                indent=2,
            )
        )
        return
    scene = load_scene(
        args.file, args.scene, width=args.width, height=args.height, fps=args.fps, msaa=args.msaa
    )
    if args.command == "render":
        scene.render(args.output, codec=args.codec, bitrate=args.bitrate)
    elif args.command == "still":
        scene.render_frame(args.time, output=args.output)
    else:
        scene.preview()


if __name__ == "__main__":
    main()
