"""Compare end-to-end export encoders on the same scene and validate every frame."""

import argparse
import hashlib
import json
import math
import platform
import statistics
from fractions import Fraction
from pathlib import Path

import av
import wgpu

from proteinmotion.cli import load_scene

ROOT = Path(__file__).resolve().parents[1]


def validate_movie(path, width, height, fps, count):
    decoded = 0
    with av.open(path) as movie:
        stream = movie.streams.video[0]
        assert (stream.width, stream.height) == (width, height)
        assert stream.codec_context.colorspace == 1
        assert stream.codec_context.color_range == 1
        for frame in movie.decode(video=0):
            assert frame.pts * frame.time_base == decoded / Fraction(str(fps))
            decoded += 1
    assert decoded == count, (decoded, count)
    return decoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=ROOT / "examples/quickstart.py")
    parser.add_argument("--scene", default="Quickstart")
    parser.add_argument("--codecs", nargs="+", default=["libx264", "auto"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--output", type=Path, default=ROOT / "output/windows-nvidia")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    scene = load_scene(args.file, args.scene, width=args.width, height=args.height, fps=args.fps).build()
    count = max(1, math.ceil(scene.duration * scene.fps))
    # Warm shader/geometry/driver caches for each codec. Renderer startup is still
    # included in each timed export; scene construction and decoding are excluded.
    for codec in args.codecs:
        scene.render(args.output / f"warmup-{codec}.mp4", codec=codec, progress=False)
    records = []
    for repeat in range(args.repeats):
        for codec in args.codecs:
            path = args.output / f"{codec}-{repeat + 1}.mp4"
            result = scene.render(path, codec=codec, progress=False)
            result["requested_codec"] = codec
            result["repeat"] = repeat + 1
            result["decoded_frames"] = validate_movie(path, args.width, args.height, args.fps, count)
            result["bytes"] = path.stat().st_size
            result["output"] = path.name
            records.append(result)
            print(
                f"{codec}: {result['seconds']:.3f}s, {result['fps']:.1f} fps; decoded {count} frames",
                flush=True,
            )
    summary = {
        codec: {
            "median_seconds": statistics.median(
                r["seconds"] for r in records if r["requested_codec"] == codec
            ),
            "median_fps": statistics.median(r["fps"] for r in records if r["requested_codec"] == codec),
        }
        for codec in args.codecs
    }
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "wgpu": wgpu.__version__,
        "pyav": av.__version__,
        "scene": args.scene,
        "scene_sha256": hashlib.sha256(args.file.read_bytes()).hexdigest(),
        "width": args.width,
        "height": args.height,
        "video_fps": args.fps,
        "msaa": scene.msaa,
        "bitrate": "20M",
        "summary": summary,
        "results": records,
        "notes": "One warmup per codec; interleaved measured runs. Includes renderer startup, scene evaluation, "
        "GPU upload, drawing, NV12 readback, encoding and container finalization. "
        "Excludes scene construction and frame validation. Desktop applications remained running. "
        "Encoders have different rate-control/quality settings; this is a throughput comparison.",
    }
    (args.output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
