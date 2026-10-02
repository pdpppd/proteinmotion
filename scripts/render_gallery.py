"""Render the website gallery from examples/gallery/catalog.json.

Run from a checkout: python scripts/render_gallery.py
Use --only ID [ID ...] to refresh some clips. Each clip is rendered at 1280 × 720,
60 fps with the scene's own Studio settings. Entries whose scene is a list are
rendered at 640 × 360 and tiled 2 × 2, for looks that cannot change mid-scene.
The manifest records source and input hashes; the website build rejects stale clips.
"""

import argparse
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

import av
import numpy as np

from proteinmotion.cli import load_scene

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "examples/gallery"
DATA = ROOT / "examples/data"
OUTPUT = ROOT / "website/public/media/gallery"
WIDTH, HEIGHT, FPS, BITRATE = 1280, 720, 60, "2500k"


def digest(path):
    data = path.read_bytes()
    if path.suffix == ".py":
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def inputs(source):
    """Data files a scene reads through DATA / "name"."""
    names = sorted(set(re.findall(r'DATA / "([^"]+)"', source.read_text())))
    missing = [name for name in names if not (DATA / name).exists()]
    if missing:
        raise FileNotFoundError(f"{source.name} reads missing inputs: {missing}")
    return [DATA / name for name in names]


def render_one(source, scene_name, output, width=WIDTH, height=HEIGHT, bitrate=BITRATE):
    scene = load_scene(source, scene_name, width=width, height=height, fps=FPS).build()
    result = scene.render(output, bitrate=bitrate, progress=False)
    return scene.duration, result


def tile(parts, output):
    """Tile four equal videos 2 × 2 into one H.264 video."""
    containers = [av.open(str(part)) for part in parts]
    try:
        streams = [c.decode(video=0) for c in containers]
        with av.open(str(output), "w") as out:
            codec = "h264_videotoolbox" if sys.platform == "darwin" else "libx264"
            stream = out.add_stream(codec, rate=FPS)
            stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
            stream.bit_rate = int(BITRATE.rstrip("k")) * 1000
            canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
            count = 0
            for frames in zip(*streams):
                for k, frame in enumerate(frames):
                    y, x = divmod(k, 2)
                    canvas[y * 360 : (y + 1) * 360, x * 640 : (x + 1) * 640] = frame.to_ndarray(
                        format="rgb24"
                    )
                canvas[359:361, :] = canvas[:, 639:641] = 11  # Thin seams between tiles.
                image = av.VideoFrame.from_ndarray(canvas, format="rgb24")
                for packet in stream.encode(image):
                    out.mux(packet)
                count += 1
            for packet in stream.encode():
                out.mux(packet)
    finally:
        for c in containers:
            c.close()
    return count


def poster(video, time, path):
    """Save the decoded frame at `time` and return the frame count and that frame's time."""
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == (WIDTH, HEIGHT), video
        assert round(float(stream.average_rate)) == FPS, video
        frames = list(container.decode(video=0))
    index = min(round(time * FPS), len(frames) - 1)
    frames[index].to_image().save(path, quality=88)
    return len(frames), index / FPS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", help="Example IDs to refresh")
    args = parser.parse_args()
    catalog = json.loads((GALLERY / "catalog.json").read_text())
    categories = {c["id"] for c in catalog["categories"]}
    ids = [e["id"] for e in catalog["examples"]]
    if len(ids) != len(set(ids)):
        parser.error("Example IDs must be unique")
    unknown = set(args.only or []) - set(ids)
    if unknown:
        parser.error(f"Unknown example IDs: {sorted(unknown)}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = {k: v for k, v in manifest.items() if k in ids}  # Drop removed examples.
    for entry in catalog["examples"]:
        key = entry["id"]
        if entry["category"] not in categories:
            raise ValueError(f"{key}: unknown category {entry['category']}")
        if args.only and key not in args.only:
            continue
        source = GALLERY / entry["file"]
        video = OUTPUT / f"{key}.mp4"
        scenes = entry["scene"] if isinstance(entry["scene"], list) else [entry["scene"]]
        if len(scenes) == 1:
            duration, result = render_one(source, scenes[0], video)
            seconds = result["seconds"]
        else:
            if len(scenes) != 4:
                raise ValueError(f"{key}: tiled entries need four scenes")
            with tempfile.TemporaryDirectory() as tmp:
                parts, durations, seconds = [], [], 0.0
                for name in scenes:
                    part = Path(tmp) / f"{name}.mp4"
                    duration, result = render_one(source, name, part, WIDTH // 2, HEIGHT // 2, "1500k")
                    parts.append(part)
                    durations.append(duration)
                    seconds += result["seconds"]
                if len(set(durations)) != 1:
                    raise ValueError(f"{key}: tiled scenes must have equal durations")
                tile(parts, video)
        frames, poster_time = poster(video, entry.get("poster", 0), OUTPUT / f"{key}.jpg")
        dependencies = [source, *inputs(source)]
        manifest[key] = {
            **{k: v for k, v in entry.items() if k != "poster"},
            "scene": scenes if len(scenes) > 1 else scenes[0],
            "source": str(source.relative_to(ROOT)),
            "duration": round(frames / FPS, 3),
            "frames": frames,
            "width": WIDTH,
            "height": HEIGHT,
            "fps": FPS,
            "video": f"media/gallery/{key}.mp4",
            "poster": f"media/gallery/{key}.jpg",
            "poster_time": poster_time,
            "video_sha256": digest(video),
            "poster_sha256": digest(OUTPUT / f"{key}.jpg"),
            "dependencies": {str(p.relative_to(ROOT)): digest(p) for p in dependencies},
            "code": source.read_text(),
        }
        ordered = {e["id"]: manifest[e["id"]] for e in catalog["examples"] if e["id"] in manifest}
        manifest_path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n")
        size = video.stat().st_size / 1e6
        print(
            f"{key}: {frames} frames, {frames / FPS:.1f} s, {size:.1f} MB, rendered in {seconds:.1f} s",
            flush=True,
        )
    missing = [k for k in ids if k not in manifest]
    if missing:
        print(f"Not yet rendered: {', '.join(missing)}")


if __name__ == "__main__":
    main()
