"""Render documentation clips and record their exact source snippets.

Run from a checkout: python scripts/render_docs_examples.py
Use --only ID to refresh one clip after adjusting it. The website build checks
source and input hashes, so all affected clips must be refreshed before publishing.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
import textwrap
from pathlib import Path

import av

from proteinmotion.cli import load_scene

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/docs_examples.py"
OUTPUT = ROOT / "website/public/media/docs"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snippet(identifier):
    text = SOURCE.read_text()
    match = re.search(
        rf"^[ \t]*# docs:start {re.escape(identifier)}\n(.*?)^[ \t]*# docs:end {re.escape(identifier)}$",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise ValueError(f"Missing source marker: {identifier}")
    return textwrap.dedent(match.group(1)).strip(), text[: match.start(1)].count("\n") + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", help="Example IDs to refresh")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("docs_examples", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    entries = [
        (key, cls.__name__, title, caption, poster, SOURCE)
        for key, cls, title, caption, poster in module.EXAMPLES
    ]
    starter = ROOT / "skills/proteinmotion-movies/assets/film.py"
    entries.append(
        ("starter", "ProteinMovie", "Starter video", "The scene created by proteinmotion init.", 6, starter)
    )
    unknown = set(args.only or []) - {entry[0] for entry in entries}
    if unknown:
        parser.error(f"Unknown example IDs: {sorted(unknown)}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for key, cls, title, caption, poster_time, source in entries:
        if args.only and key not in args.only:
            continue
        scene = load_scene(source, cls, width=1280, height=720, fps=60).build()
        video_path = OUTPUT / f"{key}.mp4"
        result = scene.render(video_path, bitrate="3M", progress=False)
        frame_count = 0
        poster_index = min(round(poster_time * scene.fps), math.ceil(scene.duration * scene.fps) - 1)
        with av.open(str(video_path)) as video:
            stream = video.streams.video[0]
            assert (stream.width, stream.height) == (1280, 720)
            assert float(stream.average_rate) == 60
            for frame_count, frame in enumerate(video.decode(video=0), start=1):
                if frame_count == poster_index + 1:
                    frame.to_image().save(OUTPUT / f"{key}.jpg", quality=92)
        assert frame_count == result["frames"]
        dependencies = (
            [source, ROOT / "examples/data/1ubq.cif", ROOT / "examples/data/2k39.cif"]
            if source == SOURCE
            else [source, starter.with_name("1ubq.cif")]
        )
        data = {
            "title": title,
            "caption": caption,
            "source": str(source.relative_to(ROOT)),
            "scene": cls,
            "duration": scene.duration,
            "frames": frame_count,
            "width": scene.width,
            "height": scene.height,
            "fps": scene.fps,
            "video": f"media/docs/{key}.mp4",
            "poster": f"media/docs/{key}.jpg",
            "poster_time": poster_index / scene.fps,
            "video_sha256": digest(video_path),
            "poster_sha256": digest(OUTPUT / f"{key}.jpg"),
            "dependencies": {str(p.relative_to(ROOT)): digest(p) for p in dependencies},
        }
        if source == SOURCE:
            data["code"], data["line"] = snippet(key)
        manifest[key] = data
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            f"{key}: {frame_count} frames decoded, {scene.duration:.1f}s, {result['seconds']:.2f}s to render",
            flush=True,
        )
    # Keep existing paired Markdown snippets identical to the rendered source.
    for doc in (ROOT / "docs").glob("*.md"):
        text = doc.read_text()
        for key, data in manifest.items():
            if "code" in data:
                text = re.sub(
                    rf"(```python output={re.escape(key)}\n).*?\n```",
                    lambda m, code=data["code"]: m[1] + code + "\n```",
                    text,
                    flags=re.DOTALL,
                )
        if text != doc.read_text():
            doc.write_text(text)


if __name__ == "__main__":
    main()
