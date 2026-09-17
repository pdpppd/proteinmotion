"""Render and validate the EEVEE documentation movie. Requires Blender 4.5+."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import av

from proteinmotion.cli import load_scene

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/eevee_focus.py"
MEDIA = ROOT / "website/public/media/docs"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only", action="store_true", help="Validate an existing 960×540/60 fps export"
    )
    args = parser.parse_args()
    scene = load_scene(SOURCE, "HelixFocus", width=960, height=540, fps=60).build()
    MEDIA.mkdir(parents=True, exist_ok=True)
    video, poster = MEDIA / "eevee-focus.mp4", MEDIA / "eevee-focus.jpg"
    if not args.validate_only:
        scene.render(video, renderer="eevee")
    count = 0
    with av.open(video) as container:
        stream = container.streams.video[0]
        assert stream.average_rate == 60
        for frame in container.decode(video=0):
            assert (frame.width, frame.height) == (960, 540)
            assert abs(float(frame.pts * frame.time_base) - count / 60) < 0.001
            if count == 180:
                frame.to_image().save(poster, quality=92)
            count += 1
    assert count == round(scene.duration * 60) == 480
    manifest_path = MEDIA / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["eevee-focus"] = dict(
        title="EEVEE lens focus",
        caption="Calmodulin: highlight residues 5–19, fade the surrounding cartoon, and change lens focus.",
        source="examples/eevee_focus.py",
        scene="HelixFocus",
        duration=scene.duration,
        frames=count,
        width=960,
        height=540,
        fps=60,
        video="media/docs/eevee-focus.mp4",
        poster="media/docs/eevee-focus.jpg",
        poster_time=3,
        video_sha256=digest(video),
        poster_sha256=digest(poster),
        dependencies={str(p.relative_to(ROOT)): digest(p) for p in (SOURCE, ROOT / "examples/data/1cll.cif")},
        code=SOURCE.read_text().strip(),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    doc = ROOT / "docs/eevee.md"
    doc.write_text(
        re.sub(
            r"(```python output=eevee-focus\n).*?\n```",
            lambda m: m[1] + SOURCE.read_text().strip() + "\n```",
            doc.read_text(),
            flags=re.DOTALL,
        )
    )
    print(f"Validated {count} EEVEE frames at 960×540, 60 fps. Updated source and output manifest.")


if __name__ == "__main__":
    main()
