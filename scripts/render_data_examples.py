"""Render numerical-property, plot, and density examples and verify every frame."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import av

from proteinmotion.cli import load_scene

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "website/public/media/docs"
EXAMPLES = [
    (
        "numerical-properties",
        "numerical_properties",
        "NumericalProperties",
        3,
        "B factors across representations",
        "1UBQ B factors control residue color and cartoon thickness.",
        ["1ubq.cif"],
    ),
    (
        "synchronized-plots",
        "synchronized_plots",
        "SynchronizedPlots",
        4,
        "NMR states with linked plots",
        "2K39 playback with a contact map, sequence strip, and Cα distance trace.",
        ["2k39.cif"],
    ),
    (
        "density-maps",
        "density_maps",
        "DensityMaps",
        6,
        "Electron density and slices",
        "1UBQ PDBe density: animate the contour and move a slice past helix 23–34.",
        ["1ubq.cif", "1ubq.ccp4"],
    ),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=[entry[0] for entry in EXAMPLES])
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    manifest_path = MEDIA / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for key, file, cls, poster_time, title, caption, data in EXAMPLES:
        if args.only and key not in args.only:
            continue
        source = ROOT / "examples" / (file + ".py")
        scene = load_scene(source, cls, width=1280, height=720, fps=60).build()
        video, poster = MEDIA / (key + ".mp4"), MEDIA / (key + ".jpg")
        if not args.validate_only:
            scene.render(video, bitrate="5M", progress=False)
        count = 0
        with av.open(video) as container:
            assert container.streams.video[0].average_rate == 60
            for frame in container.decode(video=0):
                assert (frame.width, frame.height) == (1280, 720)
                assert abs(float(frame.pts * frame.time_base) - count / 60) < 0.001
                if count == round(poster_time * 60):
                    frame.to_image().save(poster, quality=92)
                count += 1
        assert count == round(scene.duration * 60)
        dependencies = [source, *[ROOT / "examples/data" / p for p in data]]
        manifest[key] = dict(
            title=title,
            caption=caption,
            source=str(source.relative_to(ROOT)),
            scene=cls,
            duration=scene.duration,
            frames=count,
            width=1280,
            height=720,
            fps=60,
            video=f"media/docs/{key}.mp4",
            poster=f"media/docs/{key}.jpg",
            poster_time=poster_time,
            video_sha256=digest(video),
            poster_sha256=digest(poster),
            dependencies={str(p.relative_to(ROOT)): digest(p) for p in dependencies},
            code=source.read_text().strip(),
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        doc = ROOT / "docs" / (key + ".md")
        if doc.exists():
            doc.write_text(
                re.sub(
                    rf"(```python output={key}\n).*?\n```",
                    lambda m: m[1] + source.read_text().strip() + "\n```",
                    doc.read_text(),
                    flags=re.DOTALL,
                )
            )
        print(f"{key}: {count} frames decoded at 1280×720, 60 fps", flush=True)


if __name__ == "__main__":
    main()
