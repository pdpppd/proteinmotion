"""Render the DNA/RNA guides and verify the media against their scripts and inputs."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import av

from proteinmotion.cli import load_scene

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "website/public/media/docs"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("dna-styles", "rna-styles", "dna-morph"))
    args = parser.parse_args()
    manifest_path = MEDIA / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    examples = [
        (
            "dna-styles",
            "dna_styles",
            "DNAStyles",
            ("1bna",),
            5,
            "DNA base styles",
            "1BNA: slabs, rings, sticks, ladder rods, residue styling and a surface.",
        ),
        (
            "rna-styles",
            "rna_styles",
            "RNAStyles",
            ("1ehz",),
            6.5,
            "RNA regions and surfaces",
            "1EHZ tRNA: modified bases, anticodon-loop selection and a B-factor surface.",
        ),
        (
            "dna-morph",
            "dna_morph",
            "DNAMorph",
            ("1bna", "2dcg"),
            4.5,
            "DNA morph with C1′ anchors",
            "1BNA A to 2DCG A: matched C1′ positions move in order, with fades for unmatched nucleotides.",
        ),
    ]
    for key, file, cls, pdbs, poster_time, title, caption in examples:
        if args.only and args.only != key:
            continue
        source = ROOT / "examples" / (file + ".py")
        scene = load_scene(source, cls, width=1280, height=720, fps=60).build()
        video, poster = MEDIA / (key + ".mp4"), MEDIA / (key + ".jpg")
        report = scene.render(video, bitrate="6M", progress=False)
        count = 0
        with av.open(video) as container:
            assert container.streams.video[0].average_rate == 60
            for frame in container.decode(video=0):
                assert (frame.width, frame.height) == (1280, 720)
                assert abs(float(frame.pts * frame.time_base) - count / 60) < 0.001
                if count == round(poster_time * 60):
                    frame.to_image().save(poster, quality=92)
                count += 1
        assert count == max(1, math.ceil(scene.duration * scene.fps))
        dependencies = [source, *(ROOT / "examples/data" / (pdb + ".cif") for pdb in pdbs)]
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
        print(f"{key}: {count} frames decoded; render {report['seconds']:.2f}s", flush=True)
        doc = ROOT / "docs/dna-rna.md"
        if doc.exists():
            doc.write_text(
                re.sub(
                    rf"(```python output={key}\n).*?\n```",
                    lambda m: m[1] + source.read_text().strip() + "\n```",
                    doc.read_text(),
                    flags=re.DOTALL,
                )
            )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
