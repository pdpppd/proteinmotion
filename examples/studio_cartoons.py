"""Compare original and classic Studio cartoons at identical camera/lighting settings.

Run: .venv/bin/python examples/studio_cartoons.py
Writes three 3840×2160 comparison plates and a 1920×1080, 60 fps turntable.
PDB 2DRI (chain A), 1UBQ, and 4HHB (chain A) use their deposited secondary
structure. This is a display-style comparison, not a PyMOL/Chimera rendering
or a molecular dynamics simulation. The underlying coordinates are unchanged.
"""

import json
from dataclasses import asdict
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from proteinmotion import Protein, ProteinScene, StudioLook, linear
from proteinmotion.studio import StudioRenderer
from proteinmotion.video import VideoWriter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/studio-cartoons"
STYLES = ("legacy", "classic")
NAMES = {"2dri": "Ribose-binding protein · 2DRI A", "1ubq": "Ubiquitin · 1UBQ", "4hhb": "Hemoglobin · 4HHB A"}


def look(style):
    return StudioLook(cartoon_style=style, lighting="soft", material="medium", effects="clean")


class CartoonPortrait(ProteinScene):
    def __init__(self, *, code="2dri", **kwargs):
        self.code = code
        kwargs.setdefault("background", "#000000")
        super().__init__(**kwargs)

    def construct(self):
        p = Protein.from_file(
            ROOT / f"examples/data/{self.code}.cif", chains=None if self.code == "1ubq" else "A"
        )
        p.ball_and_stick(atom_scale=0.20, bond_radius=0.10).cartoon(color="#e8cda5").center()
        self.camera.frame(p, aspect=self.width / self.height, margin=1.12)
        self.camera.zoom({"2dri": 1.02, "1ubq": 1.4, "4hhb": 1.18}[self.code])
        self.camera.depth_cue = 0.1
        self.add(p)
        self.wait(0.5)
        self.play(self.camera.animate.orbit(1.2, 0.12), run_time=6, rate_func=linear)
        self.wait(0.5)


def heading(width, height, name):
    image = Image.new("RGBA", (width, height), "#10151c")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(ROOT / "src/proteinmotion/fonts/SourceSans3-Regular.otf"), height // 4)
    draw.text((height // 4, 6), name, fill="#e8cda5", font=font)
    draw.text((height // 4, height // 2), "BEFORE · Studio legacy", fill="#a8b1be", font=font)
    draw.text((width // 2 + height // 4, height // 2), "AFTER · Studio classic", fill="#8fe3c0", font=font)
    return np.asarray(image)


def pair(images, title):
    return np.concatenate([title, np.concatenate(images, axis=1)], axis=0)


def render_comparisons():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = dict(looks={s: asdict(look(s)) for s in STYLES}, stills=[], video=None)
    for code in NAMES:
        scene = CartoonPortrait(code=code, width=1920, height=2048, fps=60)
        images = []
        for style in STYLES:
            name = f"{code}-{style}-full.png"
            images.append(scene.render_frame(0, output=OUT / name, renderer="studio", look=look(style)))
        plate = Image.fromarray(pair(images, heading(3840, 112, NAMES[code])))
        name = f"{code}-comparison-4k.png"
        plate.save(OUT / name)
        plate.thumbnail((1920, 1080), Image.Resampling.LANCZOS)
        plate.convert("RGB").save(OUT / f"{code}-comparison.jpg", quality=94)
        manifest["stills"].append(dict(pdb=code.upper(), file=name, size=[3840, 2160]))
        print(name, flush=True)

    scene = CartoonPortrait(width=960, height=1008, fps=60).build()
    title = heading(1920, 72, NAMES["2dri"] + " / same camera, lighting and coordinates")
    file = OUT / "cartoon-turntable.mp4"
    with (
        StudioRenderer(960, 1008, look=look("legacy")) as before,
        StudioRenderer(960, 1008, look=look("classic")) as after,
        VideoWriter(file, 1920, 1080, 60) as video,
    ):
        for i in range(round(scene.duration * 60)):
            images = [scene.render_frame(i / 60, renderer=r) for r in (before, after)]
            video.write(pair(images, title))
    with av.open(str(file)) as movie:
        frames = sum(1 for _ in movie.decode(video=0))
    assert frames == round(scene.duration * 60)
    manifest["video"] = dict(file=file.name, fps=60, frames=frames, seconds=scene.duration, size=[1920, 1080])
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{file.name}: {frames} frames decoded", flush=True)


if __name__ == "__main__":
    render_comparisons()
