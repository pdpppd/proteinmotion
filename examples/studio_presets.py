"""Render controlled 3840 x 2160 comparisons of studio lighting, materials and effects.

    .venv/bin/python examples/studio_presets.py

Uses the same 2DRI structure as the buried-ribose movie. Each image has the same
coordinates, camera and pose; only StudioLook and the identifying caption change.
No external image editor is used: every effect is produced by the GPU renderer.
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from proteinmotion import Protein, ProteinScene, Region, Reveal, StudioLook, Text

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "studio-presets-4k"
INK, MINT, CREAM, MUTED, GOLD = "#0B1210", "#8FE3C0", "#F3EEE3", "#9EB3A8", "#FFC46B"


class PresetPortrait(ProteinScene):
    def __init__(
        self,
        *,
        heading="Studio",
        category="Lighting",
        description="Warm key · cool fill · mint rim",
        parameters="Material / glossy\nEffects / clean",
        **kwargs,
    ):
        kwargs.setdefault("background", INK)
        self.heading, self.category, self.description, self.parameters = (
            heading,
            category,
            description,
            parameters,
        )
        super().__init__(**kwargs)

    def construct(self):
        p = Protein.from_file(ROOT / "examples/data/2dri.cif", chains="A")
        p.ball_and_stick(atom_scale=0.25, bond_radius=0.105).surface(resolution=0.45).center()
        for code, color in {"H": MINT, "E": CREAM, "C": "#75A89B"}.items():
            ids = [
                i
                for i, a in enumerate(p.topology.atoms)
                if p.topology.residue_categories[a.residue_index] == "polymer"
                and p.topology.residues[a.residue_index].secondary == code
            ]
            if ids:
                Region(p, ids).set_color(color)
        ligand = p.select(resname="RIP")
        Region(p, [i for i in ligand.atom_indices if p.topology.atoms[i].element == "C"]).set_color(GOLD)
        self.camera.frame(p, margin=1.1, aspect=self.width / self.height)
        self.camera.theta, self.camera.phi, self.camera.depth_cue = 1.5, 0.25, 0.25
        right = np.array([np.cos(self.camera.theta), 0, -np.sin(self.camera.theta)])
        self.camera.target -= (
            right * self.camera.distance * np.tan(self.camera.fov / 2) * (self.width / self.height) * 0.33
        )
        self.add(p)
        self.add(
            Text(
                f"PROTEINMOTION  /  {self.category.upper()}",
                position=(0.065, 0.095),
                font_size=23,
                color=MINT,
            ),
            Text(self.heading, position=(0.063, 0.20), font_size=78, font="semibold", color=CREAM),
            Text(self.description, position=(0.067, 0.325), font_size=29, color=MUTED, line_spacing=1.6),
            Text(self.parameters, position=(0.067, 0.57), font_size=25, color=MUTED, line_spacing=1.8),
            Text("Ribose-binding protein · PDB 2DRI", position=(0.067, 0.895), font_size=23, color=MUTED),
            Text("3840 × 2160  /  GPU studio rendering", position=(0.067, 0.935), font_size=19, color=MUTED),
        )
        self.play(Reveal(self.camera, ligand, window=1.6, shape="cone", band=2), run_time=1)
        self.wait(1)


def render_gallery():
    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, description in {
        "studio": "Warm key · cool fill\nMint rim light",
        "soft": "Broad, even illumination\nGentle contrast",
        "dramatic": "Directional warm key\nDeep shadows · cool rim",
        "flat": "Front illumination\nLow-contrast illustration",
    }.items():
        entries.append(
            ("lighting", name, name.title(), description, StudioLook(lighting=name, effects="clean"))
        )
    for name, description in {
        "rough": "Matte surface\nBroad, soft highlights",
        "medium": "Satin surface\nBalanced reflections",
        "glossy": "Polished surface\nBright clearcoat highlights",
    }.items():
        entries.append(
            ("materials", name, name.title(), description, StudioLook(material=name, effects="clean"))
        )
    for name, heading, description, settings in [
        ("clean", "Clean", "All film effects disabled\nReference image", dict(effects="clean")),
        (
            "grain",
            "Film grain",
            "Luminance-based texture\nMonochrome · deterministic",
            dict(effects="clean", grain="film"),
        ),
        ("bloom", "Bloom", "Soft highlight spill\nNeutral glow", dict(effects="clean", bloom="strong")),
        (
            "halation",
            "Halation",
            "Warm highlight fringes\nRed / amber edge glow",
            dict(effects="clean", halation="warm"),
        ),
        ("subtle", "Subtle", "Gentle grain and glow\nA restrained finishing pass", dict(effects="subtle")),
        ("film", "Film", "Grain + bloom + halation\nA complete film look", dict(effects="film")),
        (
            "dreamy",
            "Dreamy",
            "Broad bloom + warm halation\nSoft, textured highlights",
            dict(effects="dreamy"),
        ),
    ]:
        entries.append(("effects", name, heading, description, StudioLook(**settings)))

    manifest = []
    for group, name, heading, description, look in entries:
        if group == "effects":
            params = f"Grain / {look.grain:g}\nBloom / {look.bloom:g}\nHalation / {look.halation:g}"
        else:
            params = f"Lighting / {look.lighting}\nMaterial / {look.material}\nEffects / clean"
        scene = PresetPortrait(
            width=3840,
            height=2160,
            msaa=4,
            heading=heading,
            category=group,
            description=description,
            parameters=params,
        )
        path = OUT / f"{group}-{name}.png"
        scene.render_frame(1.5, renderer="studio", look=look, output=path)
        manifest.append(
            dict(group=group, name=name, file=path.name, width=3840, height=2160, look=asdict(look), time=1.5)
        )
        print(f"Rendered {path.name}", flush=True)

    # Contact sheets are labeled comparisons; the source images above are all native 4K.
    boards = {
        "lighting-comparison-4k.jpg": [f"lighting-{n}.png" for n in ("studio", "soft", "dramatic", "flat")],
        "material-comparison-4k.jpg": [f"materials-{n}.png" for n in ("rough", "medium", "glossy")],
        "effects-comparison-4k.jpg": [f"effects-{n}.png" for n in ("clean", "grain", "bloom", "halation")],
        "recipes-comparison-4k.jpg": [f"effects-{n}.png" for n in ("clean", "subtle", "film", "dreamy")],
    }
    for filename, images in boards.items():
        board = Image.new("RGB", (3840, 2160), INK)
        for index, filename_in in enumerate(images):
            with Image.open(OUT / filename_in) as im:
                board.paste(
                    im.convert("RGB").resize((1920, 1080), Image.Resampling.LANCZOS),
                    ((index % 2) * 1920, (index // 2) * 1080),
                )
        if len(images) == 3:
            # A key in the fourth quadrant keeps the three-material comparison readable.
            draw = ImageDraw.Draw(board)
            font_path = ROOT / "src/proteinmotion/fonts/SourceSans3-Regular.otf"
            font = (
                ImageFont.truetype(str(font_path), 54)
                if font_path.exists()
                else ImageFont.load_default(size=54)
            )
            draw.multiline_text(
                (2100, 1320),
                "MATERIAL PRESETS\n\nrough  /  matte\nmedium  /  satin\nglossy  /  clearcoat\n\nSame lighting, pose and camera",
                font=font,
                fill=CREAM,
                spacing=22,
            )
        board.save(OUT / filename, quality=95)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    index = [
        "# Studio presets — 4K gallery",
        "",
        "Every PNG below is rendered directly at **3840 × 2160**, with 4× MSAA.",
        "The comparison sheets contain reduced copies of those originals.",
        "",
        "Source: [2DRI](https://www.rcsb.org/structure/2DRI), ribose-binding protein.",
        "",
        "Regenerate with `python examples/studio_presets.py` from the repository root.",
        "",
        "## Comparisons",
        "",
    ]
    index.extend(f"- [{name.replace('-4k.jpg', '').replace('-', ' ').title()}]({name})" for name in boards)
    for group in ("lighting", "materials", "effects"):
        index.extend(
            [
                "",
                f"## {group.title()}",
                "",
                "| Image | Lighting | Material | Grain | Bloom | Halation |",
                "| --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for entry in manifest:
            if entry["group"] == group:
                look = entry["look"]
                index.append(
                    f"| [{entry['name'].title()}]({entry['file']}) | {look['lighting']} | {look['material']} | "
                    f"{look['grain']:g} | {look['bloom']:g} | {look['halation']:g} |"
                )
    index.extend(["", "Complete settings: [manifest.json](manifest.json).", ""])
    (OUT / "README.md").write_text("\n".join(index))


if __name__ == "__main__":
    render_gallery()
