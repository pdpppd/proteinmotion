"""Eased threading of actual cartoon and ball-and-stick geometry in Studio.

Run: .venv/bin/python examples/studio_threading.py
Output: output/studio-threading-eased/ (two 1920×1080, 60 fps, 11 s movies).

PDB 2DRI chain A, including its bound ribose, matches the cartoon comparison.
The camera stays fixed while the actual representation travels into place.
Quintic easing controls travel over the full eight-second entrance.
Residues and ligands move as rigid groups; bonds between residues may stretch.
This is an illustrative entrance, not protein folding or molecular dynamics.
"""

from pathlib import Path

from proteinmotion import Protein, ProteinScene, StudioLook, Text, Thread

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/studio-threading-eased"
LOOK = StudioLook(cartoon_style="classic", lighting="soft", material="medium", effects="clean")


class ThreadRibose(ProteinScene):
    renderer = "studio"
    look = LOOK
    representation = "cartoon"

    def __init__(self, **kwargs):
        kwargs.setdefault("background", "#000000")
        kwargs.setdefault("fps", 60)
        super().__init__(**kwargs)

    def construct(self):
        p = Protein.from_file(ROOT / "examples/data/2dri.cif", chains="A")
        p.ball_and_stick(atom_scale=0.20, bond_radius=0.10).center()
        if self.representation == "cartoon":
            p.cartoon(color="#e8cda5")
        self.camera.frame(p, margin=1.12)
        self.camera.zoom(1.02)
        self.camera.depth_cue = 0.1
        self.add(
            Text("2DRI · Eased threading", position=(0.03, 0.035), font_size=25, color="#e8cda5"),
            Text(
                f"Studio / {self.representation.replace('_', ' ')} / actual geometry",
                position=(0.03, 0.075),
                font_size=19,
                color="#a8b1be",
            ),
        )
        self.wait(0.5)
        self.play(
            Thread(
                p,
                swirl=0.25,
                glow=0,
                seed=4,
            ),
            run_time=8,
        )
        self.wait(2.5)


class ThreadRiboseAtoms(ThreadRibose):
    representation = "ball_and_stick"


if __name__ == "__main__":
    for cls, name in ((ThreadRibose, "cartoon"), (ThreadRiboseAtoms, "ball-and-stick")):
        scene = cls(width=1920, height=1080)
        scene.render(OUT / f"2dri-threading-{name}.mp4")
