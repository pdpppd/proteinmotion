"""Fade proteins in and out, rotate them, and move them through the scene.

Inputs: ubiquitin (PDB 1UBQ) and the protein G B1 domain (PDB 1PGA).
Transforms move the whole model; distances are Å and angles are radians.
"""

from pathlib import Path

import numpy as np

from proteinmotion import FadeIn, FadeOut, Protein, ProteinScene, Rotate, StudioLook, Text

DATA = Path(__file__).resolve().parent.parent / "data"


class Transforms(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="studio", material="medium", effects="subtle")

    def construct(self):
        ubiquitin = Protein.from_file(DATA / "1ubq.cif").cartoon().center()
        protein_g = Protein.from_file(DATA / "1pga.cif").cartoon(color="#8fb8ff").center().shift((70, 0, 0))
        self.camera.frame(ubiquitin, margin=1.15, screen_position=(0.55, 0.52))
        self.add(Text("Fades and transforms", position=(0.05, 0.08), font_size=46, font="semibold"))
        caption = Text("FadeIn(ubiquitin)", position=(0.05, 0.145), font_size=28, color="#a3b3c7")
        self.add(caption)
        self.play(FadeIn(ubiquitin), run_time=1)
        self.play(caption.animate.set_text("Rotate(ubiquitin, np.pi)"), Rotate(ubiquitin, np.pi), run_time=2)
        self.play(
            caption.animate.set_text("Rotate(ubiquitin, np.pi / 2, axis=(1, 0, 0))"),
            Rotate(ubiquitin, np.pi / 2, axis=(1, 0, 0)),
            run_time=1.5,
        )
        self.play(
            caption.animate.set_text("animate.shift((x, y, z))"),
            ubiquitin.animate.shift((-18, 0, 0)),
            FadeIn(protein_g),
            protein_g.animate.shift((-52, 0, 0)),
            run_time=2.5,
        )
        self.play(caption.animate.set_text("animate.scale(1.4)"), protein_g.animate.scale(1.4), run_time=1.5)
        self.play(
            caption.animate.set_text("FadeOut(…)"), FadeOut(ubiquitin), FadeOut(protein_g), run_time=1.5
        )
