"""Deform coordinates with any function, then morph back.

Input: ubiquitin, PDB 1UBQ. Deform passes the current coordinates (N × 3, in
Å) to a Python function each frame; Morph returns to the saved array.
"""

from pathlib import Path

import numpy as np

from proteinmotion import Deform, Morph, Protein, ProteinScene, StudioLook, Text

DATA = Path(__file__).resolve().parent.parent / "data"


def twist(xyz):
    """Turn each slab about the y axis in proportion to its height."""
    center = xyz.mean(axis=0)
    local = xyz - center
    angle = 0.1 * local[:, 1]
    cos, sin = np.cos(angle), np.sin(angle)
    x = cos * local[:, 0] + sin * local[:, 2]
    z = -sin * local[:, 0] + cos * local[:, 2]
    return center + np.column_stack((x, local[:, 1], z))


class Deformation(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="studio", material="medium", effects="subtle")

    def construct(self):
        protein = Protein.from_file(DATA / "1ubq.cif").cartoon().center()
        rest = protein.positions.copy()
        self.camera.frame(protein, margin=1.0, screen_position=(0.6, 0.52))
        self.add(protein, Text("Deformations", position=(0.05, 0.08), font_size=46, font="semibold"))
        caption = Text("Deform(protein, twist)", position=(0.05, 0.145), font_size=28, color="#a3b3c7")
        self.add(caption)
        self.wait(0.5)
        self.play(Deform(protein, twist), run_time=2.5)
        self.play(self.camera.animate.orbit(0.6), run_time=2)
        self.play(
            caption.animate.set_text("Morph(protein, rest, align=False)"),
            Morph(protein, rest, align=False),
            run_time=2.5,
        )
        self.play(self.camera.animate.orbit(0.4), run_time=1.5)
