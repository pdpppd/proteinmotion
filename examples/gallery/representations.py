"""Cartoon, ribbon, ball-and-stick, and surface views of one protein.

Input: ubiquitin, PDB 1UBQ chain A. Each Representation() call blends the
current view into the next one; the surface is a 0.6 Å voxel SES.
"""

from pathlib import Path

from proteinmotion import Protein, ProteinScene, Representation, StudioLook, Text

DATA = Path(__file__).resolve().parent.parent / "data"


class Representations(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="studio", material="medium", effects="subtle")

    def construct(self):
        protein = Protein.from_file(DATA / "1ubq.cif").cartoon().center()
        self.camera.frame(protein, margin=0.9, screen_position=(0.6, 0.52))
        self.add(protein, Text("Representations", position=(0.05, 0.08), font_size=46, font="semibold"))
        caption = Text("Cartoon", position=(0.05, 0.145), font_size=28, color="#a3b3c7")
        self.add(caption)
        self.play(self.camera.animate.orbit(0.4), run_time=1.5)
        for name, label, options in (
            ("ribbon", "Ribbon", {}),
            ("ball_and_stick", "Ball and stick", {}),
            ("surface", "Solvent-excluded surface", {"kind": "ses", "grid_spacing": 0.6}),
            ("cartoon", "Cartoon", {}),
        ):
            self.play(
                Representation(protein, name, **options),
                caption.animate.set_text(label),
                self.camera.animate.orbit(0.35),
                run_time=1.5,
            )
            self.wait(0.6)
