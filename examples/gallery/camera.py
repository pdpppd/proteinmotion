"""Orbit, zoom, and focus the camera on a region.

Input: ubiquitin, PDB 1UBQ chain A. Camera angles are radians; self.focus()
eases the camera onto a selection and keeps it centered there.
"""

from pathlib import Path

from proteinmotion import Protein, ProteinScene, StudioLook, Text

DATA = Path(__file__).resolve().parent.parent / "data"


class CameraMoves(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="studio", material="medium", effects="subtle")

    def construct(self):
        protein = Protein.from_file(DATA / "1ubq.cif").cartoon().center()
        helix = protein.select(residues=(23, 34))
        self.camera.frame(protein, margin=0.9, screen_position=(0.6, 0.52))
        self.add(protein, Text("Camera moves", position=(0.05, 0.08), font_size=46, font="semibold"))
        caption = Text(
            "camera.animate.orbit(theta=1.2, phi=0.3)", position=(0.05, 0.145), font_size=28, color="#a3b3c7"
        )
        self.add(caption)
        self.play(self.camera.animate.orbit(theta=1.2, phi=0.3), run_time=2.5)
        self.play(
            caption.animate.set_text("camera.animate.zoom(1.4)"), self.camera.animate.zoom(1.4), run_time=1.5
        )
        self.wait(0.3)
        self.play(
            caption.animate.set_text("self.focus(helix)"), helix.animate.set_color("#ff8fb1"), run_time=0.8
        )
        self.focus(helix, margin=1.3, screen_position=(0.6, 0.52), run_time=1.5)
        self.play(self.camera.animate.orbit(theta=0.8), run_time=2)
        self.play(caption.animate.set_text("self.focus(protein)"), run_time=0.5)
        self.focus(protein, margin=0.9, screen_position=(0.6, 0.52), run_time=1.5)
        self.wait(0.5)
