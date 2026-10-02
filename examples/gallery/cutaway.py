"""Cut a window through the surface onto a buried ligand.

Input: E. coli ribose-binding protein with bound β-D-ribose (RIP A272),
PDB 2DRI, as a 0.6 Å voxel SES. The cone-shaped cutaway faces the camera and
follows it during the orbit; the atomic coordinates are not changed.
"""

from pathlib import Path

from proteinmotion import Conceal, Focus, Protein, ProteinScene, Reveal, StudioLook, Text

DATA = Path(__file__).resolve().parent.parent / "data"


class Cutaway(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="dramatic", material="medium", effects="subtle")

    def construct(self):
        protein = Protein.from_file(DATA / "2dri.cif", chains="A").center()
        protein.surface(kind="ses", grid_spacing=0.6, color="#8fe3c0")
        ligand = protein.select(resname="RIP")
        ligand.filter(element="C").set_color("#ffc46b")
        self.camera.theta, self.camera.phi = 1.5, 0.25
        self.camera.frame(protein, margin=1.1, screen_position=(0.6, 0.53))
        self.add(protein, Text("Cutaways", position=(0.05, 0.08), font_size=46, font="semibold"))
        caption = Text(
            "A buried ribose in its binding cleft", position=(0.05, 0.145), font_size=28, color="#a3b3c7"
        )
        self.add(caption)
        self.wait(1)
        self.play(
            Reveal(self.camera, ligand, window=1.8, padding=3, surface_keep=0.2),
            Focus(self.camera, ligand, margin=5, screen_position=(0.6, 0.53)),
            caption.animate.set_text("Reveal(camera, ligand)"),
            run_time=2,
        )
        self.play(self.camera.animate.orbit(0.9), run_time=4)
        self.play(
            Conceal(self.camera),
            Focus(self.camera, protein, margin=1.1, screen_position=(0.6, 0.53)),
            caption.animate.set_text("Conceal(camera)"),
            run_time=1.5,
        )
        self.wait(0.5)
