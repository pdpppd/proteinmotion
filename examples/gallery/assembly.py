"""A 21-chain chaperonin: GroEL–GroES, colored by chain.

Input: GroEL–GroES–ADP7, PDB 1AON: 14 GroEL and 7 GroES chains, about 8,000
residues. The camera turns from the side view to look down the barrel.
"""

from pathlib import Path

from proteinmotion import Protein, ProteinScene, SetOpacity, StudioLook, Text

DATA = Path(__file__).resolve().parent.parent / "data"


class LargeAssembly(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="studio", material="medium", effects="subtle")

    def construct(self):
        protein = Protein.from_file(DATA / "1aon.cif", chains=None).cartoon(color="chain").center()
        subunit = protein.select(chain="A")
        self.camera.frame(protein, margin=0.95, screen_position=(0.6, 0.52))
        self.add(protein, Text("Large assemblies", position=(0.05, 0.08), font_size=46, font="semibold"))
        caption = Text(
            "GroEL–GroES · PDB 1AON · 21 chains", position=(0.05, 0.145), font_size=28, color="#a3b3c7"
        )
        self.add(caption)
        self.play(self.camera.animate.orbit(0.8), run_time=3)
        self.play(
            SetOpacity(~subunit, 0.12), caption.animate.set_text("One GroEL subunit, chain A"), run_time=1.5
        )
        self.play(self.camera.animate.orbit(0.6), run_time=2.5)
        self.play(SetOpacity(protein, 1, scope="residues"), self.camera.animate.orbit(0, 1.2), run_time=3)
        self.wait(0.5)
