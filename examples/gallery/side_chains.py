"""Grow every side chain from its Cα, N terminus to C terminus.

Input: ubiquitin, PDB 1UBQ. ShowSideChains joins each side chain to the
cartoon at Cα; residue_delay staggers the residues in seconds.
"""

from pathlib import Path

from proteinmotion import HideSideChains, Protein, ProteinScene, ShowSideChains, StudioLook, Text

DATA = Path(__file__).resolve().parent.parent / "data"


class SideChains(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="studio", material="medium", effects="subtle")

    def construct(self):
        protein = Protein.from_file(DATA / "1ubq.cif").cartoon().center()
        self.camera.frame(protein, margin=0.9, screen_position=(0.6, 0.52))
        self.add(protein, Text("Side chains", position=(0.05, 0.08), font_size=46, font="semibold"))
        caption = Text(
            "ShowSideChains(protein, residue_delay=0.04)",
            position=(0.05, 0.145),
            font_size=28,
            color="#a3b3c7",
        )
        self.add(caption)
        self.wait(0.5)
        self.play(ShowSideChains(protein, residue_delay=0.04), self.camera.animate.orbit(0.6), run_time=4)
        self.play(self.camera.animate.orbit(0.6), run_time=2)
        self.play(
            caption.animate.set_text("HideSideChains(protein, reverse=True)"),
            HideSideChains(protein, residue_delay=0.02, reverse=True),
            self.camera.animate.orbit(0.3),
            run_time=2.5,
        )
        self.wait(0.3)
