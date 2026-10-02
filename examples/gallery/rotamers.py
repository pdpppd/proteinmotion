"""Spin side-chain χ angles through full turns.

Input: ubiquitin, PDB 1UBQ. RotateTorsions turns χ1 and χ2 of Leu8, Ile44,
and Val70 by the given amounts in degrees. Rotation alone can pass atoms
through neighbors; this shows the geometry of a χ angle, not an allowed path.
"""

from pathlib import Path

from proteinmotion import Protein, ProteinScene, RotateTorsions, ShowSideChains, StudioLook, Text, Write

DATA = Path(__file__).resolve().parent.parent / "data"


class Rotamers(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="studio", material="medium", effects="subtle")

    def construct(self):
        protein = Protein.from_file(DATA / "1ubq.cif").cartoon().center()
        patch = protein.select(residues=[8, 44, 70])
        self.camera.theta = 0.6
        self.camera.frame(protein, margin=0.85, screen_position=(0.6, 0.52))
        self.add(protein, Text("Side-chain rotamers", position=(0.05, 0.08), font_size=46, font="semibold"))
        caption = Text(
            "RotateTorsions(patch, chi1=360)", position=(0.05, 0.145), font_size=28, color="#a3b3c7"
        )
        self.add(caption)
        self.play(ShowSideChains(patch, residue_delay=0.2), run_time=1.2)
        self.focus(patch, margin=2.0, screen_position=(0.6, 0.52), run_time=1.5)
        marker = protein.select(residues=44).torsion_marker("chi1", radius=1.2, color="#f5d477")
        self.play(Write(marker), run_time=0.8)
        self.play(RotateTorsions(patch, chi1=360), run_time=3.5)
        self.play(caption.animate.set_text("RotateTorsions(patch, chi2=360)"), run_time=0.5)
        self.play(RotateTorsions(patch, chi2=360), self.camera.animate.orbit(0.4), run_time=3.5)
