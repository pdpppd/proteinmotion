"""Turn φ and ψ of one residue and watch it move on a Ramachandran plot.

Input: a 12-residue alanine peptide built as a β strand. Atoms rotate about
the N–Cα (φ) and Cα–C (ψ) bonds, so bond lengths and angles stay fixed.
Torsion angles are in degrees.
"""

from proteinmotion import (
    FadeIn,
    Protein,
    ProteinScene,
    RamachandranPlot,
    SetTorsions,
    StudioLook,
    Text,
    Unwrite,
    Write,
)


class TorsionMarkers(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="studio", material="medium", effects="subtle")

    def construct(self):
        peptide = Protein.build("AAAAAAAAAAAA", "strand").cartoon()
        middle = peptide.select(residues=6)
        middle.show_atoms()
        self.camera.phi = 0.3
        self.camera.frame(peptide.select(residues=(3, 9)), margin=1.0, screen_position=(0.3, 0.55))
        plot = RamachandranPlot(
            peptide, highlight=middle, position=(0.62, 0.14), size=(0.35, 0.7)
        ).with_panel()
        self.add(peptide, Text("Torsion angles", position=(0.05, 0.08), font_size=46, font="semibold"))
        phi = middle.torsion_marker("phi", radius=1.4, color="#f5d477")
        psi = middle.torsion_marker("psi", radius=1.4, color="#7fd4ff")
        self.play(FadeIn(plot), Write(phi), run_time=1.2)
        self.play(SetTorsions(middle, phi=-60, anchor="n"), run_time=2.5)
        self.play(Unwrite(phi), Write(psi), run_time=1)
        self.play(SetTorsions(middle, psi=-45, anchor="n"), run_time=2.5)
        self.play(self.camera.animate.orbit(0.6), run_time=2)
