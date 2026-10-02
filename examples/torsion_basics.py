"""Turn one ψ angle, then wind a whole peptide into an α helix.

A 12-residue alanine peptide is built as an extended β strand. ψ of Ala6 turns to
−45° with its atoms shown over the cartoon; then every residue moves to φ = −57°,
ψ = −47°, starting at the N terminus. DSSP assigns the helix from the final
coordinates, and the cartoon blends from coil to helix as the angles change.
"""

from proteinmotion import (
    FadeIn,
    Protein,
    ProteinScene,
    RamachandranPlot,
    SetTorsions,
    Text,
    Unwrite,
    Write,
)


class TorsionBasics(ProteinScene):
    def construct(self):
        peptide = Protein.build("AAAAAAAAAAAA", "strand").cartoon()
        middle = peptide.select(residues=6)
        middle.show_atoms()
        self.add(peptide)
        self.camera.frame(peptide, margin=1.0, aspect=self.width / self.height, screen_position=(0.32, 0.55))
        self.camera.phi = 0.3
        plot = RamachandranPlot(peptide, highlight=middle, position=(0.62, 0.14), size=(0.35, 0.7))
        self.add(Text("Torsion angles", font_size=48, font="semibold", position=(0.05, 0.07)))
        marker = middle.torsion_marker("psi", radius=1.6)
        self.play(FadeIn(plot), Write(marker), run_time=1.5)
        self.play(SetTorsions(middle, psi=-45, anchor="n"), run_time=2.5)
        self.play(Unwrite(marker), run_time=0.6)
        self.play(SetTorsions(peptide, conformation="helix", stagger=0.5), run_time=5)
        self.play(self.camera.animate.orbit(0.8), run_time=2.5)
