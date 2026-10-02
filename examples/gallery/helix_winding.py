"""Wind a β strand into an α helix, residue by residue.

Input: Baldwin's peptide AEAAAKEAAAKEAAAKA, built as an extended strand.
Backbone atoms are drawn over the cartoon. SetTorsions moves every φ, ψ to α-helix values (−57°, −47°) from the N
terminus; DSSP of the new coordinates turns the cartoon into a helix.
"""

from proteinmotion import FadeIn, Protein, ProteinScene, RamachandranPlot, SetTorsions, StudioLook, Text


class HelixWinding(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="studio", material="medium", effects="subtle")

    def construct(self):
        peptide = Protein.build("AEAAAKEAAAKEAAAKA", "strand").cartoon()
        peptide.select(atoms=["N", "CA", "C", "O"]).show_atoms()
        self.camera.phi = 0.25
        self.camera.frame(peptide, margin=1.0, screen_position=(0.33, 0.55))
        plot = RamachandranPlot(peptide, position=(0.62, 0.14), size=(0.35, 0.7)).with_panel()
        self.add(peptide, Text("Secondary structure", position=(0.05, 0.08), font_size=46, font="semibold"))
        caption = Text(
            'SetTorsions(peptide, conformation="helix", stagger=0.5)',
            position=(0.05, 0.145),
            font_size=28,
            color="#a3b3c7",
        )
        self.add(caption)
        self.play(FadeIn(plot), run_time=1)
        self.play(SetTorsions(peptide, conformation="helix", stagger=0.5), run_time=6)
        self.focus(peptide, margin=0.75, screen_position=(0.33, 0.55), run_time=1.5)
        self.play(self.camera.animate.orbit(1.5), run_time=3)
