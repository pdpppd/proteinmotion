"""A distance ruler that follows an NMR ensemble, with a synchronized trace.

Input: ubiquitin NMR ensemble, PDB 2K39 (116 models), aligned on Cα 1–70.
The ruler measures Gly10 Cα to Gly47 Cα, two loop residues, in each interpolated model. Interpolating
between deposited models illustrates ensemble variation, not dynamics.
"""

from pathlib import Path

from proteinmotion import Distance, PlayTrajectory, Protein, ProteinScene, StudioLook, Text, Write, linear

DATA = Path(__file__).resolve().parent.parent / "data"


class DistanceRuler(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="studio", material="medium", effects="subtle")

    def construct(self):
        protein = Protein.from_file(DATA / "2k39.cif").cartoon().center()
        core = protein.select(residues=(1, 70), atoms="CA")
        protein.trajectory = protein.trajectory.aligned(indices=core.atom_indices)
        start = protein.select(residues=10, atoms="CA")
        end = protein.select(residues=47, atoms="CA")
        ruler = Distance(start, end, color="#ffc46b", font_size=36, style="solid", line_width=2.5)
        trace = ruler.plot(title="Gly10 Cα → Gly47 Cα", position=(0.64, 0.2), size=(0.32, 0.32)).with_panel()
        self.camera.frame(protein, margin=0.9, screen_position=(0.33, 0.55))
        self.add(protein, Text("Distance rulers", position=(0.05, 0.08), font_size=46, font="semibold"))
        self.add(Text("2K39 · 116 NMR models", position=(0.05, 0.145), font_size=28, color="#a3b3c7"))
        self.play(Write(ruler), Write(trace), run_time=1.2)
        self.play(
            PlayTrajectory(protein, end=40), self.camera.animate.orbit(0.6), run_time=8, rate_func=linear
        )
        self.wait(0.5)
