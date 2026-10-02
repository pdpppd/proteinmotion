"""Play through the models of an NMR ensemble, side chains included.

Input: ubiquitin, PDB 2K39, 116 deposited models aligned on Cα 1–70.
PlayTrajectory interpolates between consecutive models; this illustrates the
spread of the ensemble, not motion over time.
"""

from pathlib import Path

from proteinmotion import PlayTrajectory, Protein, ProteinScene, ShowSideChains, StudioLook, Text, linear

DATA = Path(__file__).resolve().parent.parent / "data"


class NMREnsemble(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="studio", material="medium", effects="subtle")

    def construct(self):
        protein = Protein.from_file(DATA / "2k39.cif").cartoon().center()
        core = protein.select(residues=(1, 70), atoms="CA")
        protein.trajectory = protein.trajectory.aligned(indices=core.atom_indices)
        self.camera.frame(protein, margin=0.95, screen_position=(0.6, 0.52))
        self.add(protein, Text("NMR ensembles", position=(0.05, 0.08), font_size=46, font="semibold"))
        caption = Text(
            "PlayTrajectory(protein) · 2K39, 116 models",
            position=(0.05, 0.145),
            font_size=28,
            color="#a3b3c7",
        )
        self.add(caption)
        self.play(
            PlayTrajectory(protein, end=24), self.camera.animate.orbit(0.4), run_time=5, rate_func=linear
        )
        self.play(
            ShowSideChains(protein, residue_delay=0.01),
            PlayTrajectory(protein, start=24, end=29),
            caption.animate.set_text("Side chains follow the models"),
            run_time=1.5,
            rate_func=linear,
        )
        self.play(
            PlayTrajectory(protein, start=29, end=41),
            self.camera.animate.orbit(0.4),
            run_time=5,
            rate_func=linear,
        )
