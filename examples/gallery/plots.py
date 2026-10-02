"""An NMR ensemble with a live contact map, distance trace, and sequence track.

Input: ubiquitin NMR ensemble, PDB 2K39, aligned on Cα 1–70 and colored by
per-residue RMSF across the models. The plots follow the interpolated state;
state order is the deposited model order, not physical time.
"""

from pathlib import Path

from proteinmotion import (
    ColorScale,
    ContactMap,
    PlayTrajectory,
    Protein,
    ProteinScene,
    ResidueValues,
    SequenceTrack,
    StudioLook,
    Text,
    TimeSeriesPlot,
    linear,
)

DATA = Path(__file__).resolve().parent.parent / "data"


class SynchronizedPlots(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="studio", material="medium", effects="subtle")

    def construct(self):
        protein = Protein.from_file(DATA / "2k39.cif").cartoon().center()
        core = protein.select(residues=(1, 70), atoms="CA")
        protein.trajectory = protein.trajectory.aligned(indices=core.atom_indices)
        protein.color_by(ResidueValues.rmsf(protein, align=False), scale=ColorScale(0, 8))
        tail = protein.select(residues=(66, 76))
        self.camera.frame(protein, margin=0.85, screen_position=(0.31, 0.48))
        self.add(protein, Text("Synchronized plots", position=(0.05, 0.08), font_size=46, font="semibold"))
        self.add(
            Text("2K39 · colored by ensemble RMSF", position=(0.05, 0.145), font_size=28, color="#a3b3c7")
        )
        self.add(ContactMap(protein, selection=tail, position=(0.62, 0.08), size=(0.34, 0.4)).with_panel())
        self.add(
            TimeSeriesPlot.distance(
                protein.select(residues=1, atoms="CA"),
                protein.select(residues=76, atoms="CA"),
                title="N to C terminus (Cα)",
                position=(0.62, 0.52),
                size=(0.34, 0.28),
            ).with_panel()
        )
        self.add(
            SequenceTrack(
                protein,
                selection=tail,
                title="C-terminal tail 66–76",
                position=(0.05, 0.84),
                size=(0.91, 0.12),
            )
        )
        self.wait(0.5)
        self.play(PlayTrajectory(protein, end=30), run_time=9, rate_func=linear)
        self.wait(0.5)
