"""A ubiquitin NMR ensemble, contact map, sequence, and synchronized distance trace.

2K39 contains deposited NMR models. Interpolation illustrates ensemble variation;
state indices do not represent elapsed physical time. Frames are aligned on Cα
residues 1–70 to remove overall translation and rotation.
"""

from pathlib import Path

from proteinmotion import (
    ColorLegend,
    ColorScale,
    ContactMap,
    PlayTrajectory,
    Protein,
    ProteinScene,
    ResidueValues,
    SequenceTrack,
    Text,
    TimeSeriesPlot,
    linear,
)

DATA = Path(__file__).parent / "data"


class SynchronizedPlots(ProteinScene):
    def construct(self):
        protein = Protein.from_file(DATA / "2k39.cif").center()
        core = protein.select(chain="A", residues=(1, 70), atoms="CA")
        protein.trajectory = protein.trajectory.aligned(indices=core.atom_indices)
        values = ResidueValues.rmsf(protein, align=False)
        scale = ColorScale(0, 8)
        protein.color_by(values, scale=scale)
        selected = protein.select(chain="A", residues=(23, 34))
        self.add(protein, selected.highlight(style="box", padding=1.0, color="#f5d477"))
        self.camera.frame(protein, margin=1.22)
        # Offset the camera target to leave room for the plots on the right.
        self.camera.target += [19, 0, 0]
        self.add(Text("An NMR ensemble with synchronized plots", position=(0.05, 0.05), font_size=35))
        self.add(Text("2K39 · deposited model order", position=(0.05, 0.105), font_size=24))
        self.add(ContactMap(protein, selection=selected, position=(0.64, 0.16), size=(0.32, 0.36)))
        self.add(
            TimeSeriesPlot.distance(
                protein.select(residues=5, atoms="CA"),
                protein.select(residues=70, atoms="CA"),
                title="Cα 5 → Cα 70",
                position=(0.64, 0.54),
                size=(0.32, 0.29),
            )
        )
        self.add(
            SequenceTrack(
                protein,
                selection=selected,
                title="Helix: residues 23–34",
                position=(0.05, 0.85),
                size=(0.91, 0.13),
            )
        )
        self.add(
            ColorLegend(
                scale, title="Aligned ensemble RMSF", unit="Å", position=(0.05, 0.68), size=(0.29, 0.12)
            )
        )
        self.wait(0.5)
        self.play(PlayTrajectory(protein), run_time=9, rate_func=linear)
        self.wait(0.5)
