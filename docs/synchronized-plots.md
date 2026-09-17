# Plots that follow the movie

Add a time-series plot, a live Cα contact map, or a sequence strip to a scene. They render as vector overlays in both backends. Positions and sizes use fractions of the viewport, measured from its upper-left corner.

## Complete example and output

The example aligns the 2K39 ubiquitin NMR ensemble, colors it by RMSF, and plays the deposited models. The same helix selection is marked in 3D, in the contact map, and above the sequence. The horizontal axis uses state indices because an NMR ensemble has no simulation time.

```python output=synchronized-plots
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
        self.camera.target += [12, 0, 0]
        self.add(Text("An NMR ensemble with synchronized plots", position=(0.05, 0.05), font_size=35))
        self.add(Text("2K39 · deposited model order", position=(0.05, 0.105), font_size=24))
        self.add(ContactMap(protein, selection=selected, position=(0.69, 0.16), size=(0.27, 0.36)))
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
```

[Download the script](synchronized_plots.py). Run it from a repository checkout:

```bash
proteinmotion render examples/synchronized_plots.py SynchronizedPlots --fps 60 -o plots.mp4
```

## Plot a measured quantity

```python
from proteinmotion import TimeSeriesPlot

plot = TimeSeriesPlot(
    [0, 1, 2, 3], [0.2, 0.7, 0.4, 0.9],
    title="Backbone RMSD", xlabel="Time (ns)", ylabel="RMSD (Å)",
    protein=protein,
)
self.add(plot)
```

With `protein=protein`, supply one value and one time for each state in `protein.trajectory`. The cursor follows the current trajectory state, including eased interpolation and reverse playback. The example above requires a four-state trajectory. Compute RMSD or other measurements with your analysis tools and supply the resulting arrays.

Omit `protein` to use scene time in seconds as the cursor coordinate. Times must be finite and strictly increasing. A `NaN` value leaves a gap in the trace. `ylim=(low, high)` fixes the vertical range. `reveal=True` shows samples up to the cursor. Values outside `ylim` are clipped at the plot edge. Long traces are reduced for display while retaining the minimum and maximum of each sample group.

## Distance traces

```python
first = protein.select(chain="A", residues=5, atoms="CA")
second = protein.select(chain="A", residues=70, atoms="CA")
plot = TimeSeriesPlot.distance(first, second, title="Cα distance")
self.add(plot)
```

This helper reads the trajectory and measures the distance between the two selection centroids in each state. Both selections must belong to the same protein. Measurements use ångströms before scene scaling or rotation. Supply `times=` for physical trajectory times; otherwise the axis shows state indices.

The marker measures the current interpolated coordinates. The line connects measurements at the supplied states. These can differ between states because interpolating coordinates can change distance nonlinearly. For another live measurement, pass a deterministic, side-effect-free `live_value` function to `TimeSeriesPlot`.

## Contact maps and sequence tracks

```python
from proteinmotion import ContactMap, SequenceTrack

helix = protein.select(chain="A", residues=(23, 34))
self.add(helix.highlight(style="box", color="#f5d477"))
self.add(ContactMap(protein, selection=helix, cutoff=8, min_separation=3))
self.add(SequenceTrack(protein, selection=helix))
```

A contact means that two Cα atoms are at most `cutoff` ångströms apart. The diagonal and nearby residues within each chain are excluded using `min_separation`. Contacts update from the coordinates of every displayed frame. Use `region=` to limit the map to a subset; the default limit is 512 Cα residues because the calculation grows with the square of the residue count. Labels use PDB author numbers. Matrix rows follow topology order.

`SequenceTrack` uses the protein's current residue colors, including color animations. Pass `values=` and `scale=` to show a separate fixed measurement. Letters appear when tiles are wide enough. `region=` restricts the displayed sequence. Sharing a `selection` with a 3D highlight gives the views matching residue markers.

These overlays export to the movie. They also update when you seek to a frame in an interactive preview. Use `FadeIn`, `FadeOut`, or `.animate.move_to(...)` to arrange them on the timeline.
