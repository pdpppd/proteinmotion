# Numerical values, plots, and density

Use ProteinMotion 0.9+ for these APIs. Import the names below from `proteinmotion`.

## Numerical residue values

```python
values = ResidueValues.b_factors(protein)
scale = ColorScale(0, 40)
protein.color_by(values, scale=scale, thickness=(0.6, 1.8))
self.add(protein, ColorLegend(scale, title=values.name, unit=values.unit))
```

For an animated change, use `ColorByProperty(protein, values, scale=scale, thickness=(0.6, 1.8), residue_delay=0.01)` inside `play`. Color applies to all molecular representations. Thickness scales the cartoon cross-section only. Missing values use the scale's `missing` color and normal thickness.

`ResidueValues(protein, array, name="...", unit="...")` needs one value per topology residue, including retained ligands. Use `ResidueValues.from_mapping(protein, {("A", 10): value})` for a partial selection; add insertion codes to ambiguous keys. Missing residues become NaN.

`ResidueValues.rmsf(protein, alignment=region)` aligns frames to the first frame, then computes fluctuations about the mean. It uses Cα atoms by default and reads one trajectory frame at a time. Use `align=False` for already aligned data. Unwrap periodic MD coordinates first. Describe NMR variation as ensemble variation and label state indices rather than simulation time. Label B-factor fields as confidence only when the source file defines them that way.

## Synchronized overlays

```python
helix = protein.select(chain="A", residues=(23, 34))
trace = TimeSeriesPlot.distance(
    protein.select(residues=5, atoms="CA"),
    protein.select(residues=70, atoms="CA"),
    title="Cα distance",
)
self.add(trace, ContactMap(protein, selection=helix), SequenceTrack(protein, selection=helix))
self.play(PlayTrajectory(protein), run_time=8)
```

Use actual selections in the supplied structure. Distance traces measure centroids before scene transforms. `TimeSeriesPlot(times, values, protein=protein)` requires one sample per trajectory state. Without `protein`, the cursor uses scene seconds. A `live_value` callable can measure current interpolated coordinates; it must be deterministic and free of side effects. NaN leaves a trace gap. Labels should include measurement units.

`ContactMap` is a live binary Cα contact map, with an 8 Å default cutoff and nearby residues excluded. Use `region=` to limit its quadratic calculation for large proteins. `SequenceTrack` follows current residue colors, or accepts fixed `values=` and `scale=`. The same `selection` can mark a 3D region and both overlays. Panel `position` and `size` are viewport fractions; reserve space to keep the molecule visible.

## Maps and slices

```python
density = DensityMap.from_file("map.mrc")
local = density.crop(protein.select(residues=(23, 34)), padding=3)
shell = local.isosurface(1.5, units="sigma", opacity=0.3, follow=protein)
section = local.slice("z", 0.2, follow=protein)
self.add(shell, section)
self.play(section.animate.set_slice(0.8), run_time=3)
```

Confirm that the structure and map originally share coordinates. `follow=protein` applies the protein's scene transforms to the map. Density stays fixed during protein deformation. File axes, spacing, cell geometry, and origins are preserved. The default origin convention prefers nonzero ORIGIN, otherwise grid starts; use `origin="header"` or `"start"` when needed. Full crystallographic grids support periodic crops. Crops preserve the full map's sigma statistics.

For raw arrays, use `(x,y,z)` order and supply physical spacing/origin. Contours in `units="sigma"` use mean + level × standard deviation; absolute contours use raw values. Slice positions span 0–1 along a grid axis. Fixed contour meshes are cached; changing levels runs CPU marching cubes. Crop large maps and use `step_size` to reduce work. Inspect map alignment, contour coverage, slice contrast, and overlays in actual rendered frames.

The website has complete scripts with rendered output:
- [Numerical properties](https://pdpppd.github.io/proteinmotion/docs/numerical-properties/)
- [Synchronized plots](https://pdpppd.github.io/proteinmotion/docs/synchronized-plots/)
- [Density maps](https://pdpppd.github.io/proteinmotion/docs/density-maps/)
