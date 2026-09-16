# API reference

This reference covers the public authoring API in version 0.5.0. Import these names from `proteinmotion`, except `Camera` and `Renderer`, which live in their corresponding modules.

## ProteinScene

```python
ProteinScene(width=1920, height=1080, fps=30, background="#0b1220", msaa=4)
```

Subclass `ProteinScene` and implement `construct()`. `Scene` is an alias.

| Method | Behavior |
|---|---|
| `add(*proteins)` | Add proteins, highlights, text or callouts at the current timeline position |
| `play(*animations, run_time=1.0, rate_func=None)` | Run animations concurrently, then advance the timeline |
| `wait(duration=1.0)` | Hold the current scene |
| `focus(target, run_time=1.5, margin=1.25, follow=True)` | Ease the camera to a protein or region |
| `seek(time)` | Evaluate a time deterministically; supports backward seeks |
| `render(output, codec="auto", bitrate="20M")` | Export an MP4 and return timing/adapter metadata |
| `render_frame(time=0, output=None, renderer=None)` | Render one RGBA frame; optionally save an image |
| `preview()` | Open the optional interactive native preview |

Animations in separate calls run sequentially. Concurrent writers to the same object property are rejected. Configure initial styles and camera pose before the first `play()` or `wait()`.

## Protein

```python
Protein.from_file(path, chains="A")
Protein.from_trajectory(trajectory)
```

`from_file` reads PDB/mmCIF and keeps all matching models as `protein.trajectory`. The default selection excludes waters and hydrogens; other hetero atoms are retained. Multi-model inputs must have consistent selected atom identities/order.

| Method/property | Behavior |
|---|---|
| `cartoon(color="secondary")` | Helices, sheet arrows, and tubular coils |
| `ribbon(color="rainbow", width=1.05)` | Continuous ribbon representation |
| `ball_and_stick(atom_scale=0.30, bond_radius=0.14)` | Element-colored atoms and geometric bonds |
| `center()` / `shift(vector)` / `rotate(angle, axis)` / `scale(factor)` | Configure the molecular transform |
| `positions` | Current interpolated coordinates in the protein's local frame |
| `set_positions(xyz)` | Set coordinates with unchanged topology |
| `copy()` | Make an independently animated protein |
| `with_secondary_structure(assignments)` | One `H/E/C` character per residue |
| `animate` | Build fluent transform or opacity animations |

Coordinates/radii are ångströms; angles are radians. Cartoon/ribbon color accepts `secondary`, `rainbow`, `chain`, or `#RRGGBB`. Ball-and-stick uses element colors.

## Region and RegionHighlight

```python
region = protein.select(chain="A", residues=(23, 34), atoms="CA")
region = Region(protein, atom_indices)
combined = region_a | region_b
marker = region.highlight(style="box", color="#f2ba67", padding=1.5)
```

Selections use PDB author residue numbers. Tuples are inclusive ranges; lists select explicit numbers. The chain and atom-name filters also accept lists. Empty selections raise an error. Explicit atom indices are zero-based.

`region.atom_indices`, `region.residue_indices`, `region.positions`, and `region.world_positions` expose the selected atoms/residues and live coordinates. Indices are read-only. Unions require the same parent protein.

Highlight styles are `sphere`, `box`, and `atoms`. All accept `color`, `opacity`, and `padding`; boxes also use `line_width`. Highlights follow their parent coordinates and transforms. Animate their opacity, not their transform. [More examples](regions.md).

## Text and annotations

Import `Text`, `Callout`, `ResidueLabel`, `ResidueLabels`, `Write`, and `Unwrite` from `proteinmotion`.

```python
Text(text, font_size=36, font="regular", color="#edf3fc",
     position=(0.06, 0.08), align="left", line_spacing=1.3,
     ligatures=True, opacity=1)
Callout(region, text, subtitle=None, position=(0.73, 0.25),
        font_size=32, font="semibold", color="#edf3fc",
        subtitle_color="#a3b3c7", line_color=None, line_width=1.5,
        tip="dot", clamp=True, follow_opacity=True, opacity=1)
ResidueLabel(region, text=None, offset=(24, -36),
             format="three_letter", include_chain=True, font_size=26, **callout_options)
ResidueLabels(region, offsets=None, avoid_overlap=True, **label_options)
Write(target, lag_ratio=None, stroke_width=1.5, reverse=False, rate_func=linear)
Unwrite(target, lag_ratio=None, stroke_width=1.5, reverse=False, rate_func=linear)
```

Constructor options after the positional arguments are keyword-only. Text fonts are bundled `regular` / `semibold` or a TTF/OTF path. Font sizes, outline widths, and label offsets use pixels at 1080p height. Positions are normalized viewport coordinates with `(0, 0)` at the top-left. `Text.to_corner("UL", buff=0.06)` supports all four corners.

Convenience methods: `region.callout(text, **options)`, `region.label(text=None, **options)` for one residue, and `protein.label_residues(chain=None, residues=None, **options)`. Automatic names use PDB author numbers and insertion codes. Group offsets accept residue-number or `(chain, number, insertion_code)` keys.

Annotations support `set_opacity()` and `.animate.set_opacity()`. Text/callouts support `move_to()`, `shift()` and animated equivalents. A single `ResidueLabel` supports `set_offset()` and `.animate.set_offset()` instead. `Write` and `Unwrite` operate on whole annotation objects; configure their child text styles before adding them to a scene. [Full guide and runnable example](text.md).

## Camera

```python
self.camera.frame(protein, margin=1.25, aspect=self.width / self.height)
self.camera.focus(region, follow=True, aspect=self.width / self.height)
self.play(self.camera.animate.orbit(theta=0.5, phi=0.1), run_time=2)
self.play(self.camera.animate.zoom(1.3), run_time=1)
```

`frame` and `focus` are immediate setup methods. Use `self.focus(...)`, `Focus(...)`, or `camera.animate.focus(...)` for timeline animation. `theta`, `phi`, and `fov` are radians; `distance` and `target` control camera position. `depth_cue=0` disables distance fog. Region tracking follows the center with fixed zoom.

## Animations

| Constructor | Purpose |
|---|---|
| `Write(annotation)` / `Unwrite(annotation)` | Draw or erase vector glyphs and callout leaders |
| `Rotate(protein, angle, axis=(0, 1, 0))` | Rotate around the molecular centroid |
| `FadeIn(target)` / `FadeOut(target)` | Animate opacity with smooth transparency |
| `Representation(protein, name)` | Transition to `cartoon`, `ribbon`, or `ball_and_stick` |
| `Morph(protein, target, align=True)` | Morph matching atom topology or ordered coordinate arrays |
| `Deform(protein, function)` | Transform coordinates through a callable |
| `Focus(camera, region, margin=1.25, aspect=16/9, follow=True)` | Animate camera focus; evaluate after molecular motion |
| `PlayTrajectory(protein, trajectory=None, start=0, end=None, state_easing=linear)` | Interpolate between trajectory models |
| `BackboneMorph(source, target, match=..., residue_delay=0.025)` | Contact-guided morph between different proteins |

Default motion easing is quintic `smooth`. Trajectory playback uses a linear clip clock by default. `state_easing` changes the blend inside each adjacent model pair; `rate_func` changes progress through the whole clip. Built-ins include `linear`, `smooth`, `ease_in_out_sine`, and `there_and_back`.

For `BackboneMorph`, use `motion_easing` for individual residue motion; its timeline must stay linear to preserve delays in seconds. See [matching options and limitations](morphing.md).

## Trajectory

```python
Trajectory(frames, topology=None, units="angstrom")
Trajectory.from_npy(path, topology=None, units="angstrom")
Trajectory.from_mdanalysis(topology_file, trajectory_file,
                          selection="protein", stride=1)
trajectory.aligned(reference=None, indices=None)
trajectory.frame(index)
len(trajectory)
```

Coordinates have shape `(frames, atoms, 3)`. NumPy files are memory mapped; MDAnalysis files are read lazily. `aligned()` removes overall rigid motion with a proper fit on the requested atom indices. Preprocess periodic boundaries before loading. [NMR and MD guide](trajectories.md).

## Contact matching

```python
match = match_backbones(source, target, max_contact_error=0.30,
                        search_seconds=15, max_candidates=6000)
match.save("correspondence.json")
match = ContactMatch.load("correspondence.json")
print(match.report)
```

`ContactMatch.from_pairs(source, target, source_indices, target_indices)` accepts manually specified, monotone topology residue indices (not atom indices or PDB numbers). The bounded optimizer maximizes matched residue count under a configurable contact-error limit, then minimizes error. Global optimality must be checked in the report; it is not promised for large structures.

## Command line

```bash
proteinmotion doctor
proteinmotion render scene.py SceneName -o film.mp4 --fps 60
proteinmotion still scene.py SceneName -o frame.png --time 4.5
proteinmotion preview scene.py SceneName
```

Shared options: `--width`, `--height`, `--fps`, `--msaa 1|4`. Rendering also accepts `--codec` and `--bitrate`. Preview controls: drag to orbit, wheel to zoom, Space to pause, arrows to seek, Home to rewind, R to reset the camera, Escape to close.
