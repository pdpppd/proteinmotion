# Cutaways and depth tunnels

A cutaway shows a selection that other parts of the molecule hide. `Reveal` opens a window onto the selection; `Conceal` closes it. No atoms move: geometry in front of the selection fades for display only. A tunnel also shows how deep the selection lies, with a ring every 5 Å.

Cutaways, tunnels, and the depth-cue animation are drawn by the native renderer. EEVEE renders the full molecule without the cutaway.

## Example and output

This film uses GroEL–GroES, PDB 1AON. It compares the ADP pocket of subunit A, 21 Å beneath the outer surface of the ring, with residues on the ring's inner wall, 44 Å beneath it. It then shows the tunnel in a side view with the rest of the complex ghosted, flies down it, and carves the molecular surface.

```bash output=gallery-depth-tunnels
proteinmotion render examples/depth_tunnels.py DepthTunnels --fps 60 -o tunnels.mp4
```

[Source on GitHub](https://github.com/pdpppd/proteinmotion/blob/main/examples/depth_tunnels.py)

## Open a window

```python
pocket = p.select(within=4.0, of=p.select(resname="ADP", chain="A"))
self.play(Reveal(self.camera, pocket, window=1.6), run_time=2)
self.play(self.camera.animate.orbit(1.2), run_time=6)   # The window follows the camera.
self.play(Conceal(self.camera), run_time=1.5)
```

The default window is a cone from the camera to the selection, which appears as a fixed circle on the screen. Geometry between the camera and the selection fades inside it; the selection and everything behind it stay. The window stays aimed at the selection as the camera or the molecule moves.

| Option | Effect |
|---|---|
| `window=1.35` | Window radius, as a multiple of the selection's bounding radius |
| `softness=0.22` | Fraction of the window used for its fading edge |
| `band=4.0` | Depth in Å over which geometry just in front of the selection fades back in |
| `shape="cone"` | `"cone"` for a window, or `"tunnel"` for a depth tunnel |

Atoms keep the selection's whole bounding sphere, so ligands and side chains in it stay visible. Cartoons and molecular surfaces are carved closer to its center, because they often pass in front of a ligand or enclose it. Through the window, a molecular surface shows its inner walls.

A second `Reveal` on the same selection changes the window size smoothly, for example to widen it as the camera pulls back. `Conceal(self.camera)` closes the open window.

## Show depth with a tunnel

```python
self.play(Reveal(self.camera, site, window=1.3, shape="tunnel"), run_time=2)
depth = self.camera.cutaway_geometry()[3]     # Å from the selection's center to open space
```

A tunnel is a cylinder drilled along the current view. It then stays fixed to the molecule, so camera motion shows its walls in perspective. A ring marks every 5 Å from the outer surface, with a brighter ring every 10 Å; `rings=` changes the spacing and `wall=` the wall opacity.

The tunnel ends where its centerline first reaches open space: the last atom within 6 Å of the axis before a gap longer than 8 Å, such as solvent or an internal chamber. The depth therefore depends on the direction. Report it with its view, for example "21 Å beneath the outer surface, seen from outside the ring".

These shots make depth easiest to read:

- Small camera moves near the tunnel axis, so the rings slide past each other.
- A side view with the rest of the molecule faded, for example `SetOpacity(rest, 0.18)`, so the whole tunnel is visible.
- A fly-down: align the camera with the tunnel, then `self.camera.animate.zoom(...)`.

## Depth cue and view search

Distance fog darkens geometry far from the camera. `self.camera.animate.depth_cue(0.7)` eases the fog strength and can be combined with camera moves:

```python
self.play(self.camera.animate.orbit(0.3).depth_cue(0.7), run_time=2)
```

`self.camera.clearest_view(region)` returns the camera angles `(theta, phi)` with the fewest atoms between the camera and the region. Use it to choose a starting view before a `Reveal`.
