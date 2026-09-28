# Cutaways, depth tunnels, and threading

Requires ProteinMotion 0.12.0 or later. Cutaways, tunnels, fog, and wire glow render in both the native renderer and EEVEE.

## Reveal a hidden selection

```python
site = p.select(within=4.0, of=p.select(chain="A", resname="ADP"))
self.play(Reveal(self.camera, site, window=1.6), run_time=2)  # Cone window facing the camera.
self.play(self.camera.animate.orbit(1.0), run_time=5)  # The window follows the camera.
self.play(Conceal(self.camera), run_time=1.5)
```

Geometry between the camera and the selection fades inside the window; no atoms move. Atoms keep the selection's bounding sphere, so ligands and side chains stay; cartoons and surfaces are carved closer to its center, and surfaces show their inner walls. A second `Reveal` on the open selection resizes the window smoothly. One cutaway is open at a time.

## Show depth with a tunnel

```python
self.play(Reveal(self.camera, site, window=1.3, shape="tunnel"), run_time=2)
depth = self.camera.cutaway_geometry()[3]  # Å from the selection center to open space.
```

The tunnel is drilled along the current view and then fixed to the molecule, with a ring every 5 Å (`rings=`) from where its centerline reaches open space (the first gap longer than 8 Å). Depth depends on direction: choose the view deliberately, for example straight in from the outside of a complex, and state it in captions ("21 Å beneath the outer surface, seen from outside the ring"). A tunnel that starts at open space is short; that is a real result, not a failure.

Depth reads best with:

- small orbits near the tunnel axis (large orbits hide a tunnel inside opaque protein);
- a side view with the rest faded: `rest = Region(p, np.setdiff1d(np.arange(len(p.topology.atoms)), keep.atom_indices))`, then `SetOpacity(rest, 0.18)` and a highlight sphere on the target;
- a fly-down: orbit to face the tunnel axis, `Focus` on the target, then `self.camera.animate.zoom(3)`;
- stronger fog in close shots: `self.camera.animate.orbit(0.3).depth_cue(0.7)`.

`Focus` and camera `animate` builders both write the camera; put them in separate `play` calls. `camera.clearest_view(region)` returns the least-occluded `(theta, phi)`.

## Thread chains into view

```python
self.camera.frame(p, aspect=self.width / self.height)
self.play(Thread(p, self.camera, stagger=0.16, glow_color="#bfefff"), run_time=7)
self.play(Unthread(p, self.camera, stagger=0.16), run_time=5)
```

By default the actual cartoon, ribbon, or atoms and bonds follow a route through the C terminus toward the N terminus. Geometry travel uses quintic `smooth` easing over the full clip. Try `swirl=0.25` and an eight-second entrance. Surface representation raises `ValueError`. Residues and ligands move as rigid groups, but inter-residue bonds may stretch. Coordinates restore exactly at the end. Do not combine with concurrent coordinate or representation animations.

Pass the scene camera and `aspect=self.width/self.height` for off-screen entry. Options include `easing`, `stagger` (per-chain delay as a fraction of the clip), `swirl`, `seed`, and the optional halo `glow` (size relative to `radius`; 0 disables), `glow_color`, `glow_brightness`. Plan 5–8 s and frame the resting protein first. Describe these animations as illustrations, never as folding or physical motion.

`mode="wire"` preserves the original wire-to-protein fade, with sine easing by default and `settle` controlling the crossfade in wire mode only. Wire colors follow cartoon tints, `radius` and `head` size the wire. `examples/studio_threading.py` renders actual geometry in cartoon and ball-and-stick; the hemoglobin example explicitly selects wire mode.

Examples: `examples/depth_tunnels.py` (GroEL–GroES, 1AON) and `examples/thread_hemoglobin.py` (4HHB).
