# Animation and state changes

## Timeline invariants

`ProteinScene` snapshots objects on `add` and the camera on the first `play`/`wait`. Each animation binds to the preceding endpoint; the compiled timeline can be sought forward or backward. Avoid direct mutations after those snapshots. A `play` rejects concurrent writers to the same property: combine camera `.orbit(...).zoom(...)` in one builder, but sequence it with `Focus`, which also owns the camera channel. Separate color/opacity selections may run concurrently when they do not overlap.

```python
self.play(Rotate(p, 0.6), run_time=2)
self.play(p.animate.shift((4, 0, 0)).rotate(0.4), run_time=2)
self.play(self.camera.animate.orbit(theta=0.35, phi=0.08).zoom(1.1), run_time=2)
```

`Focus(camera, region, margin=..., aspect=..., follow=True)` evaluates after geometry changes in the same clip. It follows a region's center afterward without continually changing zoom. Use `self.focus(region, margin=1.7, run_time=2)` for a sequential focus with the scene's aspect ratio.

## Representations and styles

Initialize before `add`:

```python
p = Protein.from_file(path, chains="A").ball_and_stick(atom_scale=0.28, bond_radius=0.10)
p.surface(kind="ses", resolution=0.7).cartoon().center()
```

The chained calls configure radii and surface options, then select cartoon as the initial view. Surface kinds are `vdw`, `sas`, and approximate `ses`; smaller resolution means a finer and more expensive grid.

Animate changes:

```python
self.play(Representation(p, "ribbon"), run_time=1.5)
self.play(Representation(p, "ball_and_stick"), run_time=1.5)
self.play(Representation(p, "surface"), run_time=1.5)
helix = p.select(chain="A", residues=(23, 34))
self.play(Colorize(helix, "#50e0d0", residue_delay=0.05), run_time=2)
self.play(SetOpacity(helix, 0.15, residue_delay=0.05), run_time=2)
self.play(Colorize(helix, None), SetOpacity(helix, 1), run_time=1.5)
```

Cartoons and ribbons draw ligands, ions, and loaded waters as ball-and-stick by default. `ShowSideChains`, `ShowAtoms`, and their `Hide` counterparts add or remove atoms residue by residue; see [ligands, ions, and side chains](ligands-and-side-chains.md).

Colors and residue opacity carry across representations. `Colorize(..., None)` restores the base palette. For staggered appearance tracks, keep the clip clock linear (the default for these animations); their internal easing handles each residue. `run_time` must exceed `(residue_count - 1) * residue_delay`. Do not add a global smooth `rate_func` to such clips.

## Ensembles and trajectory readers

A multi-model PDB/mmCIF loaded with `Protein.from_file` exposes `p.trajectory`. To remove global drift, align on a stable set of Cαs before creating the display object:

```python
loaded = Protein.from_file(ensemble_path, chains="A")
core = loaded.select(residues=(5, 65), atoms="CA")
aligned = loaded.trajectory.aligned(indices=core.atom_indices)
p = Protein.from_trajectory(aligned).cartoon().center()
self.add(p)
self.camera.frame(p, aspect=self.width / self.height)
self.play(PlayTrajectory(p, start=0, end=min(2, len(aligned) - 1), state_easing=smooth), run_time=8)
```

Use `Trajectory.from_mdanalysis(topology_file, trajectory_file, selection="protein")` for lazy XTC/DCD/TRR readers (`md` extra). `Trajectory.from_npy(path, topology=...)` reads `(frames, atoms, 3)` arrays with bounded memory. Atom identities/order must match the displayed protein. Reader coordinates are Å; do not rescale them as if they were still native XTC nanometers.

`PlayTrajectory` starts exactly at its selected frame; it does not bridge from the current coordinates. If those differ, first `Morph(p, trajectory.frame(start), align=False)` with an appropriate duration. After a protein morph, align any incoming trajectory to the current destination pose before playback. Otherwise the first state can jump in world space.

Use linear interpolation for dense MD frames to preserve uniform playback timing. For sparse NMR conformers, slower transitions with `state_easing=smooth` can be clearer; each state then eases to rest. NMR model order is not physical time. Do not infer a dynamics simulation from a multi-state file.

## Deformation and same-topology morphs

```python
rest = p.positions.copy()
self.play(Deform(p, lambda xyz: xyz * [1.12, 1.0, 1.0]), run_time=2)
self.play(Morph(p, rest, align=False), run_time=2)
```

`Morph(p, other_protein)` requires matching atom identities, or an explicit map covering every source atom. Raw coordinate arrays must follow source atom order. These operations retain topology; use `BackboneMorph` for different protein or nucleic-acid topologies. DNA/RNA alignment uses C1′ anchors.

## Different-protein contact matching

```python
source = Protein.from_file(source_path, chains="A").ball_and_stick().center()
target = Protein.from_file(target_path, chains="A").ball_and_stick().center()
match = match_backbones(source, target, max_contact_error=0.30, search_seconds=15, max_candidates=6000)
match.save("correspondence.json")
self.add(source)
self.camera.frame(source, target, margin=1.3, aspect=self.width / self.height)
self.play(
    BackboneMorph(source, target, match=match, residue_delay=0.025, fade_out=(0, 0.35), fade_in=(0.65, 1)),
    run_time=8,
)
self.play(Rotate(target, 0.5), run_time=2)
```

Match preparation can be expensive: save and reuse `ContactMatch.load(...)`. Matching maximizes correspondence size subject to a configurable contact-map error limit, then minimizes error among equal-sized sets. A bounded/pruned search does not guarantee a global optimum; report the actual result rather than promising one.

`BackboneMorph` accepts increasing source/target residue indices and translates whole matched residues with Cα for proteins or C1′ for DNA/RNA. Nucleotide delays follow deposited 5′-to-3′ order. Unmatched residues fade. Atoms in different side chains are crossfaded, not individually mapped. The morph owns both proteins' geometry, transforms, controls and opacity: use camera motion concurrently, not an independent `Rotate` on an endpoint. For K matches, duration must exceed `(K-1)*residue_delay`; the remainder is each residue's motion time. Per-residue easing is controlled with `motion_easing="smooth"`.

The target is rigidly aligned when `align=True`. Continue using that same destination object after the morph; the source becomes hidden. Do not reload/recenter the destination between scenes in a continuous tour. Bond lengths and physical energetics are not constrained by this visual interpolation.
