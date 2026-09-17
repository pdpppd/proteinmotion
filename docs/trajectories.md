# NMR and MD trajectories

```python
# PDB MODEL records / mmCIF model numbers, with strict identity checks:
p = Protein.from_file("ensemble.pdb").cartoon().center()
self.add(p)
self.camera.frame(p)
self.play(PlayTrajectory(p), run_time=6)

# Large NumPy trajectories are memory mapped (frames, atoms, 3):
trajectory = Trajectory.from_npy("coordinates.npy", units="angstrom")
self.play(PlayTrajectory(p, trajectory), run_time=10)

# XTC, DCD, TRR, and other MDAnalysis-supported formats are read lazily:
trajectory = Trajectory.from_mdanalysis(
    "topology.pdb", "trajectory.xtc", selection="protein and not name H*", stride=2
)
trajectory = trajectory.aligned()  # optional removal of overall rigid-body motion
p = Protein.from_trajectory(trajectory).ribbon().center()
self.add(p)
self.camera.frame(p)
self.play(PlayTrajectory(p, start=0, end=len(trajectory) - 1), run_time=10)
```

`state_easing=smooth` eases interpolation within each adjacent pair of models;
the default `state_easing=linear` preserves constant interpolation speed for MD.
This is separate from `rate_func`, which controls progress through the entire clip.

Playback holds two coordinate states on the GPU and caches three decoded frames on the CPU per trajectory animation. PyAV streams exported video through three readback buffers.

MDAnalysis converts coordinates to ångströms. For NumPy files in nanometers, pass `units="nm"`. Raw arrays must follow the protein’s atom order. Sources with topology information are checked for matching atom counts and identities.

**Load whole, unwrapped molecules.** Prepare periodic boundaries before importing a trajectory. Wrapped coordinates can create long bonds and incorrect interpolated paths. You can also align frames to remove overall motion. The MD adapter reads frames sequentially on one thread.

Secondary structure comes from PDB/mmCIF annotations. When assignments are missing, cartoons use coils and structure files produce a warning. MD topologies also default to coils.

To supply assignments, call `p.with_secondary_structure("HHHCCEEE…")` before adding the protein. Use one `H`, `E`, or `C` per topology residue. The assignments stay fixed during playback. Helix geometry follows the Cα backbone.

## Ubiquitin NMR example

`examples/nmr_regions.py` uses [PDB 2K39](https://www.rcsb.org/structure/2K39),
an RDC-derived solution NMR ensemble of ubiquitin: **116 deposited models, 76 residues,
602 selected heavy atoms per model**. Atom identities/order are checked in every model.
Each conformer is rigidly aligned on the Cα atoms of residues 1–70, leaving the
C-terminal tail free to move:

```python
p = Protein.from_file("examples/data/2k39.cif", chains="A")
core = p.select(residues=(1, 70), atoms="CA")
aligned = p.trajectory.aligned(indices=core.atom_indices)
p = Protein.from_trajectory(aligned).cartoon().center()
self.add(p)
self.camera.frame(p, aspect=self.width / self.height)
self.play(PlayTrajectory(p, state_easing=smooth), run_time=23)
```

```bash
proteinmotion render examples/nmr_regions.py RegionTour -o nmr-region-tour.mp4 --fps 60
proteinmotion render examples/nmr_regions.py NMRStates -o nmr-116-states.mp4 --fps 60
proteinmotion render examples/nmr_regions.py NMRAtoms -o nmr-116-states-atoms.mp4 --fps 60
```

`RegionTour` highlights residues 23–34 with a gold sphere and box, then residues 71–76 with cyan atom highlights and a box. It switches to ball-and-stick for the tail. The other two scenes play all 116 models in cartoon and ball-and-stick.

NMR models describe structural variation. Their deposited order is not a time sequence. The animation interpolates between them over a duration chosen for viewing. See the [ensemble report](nmr-ensemble-report.json) for source and alignment details.