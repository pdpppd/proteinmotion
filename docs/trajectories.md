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

Only two coordinate states reside in GPU buffers; playback caches three decoded CPU
frames per trajectory animation. MDAnalysis converts its coordinates to ångströms.
For `.npy`, pass `units='nm'` when appropriate. Raw arrays cannot verify atom identity;
the caller must preserve atom order. Topology-backed sources verify both count and keys.
Long movies are streamed through PyAV with three readback buffers, not kept in RAM.

**Preprocess periodic boundaries before loading.** ProteinMotion does not unwrap
molecules or apply minimum-image interpolation. Wrapped coordinates can create long
bonds or incorrect interpolated paths. Supply whole, unwrapped molecules; align if
desired. The MD adapter is a sequential, single-threaded reader.

Secondary structure comes from PDB/mmCIF annotations. Without assignments, cartoon
renders coils and emits a warning for structure files. MD topologies default to coils.
Supply external DSSP-style assignments with `p.with_secondary_structure('HHHCCEEE…')`,
one `H/E/C` character per topology residue, before adding/rendering. Assignments remain
fixed over a trajectory. Cartoon helices follow the alpha-carbon backbone; they are
not idealized cylindrical helices.

### A real NMR ensemble

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

`RegionTour` visits residues 23–34 with gold sphere/box annotations, then residues
71–76 with cyan atom halos and a box. It switches to ball-and-stick for the tail.
The other two scenes play **all 116 states** in cartoon and ball-and-stick.
Deposited NMR model order is **not a physical time sequence**. Interpolation is an
illustrative transition between conformers, not a simulated molecular pathway.
The 23 s playback duration is chosen for viewing. Provenance and alignment details
are recorded in [the ensemble report](nmr-ensemble-report.json).
