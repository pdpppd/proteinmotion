"""Random-access trajectories with bounded frame caches.

A Trajectory wraps any sequence of (atoms, 3) coordinate frames: a list of arrays, a
memory-mapped .npy file, an MDAnalysis reader, or an aligned view of another trajectory.
Frames are read and validated only when asked for, so memory-mapped files and MD
readers never load a whole simulation. Protein.from_file() stores NMR models as a
Trajectory, PlayTrajectory (animation.py) plays one through a FrameCache, and
time-series plots read its timestamps.
"""

from collections import OrderedDict

import numpy as np

from .math3d import align_coordinates
from .structure import Atom, Topology, coordinates, infer_bonds, make_chains, residue_record, with_dssp


class Trajectory:
    """Frames in ångströms, shaped (states, atoms, 3). No physical dynamics are inferred."""

    def __init__(self, frames, *, topology=None, units="angstrom", times=None, time_unit="ps"):
        if units not in ("angstrom", "nm"):
            raise ValueError("units must be 'angstrom' or 'nm'")
        if len(frames) < 1:
            raise ValueError("Trajectory must contain at least one frame")
        self._times = None if times is None else self._validate_times(times, len(frames))
        # A callable that returns the timestamps or None, read on first use of ``times``. MD
        # readers must seek through the whole file to collect them, so they wait until needed.
        self._time_reader = None
        self.time_unit = str(time_unit)
        # Any object with len() and indexing that returns one (atoms, 3) frame.
        self._frames = frames
        # Factor that converts the stored units to Å.
        self._factor = 10.0 if units == "nm" else 1.0
        self.topology = topology
        self.n_atoms = len(self.frame(0))
        if topology is not None and len(topology.atoms) != self.n_atoms:
            raise ValueError("Trajectory atom count does not match topology")

    @staticmethod
    def _validate_times(times, count):
        """A read-only float array of ``count`` finite, strictly increasing timestamps."""
        values = np.array(times, dtype=float, copy=True)
        if values.shape != (count,) or not np.isfinite(values).all() or (np.diff(values) <= 0).any():
            raise ValueError("times needs one finite, strictly increasing timestamp per frame")
        values.flags.writeable = False
        return values

    @property
    def times(self):
        """Physical timestamps, or None when the input has no physical clock."""
        if self._time_reader is not None:
            values = self._time_reader()
            self._times = None if values is None else self._validate_times(values, len(self))
            self._time_reader = None
        return self._times

    def __len__(self):
        return len(self._frames)

    def frame(self, index):
        """Read one frame as a float32 (atoms, 3) array in Å."""
        if not 0 <= index < len(self):
            raise IndexError(index)
        a = np.asarray(self._frames[index], dtype=np.float32)
        if self._factor != 1:
            a = a * self._factor
        if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
            raise ValueError(f"Invalid coordinates in frame {index}")
        # __init__ reads frame 0 to learn n_atoms, so the attribute may not exist yet.
        if hasattr(self, "n_atoms") and len(a) != self.n_atoms:
            raise ValueError(f"Atom count changes in frame {index}")
        return a

    @classmethod
    def from_npy(cls, path, *, topology=None, units="angstrom", times=None, time_unit="ps"):
        """Memory-mapped .npy file; reads only frames needed for playback."""
        frames = np.load(path, mmap_mode="r", allow_pickle=False)
        if frames.ndim != 3 or frames.shape[2] != 3:
            raise ValueError("Expected a .npy array shaped (frames, atoms, 3)")
        return cls(frames, topology=topology, units=units, times=times, time_unit=time_unit)

    @classmethod
    def from_mdanalysis(cls, topology_file, trajectory_file, *, selection="protein or nucleic", stride=1):
        """Lazy XTC/DCD/TRR/etc. reader. Install proteinmotion[md]."""
        try:
            import MDAnalysis as mda
        except ImportError as exc:
            raise ImportError('Install MD readers with pip install "proteinmotion[md]"') from exc
        if not isinstance(stride, int) or stride < 1:
            raise ValueError("stride must be a positive integer")
        universe = mda.Universe(str(topology_file), str(trajectory_file))
        group = universe.select_atoms(selection)
        if not len(group):
            raise ValueError("MD selection contains no atoms")
        # Build the topology from the selected atoms, one residue at a time.
        atoms, residues = [], []
        for ri, res in enumerate(group.residues):
            selected = res.atoms.intersection(group)
            names = {}
            for a in selected:
                names[a.name] = len(atoms)
                try:
                    element = a.element.title()
                except (AttributeError, mda.exceptions.NoDataError):
                    # Topologies without element records (GRO files, for example): guess from
                    # the first letter of the name after any leading digits, so "1HB" is H.
                    element = a.name.lstrip("0123456789")[0].upper()
                # PDB-style chain ID if the topology has one, otherwise the segment ID.
                chain = getattr(a, "chainID", "") or a.segid
                atoms.append(
                    Atom(chain, int(a.resid), getattr(a, "icode", "").strip(), a.resname, a.name, element, ri)
                )
            residues.append(
                residue_record(
                    atoms[-1].chain,
                    int(res.resid),
                    getattr(res, "icode", "").strip(),
                    res.resname,
                    names,
                    atoms,
                )
            )
        # AtomGroup order is authoritative; reject unusual interleaved residue order.
        if [a.name for a in atoms] != list(group.names):
            raise ValueError("Selection atoms must be grouped by residue")
        xyz = coordinates(group.positions)
        topo = Topology(
            tuple(atoms), tuple(residues), infer_bonds(atoms, xyz), make_chains(residues, xyz, atoms)
        )
        # MD topologies carry no helix/sheet records; assign them from the first frame.
        topo = with_dssp(topo, xyz)
        source = _MDFrames(universe, group.indices, stride)
        result = cls(source, topology=topo, time_unit="ps")
        result._time_reader = source.times
        return result

    def aligned(self, reference=None, *, indices=None, selection=None):
        """A lazy trajectory with every frame superposed on ``reference`` (default: frame 0).

        The Kabsch fit uses ``indices`` or the atoms of ``selection`` when given, all atoms
        otherwise, and moves the whole frame. Timestamps carry over.
        """
        if selection is not None:
            from .regions import Region

            if indices is not None or not isinstance(selection, Region):
                raise ValueError("Use selection=Region or indices, not both")
            if self.topology is None or selection.protein.topology.keys != self.topology.keys:
                raise ValueError("Alignment selection must match the trajectory topology")
            indices = selection.atom_indices
        if indices is not None and len(indices) < 3:
            raise ValueError("Alignment needs at least three selected atoms")
        reference = coordinates(self.frame(0) if reference is None else reference, self.n_atoms)
        parent = self

        # Frames are aligned one at a time as they are read. parent.frame() has already
        # converted them to Å, so the new trajectory keeps the default units.
        class Aligned:
            def __len__(self):
                return len(parent)

            def __getitem__(self, i):
                return align_coordinates(parent.frame(i), reference, indices)

        result = Trajectory(Aligned(), topology=self.topology, time_unit=self.time_unit)
        result._time_reader = lambda: self.times
        return result


class _MDFrames:
    """Frames of an MDAnalysis Universe for the selected atom indices; frame i is reader frame i × stride."""

    def __init__(self, universe, indices, stride):
        self.universe, self.indices, self.stride = universe, indices, stride

    def __len__(self):
        # Ceiling division: frames 0, stride, 2 × stride, … that exist in the file.
        return (len(self.universe.trajectory) + self.stride - 1) // self.stride

    def times(self):
        """Timestamps in ps of the strided frames, or None when the file has no clock."""
        # Indexing the reader seeks it. Return to the frame it was on afterwards.
        reader = self.universe.trajectory
        current = reader.frame
        try:
            values = [reader[i * self.stride].time for i in range(len(self))]
        finally:
            reader[current]
        # A file written without a clock, such as an XTC saved from static models, reports
        # time 0 for every frame. Times that do not strictly increase are not a clock.
        if (np.diff(values) <= 0).any():
            return None
        return values

    def __getitem__(self, i):
        # Seeking loads the frame into universe.atoms.positions, which the next seek
        # overwrites, so return a copy.
        self.universe.trajectory[i * self.stride]
        return np.array(self.universe.atoms.positions[self.indices], dtype=np.float32, copy=True)


class FrameCache:
    """Least-recently-used cache of validated, read-only frames, at most ``maxsize`` of them.

    PlayTrajectory reads the two frames around the current time through it.
    """

    def __init__(self, trajectory, maxsize=3):
        self.trajectory, self.maxsize = trajectory, maxsize
        self.cache = OrderedDict()

    def get(self, i):
        if i not in self.cache:
            self.cache[i] = coordinates(self.trajectory.frame(i), self.trajectory.n_atoms)
        self.cache.move_to_end(i)
        while len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)
        return self.cache[i]
