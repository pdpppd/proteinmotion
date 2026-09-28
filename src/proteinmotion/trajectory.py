"""Random-access trajectories with bounded frame caches."""

from collections import OrderedDict

import numpy as np

from .math3d import align_coordinates
from .structure import Atom, Topology, coordinates, infer_bonds, make_chains, residue_record


class Trajectory:
    """Frames in ångströms, shaped (states, atoms, 3). No physical dynamics are inferred."""

    def __init__(self, frames, *, topology=None, units="angstrom", times=None, time_unit="ps"):
        if units not in ("angstrom", "nm"):
            raise ValueError("units must be 'angstrom' or 'nm'")
        if len(frames) < 1:
            raise ValueError("Trajectory must contain at least one frame")
        self._times = None if times is None else self._validate_times(times, len(frames))
        self._time_reader = None
        self.time_unit = str(time_unit)
        self._frames = frames
        self._factor = 10.0 if units == "nm" else 1.0
        self.topology = topology
        self.n_atoms = len(self.frame(0))
        if topology is not None and len(topology.atoms) != self.n_atoms:
            raise ValueError("Trajectory atom count does not match topology")

    @staticmethod
    def _validate_times(times, count):
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
        if not 0 <= index < len(self):
            raise IndexError(index)
        a = np.asarray(self._frames[index], dtype=np.float32)
        if self._factor != 1:
            a = a * self._factor
        if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
            raise ValueError(f"Invalid coordinates in frame {index}")
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
        atoms, residues = [], []
        for ri, res in enumerate(group.residues):
            selected = res.atoms.intersection(group)
            names = {}
            for a in selected:
                names[a.name] = len(atoms)
                try:
                    element = a.element.title()
                except (AttributeError, mda.exceptions.NoDataError):
                    element = a.name.lstrip("0123456789")[0].upper()
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
        source = _MDFrames(universe, group.indices, stride)
        result = cls(source, topology=topo, time_unit="ps")
        result._time_reader = source.times
        return result

    def aligned(self, reference=None, *, indices=None, selection=None):
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

        class Aligned:
            def __len__(self):
                return len(parent)

            def __getitem__(self, i):
                return align_coordinates(parent.frame(i), reference, indices)

        result = Trajectory(Aligned(), topology=self.topology, time_unit=self.time_unit)
        result._time_reader = lambda: self.times
        return result


class _MDFrames:
    def __init__(self, universe, indices, stride):
        self.universe, self.indices, self.stride = universe, indices, stride

    def __len__(self):
        return (len(self.universe.trajectory) + self.stride - 1) // self.stride

    def times(self):
        reader = self.universe.trajectory
        current = reader.frame
        try:
            return [reader[i * self.stride].time for i in range(len(self))]
        finally:
            reader[current]

    def __getitem__(self, i):
        self.universe.trajectory[i * self.stride]
        return np.array(self.universe.atoms.positions[self.indices], dtype=np.float32, copy=True)


class FrameCache:
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
