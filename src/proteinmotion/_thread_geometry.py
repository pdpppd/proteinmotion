"""Deterministic molecular transport along threading routes (an illustration, not dynamics).

``Thread(mode="geometry")`` (thread.py) builds one route per chain: a flight curve from
off-screen that joins the chain's backbone curve at the C terminus and runs along it to
the N terminus. ``ThreadGeometry`` slides every residue along its chain's route like a
bead on a wire. Each residue (and each ligand, ion or water, carried by a nearby
residue) keeps its own shape: its atoms ride in a frame that is parallel-transported
along the route. At the end of the clip every residue sits at its own place on the
backbone, which reproduces the original coordinates.
"""

import numpy as np
from scipy.spatial import cKDTree

from .math3d import normalize


def _perpendicular(tangent):
    """A unit vector perpendicular to ``tangent``, from whichever world axis is far from it."""
    axis = np.array([0.0, 1.0, 0.0]) if abs(tangent[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    return normalize(np.cross(tangent, axis))


class TransportPath:
    """Arc-length sampling with orthonormal, parallel-transported residue frames."""

    def __init__(self, points):
        points = np.asarray(points, float)
        # Repeated points would give zero-length steps and undefined tangents.
        keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]
        self.points = points[keep]
        if len(self.points) < 2:
            raise ValueError("Thread needs a nonzero-length route")
        # Arc length (Å) at each point, from the start of the route.
        self.s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))]
        self.tangent = normalize(np.gradient(self.points, self.s, axis=0))
        # Parallel transport: each normal is the previous one with its component along the
        # new tangent removed. The frame then turns only as much as the route bends.
        self.normal = np.empty_like(self.points)
        self.normal[0] = _perpendicular(self.tangent[0])
        for i in range(1, len(self.points)):
            t = self.tangent[i]
            n = self.normal[i - 1] - t * np.dot(self.normal[i - 1], t)
            self.normal[i] = normalize(n) if np.linalg.norm(n) > 1e-7 else _perpendicular(t)

    def sample(self, distance):
        """Positions (n, 3) and frames (n, 3, 3) at arc lengths ``distance`` (Å).

        Frame columns are the tangent, the normal and their cross product. Distances
        beyond either end continue in a straight line along the end tangent.
        """
        distance = np.atleast_1d(distance)

        def interpolate(values):
            return np.column_stack([np.interp(distance, self.s, values[:, k]) for k in range(3)])

        xyz = interpolate(self.points)
        # Upstream atoms extend out of frame instead of collapsing at the entry point.
        xyz += np.minimum(distance, 0)[:, None] * self.tangent[0]
        xyz += np.maximum(distance - self.s[-1], 0)[:, None] * self.tangent[-1]
        # Interpolated vectors are no longer orthonormal; restore that, using any
        # perpendicular where the normal vanishes.
        tangent = normalize(interpolate(self.tangent))
        normal = interpolate(self.normal)
        normal -= tangent * np.sum(normal * tangent, axis=1, keepdims=True)
        for i in np.flatnonzero(np.linalg.norm(normal, axis=1) < 1e-7):
            normal[i] = _perpendicular(tangent[i])
        normal = normalize(normal)
        frames = np.stack((tangent, normal, np.cross(tangent, normal)), axis=2)
        return xyz, frames


class ThreadGeometry:
    """Move real atom groups, preserving each residue/ligand's internal geometry.

    ``strands`` are thread._Strand routes, one per chain in ``topology.chains`` order.
    Residues are stacked chain by chain, each chain from its C terminus to its N
    terminus (the order of its route); ``slices`` gives each chain's part of that stack
    and ``trace`` the stacked trace atoms. ``owners`` maps every atom to the stacked
    residue that carries it, and ``offsets`` holds its position in that residue's frame.
    """

    def __init__(self, protein, strands):
        topology = protein.topology
        self.reference = protein.positions.copy()
        self.paths = [TransportPath(s.path) for s in strands]
        self.distances, self.slices, trace = [], [], []
        residue_owner = np.full(len(topology.residues), -1, int)
        cursor = 0
        for chain, strand in zip(topology.chains, strands):
            # Routes start at the C terminus, so stack each chain's residues in reverse.
            reverse = chain[::-1]
            n = len(reverse)
            self.slices.append(slice(cursor, cursor + n))
            # _catmull samples eight points per residue and includes its final endpoint.
            # Every eighth sample is a residue's trace atom; its arc length past the flight
            # is where that residue comes to rest.
            self.distances.append(strand.flight_length + strand.backbone_s[::8])
            residue_owner[reverse] = np.arange(cursor, cursor + n)
            trace.extend(topology.residues[i].trace_atom for i in reverse)
            cursor += n
        self.trace = np.array(trace)
        anchors = self.reference[self.trace]
        self.residue_ids = np.array([a.residue_index for a in topology.atoms])
        # Keep a whole ligand/ion/water with one nearby traced residue, preferring
        # its author chain. Assigning each atom separately would tear ligands apart.
        chain_names = np.array([topology.atoms[i].chain for i in self.trace])
        trees = {}
        for ri in np.flatnonzero(residue_owner < 0):
            name = topology.residues[ri].chain
            if name not in trees:
                ids = np.flatnonzero(chain_names == name)
                if not len(ids):
                    ids = np.arange(len(anchors))
                trees[name] = ids, cKDTree(anchors[ids])
            ids, tree = trees[name]
            atom_ids = np.flatnonzero(self.residue_ids == ri)
            if len(atom_ids):
                _, nearest = tree.query(self.reference[atom_ids].mean(0))
                residue_owner[ri] = ids[nearest]
        self.owners = residue_owner[self.residue_ids]
        # Each atom's offset from its owner's trace atom, in the owner's frame at rest:
        # offset = Fᵀ (x - anchor), so that x = anchor + F offset wherever the owner goes.
        frames = np.concatenate([path.sample(s)[1] for path, s in zip(self.paths, self.distances)])
        self.offsets = np.einsum("ni,nij->nj", self.reference - anchors[self.owners], frames[self.owners])

    def evaluate(self, clocks, easing):
        """Atom coordinates for per-chain clocks in [0, 1]; ``easing`` maps a clock to travel.

        Returns float32 coordinates (atoms, 3), the position of each chain's leading
        residue (its N terminus), and each head's glow strength in [0, 1].
        """
        anchors, frames, strengths = [], [], []
        for path, target, clock in zip(self.paths, self.distances, clocks):
            progress = float(easing(clock))
            if not np.isfinite(progress) or not 0 <= progress <= 1:
                raise ValueError("easing must return a finite value in [0, 1]")
            # The whole chain sits one route length short of its rest positions at the
            # start and slides forward together. Residues not yet on the route queue up
            # in a straight line before its start.
            remaining = path.s[-1] * (1 - progress)
            xyz, basis = path.sample(target - remaining)
            anchors.append(xyz)
            frames.append(basis)
            # The glow stays bright for most of the trip and fades shortly before arrival.
            strengths.append(1 - progress**6 if clock > 0 else 0.0)
        anchors, frames = np.concatenate(anchors), np.concatenate(frames)
        xyz = anchors[self.owners] + np.einsum("nij,nj->ni", frames[self.owners], self.offsets)
        # Settled chains are exact, including when other chains are still arriving.
        for section, clock in zip(self.slices, clocks):
            if clock >= 1:
                mask = (self.owners >= section.start) & (self.owners < section.stop)
                xyz[mask] = self.reference[mask]
        # The last stacked residue of each chain is its N terminus, which leads.
        heads = xyz[[self.trace[section.stop - 1] for section in self.slices]]
        return xyz.astype(np.float32), heads, np.array(strengths)
