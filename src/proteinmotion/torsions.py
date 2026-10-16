"""Backbone and side-chain torsion angles, measured and animated in degrees.

φ, ψ and ω follow the IUPAC definitions: φ(i) is C(i−1)–N–Cα–C, ψ(i) is
N–Cα–C–N(i+1), and ω(i) is Cα–C–N(i+1)–Cα(i+1), the peptide bond that follows
residue i. χ1–χ5 use the standard side-chain atoms. Changing a torsion rotates
every atom on one side of its bond about that bond, so bond lengths and angles
never change. Bonds inside rings (the proline ring, aromatic side chains) cannot
rotate. Disulfides, metal contacts and other links between non-adjacent residues
are left in place and stretch when the chain moves.

How a torsion turns: the bonds within residues and between sequence neighbors
(``_kinematic_bonds``) are walked depth first from one anchor atom per chain, which
gives a spanning tree (``_Tree``). Turning the bond u–v, where u is v's parent in the
tree, rotates v and every atom beyond it; the anchor's side stays still. A bond on a
cycle of the bond graph lies in a ring and is reported as locked. ``_Plan`` collects
the turns of one call, and ``rotate_bonds`` applies any fraction of them to the
coordinates captured when the plan was made, so every frame is computed on its own.

``Protein.torsions`` and ``Region.torsions`` call ``measure``; ``set_torsions`` changes
coordinates at once; ``SetTorsions`` and ``RotateTorsions`` animate; ``TorsionMarker``
draws one torsion on screen. ``StructureMorph`` (conformations.py) reuses ``_tree`` and
``rotate_bonds`` to turn side chains, and ``RamachandranPlot`` reads the paths that
``SetTorsions`` stores in ``Protein._torsion_motion``.
"""

import unicodedata

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .animation import Animation
from .annotations import Annotation, Text
from .math3d import color as parse_color
from .rates import evaluate, linear, resolve
from .structure import coordinates

# Column order of the torsion table (torsion_atoms) and of TorsionAngles.values:
# columns 0–2 are φ, ψ, ω and columns 3–7 are χ1–χ5.
KINDS = ("phi", "psi", "omega", "chi1", "chi2", "chi3", "chi4", "chi5")
SYMBOLS = {
    "phi": "φ",
    "psi": "ψ",
    "omega": "ω",
    "chi1": "χ₁",
    "chi2": "χ₂",
    "chi3": "χ₃",
    "chi4": "χ₄",
    "chi5": "χ₅",
}
# (φ, ψ) in degrees: Pauling α helix, 3₁₀ and π helices, β strands, polyproline II,
# the fully extended chain and the left-handed helix region.
CONFORMATIONS = {
    "helix": (-57.0, -47.0),
    "alpha": (-57.0, -47.0),
    "310": (-49.0, -26.0),
    "pi": (-57.0, -70.0),
    "strand": (-135.0, 135.0),
    "beta": (-135.0, 135.0),
    "antiparallel": (-139.0, 135.0),
    "parallel": (-119.0, 113.0),
    "ppii": (-75.0, 145.0),
    "extended": (180.0, 180.0),
    "left": (57.0, 47.0),
}
# The four atoms of χ1, χ2, ... for each residue name, in standard PDB atom names.
# Residues missing here (Gly, Ala) have no χ angles.
_CHI = {
    "ARG": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "NE"),
            ("CG", "CD", "NE", "CZ"), ("CD", "NE", "CZ", "NH1")),
    "ASN": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")),
    "ASP": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")),
    "CYS": (("N", "CA", "CB", "SG"),),
    "GLN": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")),
    "GLU": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")),
    "HIS": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")),
    "ILE": (("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1")),
    "LEU": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "LYS": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "CE"),
            ("CG", "CD", "CE", "NZ")),
    "MET": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"), ("CB", "CG", "SD", "CE")),
    "MSE": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SE"), ("CB", "CG", "SE", "CE")),
    "PHE": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "PRO": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")),
    "SER": (("N", "CA", "CB", "OG"),),
    "THR": (("N", "CA", "CB", "OG1"),),
    "TRP": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "TYR": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "VAL": (("N", "CA", "CB", "CG1"),),
}  # fmt: skip


def wrap(degrees):
    """Wrap angles to (−180°, 180°]."""
    # np.mod returns [0, 360), so 180 − mod(180 − x) lands in (−180, 180].
    return 180.0 - np.mod(180.0 - np.asarray(degrees, dtype=float), 360.0)


def dihedrals(xyz, quads):
    """Signed dihedral angles in degrees for (…, 4) atom indices; NaN where any index is −1."""
    quads = np.asarray(quads, dtype=int)
    shape = quads.shape[:-1]
    quads = quads.reshape(-1, 4)
    result = np.full(len(quads), np.nan)
    valid = (quads >= 0).all(1)
    if valid.any():
        # For atoms a–b–c–d, project the outer bonds b→a and c→d onto the plane normal to
        # the central bond b→c (v and w). The dihedral is the signed angle from v to w,
        # positive when it turns clockwise looking from b toward c, as IUPAC defines it.
        a, b, c, d = (np.asarray(xyz, dtype=np.float64)[quads[valid, k]] for k in range(4))
        b1 = c - b
        b1 /= np.maximum(np.linalg.norm(b1, axis=1, keepdims=True), 1e-12)
        v = (a - b) - np.sum((a - b) * b1, axis=1, keepdims=True) * b1
        w = (d - c) - np.sum((d - c) * b1, axis=1, keepdims=True) * b1
        result[valid] = np.degrees(np.arctan2(np.sum(np.cross(b1, v) * w, axis=1), np.sum(v * w, axis=1)))
    return result.reshape(shape)


def _bonded(topology):
    """Set of bonded atom pairs, in both orders, cached on the topology."""
    # Caches live in the topology's __dict__, next to its cached_property values, so every
    # protein that shares a topology shares them too.
    cache = topology.__dict__.get("_torsion_bond_set")
    if cache is None:
        cache = {(int(a), int(b)) for a, b in topology.bonds} | {(int(b), int(a)) for a, b in topology.bonds}
        topology.__dict__["_torsion_bond_set"] = cache
    return cache


def torsion_atoms(topology):
    """(residues, 8, 4) atom indices for φ, ψ, ω and χ1–χ5; −1 where a torsion is undefined."""
    cached = topology.__dict__.get("_torsion_atoms")
    if cached is not None:
        return cached
    names = topology.residue_atoms
    residues = topology.residues
    bonds = _bonded(topology)
    table = np.full((len(residues), len(KINDS), 4), -1, dtype=np.int64)

    def peptide(i, j):
        """Residues i and j are consecutive amino acids joined by a C–N bond."""
        if not (0 <= i < len(residues) and 0 <= j < len(residues)):
            return False
        a, b = residues[i], residues[j]
        if a.chain != b.chain or a.is_nucleic or b.is_nucleic or a.ca < 0 or b.ca < 0:
            return False
        carbon, nitrogen = names[i].get("C"), names[j].get("N")
        return carbon is not None and nitrogen is not None and (carbon, nitrogen) in bonds

    for i, residue in enumerate(residues):
        own = names[i]
        if residue.is_nucleic or residue.ca < 0:
            continue
        n, ca, c = own.get("N"), own.get("CA"), own.get("C")
        # φ needs the previous residue's C and ψ the next residue's N, so a chain end or a
        # break leaves them undefined. ω also needs the next residue's Cα.
        if peptide(i - 1, i) and None not in (n, ca, c):
            table[i, 0] = (names[i - 1]["C"], n, ca, c)
        if peptide(i, i + 1) and None not in (n, ca, c):
            table[i, 1] = (n, ca, c, names[i + 1]["N"])
            if "CA" in names[i + 1]:
                table[i, 2] = (ca, c, names[i + 1]["N"], names[i + 1]["CA"])
        for k, quad in enumerate(_CHI.get(residue.name.upper(), ())):
            if all(name in own for name in quad):
                table[i, 3 + k] = [own[name] for name in quad]
    # Shared through the cache, so read-only.
    table.flags.writeable = False
    topology.__dict__["_torsion_atoms"] = table
    return table


def _kind(name):
    """The KINDS key for a torsion name, accepting Greek letters such as "φ", "χ1" or "χ₁"."""
    # NFKC turns subscript digits into plain ones, so the SYMBOLS labels ("χ₁") read
    # back, and the variant ϕ into φ.
    key = unicodedata.normalize("NFKC", str(name)).lower()
    key = key.replace("χ", "chi").replace("φ", "phi").replace("ψ", "psi").replace("ω", "omega")
    if key not in KINDS:
        raise ValueError(f"Unknown torsion {name!r}; choose from {', '.join(KINDS)}")
    return key


class TorsionAngles:
    """Torsion angles in degrees for a set of residues; NaN where a torsion is undefined.

    ``angles.phi`` returns one value per residue. ``angles[23]`` or
    ``angles["A", 23]`` returns a dictionary for one residue.
    """

    def __init__(self, protein, residue_indices, values):
        self.protein = protein
        self.residue_indices = np.asarray(residue_indices, dtype=int)
        # (residues, 8) in degrees, columns in KINDS order.
        self.values = np.asarray(values, dtype=float)
        self.values.flags.writeable = False

    def __len__(self):
        return len(self.residue_indices)

    def __getattr__(self, name):
        if name in KINDS:
            return self.values[:, KINDS.index(name)]
        raise AttributeError(name)

    @property
    def residue_ids(self):
        """Author identities (chain, residue number, insertion code)."""
        residues = self.protein.topology.residues
        return tuple((residues[i].chain, residues[i].resid, residues[i].icode) for i in self.residue_indices)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        # A key matches a leading part of (chain, residue number, insertion code), or a
        # bare residue number in any chain.
        rows = [
            k
            for k, (chain, resid, icode) in enumerate(self.residue_ids)
            if (chain, resid, icode)[: len(key)] == key or (len(key) == 1 and resid == key[0])
        ]
        if len(rows) != 1:
            raise KeyError(f"{key!r} matches {len(rows)} residues; include the chain or insertion code")
        return {kind: float(value) for kind, value in zip(KINDS, self.values[rows[0]])}

    def __repr__(self):
        residues = self.protein.topology.residues
        lines = ["residue        phi     psi   omega    chi1    chi2"]
        # Show the first 12 residues and the first five torsions (φ, ψ, ω, χ1, χ2).
        for i, row in zip(self.residue_indices[:12], self.values[:12]):
            r = residues[i]
            label = f"{r.chain}:{r.name.title()}{r.resid}{r.icode}"
            lines.append(
                f"{label:<12}" + "".join(f"{v:8.1f}" if np.isfinite(v) else "       –" for v in row[:5])
            )
        if len(self) > 12:
            lines.append(f"… {len(self) - 12} more residues")
        return "\n".join(lines)


def measure(protein, residue_indices=None, xyz=None):
    """TorsionAngles for the given topology residues (default: all) at current coordinates."""
    table = torsion_atoms(protein.topology)
    if residue_indices is None:
        residue_indices = np.arange(len(protein.topology.residues))
    residue_indices = np.asarray(residue_indices, dtype=int)
    xyz = protein.positions if xyz is None else xyz
    return TorsionAngles(protein, residue_indices, dihedrals(xyz, table[residue_indices]))


def _kinematic_bonds(topology):
    """Covalent bonds within residues and between sequence neighbors of one chain.

    Links between non-adjacent residues (disulfides, metal contacts, ligands bonded
    to several residues) are left out, so they never lock the chain.
    """
    cached = topology.__dict__.get("_kinematic_bonds")
    if cached is None:
        bonds = np.asarray(topology.bonds, dtype=int).reshape(-1, 2)
        owner = np.array([a.residue_index for a in topology.atoms])
        chains = np.array([r.chain for r in topology.residues])
        ra, rb = owner[bonds[:, 0]], owner[bonds[:, 1]]
        keep = (ra == rb) | ((abs(ra - rb) == 1) & (chains[ra] == chains[rb]))
        cached = bonds[keep]
        topology.__dict__["_kinematic_bonds"] = cached
    return cached


class _Tree:
    """A spanning tree of kinematic bonds whose subtrees are contiguous in preorder.

    A bond that lies on a cycle, such as one inside a proline or aromatic ring,
    cannot rotate.

    ``order`` lists the atoms in depth-first preorder and ``pre[atom]`` is the atom's
    place in it. ``parent[atom]`` is the atom one bond closer to the root (−1 for a
    root), and ``size[atom]`` counts the atoms in its subtree, itself included.
    ``low[atom]`` is Tarjan's low link: the smallest preorder index the subtree reaches
    through one bond outside the tree.
    """

    def __init__(self, topology, roots):
        n = len(topology.atoms)
        adjacency = [[] for _ in range(n)]
        for a, b in _kinematic_bonds(topology):
            adjacency[a].append(b)
            adjacency[b].append(a)
        parent = np.full(n, -1, dtype=int)
        pre = np.full(n, -1, dtype=int)
        low = np.zeros(n, dtype=int)
        order = []
        # Walk from the anchor roots first. The sweep over every atom afterwards starts a
        # new walk for any atom the roots did not reach.
        for root in [*roots, *range(n)]:
            if pre[root] >= 0:
                continue
            pre[root] = low[root] = len(order)
            order.append(root)
            # An explicit stack rather than recursion: a chain is thousands of bonds deep,
            # past Python's recursion limit. Each entry keeps its neighbor iterator, so a
            # node resumes where it left off when the walk returns to it.
            stack = [(root, iter(adjacency[root]))]
            while stack:
                node, neighbors = stack[-1]
                for other in neighbors:
                    if pre[other] < 0:
                        # A new atom: make it a child and descend into it.
                        parent[other] = node
                        pre[other] = low[other] = len(order)
                        order.append(other)
                        stack.append((other, iter(adjacency[other])))
                        break
                    if other != parent[node]:
                        # A bond back to an atom already visited closes a ring.
                        low[node] = min(low[node], pre[other])
                else:
                    # Every neighbor is done: pass the low link up to the parent.
                    stack.pop()
                    if parent[node] >= 0:
                        low[parent[node]] = min(low[parent[node]], low[node])
        self.order = np.array(order, dtype=int)
        # Children follow their parents in preorder, so a reverse sweep finishes every
        # subtree before adding its size to the parent.
        size = np.ones(n, dtype=int)
        for node in self.order[::-1]:
            if parent[node] >= 0:
                size[parent[node]] += size[node]
        self.parent, self.pre, self.low, self.size = parent, pre, low, size

    def edge(self, b, c):
        """(fixed, moving) atoms of bond b–c, or None if the bond cannot rotate.

        The moving atom is the one farther from the root, so the root's side stays still.
        Bonds outside the tree (ring closures, or atoms that are not bonded) give None too.
        """
        for u, v in ((b, c), (c, b)):
            if self.parent[v] == u:
                # A tree edge on a cycle (low link at or above u) lies in a ring.
                return (u, v) if self.low[v] > self.pre[u] else None
        return None

    def subtree(self, v):
        """Slice of ``order`` holding v and every atom beyond it from the root."""
        return slice(self.pre[v], self.pre[v] + self.size[v])


def _roots(protein, anchor):
    """One root atom per connected component; the root's side of every torsion stays still."""
    topology = protein.topology
    n = len(topology.atoms)
    # Components of the kinematic bond graph: each chain, and each ligand, ion or water.
    bonds = _kinematic_bonds(topology)
    graph = coo_matrix((np.ones(len(bonds)), (bonds[:, 0], bonds[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    owner = np.array([a.residue_index for a in topology.atoms])
    anchored = None
    if anchor is not None and not isinstance(anchor, str):
        from .regions import Region

        if not isinstance(anchor, Region) or anchor.protein is not protein:
            raise ValueError("anchor must be 'center', 'n', 'c' or a Region of this protein")
        anchored = set(anchor.residue_indices)
    elif anchor not in ("center", "auto", "n", "c", None):
        raise ValueError("anchor must be 'center', 'n', 'c' or a Region of this protein")
    roots = []
    for label in np.unique(labels):
        atoms = np.flatnonzero(labels == label)
        residues = sorted(set(owner[atoms]))
        traced = [r for r in residues if topology.residues[r].trace_atom >= 0]
        # A Region anchor picks its root among its own residues in this component.
        if anchored is not None and anchored & set(traced):
            traced = [r for r in traced if r in anchored]
        # Ligands, ions and waters have no trace atom; any atom serves as their root.
        if not traced:
            roots.append(int(atoms[0]))
            continue
        if anchor == "n":
            chosen = traced[0]
        elif anchor == "c":
            chosen = traced[-1]
        else:
            # The residue that splits the component's atoms in half stays still.
            counts = np.cumsum([np.count_nonzero(owner[atoms] == r) for r in traced])
            chosen = traced[int(np.searchsorted(counts, counts[-1] / 2))]
        roots.append(int(topology.residues[chosen].trace_atom))
    return roots


def _tree(protein, anchor):
    """The _Tree for an anchor; trees for string anchors and None are cached on the topology."""
    if not isinstance(anchor, (str, type(None))):
        return _Tree(protein.topology, _roots(protein, anchor))  # Region anchors are not cached
    cache = protein.topology.__dict__.setdefault("_torsion_trees", {})
    if anchor not in cache:
        cache[anchor] = _Tree(protein.topology, _roots(protein, anchor))
    return cache[anchor]


def _rotate(points, origin, axis, angle):
    """Rotate (n, 3) points by ``angle`` radians about the line through origin along axis.

    Rodrigues' formula: p′ = p cos θ + (k × p) sin θ + k (k · p)(1 − cos θ) for p measured
    from the origin and unit axis k.
    """
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    c, s = np.cos(angle), np.sin(angle)
    shifted = points - origin
    return origin + shifted * c + np.cross(axis, shifted) * s + np.outer(shifted @ axis, axis) * (1 - c)


class _Plan:
    """Rotations that change a protein's torsions, applied from shared start coordinates.

    The plan is made once, from the coordinates at that moment (``start``). ``rows`` holds
    one rotation each: (preorder index of the moving atom, fixed atom u, moving atom v,
    change in radians, topology residue index, torsion kind, position in ``residues``).
    """

    def __init__(self, protein, residues, requested, *, relative=False, anchor="center", path="allowed"):
        if path not in ("allowed", "shortest"):
            raise ValueError("path must be 'allowed' or 'shortest'")
        self.protein = protein
        table = torsion_atoms(protein.topology)
        start = np.asarray(protein.positions, dtype=np.float64)
        self.start = start
        tree = _tree(protein, anchor)
        self.order = tree.order
        # A Region anchor is also held in place as a whole, even if its own torsions change.
        # Superposing it back needs at least three atoms.
        self.hold = None if isinstance(anchor, (str, type(None))) else anchor.atom_indices
        if self.hold is not None and len(self.hold) < 3:
            self.hold = None
        rows, locked, missing = [], [], []
        # The change for each requested torsion in degrees, keyed by (position in residues,
        # KINDS column). A target value turns the short way, wrapped to (−180°, 180°]; a
        # relative value is used as given, so it can exceed a half turn.
        current = dihedrals(start, table[residues])
        changes = {}
        for column, kind in enumerate(KINDS):
            values = requested.get(kind)
            if values is not None:
                for k, value in enumerate(values):
                    if np.isfinite(value) and np.isfinite(current[k, column]):
                        changes[k, column] = (
                            float(value) if relative else float(wrap(value - current[k, column]))
                        )
        if path == "allowed" and not relative:
            _allowed_paths(protein, residues, current, changes)
        # Turn each change into a rotation about the torsion's central bond, or record why
        # it cannot be made: the residue lacks the torsion, or its bond lies in a ring.
        for column, kind in enumerate(KINDS):
            values = requested.get(kind)
            if values is None:
                continue
            for k, (ri, value) in enumerate(zip(residues, values)):
                if value is None or not np.isfinite(value):
                    continue
                quad = table[ri, column]
                if quad[0] < 0:
                    missing.append((ri, kind))
                    continue
                delta = changes[k, column]
                edge = tree.edge(quad[1], quad[2])
                if edge is None:
                    locked.append((ri, kind))
                    continue
                if abs(delta) > 1e-9:
                    u, v = edge
                    rows.append((tree.pre[v], u, v, np.radians(delta), ri, kind, k))
        if not rows and (locked or missing):
            names = ", ".join(sorted({kind for _, kind in locked + missing}))
            reason = "lie in rings (for example proline φ)" if locked else "are undefined for these residues"
            raise ValueError(f"The requested torsions ({names}) {reason}; nothing can rotate")
        # Order the rotations from the root outward. Each one reads its axis from
        # coordinates already moved by the rotations before it, and a fixed order gives
        # the same floating-point result every time a frame is evaluated.
        rows.sort(key=lambda row: row[0])
        self.rows = rows
        self.locked, self.missing = tuple(locked), tuple(missing)
        self.slices = [tree.subtree(v) for _, _, v, *_ in rows]
        self.deltas = np.array([row[3] for row in rows])
        self.residues = np.array([row[4] for row in rows], dtype=int)
        self.kinds = tuple(row[5] for row in rows)

    def coordinates(self, fractions=1.0):
        """Coordinates with each rotation applied by the given fraction of its change."""
        bonds = [(u, v, delta) for _, u, v, delta, *_ in self.rows]
        result = rotate_bonds(self.start, self.order, bonds, self.slices, fractions)
        if self.hold is not None:
            from .math3d import align_coordinates

            # Superpose the held region back onto its start pose (Kabsch).
            return coordinates(align_coordinates(result, self.start, self.hold))
        return coordinates(result)


def rotate_bonds(start, order, bonds, slices, fractions=1.0):
    """Rotate each bond's moving subtree by a fraction of its angle (radians), in preorder."""
    fractions = np.broadcast_to(np.asarray(fractions, dtype=float), (len(bonds),))
    # Work in preorder, where each subtree is one contiguous slice. position[atom] is the
    # atom's row in that order.
    xyz = np.asarray(start, dtype=np.float64)[order]
    position = np.empty(len(order), dtype=int)
    position[order] = np.arange(len(order))
    for (u, v, delta), part, fraction in zip(bonds, slices, fractions):
        if fraction == 0:
            continue
        # The axis runs from u to v at their current positions, so a rotation applied
        # earlier to an enclosing subtree carries this bond along with it.
        origin = xyz[position[u]]
        xyz[part] = _rotate(xyz[part], origin, xyz[position[v]] - origin, delta * fraction)
    result = np.empty_like(xyz)
    result[order] = xyz
    return result


def _allowed_paths(protein, residues, current, changes):
    """Turn φ and ψ the way that keeps each residue's path in allowed Ramachandran regions.

    Each residue's (φ, ψ) moves in a straight line in torsion space. Of the (up to)
    four directions that reach the target, keep the one whose path spends the most
    time in favored and allowed regions; ties go to the shorter turn.
    """
    from .ramachandran import regions

    topology = protein.topology
    table = torsion_atoms(topology)
    # 49 samples along each candidate path, from start (t = 0) to target (t = 1).
    t = np.linspace(0, 1, 49)[:, None]
    for k, ri in enumerate(residues):
        # Each changing angle can reach its target the short way (delta) or the long way
        # round (delta ∓ 360°).
        options = []
        for column in (0, 1):
            delta = changes.get((k, column))
            if delta is None or abs(delta) < 1e-9:
                options.append((0.0,))
            else:
                options.append((delta, delta - 360.0 * np.sign(delta)))
        if len(options[0]) == len(options[1]) == 1:
            continue
        name = topology.residues[ri].name
        # Pre-proline means peptide-bonded to the proline. ψ is defined only across such a
        # bond in the same chain, so a chain end or a break is never pre-proline.
        following = topology.residues[ri + 1].name if table[ri, 1, 0] >= 0 else ""
        background = (
            "glycine"
            if name == "GLY"
            else "proline"
            if name == "PRO"
            else "preproline"
            if following == "PRO"
            else "general"
        )
        levels = regions(background)
        step = 360 / levels.shape[0]
        # A terminal residue lacks φ or ψ; score its path at a typical β value instead.
        start = np.where(np.isfinite(current[k, :2]), current[k, :2], [-120.0, 140.0])
        best = None
        for dphi in options[0]:
            for dpsi in options[1]:
                track = wrap(start + t * [dphi, dpsi])
                cells = np.clip(((track + 180) // step).astype(int), 0, levels.shape[0] - 1)
                # Mean region level along the path (2 favored, 1 allowed, 0 outside), less
                # 1/3600 per degree turned. The turn cost is small next to the levels and
                # settles near ties in favour of the shorter turn.
                score = levels[cells[:, 1], cells[:, 0]].mean() - (abs(dphi) + abs(dpsi)) / 3600
                if best is None or score > best[0] + 1e-9:
                    best = (score, dphi, dpsi)
        for column, delta in zip((0, 1), best[1:]):
            if (k, column) in changes:
                changes[k, column] = delta


def _targets(protein, residues, values, conformation):
    """Per-kind arrays aligned with ``residues``; None where a kind is not requested.

    Returns {kind: (len(residues),) degrees}. NaN entries leave that residue's torsion as it is.
    """
    requested = {}
    if conformation is not None:
        key = str(conformation).lower().replace("-", "").replace("_", "").replace(" ", "")
        if key not in CONFORMATIONS:
            raise ValueError(f"Unknown conformation {conformation!r}; choose from {', '.join(CONFORMATIONS)}")
        phi, psi = CONFORMATIONS[key]
        # Explicit angles override the conformation's φ and ψ.
        values = {"phi": phi, "psi": psi, **{k: v for k, v in values.items() if v is not None}}
    topology = protein.topology
    for name, value in values.items():
        if value is None:
            continue
        kind = _kind(name)
        if isinstance(value, dict):
            # {residue number or (chain, residue number[, insertion code]): angle}
            identities = [
                (topology.residues[ri].chain, topology.residues[ri].resid, topology.residues[ri].icode)
                for ri in residues
            ]
            column = np.full(len(residues), np.nan)
            for key, angle in value.items():
                key = key if isinstance(key, tuple) else (key,)
                rows = [
                    k
                    for k, identity in enumerate(identities)
                    if identity[: len(key)] == key or (len(key) == 1 and identity[1] == key[0])
                ]
                if len(rows) != 1:
                    raise ValueError(f"Residue {key!r} matches {len(rows)} selected residues")
                column[rows[0]] = float(angle)
        elif np.ndim(value) == 0:
            column = np.full(len(residues), float(value))
        else:
            column = np.asarray(value, dtype=float)
            if column.shape != (len(residues),):
                raise ValueError(f"{kind} needs one value per selected residue ({len(residues)})")
        if np.isinf(column).any():
            raise ValueError("Torsion angles must be finite degrees")
        requested[kind] = column
    if not requested:
        raise ValueError("Supply at least one torsion (phi, psi, omega, chi1–chi5) or a conformation")
    return requested


def _residues(target):
    """(protein, topology indices of the amino acids with a Cα) for a Protein or Region."""
    from .protein import Protein
    from .regions import Region

    if isinstance(target, Region):
        target._validate()
        protein, residues = target.protein, target.residue_indices
    elif isinstance(target, Protein):
        protein, residues = target, np.arange(len(target.topology.residues))
    else:
        raise TypeError("Torsions need a Protein or a Region")
    residues = np.array(
        [
            r
            for r in residues
            if protein.topology.residues[r].ca >= 0 and not protein.topology.residues[r].is_nucleic
        ],
        dtype=int,
    )
    if not len(residues):
        raise ValueError("Selection contains no amino acids with torsion angles")
    return protein, residues


def _changed_secondary(protein, before, after):
    """Current cartoon weights, with DSSP states that differ between two conformations replaced.

    Residues the change does not affect keep their current (possibly deposited) assignment.
    """
    from .secondary import assign, weights

    old, new = assign(protein.topology, before), assign(protein.topology, after)
    changed = np.array([a != b for a, b in zip(old, new)], dtype=bool)
    result = np.where(changed[:, None], weights(new), protein._secondary).astype(np.float32)
    result.flags.writeable = False
    return result


def set_torsions(
    target, *, conformation=None, anchor="center", secondary="auto", relative=False, path="allowed", **values
):
    """Change torsions immediately, without animation. Angles are in degrees."""
    if secondary not in ("auto", "keep", None):
        raise ValueError("secondary must be 'auto' or 'keep'")
    protein, residues = _residues(target)
    requested = _targets(protein, residues, values, conformation)
    plan = _Plan(protein, residues, requested, relative=relative, anchor=anchor, path=path)
    xyz = plan.coordinates(1.0)
    if secondary == "auto":
        protein._secondary = _changed_secondary(protein, plan.start, xyz)
    protein.set_positions(xyz)
    return target


class SetTorsions(Animation):
    """Animate backbone and side-chain torsions to new values, in degrees.

    Give ``phi``, ``psi``, ``omega`` or ``chi1``–``chi5`` as one value for every
    selected residue, one value per residue, or a ``{residue_number: angle}``
    dictionary. ``conformation="helix"`` (or strand, ppii, 310, pi, extended, left)
    sets φ and ψ together. φ and ψ turn the way that keeps each residue in allowed
    Ramachandran regions (``path="allowed"``); ``path="shortest"`` takes the
    smaller turn, as χ angles always do. Atoms rotate
    about the torsion's bond, so bond lengths and angles stay fixed; the side of
    each bond nearest ``anchor`` stays still ("center" keeps the middle of each
    chain in place, "n" or "c" a terminus, or pass a Region to hold it in place
    as a whole).
    ``secondary="auto"`` blends the cartoon to the DSSP assignment of the final
    conformation. ``stagger`` (a fraction of the clip) starts residues one after
    another, N to C, or C to N with ``reverse=True``.
    Torsions in rings, such as proline φ, are skipped; ``locked`` lists them.
    """

    # The scene refuses two concurrent animations that write the same channel of a protein.
    channels = frozenset({"geometry", "secondary"})
    # The clip clock stays linear so stagger delays are true fractions of the clip; each
    # residue applies ``easing`` to its own progress (_progress).
    requires_linear_timeline = True
    relative = False

    def __init__(
        self,
        target,
        *,
        phi=None,
        psi=None,
        omega=None,
        chi1=None,
        chi2=None,
        chi3=None,
        chi4=None,
        chi5=None,
        conformation=None,
        anchor="center",
        secondary="auto",
        path="allowed",
        stagger=None,
        reverse=False,
        easing="smooth",
    ):
        self.protein, self.residues = _residues(target)
        super().__init__(self.protein, rate_func=linear)
        values = dict(phi=phi, psi=psi, omega=omega, chi1=chi1, chi2=chi2, chi3=chi3, chi4=chi4, chi5=chi5)
        self.requested = _targets(self.protein, self.residues, values, conformation)
        if secondary not in ("auto", "keep", None):
            raise ValueError("secondary must be 'auto' or 'keep'")
        if stagger is not None and (not np.isfinite(stagger) or not 0 <= stagger < 1):
            raise ValueError("stagger is the fraction of the clip used for delays, in [0, 1)")
        if path not in ("allowed", "shortest"):
            raise ValueError("path must be 'allowed' or 'shortest'")
        self.anchor, self.secondary, self.path = anchor, secondary == "auto", path
        self.stagger, self.reverse, self.easing = stagger, bool(reverse), resolve(easing)
        _roots(self.protein, anchor)  # validate the anchor now

    def _set_easing(self, easing):
        self.easing = resolve(easing)

    @property
    def locked(self):
        """(residue index, torsion) pairs that could not rotate because they lie in rings."""
        return self.plan.locked

    def bind(self):
        super().bind()
        protein = self.protein
        # The plan is made from the coordinates at the start of the clip. apply() always
        # works from them, not from the previous frame, so seeking to any time gives the
        # same result.
        self.plan = _Plan(
            protein, self.residues, self.requested, relative=self.relative, anchor=self.anchor, path=self.path
        )
        # Stagger: rank each rotation by its residue among the residues that change (0 is
        # the most N-terminal). Residue starts are spread evenly over [0, stagger] of the
        # clip, and each residue turns over the remaining 1 − stagger.
        ranks = np.searchsorted(np.unique(self.plan.residues), self.plan.residues)
        count = max(1, len(np.unique(self.plan.residues)))
        if self.reverse:
            ranks = count - 1 - ranks
        fraction = 0.0 if self.stagger is None else self.stagger
        self.begin = ranks * fraction / max(1, count - 1)
        self.span = 1 - fraction
        self.start_weights = protein._secondary
        self.end_weights = (
            _changed_secondary(protein, self.plan.start, self.plan.coordinates(1.0))
            if self.secondary
            else self.start_weights
        )
        # Rows of the plan that belong to each changing residue.
        self.residue_rows = {
            ri: np.flatnonzero(self.plan.residues == ri) for ri in np.unique(self.plan.residues)
        }
        # The path each residue's (φ, ψ) point takes, for Ramachandran plots.
        table = torsion_atoms(protein.topology)
        shown = np.array(sorted(self.residue_rows), dtype=int)
        self.path_residues = shown
        self.path_start = dihedrals(self.plan.start, table[shown][:, :2]) if len(shown) else np.empty((0, 2))
        self.path_delta = np.zeros((len(shown), 2))
        for k, ri in enumerate(shown):
            for row in self.residue_rows[ri]:
                kind = self.plan.kinds[row]
                if kind in ("phi", "psi"):
                    self.path_delta[k, ("phi", "psi").index(kind)] = np.degrees(self.plan.deltas[row])

    def _progress(self, alpha):
        """Eased progress in [0, 1] of each plan row at clip time ``alpha``."""
        local = np.clip((float(alpha) - self.begin) / self.span, 0, 1)
        return np.array([evaluate(self.easing, t) for t in local]) if len(local) else local

    def apply(self, alpha):
        protein = self.protein
        fractions = self._progress(alpha)
        xyz = self.plan.coordinates(fractions)
        # Both ends of the protein's coordinate blend are the new coordinates, so
        # ``positions`` returns them unchanged.
        protein._pair(xyz, xyz, alpha)
        if self.secondary:
            # Residues without their own torsion changes (strand partners) follow the clip.
            blend = np.full(len(self.start_weights), float(evaluate(self.easing, alpha)))
            for ri, rows in self.residue_rows.items():
                blend[ri] = fractions[rows].mean()
            weights = (1 - blend[:, None]) * self.start_weights + blend[:, None] * self.end_weights
            weights = weights.astype(np.float32)
            weights.flags.writeable = False
            protein._secondary = weights
        # For RamachandranPlot: (residues, start (φ, ψ), change (Δφ, Δψ), clip progress),
        # angles in degrees.
        protein._torsion_motion = (self.path_residues, self.path_start, self.path_delta, float(alpha))


class RotateTorsions(SetTorsions):
    """Rotate torsions by the given amounts in degrees, rather than to target values.

    Use it to spin a side chain or turn a bond through more than 180°:
    ``RotateTorsions(residue, chi1=120)``.
    """

    relative = True


def _degrees(value, precision):
    """An angle label such as "−47°", with a true minus sign and never "−0°"."""
    text = f"{abs(value):.{precision}f}"
    return ("−" if value < 0 and float(text) != 0 else "") + text + "°"


class TorsionMarker(Annotation):
    """A live protractor for one torsion: its bond, the angle's arc and its value.

    The arc turns about the torsion's central bond from the first atom's direction
    to the last atom's (Newman projection), so it opens and closes as the torsion
    changes. ``Write`` draws the bond and arc, then the label. The label sits on a
    translucent ``backing`` card (None to omit it).
    """

    def __init__(
        self,
        target,
        torsion="phi",
        *,
        radius=1.25,
        color="#f5d477",
        font_size=30,
        precision=0,
        label=None,
        show_value=True,
        line_width=2.6,
        follow_opacity=True,
        backing="#0b1220",
        opacity=1.0,
    ):
        from .regions import Region

        super().__init__()
        if not isinstance(target, Region):
            raise TypeError("TorsionMarker needs a single-residue Region, e.g. protein.select(residues=8)")
        # The card is 72% opaque.
        self.backing = None if backing is None else (parse_color(backing), 0.72)
        target._validate()
        if len(target.residue_indices) != 1:
            raise ValueError("TorsionMarker needs exactly one residue")
        for name, value in (("radius", radius), ("font_size", font_size), ("line_width", line_width)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(precision, int) or not 0 <= precision <= 3:
            raise ValueError("precision must be between 0 and 3")
        self.kind = _kind(torsion)
        self.protein, self.residue = target.protein, int(target.residue_indices[0])
        quad = torsion_atoms(self.protein.topology)[self.residue, KINDS.index(self.kind)]
        if quad[0] < 0:
            r = self.protein.topology.residues[self.residue]
            raise ValueError(f"{r.name.title()} {r.resid} has no {self.kind} torsion")
        self.quad = np.array(quad, dtype=int)
        self.radius, self.precision, self.line_width = float(radius), precision, float(line_width)
        self.symbol = SYMBOLS[self.kind] if label is None else str(label)
        self.show_value, self.follow_opacity = bool(show_value), bool(follow_opacity)
        self.color = parse_color(color)
        self.title = Text(self._caption(), font_size=font_size, font="semibold", color=color)
        self.set_opacity(opacity)

    @property
    def value(self):
        """The current torsion in degrees."""
        return float(dihedrals(self.protein.positions, self.quad[None])[0])

    def _caption(self):
        return f"{self.symbol}  {_degrees(self.value, self.precision)}" if self.show_value else self.symbol

    @property
    def glyph_count(self):
        return self.title.glyph_count

    @property
    def text_progress(self):
        # Write draws the strokes over its first 45% (_strokes) and the label from 30% on.
        return float(np.clip((self._write - 0.3) / 0.7, 0, 1))

    def layout(self, camera, width, height):
        from .annotations import AnnotationLayout, Placement
        from .distances import project

        frame = self._newman_frame()
        if frame is None:
            return AnnotationLayout([], [])
        b, c, axis, u, w = frame
        theta = np.radians(self.value)
        # The arc lies in the plane normal to the bond, centered on its midpoint, with a
        # radius of ``radius`` Å. It starts along u and turns by the torsion about the bond,
        # which brings it to w.
        center = (b + c) / 2
        r = self.radius * self.protein.size
        steps = np.linspace(0, 1, 33)[:, None]
        arc3 = center + r * (np.cos(steps * theta) * u + np.sin(steps * theta) * np.cross(axis, u))
        # Screen pixels; draw nothing if any point is behind the camera or off screen.
        points = [project(point, camera, width, height) for point in (b, c, center, *arc3)]
        if any(point is None for point in points):
            return AnnotationLayout([], [])
        pb, pc, pcenter, arc = points[0], points[1], points[2], np.array(points[3:])
        # Spokes run along u and w a little past the arc; the label sits farther out on the
        # arc's bisector.
        spoke_u = project(center + 1.25 * r * u, camera, width, height)
        spoke_w = project(center + 1.25 * r * w, camera, width, height)
        middle = np.cos(theta / 2) * u + np.sin(theta / 2) * np.cross(axis, u)
        anchor = project(center + 2.1 * r * middle, camera, width, height)
        if spoke_u is None or spoke_w is None or anchor is None:
            return AnnotationLayout([], [])
        # Sizes are in pixels at a 1080-pixel render height.
        scale = height / 1080
        visibility = float(self.protein.atom_opacities[self.quad].min()) if self.follow_opacity else 1.0
        leaders = self._strokes(pb, pc, pcenter, arc, spoke_u, spoke_w, theta, scale, visibility)
        self.title.set_text(self._caption())
        size = np.array([self.title.geometry.width, self.title.geometry.height]) * scale
        origin = anchor - size / 2
        triangles = None
        if self.backing is not None:
            triangles = self._backing_triangles(origin, size, scale, visibility)
        return AnnotationLayout(
            [Placement(self.title, origin, opacity=visibility)],
            leaders,
            np.r_[origin, origin + size],
            triangles=triangles,
        )

    def _newman_frame(self):
        """World-space bond and Newman-projection directions, or None if they are degenerate.

        Returns (b, c, axis, u, w): the central bond's atoms, the bond's unit direction b→c,
        and unit vectors normal to the bond pointing toward the first and the last atom.
        """
        # Model to world coordinates, so the marker follows the protein's pose.
        m = self.protein.model_matrix
        a, b, c, d = self.protein.positions[self.quad].astype(float) @ m[:3, :3].T + m[:3, 3]
        axis = c - b
        length = np.linalg.norm(axis)
        if length < 1e-6:
            return None
        axis /= length
        u = (a - b) - np.dot(a - b, axis) * axis
        w = (d - c) - np.dot(d - c, axis) * axis
        # An outer atom on the bond's line has no direction in the projection.
        if np.linalg.norm(u) < 1e-6 or np.linalg.norm(w) < 1e-6:
            return None
        u, w = u / np.linalg.norm(u), w / np.linalg.norm(w)
        return b, c, axis, u, w

    def _strokes(self, pb, pc, pcenter, arc, spoke_u, spoke_w, theta, scale, visibility):
        """Leaders for the bond, the two spokes, the arc and its arrowhead, in screen pixels."""
        from .annotations import Leader, _partial_path

        # Write draws every stroke over its first 45%.
        drawn = float(np.clip(self._write / 0.45, 0, 1))
        leaders = []
        bond = _partial_path(np.array([pb, pc]), drawn)
        if len(bond) > 1:
            leaders.append(Leader(bond, self.color, self.line_width * 1.6 * scale, 0.9 * visibility))
        # A (2, 2, 2) array draws two separate segments, both growing out from the center.
        spokes = np.array([[pcenter, spoke_u], [pcenter, spoke_w]])
        if drawn > 0:
            partial = pcenter + (spokes[:, 1] - pcenter) * drawn
            leaders.append(
                Leader(
                    np.stack((spokes[:, 0], partial), 1),
                    self.color,
                    self.line_width * 0.6 * scale,
                    0.55 * visibility,
                )
            )
        path = _partial_path(arc, drawn)
        if len(path) > 1:
            leaders.append(Leader(path, self.color, self.line_width * scale, visibility))
        # The arrowhead shows which way the torsion turns. Arcs under 12° are too short for one.
        if drawn >= 1 and abs(theta) > np.radians(12):
            barbs = self._arrowhead(arc, scale)
            if barbs is not None:
                leaders.append(Leader(barbs, self.color, self.line_width * scale, visibility))
        return leaders

    def _arrowhead(self, arc, scale):
        """Two barbs at the end of the screen-space arc, or None if its last step has no direction."""
        tip, before = arc[-1], arc[-4]
        direction = tip - before
        if np.linalg.norm(direction) > 1e-6:
            direction /= np.linalg.norm(direction)
            side = np.array([-direction[1], direction[0]])
            # Barbs 9 pixels long (at 1080p), spread 0.55 to the side: about 29° each way.
            head = 9 * scale
            return np.array(
                [
                    [tip, tip - head * direction + 0.55 * head * side],
                    [tip, tip - head * direction - 0.55 * head * side],
                ]
            )
        return None

    def _backing_triangles(self, origin, size, scale, visibility):
        """The label's backing card as two triangles: (6, 6) float32 rows of x, y, r, g, b, a."""
        # A soft dark card keeps the value readable over atoms.
        lo, hi = origin - [10 * scale, 6 * scale], origin + size + [10 * scale, 8 * scale]
        corners = np.array(
            [
                [lo[0], lo[1]],
                [hi[0], lo[1]],
                [hi[0], hi[1]],
                [lo[0], lo[1]],
                [hi[0], hi[1]],
                [lo[0], hi[1]],
            ]
        )
        # The card fades in with the label.
        alpha = self.backing[1] * visibility * self.text_progress
        return np.column_stack((corners, np.broadcast_to(np.r_[self.backing[0], alpha], (6, 4)))).astype(
            np.float32
        )
