"""Morph between two experimental structures of the same protein.

``match_structures`` pairs chains by sequence (and position, for identical
copies), residues by number or sequence alignment, and atoms by name, including
ligands such as hemes, and finds the rigid core that the two states share.
``StructureMorph`` superposes the target on the source and moves every matched residue and ligand along a screw path: the rotation and
translation that carry it to its target position are applied in proportion, so
rigid domains swing along arcs instead of cutting across chords and keep their
bond lengths. Side chains turn about their bonds toward the target χ angles,
and the small remainder blends in each residue's moving frame; bond lengths then
relax toward values interpolated between the structures. Atoms present in only
one structure fade out or in while moving with their residue or nearest
neighbor. At the end the target structure replaces the source, styled as the
source was.
"""

from dataclasses import dataclass, field

import gemmi
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .animation import Animation
from .rates import evaluate, linear, resolve
from .structure import coordinates

BACKBONE = frozenset(
    {"N", "CA", "C", "O", "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"}
)


@dataclass
class StructureMatch:
    """Atom correspondence between two structures, with the residue and chain pairs behind it.

    ``core`` holds the source's matched Cα/C4′ atoms that superpose within 2 Å: the
    part of the structure that does not move between the states.
    """

    source_atoms: np.ndarray
    target_atoms: np.ndarray
    residue_pairs: list = field(default_factory=list)
    chain_pairs: list = field(default_factory=list)
    core: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    core_rmsd: float = float("nan")

    @property
    def report(self):
        """Chain pairs with sequence identity, counts of matched residues and atoms, and the rigid core."""
        return {
            "chains": [(a, b, round(identity, 3)) for a, b, identity in self.chain_pairs],
            "matched_residues": len(self.residue_pairs),
            "matched_atoms": len(self.source_atoms),
            "core_atoms": len(self.core),
            "core_rmsd": round(self.core_rmsd, 2),
        }

    def _anchors(self, source):
        """Matched trace-atom pairs of the rigid core, or of every residue if there is none."""
        trace = {r.trace_atom for r in source.topology.residues if r.trace_atom >= 0}
        allowed = set(self.core) if len(self.core) >= 3 else trace
        return [(a, b) for a, b in zip(self.source_atoms, self.target_atoms) if a in trace and a in allowed]


def _letters(topology, indices):
    out = []
    for i in indices:
        r = topology.residues[i]
        if r.is_nucleic:
            out.append(r.base or "N")
        else:
            code = gemmi.find_tabulated_residue(r.name)
            out.append((code.one_letter_code.upper() if code is not None else "X") or "X")
    return "".join(out)


def _polymer_chains(topology):
    chains = {}
    categories = topology.residue_categories
    for i, r in enumerate(topology.residues):
        if categories[i] == "polymer":
            chains.setdefault(r.chain, []).append(i)
    return chains


def _align_residues(source, target, s_ids, t_ids, *, min_block=4):
    """Residue index pairs: by author number when the numbering agrees, else by sequence.

    Aligned runs shorter than ``min_block`` residues between gaps are dropped: such
    fragments (a loop paired with part of a longer stem, a last residue paired across
    a terminal insertion) are rarely the same structure, and would fly across the morph.
    """
    s_res, t_res = source.topology.residues, target.topology.residues
    by_number = {(t_res[j].resid, t_res[j].icode): j for j in t_ids}
    numbered = [(i, by_number.get((s_res[i].resid, s_res[i].icode))) for i in s_ids]
    same = [(i, j) for i, j in numbered if j is not None and s_res[i].name == t_res[j].name]
    if len(same) >= 0.8 * min(len(s_ids), len(t_ids)):
        return same
    a, b = _letters(source.topology, s_ids), _letters(target.topology, t_ids)
    result = gemmi.align_string_sequences(list(a), list(b), [])
    qa, qb = result.add_gaps(a, 1), result.add_gaps(b, 2)
    blocks, ia, ib = [[]], 0, 0
    for x, y in zip(qa, qb):
        if x != "-" and y != "-":
            blocks[-1].append((s_ids[ia], t_ids[ib]))
        elif blocks[-1]:
            blocks.append([])
        ia += x != "-"
        ib += y != "-"
    return [pair for block in blocks if len(block) >= min_block for pair in block]


def _identity(source, target, s_ids, t_ids):
    a, b = _letters(source.topology, s_ids), _letters(target.topology, t_ids)
    return gemmi.align_string_sequences(list(a), list(b), []).calculate_identity() / 100


def _trace(protein, indices):
    return np.array([protein.topology.residues[i].trace_atom for i in indices])


def match_structures(source, target, *, chains=None):
    """Pair chains, residues, ligands and atoms of two structures of the same protein.

    ``chains`` maps source chain IDs to target chain IDs; by default chains with the
    same ID and sequence are paired, and identical copies (homo-oligomers) are
    paired by position after superposing the first pair.
    """
    s_chains, t_chains = _polymer_chains(source.topology), _polymer_chains(target.topology)
    if not s_chains or not t_chains:
        raise ValueError("Both structures need at least one protein or nucleic-acid chain")
    pairs = []
    if chains is not None:
        for a, b in chains.items():
            if a not in s_chains or b not in t_chains:
                raise ValueError(f"Chain pair {a!r} → {b!r} is not in the structures")
            pairs.append((a, b, _identity(source, target, s_chains[a], t_chains[b])))
    else:
        identities = {
            (a, b): _identity(source, target, s_chains[a], t_chains[b]) for a in s_chains for b in t_chains
        }
        used, fit = set(), None
        for a in s_chains:
            options = [b for b in t_chains if b not in used and identities[a, b] >= 0.3]
            if not options:
                continue
            best = max(identities[a, b] for b in options)
            close = [b for b in options if identities[a, b] >= best - 0.05]
            if a in close and (fit is None or len(close) == 1):
                chosen = a
            elif fit is not None and len(close) > 1:
                # Identical copies: choose the one in the same place after superposition.
                center = source.positions[_trace(source, s_chains[a])].mean(0)
                distances = [
                    np.linalg.norm(fit(target.positions[_trace(target, t_chains[b])]).mean(0) - center)
                    for b in close
                ]
                chosen = close[int(np.argmin(distances))]
            else:
                chosen = a if a in close else close[0]
            used.add(chosen)
            pairs.append((a, chosen, identities[a, chosen]))
            if fit is None:
                residue_pairs = _align_residues(source, target, s_chains[a], t_chains[chosen])
                if len(residue_pairs) >= 3:
                    s_xyz = source.positions[_trace(source, [i for i, _ in residue_pairs])].astype(float)
                    t_xyz = target.positions[_trace(target, [j for _, j in residue_pairs])].astype(float)
                    rotation, translation = _kabsch(t_xyz, s_xyz)
                    fit = lambda xyz, r=rotation, t=translation: xyz @ r.T + t  # noqa: E731
    if not pairs:
        raise ValueError("No chains of the target match the source sequences")
    residue_pairs = []
    for a, b, _ in pairs:
        residue_pairs.extend(_align_residues(source, target, s_chains[a], t_chains[b]))
    # Ligands, ions and waters: same chain mapping, name and number, else the nearest same-named group.
    chain_map = {a: b for a, b, _ in pairs}
    s_categories, t_categories = source.topology.residue_categories, target.topology.residue_categories
    s_other = [i for i, c in enumerate(s_categories) if c != "polymer"]
    t_other = {}
    for j, c in enumerate(t_categories):
        if c != "polymer":
            r = target.topology.residues[j]
            t_other.setdefault((r.chain, r.name, r.resid), j)
    taken = {j for _, j in residue_pairs}
    for i in s_other:
        r = source.topology.residues[i]
        j = t_other.get((chain_map.get(r.chain, r.chain), r.name, r.resid))
        if j is not None and j not in taken:
            residue_pairs.append((i, j))
            taken.add(j)
    s_names, t_names = source.topology.residue_atoms, target.topology.residue_atoms
    xs, xt = np.asarray(source.positions, dtype=float), np.asarray(target.positions, dtype=float)
    source_atoms, target_atoms = [], []
    for i, j in residue_pairs:
        partner = {name: t_names[j][name] for name in s_names[i] if name in t_names[j]}
        _symmetric_names(source.topology.residues[i].name, s_names[i], partner, t_names[j], xs, xt)
        for name, a in s_names[i].items():
            if name in partner:
                source_atoms.append(a)
                target_atoms.append(partner[name])
    order = np.argsort(source_atoms)
    source_atoms = np.asarray(source_atoms, dtype=int)[order]
    target_atoms = np.asarray(target_atoms, dtype=int)[order]
    trace = {r.trace_atom for r in source.topology.residues if r.trace_atom >= 0}
    anchors = np.array([(a, b) for a, b in zip(source_atoms, target_atoms) if a in trace], dtype=int)
    core, rmsd = np.empty(0, dtype=int), float("nan")
    if len(anchors) >= 3:
        keep, rmsd = _rigid_core(xt[anchors[:, 1]], xs[anchors[:, 0]])
        core = anchors[keep, 0]
    return StructureMatch(source_atoms, target_atoms, residue_pairs, pairs, core, rmsd)


def _rigid_core(moving, reference, cutoff=2.0):
    """Pairs that superpose within ``cutoff`` Å, found by pruning the worst pairs and refitting.

    Each round removes the worst tenth of the pairs beyond the cutoff, as in ChimeraX
    matchmaker, so a large moving domain cannot drag the fit away from the rest.
    """
    keep = np.ones(len(moving), bool)
    floor = max(3, int(0.05 * len(moving)))
    while True:
        rotation, translation = _kabsch(moving[keep], reference[keep])
        deviation = np.linalg.norm(moving @ rotation.T + translation - reference, axis=1)
        far = np.flatnonzero(keep & (deviation > cutoff))
        if not len(far) or keep.sum() - 1 < floor:
            break
        drop = far[np.argsort(-deviation[far])][: max(1, int(np.ceil(0.1 * keep.sum())))]
        drop = drop[: keep.sum() - floor]
        keep[drop] = False
    return keep, float(np.sqrt(np.mean(deviation[keep] ** 2)))


# Atom names that label chemically equivalent positions; structures may name them either way.
SYMMETRIC = {
    "PHE": (("CD1", "CD2"), ("CE1", "CE2")),
    "TYR": (("CD1", "CD2"), ("CE1", "CE2")),
    "ASP": (("OD1", "OD2"),),
    "GLU": (("OE1", "OE2"),),
    "ARG": (("NH1", "NH2"),),
    "LEU": (("CD1", "CD2"),),
    "VAL": (("CG1", "CG2"),),
}


def _symmetric_names(residue, source_names, partner, target_names, xs, xt):
    """Swap symmetric atom pairs when the other naming lies closer, after superposing the residues."""
    pairs = SYMMETRIC.get(residue.upper())
    if not pairs or not all(a in partner and b in partner for a, b in pairs):
        return
    core = [n for n in ("N", "CA", "C", "CB") if n in partner]
    if len(core) < 3:
        return
    rotation, translation = _kabsch(xs[[source_names[n] for n in core]], xt[[partner[n] for n in core]])

    def cost(swapped):
        total = 0.0
        for a, b in pairs:
            for name, other in ((a, b), (b, a)):
                target = target_names[other] if swapped else target_names[name]
                total += np.sum((xs[source_names[name]] @ rotation.T + translation - xt[target]) ** 2)
        return total

    if cost(True) < cost(False):
        for a, b in pairs:
            partner[a], partner[b] = target_names[b], target_names[a]


def _kabsch(moving, reference):
    """Proper rotation R and translation t with moving @ R.T + t ≈ reference."""
    mc, rc = moving.mean(0), reference.mean(0)
    u, _, vt = np.linalg.svd((moving - mc).T @ (reference - rc))
    d = np.sign(np.linalg.det(u @ vt))
    rotation = (u @ np.diag([1, 1, d]) @ vt).T
    return rotation, rc - mc @ rotation.T


def _screws(rotations, translations):
    """Screw parameters (angle, axis, point on the axis, translation along it) per rigid motion."""
    vectors = Rotation.from_matrix(rotations).as_rotvec()
    angles = np.linalg.norm(vectors, axis=1)
    axes = np.where(angles[:, None] > 1e-8, vectors / np.maximum(angles, 1e-12)[:, None], [0.0, 0.0, 1.0])
    along = np.sum(translations * axes, axis=1, keepdims=True) * axes
    points = np.zeros_like(translations)
    turning = angles > 1e-6
    for k in np.flatnonzero(turning):
        # The axis passes through p with (I − R) p = t − (t·u) u; take the solution nearest the origin.
        points[k] = np.linalg.lstsq(np.eye(3) - rotations[k], translations[k] - along[k], rcond=None)[0]
    along[~turning] = translations[~turning]
    return angles, axes, points, along


def _rotations(angles, axes):
    """Rodrigues rotation matrices, one per (angle, axis)."""
    k = np.zeros((len(axes), 3, 3))
    k[:, 0, 1], k[:, 0, 2], k[:, 1, 2] = -axes[:, 2], axes[:, 1], -axes[:, 0]
    k[:, 1, 0], k[:, 2, 0], k[:, 2, 1] = axes[:, 2], -axes[:, 1], axes[:, 0]
    s, c = np.sin(angles)[:, None, None], np.cos(angles)[:, None, None]
    return np.eye(3) + s * k + (1 - c) * (k @ k)


def _owners(protein, groups_of_residue, xyz, matched_atoms):
    """Rigid group of every atom: its residue's group, a sequence neighbor's, or the nearest matched atom's."""
    topology = protein.topology
    group = np.full(len(topology.atoms), -1, dtype=int)
    residue_group = np.full(len(topology.residues), -1, dtype=int)
    for ri, g in groups_of_residue.items():
        residue_group[ri] = g
    # Unmatched polymer residues take the nearest matched residue of their chain.
    for chain in topology.chains:
        known = [k for k, ri in enumerate(chain) if residue_group[ri] >= 0]
        if known:
            for k, ri in enumerate(chain):
                if residue_group[ri] < 0:
                    residue_group[ri] = residue_group[chain[min(known, key=lambda m: abs(m - k))]]
    owner = np.array([a.residue_index for a in topology.atoms])
    group = residue_group[owner]
    loose = np.flatnonzero(group < 0)
    if len(loose):
        tree = cKDTree(xyz[matched_atoms])
        _, nearest = tree.query(xyz[loose])
        group[loose] = group[matched_atoms[nearest]]
    return group


class StructureMorph(Animation):
    """Morph between two experimental structures of the same protein, such as two states.

    The structures can differ in chains, missing residues, ligands and atoms:
    match_structures() pairs them. ``align`` (a Region of the source) chooses the atoms
    used to superpose the target; by default the rigid core, the matched Cα/C4′ atoms
    that superpose within 2 Å, so moving domains do not shift the frame. Matched
    residues and ligands move along screw paths, so rigid domains rotate along arcs
    with fixed bond lengths; side-chain changes blend in each residue's frame.
    Atoms in only one structure fade out (``fade_out``) or in (``fade_in``),
    given as fractions of the clip. The target then replaces the source, taking
    the source's place, representation and colors (``style="source"``; use
    ``style="target"`` to keep the target's own). ``secondary="auto"`` blends
    residues whose DSSP state differs between the structures. The path is an
    illustration of the change between two endpoints, not a calculated pathway.
    """

    channels = frozenset({"geometry", "transform", "opacity", "controls", "secondary"})
    requires_linear_timeline = True

    def __init__(
        self,
        source,
        target,
        *,
        align=None,
        chains=None,
        match=None,
        fade_out=(0.0, 0.35),
        fade_in=(0.65, 1.0),
        style="source",
        secondary="auto",
        easing="smooth",
    ):
        from .protein import Protein

        if not isinstance(source, Protein) or not isinstance(target, Protein) or source is target:
            raise ValueError("StructureMorph needs two different Protein objects")
        if align is not None and (getattr(align, "protein", None) is not source):
            raise ValueError("align must be a Region of the source structure")
        for interval in (fade_out, fade_in):
            if len(interval) != 2 or not 0 <= interval[0] < interval[1] <= 1:
                raise ValueError("Fade intervals must satisfy 0 <= start < end <= 1")
        if style not in ("source", "target"):
            raise ValueError("style must be 'source' or 'target'")
        if secondary not in ("auto", "keep"):
            raise ValueError("secondary must be 'auto' or 'keep'")
        super().__init__(source, rate_func=linear)
        self.destination, self.align, self.chains, self.match = target, align, chains, match
        self.fade_out, self.fade_in, self.style = fade_out, fade_in, style
        self.secondary, self.easing = secondary == "auto", resolve(easing)

    def _set_easing(self, easing):
        self.easing = resolve(easing)

    @property
    def targets(self):
        return (self.target, self.destination)

    @property
    def result(self):
        """The target structure, which replaces the source when the morph ends."""
        return self.destination

    def bind(self):
        super().bind()
        source, dest = self.target, self.destination
        match = self.match if self.match is not None else match_structures(source, dest, chains=self.chains)
        if len(match.source_atoms) < 3:
            raise ValueError("The structures share fewer than three atoms")
        self.match_result = match
        xs = np.asarray(source.positions, dtype=np.float64)
        xt = np.asarray(dest.positions, dtype=np.float64)
        # Superpose the target on the source by the rigid core's trace atoms (or the align region).
        trace = {r.trace_atom for r in source.topology.residues if r.trace_atom >= 0}
        if self.align is not None:
            allowed = set(self.align.atom_indices)
            anchors = [
                (a, b) for a, b in zip(match.source_atoms, match.target_atoms) if a in trace and a in allowed
            ]
        else:
            anchors = match._anchors(source)
        if len(anchors) < 3:
            anchors = list(zip(match.source_atoms, match.target_atoms))
        anchors = np.asarray(anchors, dtype=int)
        rotation, translation = _kabsch(xt[anchors[:, 1]], xs[anchors[:, 0]])
        xt = xt @ rotation.T + translation
        # One rigid group per matched residue or ligand.
        groups, rotations, translations = {}, [], []
        s_owner = np.array([a.residue_index for a in source.topology.atoms])
        pair_of = dict(zip(match.source_atoms, match.target_atoms))
        for i, j in match.residue_pairs:
            atoms = [a for a in np.flatnonzero(s_owner == i) if a in pair_of]
            if not atoms:
                continue
            core = [a for a in atoms if source.topology.atoms[a].name in BACKBONE]
            core = core if len(core) >= 3 else atoms
            if len(core) >= 3:
                r, t = _kabsch(xs[core], xt[[pair_of[a] for a in core]])
            else:
                r, t = np.eye(3), (xt[[pair_of[a] for a in core]] - xs[core]).mean(0)
            groups[i] = len(rotations)
            rotations.append(r)
            translations.append(t)
        rotations, translations = np.array(rotations), np.array(translations)
        self.angles, self.axes, self.points, self.along = _screws(rotations, translations)
        matched = np.asarray(match.source_atoms)
        self.source_group = _owners(source, groups, xs, matched)
        target_groups = {j: groups[i] for i, j in match.residue_pairs if i in groups}
        self.dest_group = _owners(dest, target_groups, xt, np.asarray(match.target_atoms))
        # Side chains turn about their bonds toward the target χ angles, the short way round.
        from .torsions import _tree, dihedrals, rotate_bonds, torsion_atoms, wrap

        tree = _tree(source, "center")
        table = torsion_atoms(source.topology)
        quads = [
            table[i, column]
            for i, _ in match.residue_pairs
            for column in range(3, 8)
            if table[i, column][0] >= 0 and all(a in pair_of for a in table[i, column])
        ]
        rows = []
        if quads:
            quads = np.array(quads)
            change = wrap(dihedrals(xt, np.vectorize(pair_of.get)(quads)) - dihedrals(xs, quads))
            for quad, delta in zip(quads, np.radians(change)):
                edge = tree.edge(quad[1], quad[2])
                if edge is not None and abs(delta) > 1e-6:
                    rows.append((tree.pre[edge[1]], edge[0], edge[1], delta))
        rows.sort(key=lambda row: row[0])
        self.chi_order = tree.order
        self.chi_bonds = [(u, v, delta) for _, u, v, delta in rows]
        self.chi_slices = [tree.subtree(v) for _, _, v, _ in rows]
        turned = rotate_bonds(xs, tree.order, self.chi_bonds, self.chi_slices, 1.0) if rows else xs
        # What remains after the rigid motion and χ turns blends in each residue's moving frame.
        self.xs = xs
        residual = np.zeros_like(xs)
        g = self.source_group[matched]
        rigid = np.einsum("nij,nj->ni", rotations[g], turned[matched]) + translations[g]
        residual[matched] = np.einsum("nji,nj->ni", rotations[g], xt[match.target_atoms] - rigid)
        self.residual = residual
        # Bond lengths relax toward lengths interpolated between the two structures.
        bonds = np.asarray(source.topology.bonds, dtype=int).reshape(-1, 2)
        partner = np.full(len(xs), -1, dtype=int)
        partner[match.source_atoms] = match.target_atoms
        self.relax_bonds = bonds
        self.rest_source = np.linalg.norm(xs[bonds[:, 0]] - xs[bonds[:, 1]], axis=1)
        self.rest_target = self.rest_source.copy()
        both = (partner[bonds[:, 0]] >= 0) & (partner[bonds[:, 1]] >= 0)
        self.rest_target[both] = np.linalg.norm(
            xt[partner[bonds[both, 0]]] - xt[partner[bonds[both, 1]]], axis=1
        )
        self.relax_degree = np.maximum(np.bincount(bonds.ravel(), minlength=len(xs)), 1)
        # Target-only atoms start carried back to the source frame by their group's motion.
        self.dest_matched = np.zeros(len(dest.topology.atoms), bool)
        self.dest_matched[match.target_atoms] = True
        self.dest_partner = np.full(len(dest.topology.atoms), -1, dtype=int)
        self.dest_partner[match.target_atoms] = match.source_atoms
        gd = self.dest_group
        self.dest_start = np.einsum("nji,nj->ni", rotations[gd], xt - translations[gd])
        self.dest_end = xt
        # Opacity: the source draws matched atoms until the end; the target then takes over.
        n_s, n_d = len(source.topology.atoms), len(dest.topology.atoms)
        s_matched = np.zeros(n_s, bool)
        s_matched[matched] = True
        self.source_controls = self._controls(n_s, s_matched, 1, 0, self.fade_out)
        self.dest_controls = self._controls(n_d, self.dest_matched, 0, 1, self.fade_in)
        self.dest_final_controls = dest._controls
        self.source_opacity, self.dest_opacity = source.opacity, dest.opacity or 1.0
        # Secondary structure: blend residues whose DSSP state changes; the target inherits the result.
        self.start_secondary = source._secondary
        self.end_secondary = source._secondary
        self.dest_secondary = dest._secondary
        if self.secondary:
            from .secondary import assign, weights

            before = assign(source.topology, xs)
            after = assign(dest.topology, xt)
            end = np.array(source._secondary, dtype=np.float32)
            dest_secondary = np.array(dest._secondary, dtype=np.float32)
            final = weights(after)
            for i, j in match.residue_pairs:
                if before[i] != after[j]:
                    end[i] = final[j]
                dest_secondary[j] = end[i]
            end.flags.writeable = dest_secondary.flags.writeable = False
            self.end_secondary, self.dest_secondary = end, dest_secondary
        self.dest_style = self._style(source, dest, match) if self.style == "source" else None

    @staticmethod
    def _controls(count, matched, start, end, fade):
        controls = np.zeros((count, 8), np.float32)
        controls[:, 1] = 1
        # Matched atoms switch objects at the very end; the rest fade over their interval.
        controls[matched, 4:] = [start, end, 1 - 1e-4, 1e-4]
        controls[~matched, 4:] = [start, end, fade[0], fade[1] - fade[0]]
        controls.flags.writeable = False
        return controls

    @staticmethod
    def _style(source, dest, match):
        """Representation, colors and residue overrides of the source, mapped onto the target."""
        appearance = np.array(dest._appearance)
        appearance[match.target_atoms] = source._appearance[match.source_atoms]
        appearance.flags.writeable = False
        names = (
            "color_scheme", "atom_scale", "bond_radius", "ribbon_width", "surface_opacity",
            "backbone_radius", "base_thickness", "base_radius",
        )  # fmt: skip
        values = {k: getattr(source, k) for k in names}
        values["representation"] = np.array(source.representation, copy=True)
        values["base_style"] = np.array(source.base_style, copy=True)
        mixes = (source._color_mix, source._opacity_mix, source._detail_mix)
        return values, appearance, mixes, source._surface_options

    def _relax(self, xyz, alpha, iterations=12):
        """Nudge bonds toward lengths interpolated between the structures (Jacobi projection)."""
        bonds, n = self.relax_bonds, len(xyz)
        rest = (1 - alpha) * self.rest_source + alpha * self.rest_target
        for _ in range(iterations):
            d = xyz[bonds[:, 1]] - xyz[bonds[:, 0]]
            length = np.linalg.norm(d, axis=1)
            correction = ((length - rest) / np.maximum(length, 1e-9))[:, None] * d * 0.5
            for k in range(3):
                xyz[:, k] += (
                    np.bincount(bonds[:, 0], correction[:, k], n)
                    - np.bincount(bonds[:, 1], correction[:, k], n)
                ) / self.relax_degree
        return xyz

    def _positions(self, alpha):
        from .torsions import rotate_bonds

        a = float(alpha)
        rotations = _rotations(self.angles * a, self.axes)
        g = self.source_group
        base = (
            rotate_bonds(self.xs, self.chi_order, self.chi_bonds, self.chi_slices, a)
            if self.chi_bonds
            else self.xs
        )
        moved = base - self.points[g] + a * self.residual
        xs = np.einsum("nij,nj->ni", rotations[g], moved) + self.points[g] + a * self.along[g]
        if 0 < a < 1 and len(self.relax_bonds):
            xs = self._relax(xs, a)
        gd = self.dest_group
        xd = np.einsum("nij,nj->ni", rotations[gd], self.dest_start - self.points[gd]) + self.points[gd]
        xd += a * self.along[gd]
        partners = self.dest_partner >= 0
        xd[partners] = xs[self.dest_partner[partners]]
        return xs, xd

    def apply(self, alpha):
        source, dest = self.target, self.destination
        eased = evaluate(self.easing, alpha)
        xs, xd = self._positions(eased)
        if alpha >= 1:
            # Land exactly on the target coordinates.
            partners = self.dest_partner >= 0
            xs[self.dest_partner[partners]] = self.dest_end[partners]
            xd = self.dest_end
        xs, xd = coordinates(xs), coordinates(xd)
        source._pair(xs, xs, alpha)
        dest._pair(xd, xd, alpha)
        for name in ("position", "orientation"):
            setattr(dest, name, np.array(getattr(source, name), copy=True))
        dest.size = source.size
        if self.dest_style is not None:
            values, appearance, mixes, surface = self.dest_style
            for key, value in values.items():
                setattr(dest, key, value)
            dest._appearance = appearance
            dest._color_mix, dest._opacity_mix, dest._detail_mix = mixes
            dest._surface_options = surface
        source._controls = self.source_controls
        dest._controls = self.dest_final_controls if alpha >= 1 else self.dest_controls
        source.opacity = self.source_opacity if alpha < 1 else 0.0
        dest.opacity = self.dest_opacity
        blend = float(eased)
        weights = ((1 - blend) * self.start_secondary + blend * self.end_secondary).astype(np.float32)
        weights.flags.writeable = False
        source._secondary = weights
        dest._secondary = self.dest_secondary


@dataclass(frozen=True)
class DomainMotion:
    """Rigid motion of a domain between two structures, in the source's model frame.

    ``angle`` is the rotation in degrees about ``axis`` (a unit vector) through
    ``point``; ``translation`` is the shift along the axis in Å.
    """

    angle: float
    axis: np.ndarray
    point: np.ndarray
    translation: float


def domain_motion(source, target, moving, fixed=None, *, match=None):
    """How a domain (a Region of ``source``) turns and shifts between two structures.

    The target is first superposed on the source using the matched Cα/C4′ atoms of
    ``fixed`` (default: the rigid core of the match). The motion is then the screw that
    carries the domain's matched trace atoms from the source to the target.
    """
    match = match_structures(source, target) if match is None else match
    trace = {r.trace_atom for r in source.topology.residues if r.trace_atom >= 0}
    pairs = [(a, b) for a, b in zip(match.source_atoms, match.target_atoms) if a in trace]
    if fixed is not None:
        fixed_atoms = set(fixed.atom_indices)
        anchors = [(a, b) for a, b in pairs if a in fixed_atoms]
    else:
        anchors = match._anchors(source)
    domain_atoms = set(moving.atom_indices)
    domain = [(a, b) for a, b in pairs if a in domain_atoms]
    if len(anchors) < 3 or len(domain) < 3:
        raise ValueError("The fixed and moving regions each need three matched trace atoms")
    xs, xt = np.asarray(source.positions, dtype=float), np.asarray(target.positions, dtype=float)
    rotation, translation = _kabsch(xt[[b for _, b in anchors]], xs[[a for a, _ in anchors]])
    xt = xt @ rotation.T + translation
    rotation, translation = _kabsch(xs[[a for a, _ in domain]], xt[[b for _, b in domain]])
    angles, axes, points, along = _screws(rotation[None], translation[None])
    axis, point = axes[0], points[0]
    # Report the axis point nearest the domain's center.
    center = xs[[a for a, _ in domain]].mean(0)
    point = point + np.dot(center - point, axis) * axis
    return DomainMotion(float(np.degrees(angles[0])), axis, point, float(np.dot(along[0], axis)))
