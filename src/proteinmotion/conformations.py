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

Scripts play ``StructureMorph`` in a scene; ``match_structures`` returns the
``StructureMatch`` it uses, and ``domain_motion`` reports one domain's rigid motion as
a screw (angle, axis, shift). For two different proteins, whose atoms cannot be paired
by name, ``BackboneMorph`` (backbone.py) with the contact matching of matching.py does
the job instead.
"""

from dataclasses import dataclass, field

import gemmi
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .animation import Animation
from .math3d import kabsch
from .rates import evaluate, linear, resolve
from .structure import coordinates

# Backbone atom names of amino acids and nucleotides. A residue with at least three of
# them matched takes its rigid motion from the backbone alone, so a side chain or base
# that changes does not tilt it.
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
    """One-letter sequence for alignment: base letters (N if unknown) or amino-acid codes (X)."""
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
    """Residue indices of the polymer residues of each chain, keyed by chain ID."""
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
    # Trust the author numbering when it pairs residues of the same name for at least
    # 80% of the shorter chain.
    by_number = {(t_res[j].resid, t_res[j].icode): j for j in t_ids}
    numbered = [(i, by_number.get((s_res[i].resid, s_res[i].icode))) for i in s_ids]
    same = [(i, j) for i, j in numbered if j is not None and s_res[i].name == t_res[j].name]
    if len(same) >= 0.8 * min(len(s_ids), len(t_ids)):
        return same
    # Otherwise align the sequences, then walk the gapped strings and collect the runs
    # of aligned residues between gaps. ia and ib index the next residue of each sequence.
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
    """Sequence identity of two residue lists after global alignment, as a fraction in [0, 1]."""
    a, b = _letters(source.topology, s_ids), _letters(target.topology, t_ids)
    return gemmi.align_string_sequences(list(a), list(b), []).calculate_identity() / 100


def _trace(protein, indices):
    """Trace atom (Cα; C4′ or P for nucleotides) of each residue; −1 where it is missing."""
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
    pairs = _pair_chains(source, target, s_chains, t_chains, chains)
    if not pairs:
        raise ValueError("No chains of the target match the source sequences")
    # Residues of each chain pair, by number or by sequence; then ligands, ions and waters.
    residue_pairs = []
    for a, b, _ in pairs:
        residue_pairs.extend(_align_residues(source, target, s_chains[a], t_chains[b]))
    _pair_other_residues(source, target, pairs, residue_pairs)
    xs, xt = np.asarray(source.positions, dtype=float), np.asarray(target.positions, dtype=float)
    source_atoms, target_atoms = _pair_atoms(source, target, residue_pairs, xs, xt)
    # The rigid core: matched trace atoms that superpose within 2 Å.
    trace = {r.trace_atom for r in source.topology.residues if r.trace_atom >= 0}
    anchors = np.array([(a, b) for a, b in zip(source_atoms, target_atoms) if a in trace], dtype=int)
    core, rmsd = np.empty(0, dtype=int), float("nan")
    if len(anchors) >= 3:
        keep, rmsd = _rigid_core(xt[anchors[:, 1]], xs[anchors[:, 0]])
        core = anchors[keep, 0]
    return StructureMatch(source_atoms, target_atoms, residue_pairs, pairs, core, rmsd)


def _pair_chains(source, target, s_chains, t_chains, chains):
    """(source chain, target chain, sequence identity) triples, from ``chains`` or by sequence."""
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
        # fit maps target coordinates onto the source. It comes from the first chain pair
        # with at least three aligned residues.
        used, fit = set(), None
        for a in s_chains:
            # Unused target chains with at least 30% identity are candidates. Those within
            # 0.05 of the best are equally good, like the copies in a homo-oligomer.
            options = [b for b in t_chains if b not in used and identities[a, b] >= 0.3]
            if not options:
                continue
            best = max(identities[a, b] for b in options)
            close = [b for b in options if identities[a, b] >= best - 0.05]
            # Prefer the same chain ID, unless several copies are equally good and the
            # superposition can tell them apart by position.
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
                # Superpose by the trace atoms of this pair's aligned residues.
                residue_pairs = _align_residues(source, target, s_chains[a], t_chains[chosen])
                if len(residue_pairs) >= 3:
                    s_xyz = source.positions[_trace(source, [i for i, _ in residue_pairs])].astype(float)
                    t_xyz = target.positions[_trace(target, [j for _, j in residue_pairs])].astype(float)
                    rotation, translation = _kabsch(t_xyz, s_xyz)
                    fit = lambda xyz, r=rotation, t=translation: xyz @ r.T + t  # noqa: E731
    return pairs


def _pair_other_residues(source, target, pairs, residue_pairs):
    """Append ligand, ion and water pairs to ``residue_pairs``, each target residue used once."""
    # Ligands, ions and waters: same chain mapping, name and number. Groups on unpaired
    # chains keep their chain ID.
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


def _pair_atoms(source, target, residue_pairs, xs, xt):
    """Matched atom indices, (source_atoms, target_atoms), sorted by source atom."""
    s_names, t_names = source.topology.residue_atoms, target.topology.residue_atoms
    source_atoms, target_atoms = [], []
    # Atoms pair by name within each residue pair; symmetric names such as Phe CD1/CD2
    # may swap to whichever naming fits better.
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
    return source_atoms, target_atoms


def _rigid_core(moving, reference, cutoff=2.0):
    """Pairs that superpose within ``cutoff`` Å, found by pruning the worst pairs and refitting.

    Each round removes the worst tenth of the pairs beyond the cutoff, as in ChimeraX
    matchmaker, so a large moving domain cannot drag the fit away from the rest.
    """
    keep = np.ones(len(moving), bool)
    # Keep at least 5% of the pairs, and never fewer than the three a rotation needs.
    floor = max(3, int(0.05 * len(moving)))
    while True:
        rotation, translation = _kabsch(moving[keep], reference[keep])
        deviation = np.linalg.norm(moving @ rotation.T + translation - reference, axis=1)
        far = np.flatnonzero(keep & (deviation > cutoff))
        if not len(far) or keep.sum() - 1 < floor:
            break
        # Drop the farthest pairs, up to a tenth of the kept pairs (at least one) per round,
        # and no more than the floor allows.
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
    """Swap symmetric atom pairs when the other naming lies closer, after superposing the residues.

    ``partner`` maps source atom names to target atom indices and is updated in place.
    """
    pairs = SYMMETRIC.get(residue.upper())
    if not pairs or not all(a in partner and b in partner for a, b in pairs):
        return
    # Superpose by N, CA, C and CB, whose names do not depend on the convention.
    core = [n for n in ("N", "CA", "C", "CB") if n in partner]
    if len(core) < 3:
        return
    rotation, translation = _kabsch(xs[[source_names[n] for n in core]], xt[[partner[n] for n in core]])

    # Summed squared deviation of the symmetric atoms under one naming or the other.
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
    """Proper rotation R and translation t with moving @ R.T + t ≈ reference.

    The column-vector form of math3d.kabsch. Screw motions below act on single
    points as ``R @ x``, so this module keeps R in that orientation.
    """
    rotation, translation = kabsch(moving, reference)
    return rotation.T, translation


def _screws(rotations, translations):
    """Screw parameters (angle, axis, point on the axis, translation along it) per rigid motion.

    Takes (k, 3, 3) column-vector rotations R and (k, 3) translations t in Å. Returns
    angles (k,) in radians, unit axes u (k, 3), points p (k, 3) and shifts d (k, 3) with
    R x + t = R (x − p) + p + d. For a real turn d is the part of t along u; with almost
    no turn, p = 0 and d = t.
    """
    # By Chasles' theorem every rigid motion is a turn about one axis plus a shift along
    # that axis. Playing a fraction of the turn and of the shift moves points on helices.
    vectors = Rotation.from_matrix(rotations).as_rotvec()
    angles = np.linalg.norm(vectors, axis=1)
    # A motion with no turn has no axis of its own; z stands in.
    axes = np.where(angles[:, None] > 1e-8, vectors / np.maximum(angles, 1e-12)[:, None], [0.0, 0.0, 1.0])
    along = np.sum(translations * axes, axis=1, keepdims=True) * axes
    points = np.zeros_like(translations)
    turning = angles > 1e-6
    for k in np.flatnonzero(turning):
        # The axis passes through p with (I − R) p = t − (t·u) u; take the solution nearest the origin.
        points[k] = np.linalg.lstsq(np.eye(3) - rotations[k], translations[k] - along[k], rcond=None)[0]
    # Below 1e-6 rad the axis point is ill-conditioned, so the whole translation is the shift.
    along[~turning] = translations[~turning]
    return angles, axes, points, along


def _rotations(angles, axes):
    """Rodrigues rotation matrices, one per (angle, axis)."""
    # k is the cross-product matrix K of each unit axis: R = I + sin θ K + (1 − cos θ) K².
    k = np.zeros((len(axes), 3, 3))
    k[:, 0, 1], k[:, 0, 2], k[:, 1, 2] = -axes[:, 2], axes[:, 1], -axes[:, 0]
    k[:, 1, 0], k[:, 2, 0], k[:, 2, 1] = axes[:, 2], -axes[:, 1], axes[:, 0]
    s, c = np.sin(angles)[:, None, None], np.cos(angles)[:, None, None]
    return np.eye(3) + s * k + (1 - c) * (k @ k)


def _trace_neighbours(protein, pair_of, reach=2):
    """Matched trace atoms of each one-atom residue and its sequence neighbours.

    A Cα-only residue has no backbone of its own to define a rotation, so its frame
    comes from the trace atoms within ``reach`` residues along the chain.
    """
    topology = protein.topology
    names = topology.residue_atoms
    result = {}
    for chain in topology.chains:
        chain = [int(i) for i in chain]
        trace = [topology.residues[i].trace_atom for i in chain]
        for k, ri in enumerate(chain):
            if len(names[ri]) > 1:
                continue
            # The window includes the residue's own trace atom; three are the fewest that
            # define a rotation.
            window = [a for a in trace[max(0, k - reach) : k + reach + 1] if a in pair_of]
            if len(window) >= 3:
                result[ri] = window
    return result


def _bounded_turn(rotvec, start, end, bow=0.15, floor=3.0):
    """Rotation and translation carrying one atom from start to end on a gentle screw path.

    A one-atom residue's rotation only shapes its path. The turn is scaled down until
    the path's midpoint lies within ``bow`` of the distance travelled (at least
    ``floor`` Å) of the straight line.
    """
    allowed = max(floor, bow * float(np.linalg.norm(end - start)))
    for _ in range(8):
        # Whatever the turn, t brings the atom exactly to ``end``; only the path changes.
        r = Rotation.from_rotvec(rotvec).as_matrix()
        t = end - r @ start
        # Where the screw path puts the atom half-way through the morph.
        angle, axis, point, along = (v[0] for v in _screws(r[None], t[None]))
        middle = _rotations(np.array([angle / 2]), axis[None])[0] @ (start - point) + point + along / 2
        if np.linalg.norm(middle - (start + end) / 2) <= allowed:
            break
        rotvec = rotvec * 0.6
    else:
        # No turn was gentle enough after eight tries: move in a straight line.
        r, t = np.eye(3), end - start
    return r, t


def _flatness(points):
    """How far points depart from a line: second over first singular value of their spread.

    0 for points on a line, up to 1 when they spread equally in two directions.
    """
    values = np.linalg.svd(points - points.mean(0), compute_uv=False)
    return float(values[1] / max(values[0], 1e-9))


def _virtual_trace_bonds(protein):
    """Pairs of consecutive trace atoms in chains whose residues have only a trace atom."""
    topology = protein.topology
    names = topology.residue_atoms
    pairs = []
    # Chains are already split where consecutive trace atoms lie far apart, so each pair
    # here is a real chain neighbour.
    for chain in topology.chains:
        chain = [int(i) for i in chain]
        if all(len(names[i]) == 1 for i in chain):
            trace = [topology.residues[i].trace_atom for i in chain]
            pairs.extend(zip(trace[:-1], trace[1:]))
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def _owners(protein, groups_of_residue, xyz, matched_atoms):
    """Rigid group of every atom: its residue's group, a sequence neighbor's, or the nearest matched atom's."""
    topology = protein.topology
    residue_group = np.full(len(topology.residues), -1, dtype=int)
    for ri, g in groups_of_residue.items():
        residue_group[ri] = g
    # Unmatched polymer residues take the nearest matched residue of their chain.
    # A tie goes to the residue earlier in the chain.
    for chain in topology.chains:
        known = [k for k, ri in enumerate(chain) if residue_group[ri] >= 0]
        if known:
            for k, ri in enumerate(chain):
                if residue_group[ri] < 0:
                    residue_group[ri] = residue_group[chain[min(known, key=lambda m: abs(m - k))]]
    owner = np.array([a.residue_index for a in topology.atoms])
    group = residue_group[owner]
    # Atoms still without a group (unmatched ligands and waters, or chains with no matched
    # residue) follow the nearest matched atom in space.
    loose = np.flatnonzero(group < 0)
    if len(loose):
        tree = cKDTree(xyz[matched_atoms])
        _, nearest = tree.query(xyz[loose])
        group[loose] = group[matched_atoms[nearest]]
    return group


def _rigid_groups(source, match, pair_of, xs, xt):
    """One rigid motion per matched residue or ligand, carrying it from source to target.

    ``xt`` is the target already superposed on the source. Returns ``groups`` (source
    residue index → group number), the groups' rotations R (g, 3, 3, acting as R @ x) and
    their translations t (g, 3) in Å, so that R @ x + t carries a source atom near its partner.
    """
    groups, rotations, translations = {}, [], []
    s_owner = np.array([a.residue_index for a in source.topology.atoms])
    neighbours = _trace_neighbours(source, pair_of)
    for i, j in match.residue_pairs:
        atoms = [a for a in np.flatnonzero(s_owner == i) if a in pair_of]
        if not atoms:
            continue
        core = [a for a in atoms if source.topology.atoms[a].name in BACKBONE]
        core = core if len(core) >= 3 else atoms
        if len(core) < 3 and len(neighbours.get(i, ())) >= 3:
            # Cα-only models: the residue turns with the trace of its sequence
            # neighbours, as far as they move rigidly. Where the chain bends, or
            # where the window is nearly straight and its turn is ill-defined, a
            # fitted rotation would swing the residue around a distant axis, so the
            # turn is scaled down and the residue moves more directly to its target.
            r, t = _trace_turn(xs, xt, neighbours[i], pair_of, atoms[0])
        elif len(core) >= 3:
            r, t = _kabsch(xs[core], xt[[pair_of[a] for a in core]])
        else:
            # One or two matched atoms cannot define a turn: shift by their mean displacement.
            r, t = np.eye(3), (xt[[pair_of[a] for a in core]] - xs[core]).mean(0)
        groups[i] = len(rotations)
        rotations.append(r)
        translations.append(t)
    return groups, np.array(rotations), np.array(translations)


def _trace_turn(xs, xt, window, pair_of, own):
    """Rotation and translation of a Cα-only residue, from the trace atoms around it.

    ``window`` lists the matched trace atoms near the residue (see _trace_neighbours),
    and ``own`` is its single matched atom.
    """
    target = xt[[pair_of[a] for a in window]]
    r, _ = _kabsch(xs[window], target)
    spread = min(_flatness(xs[window]), _flatness(target))
    # RMS misfit of the window after the best rigid fit. Removing the mean error is the
    # same as applying the best translation.
    error = xs[window] @ r.T - target
    misfit = np.sqrt(max(float(np.mean(np.sum((error - error.mean(0)) ** 2, axis=1))), 0.0))
    # Keep the whole turn while the window fits within 0.5 Å and its flatness is at least
    # 0.35; keep none once the misfit reaches 1.5 Å or the window is straight.
    rigid = float(np.clip((1.5 - misfit) / 1.0, 0.0, 1.0) * np.clip(spread / 0.35, 0.0, 1.0))
    return _bounded_turn(Rotation.from_matrix(r).as_rotvec() * rigid, xs[own], xt[pair_of[own]])


class StructureMorph(Animation):
    """Morph between two experimental structures of the same protein, such as two states.

    Cα-only models, such as predicted Cα traces, also work: each residue turns with
    the trace of its sequence neighbours, and consecutive Cα atoms keep their spacing.

    The structures can differ in chains, missing residues, ligands and atoms:
    match_structures() pairs them. ``align`` (a Region of the source) chooses the atoms
    used to superpose the target (``align=False`` keeps the target where it is, for
    structures already superposed); by default the rigid core, the matched Cα/C4′ atoms
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
        if align is not None and align is not False and (getattr(align, "protein", None) is not source):
            raise ValueError("align must be a Region of the source structure, or False")
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

    # The clip clock stays linear (rate_func=linear), because the fade intervals are fractions
    # of the clip. An easing passed to play() shapes the motion instead.
    def _set_easing(self, easing):
        self.easing = resolve(easing)

    @property
    def targets(self):
        # The morph drives both structures, so the scene adds the target and locks its channels too.
        return (self.target, self.destination)

    @property
    def result(self):
        """The target structure, which replaces the source when the morph ends."""
        return self.destination

    def bind(self):
        # How the morph works. Every matched residue or ligand is a rigid group with its
        # own rotation R and translation t from source to target. Any rigid motion is a
        # screw: a turn about one axis plus a shift along it. apply() plays the fraction
        # alpha of each group's turn and shift, so its atoms move on helices, and a domain
        # whose groups share one motion swings on an arc and keeps its shape, where
        # straight-line interpolation would cut across the chord and squeeze it. What the
        # rigid motion leaves over is handled by χ turns of the side chains and by a
        # residual offset stored in each group's own frame, which turns with the group.
        # Bonds then relax toward interpolated lengths. Atoms present in only one
        # structure ride the group of their residue or of a nearby atom while they fade.
        super().bind()
        source, dest = self.target, self.destination
        match = self.match if self.match is not None else match_structures(source, dest, chains=self.chains)
        if len(match.source_atoms) < 3:
            raise ValueError("The structures share fewer than three atoms")
        self.match_result = match
        # Work in double precision; apply() hands float32 copies to the renderer.
        xs = np.asarray(source.positions, dtype=np.float64)
        xt = self._superpose(match, xs, np.asarray(dest.positions, dtype=np.float64))
        pair_of = dict(zip(match.source_atoms, match.target_atoms))
        groups, rotations, translations = _rigid_groups(source, match, pair_of, xs, xt)
        # Each group's motion as a screw, which _positions plays in proportion to alpha.
        self.angles, self.axes, self.points, self.along = _screws(rotations, translations)
        # Every atom of both structures, matched or not, moves with one group.
        matched = np.asarray(match.source_atoms)
        self.source_group = _owners(source, groups, xs, matched)
        target_groups = {j: groups[i] for i, j in match.residue_pairs if i in groups}
        self.dest_group = _owners(dest, target_groups, xt, np.asarray(match.target_atoms))
        turned = self._bind_side_chains(source, match, pair_of, xs, xt)
        # What remains after the rigid motion and χ turns blends in each residue's moving frame.
        # rigid is where the full screw puts each matched atom after its χ turns. R.T maps the
        # leftover offset into the group's starting frame; _positions adds a fraction of it
        # before turning the group, so the offset turns with the residue.
        self.xs = xs
        residual = np.zeros_like(xs)
        g = self.source_group[matched]
        rigid = np.einsum("nij,nj->ni", rotations[g], turned[matched]) + translations[g]
        residual[matched] = np.einsum("nji,nj->ni", rotations[g], xt[match.target_atoms] - rigid)
        self.residual = residual
        self._bind_relaxation(source, match, xs, xt)
        self._bind_target_start(dest, match, rotations, translations, xt)
        self._bind_fades(source, dest, matched)
        self._bind_secondary(source, dest, match, xs, xt)
        self.dest_style = self._style(source, dest, match) if self.style == "source" else None

    def _superpose(self, match, xs, xt):
        """Target coordinates (atoms, 3) superposed on the source, unless ``align=False``."""
        # Superpose the target on the source by the rigid core's trace atoms (or the align region).
        source = self.target
        trace = {r.trace_atom for r in source.topology.residues if r.trace_atom >= 0}
        if self.align is not None and self.align is not False:
            allowed = set(self.align.atom_indices)
            anchors = [
                (a, b) for a, b in zip(match.source_atoms, match.target_atoms) if a in trace and a in allowed
            ]
        else:
            anchors = match._anchors(source)
        # Too few trace atoms in the region or core: fit every matched atom instead.
        if len(anchors) < 3:
            anchors = list(zip(match.source_atoms, match.target_atoms))
        anchors = np.asarray(anchors, dtype=int)
        if self.align is not False:
            rotation, translation = _kabsch(xt[anchors[:, 1]], xs[anchors[:, 0]])
            xt = xt @ rotation.T + translation
        return xt

    def _bind_side_chains(self, source, match, pair_of, xs, xt):
        """Plan the χ turns of matched side chains; return the source with every turn made."""
        # Side chains turn about their bonds toward the target χ angles, the short way round.
        from .torsions import _tree, dihedrals, rotate_bonds, torsion_atoms, wrap

        tree = _tree(source, "center")
        table = torsion_atoms(source.topology)
        # Columns 3:8 of the torsion table are χ1–χ5. Use those whose four atoms are all matched.
        quads = [
            table[i, column]
            for i, _ in match.residue_pairs
            for column in range(3, 8)
            if table[i, column][0] >= 0 and all(a in pair_of for a in table[i, column])
        ]
        rows = []
        if quads:
            quads = np.array(quads)
            # Change of each χ in degrees, wrapped to (−180°, 180°] so it turns the short way.
            change = wrap(dihedrals(xt, np.vectorize(pair_of.get)(quads)) - dihedrals(xs, quads))
            for quad, delta in zip(quads, np.radians(change)):
                # (fixed, moving) atoms of the χ bond in the kinematic tree; None for a ring bond.
                edge = tree.edge(quad[1], quad[2])
                if edge is not None and abs(delta) > 1e-6:
                    rows.append((tree.pre[edge[1]], edge[0], edge[1], delta))
        # Order the turns by the preorder of their moving atom, as SetTorsions does, so a
        # bond nearer the tree root turns first and carries the bonds beyond it.
        rows.sort(key=lambda row: row[0])
        self.chi_order = tree.order
        self.chi_bonds = [(u, v, delta) for _, u, v, delta in rows]
        self.chi_slices = [tree.subtree(v) for _, _, v, _ in rows]
        return rotate_bonds(xs, tree.order, self.chi_bonds, self.chi_slices, 1.0) if rows else xs

    def _bind_relaxation(self, source, match, xs, xt):
        """Bonds, rest lengths and pass count for the bond relaxation in _relax."""
        # Bond lengths relax toward lengths interpolated between the two structures.
        # Cα-only chains have no bonds, so consecutive trace atoms are joined virtually.
        bonds = np.asarray(source.topology.bonds, dtype=int).reshape(-1, 2)
        virtual = _virtual_trace_bonds(source)
        bonds = np.concatenate([bonds, virtual])
        # A trace residue has no internal geometry to hold its neighbours, so Cα-only
        # chains need many more passes to keep 3.8 Å spacing through large motions.
        self.relax_iterations = 200 if len(virtual) else 12
        partner = np.full(len(xs), -1, dtype=int)
        partner[match.source_atoms] = match.target_atoms
        self.relax_bonds = bonds
        # A bond reaches its target length if both its atoms are matched; otherwise it
        # keeps its source length.
        self.rest_source = np.linalg.norm(xs[bonds[:, 0]] - xs[bonds[:, 1]], axis=1)
        self.rest_target = self.rest_source.copy()
        both = (partner[bonds[:, 0]] >= 0) & (partner[bonds[:, 1]] >= 0)
        self.rest_target[both] = np.linalg.norm(
            xt[partner[bonds[both, 0]]] - xt[partner[bonds[both, 1]]], axis=1
        )
        # Bonds per atom (at least 1), to average the corrections each atom receives.
        self.relax_degree = np.maximum(np.bincount(bonds.ravel(), minlength=len(xs)), 1)

    def _bind_target_start(self, dest, match, rotations, translations, xt):
        """Starting coordinates of the target's atoms, and which have a source partner."""
        # Target-only atoms start carried back to the source frame by their group's motion.
        # R.T (x − t) undoes the group's motion, so the atom's screw path ends back on x.
        # Matched atoms are drawn at their source partner's position instead.
        self.dest_matched = np.zeros(len(dest.topology.atoms), bool)
        self.dest_matched[match.target_atoms] = True
        self.dest_partner = np.full(len(dest.topology.atoms), -1, dtype=int)
        self.dest_partner[match.target_atoms] = match.source_atoms
        gd = self.dest_group
        self.dest_start = np.einsum("nji,nj->ni", rotations[gd], xt - translations[gd])
        self.dest_end = xt

    def _bind_fades(self, source, dest, matched):
        """Per-atom opacity controls for both structures, and their overall opacities."""
        # Opacity: the source draws matched atoms until the end; the target then takes over.
        # Unmatched source atoms fade out over fade_out, and target-only atoms fade in over
        # fade_in, both while moving with their group.
        n_s, n_d = len(source.topology.atoms), len(dest.topology.atoms)
        s_matched = np.zeros(n_s, bool)
        s_matched[matched] = True
        self.source_controls = self._controls(n_s, s_matched, 1, 0, self.fade_out)
        self.dest_controls = self._controls(n_d, self.dest_matched, 0, 1, self.fade_in)
        self.dest_final_controls = dest._controls
        # A target that starts hidden (opacity 0) is drawn at full opacity; its controls fade it in.
        self.source_opacity, self.dest_opacity = source.opacity, dest.opacity or 1.0

    def _bind_secondary(self, source, dest, match, xs, xt):
        """Cartoon secondary-structure weights at the start and end of the morph, and for the target."""
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

    @staticmethod
    def _controls(count, matched, start, end, fade):
        """Read-only (count, 8) per-atom controls taking opacity from ``start`` to ``end``."""
        # Columns 0:2 are each atom's progress start and length; column 2 stays 0, so every
        # atom follows the clip's progress together. Columns 4:6 are the opacity before and
        # after the fade, and 6:8 the fade's start and length as fractions of the clip.
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

    def _relax(self, xyz, alpha, iterations=None):
        """Nudge bonds toward lengths interpolated between the structures (Jacobi projection).

        ``xyz`` (atoms, 3) in Å is changed in place and returned.
        """
        bonds, n = self.relax_bonds, len(xyz)
        rest = (1 - alpha) * self.rest_source + alpha * self.rest_target
        # Each pass measures every bond from the same coordinates (Jacobi, not Gauss–Seidel).
        # A bond off its rest length asks both its atoms to move half the error along it,
        # toward each other if it is too long. Each atom then takes the average of the
        # requests from its bonds, which keeps an atom with many bonds from overshooting.
        for _ in range(self.relax_iterations if iterations is None else iterations):
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
        """Source and target coordinates (atoms, 3) in Å at eased progress ``alpha`` (0 to 1)."""
        from .torsions import rotate_bonds

        a = float(alpha)
        # Each group turns by a × its screw angle about its own axis.
        rotations = _rotations(self.angles * a, self.axes)
        g = self.source_group
        # χ turns by the same fraction, made before the rigid motion.
        base = (
            rotate_bonds(self.xs, self.chi_order, self.chi_bonds, self.chi_slices, a)
            if self.chi_bonds
            else self.xs
        )
        # x(a) = R(a θ) (x_χ(a) + a r − p) + p + a d, with r the residual, p a point on the
        # screw axis and d the shift along it. At a = 1 this is R x_χ + t + R r, the target.
        moved = base - self.points[g] + a * self.residual
        xs = np.einsum("nij,nj->ni", rotations[g], moved) + self.points[g] + a * self.along[g]
        # The endpoints are the deposited structures; relax only in between.
        if 0 < a < 1 and len(self.relax_bonds):
            xs = self._relax(xs, a)
        # Target atoms ride their group's screw from dest_start; matched ones then sit
        # exactly on their source partner, so the two structures coincide.
        gd = self.dest_group
        xd = np.einsum("nij,nj->ni", rotations[gd], self.dest_start - self.points[gd]) + self.points[gd]
        xd += a * self.along[gd]
        partners = self.dest_partner >= 0
        xd[partners] = xs[self.dest_partner[partners]]
        return xs, xd

    def apply(self, alpha):
        source, dest = self.target, self.destination
        # alpha is the linear clip clock, which times the fades; the motion follows the eased value.
        eased = evaluate(self.easing, alpha)
        xs, xd = self._positions(eased)
        if alpha >= 1:
            # Land exactly on the target coordinates.
            partners = self.dest_partner >= 0
            xs[self.dest_partner[partners]] = self.dest_end[partners]
            xd = self.dest_end
        xs, xd = coordinates(xs), coordinates(xd)
        # Both ends of each pair are the current coordinates; the mix carries the clip clock
        # to the per-atom opacity controls.
        source._pair(xs, xs, alpha)
        dest._pair(xd, xd, alpha)
        # The target is drawn in the source's place, with the source's look for style="source".
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
        # At the end the target takes back its own controls and the source is hidden.
        source._controls = self.source_controls
        dest._controls = self.dest_final_controls if alpha >= 1 else self.dest_controls
        source.opacity = self.source_opacity if alpha < 1 else 0.0
        dest.opacity = self.dest_opacity
        # Cartoon secondary-structure weights blend with the eased progress.
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
    # Superpose the target on the fixed part, then fit the domain alone: what is left is
    # the domain's motion relative to the fixed part.
    rotation, translation = _kabsch(xt[[b for _, b in anchors]], xs[[a for a, _ in anchors]])
    xt = xt @ rotation.T + translation
    rotation, translation = _kabsch(xs[[a for a, _ in domain]], xt[[b for _, b in domain]])
    angles, axes, points, along = _screws(rotation[None], translation[None])
    axis, point = axes[0], points[0]
    # Report the axis point nearest the domain's center.
    center = xs[[a for a, _ in domain]].mean(0)
    point = point + np.dot(center - point, axis) * axis
    return DomainMotion(float(np.degrees(angles[0])), axis, point, float(np.dot(along[0], axis)))
