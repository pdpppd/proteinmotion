"""DSSP secondary structure from backbone coordinates, and blendable cartoon states.

``assign`` follows the hydrogen-bond energy and pattern rules of Kabsch and Sander
(Biopolymers 22, 2577–2637, 1983) as implemented in DSSP 2: C=O···H–N electrostatic
energies, n-turns, minimal helices, and β-bridges grouped into ladders with bulges.
Results are reduced to the three states a cartoon draws: helix (H: α, 3₁₀ and π
helices), strand (E: ladders of two or more bridges) and coil (C: everything else,
including isolated bridges, turns and bends).

Chains with Cα atoms but no backbone N, C and O, such as coarse-grained models and
predicted Cα traces, are assigned from Cα geometry instead (``trace_states``).

The cartoon stores secondary structure as one (coil, helix, strand) weight row per
residue (``weights``), so animations can blend between assignments; ``codes`` turns
weights back into an H/E/C string. ``assign`` runs when a structure is loaded with
``secondary="dssp"`` or without helix and sheet records (``with_dssp`` in structure.py),
for ``Protein.build`` and ``Protein.assign_secondary_structure``, in the
``SecondaryStructure`` animation, and after torsion changes (torsions.py) and
structure morphs (conformations.py).
"""

import numpy as np
from scipy.spatial import cKDTree

# Column order of the cartoon weights: coil, helix, strand.
STATES = "CHE"
# Kabsch–Sander partial charges, ±0.42e on C and O and ±0.20e on N and H, times the
# Coulomb factor 332 kcal·Å/mol: E = q1·q2·f·(1/r_ON + 1/r_CH − 1/r_OH − 1/r_CN).
_ENERGY = 0.084 * 332.0  # q1·q2·f in kcal·Å/mol
_HBOND = -0.5  # kcal/mol; weaker pairs are not hydrogen bonds
_FLOOR = -9.9  # kcal/mol; DSSP's lower bound for overlapping atoms


def weights(codes):
    """Coil, helix and strand weights, one row per residue, for an H/E/C string."""
    if any(c not in STATES for c in codes):
        raise ValueError("Secondary structure codes must be H, E or C")
    result = np.zeros((len(codes), 3), np.float32)
    result[np.arange(len(codes)), [STATES.index(c) for c in codes]] = 1
    # Scene snapshots keep this array without copying it, so it must never change in place.
    result.flags.writeable = False
    return result


def codes(weights):
    """The dominant H/E/C state of each residue."""
    return "".join(STATES[i] for i in np.argmax(np.asarray(weights), axis=1))


def _backbone(topology):
    """Topology residue indices with N, CA, C and O atom indices for complete amino acids.

    Returns an int array of shape (complete residues, 5): the topology residue index, then
    the N, Cα, C and O atom indices. Its rows are the residues DSSP works on.
    """
    rows = []
    for ri, (residue, names) in enumerate(zip(topology.residues, topology.residue_atoms)):
        if residue.is_nucleic or residue.ca < 0:
            continue
        ids = [names.get(name, -1) for name in ("N", "CA", "C", "O")]
        if min(ids) >= 0:
            rows.append((ri, *ids))
    return np.array(rows, dtype=int).reshape(-1, 5)


def _hydrogen_bonds(n, ca, c, o, h):
    """{(acceptor, donor)} for C=O(acceptor)···H–N(donor), keeping each donor's two best.

    ``n``, ``ca``, ``c``, ``o`` and ``h`` are (residues, 3) coordinates in Å, one row per
    backbone-table residue; ``h`` is NaN where a residue has no amide hydrogen. The
    returned indices are rows of that table.
    """
    # DSSP only scores pairs whose Cα atoms lie within 9 Å of each other.
    pairs = cKDTree(ca).query_pairs(9.0, output_type="ndarray")
    if not len(pairs):
        return set()
    lo, hi = pairs.min(1), pairs.max(1)
    # Score each pair both ways: lo's N–H to hi's C=O, and hi's N–H to lo's C=O.
    # As in DSSP, residue i+1 never donates to the C=O of residue i.
    forward = hi - lo > 1
    donor = np.concatenate((lo, hi[forward]))
    acceptor = np.concatenate((hi, lo[forward]))
    # Residues without an amide H (chain starts, residues after a break, proline) cannot donate.
    keep = np.isfinite(h[donor, 0])
    donor, acceptor = donor[keep], acceptor[keep]
    if not len(donor):
        return set()
    r_on = np.linalg.norm(o[acceptor] - n[donor], axis=1)
    r_ch = np.linalg.norm(c[acceptor] - h[donor], axis=1)
    r_oh = np.linalg.norm(o[acceptor] - h[donor], axis=1)
    r_cn = np.linalg.norm(c[acceptor] - n[donor], axis=1)
    # Atoms closer than 0.5 Å get the floor energy, as in DSSP, rather than a huge
    # Coulomb term. Coincident atoms divide by zero here; the floor replaces them below.
    close = np.minimum.reduce((r_on, r_ch, r_oh, r_cn)) < 0.5
    with np.errstate(divide="ignore"):
        energy = _ENERGY * (1 / r_on + 1 / r_ch - 1 / r_oh - 1 / r_cn)
    energy = np.where(close, _FLOOR, np.maximum(energy, _FLOOR))
    bonded = energy < _HBOND
    donor, acceptor, energy = donor[bonded], acceptor[bonded], energy[bonded]
    # DSSP remembers only the two lowest-energy acceptors of each N–H. Sort by donor, then
    # by energy, and keep the first two rows of every donor.
    order = np.lexsort((energy, donor))
    result, previous, rank = set(), -1, 0
    for k in order:
        rank = rank + 1 if donor[k] == previous else 0
        previous = donor[k]
        if rank < 2:
            result.add((int(acceptor[k]), int(donor[k])))
    return result


def _ladders(bridges, chains):
    """Group sorted (i, j, parallel) bridges into DSSP ladders, joined across bulges.

    ``bridges`` hold backbone-table rows with i < j, and ``chains`` the chain ID of each
    row. Each ladder is a dict: ``"i"`` and ``"j"``, the ascending rows of its two strands;
    ``"parallel"``; and ``"size"``, the number of bridges in it.
    """
    # Step 1, as in DSSP: a bridge extends a ladder when it follows the ladder's last
    # bridge on strand i and its partner continues strand j, forward for a parallel
    # ladder and backward for an antiparallel one. Otherwise it starts a new ladder.
    ladders = []
    for i, j, parallel in bridges:
        for ladder in ladders:
            if ladder["parallel"] != parallel or i != ladder["i"][-1] + 1:
                continue
            if parallel and ladder["j"][-1] + 1 == j:
                ladder["i"].append(i)
                ladder["j"].append(j)
                break
            if not parallel and ladder["j"][0] - 1 == j:
                ladder["i"].append(i)
                ladder["j"].insert(0, j)
                break
        else:
            ladders.append({"i": [i], "j": [j], "parallel": parallel, "size": 1})
            continue
        ladder["size"] += 1

    # DSSP compares unsigned residue numbers, so a negative gap wraps to a huge value
    # and never passes a cutoff.
    def gap(a, b):
        return a - b if a >= b else 1 << 30

    # Step 2: join two ladders of the same type across a β-bulge, a break of at most one
    # extra residue on one strand and at most four on the other (row gaps under 3 and 6).
    # The names follow the DSSP source: ibi and iei are the first and last i rows of
    # ladder a, jbi and jei its first and last j rows; ibj, iej, jbj and jej are the same
    # for ladder b.
    k = 0
    while k < len(ladders):
        m = k + 1
        while m < len(ladders):
            a, b = ladders[k], ladders[m]
            ibi, iei, jbi, jei = a["i"][0], a["i"][-1], a["j"][0], a["j"][-1]
            ibj, iej, jbj, jej = b["i"][0], b["i"][-1], b["j"][0], b["j"][-1]
            # Same type, each strand on one chain, b starting soon after a on strand i,
            # and no overlap between their i ranges.
            joinable = (
                a["parallel"] == b["parallel"]
                and chains[min(ibi, ibj)] == chains[max(iei, iej)]
                and chains[min(jbi, jbj)] == chains[max(jei, jej)]
                and gap(ibj, iei) < 6
                and not (iei >= ibj and ibi <= iej)
            )
            if joinable:
                # On strand j, b follows a in a parallel ladder and precedes it in an
                # antiparallel one, so the j gap is measured from opposite ends.
                if a["parallel"]:
                    bulge = (gap(jbj, jei) < 6 and gap(ibj, iei) < 3) or gap(jbj, jei) < 3
                else:
                    bulge = (gap(jbi, jej) < 6 and gap(ibj, iei) < 3) or gap(jbi, jej) < 3
                if bulge:
                    a["i"] = sorted(set(a["i"]) | set(b["i"]))
                    a["j"] = sorted(set(a["j"]) | set(b["j"]))
                    a["size"] += b["size"]
                    # The next ladder slides into slot m; test it against the grown ladder.
                    ladders.pop(m)
                    continue
            m += 1
        k += 1
    return ladders


def trace_states(ca):
    """H/E/C for one Cα-only chain, from Cα–Cα distances, angles and virtual dihedrals.

    A helix needs the (i, i+2), (i, i+3) and (i, i+4) distances and the virtual dihedral
    of an α helix over five residues. A strand needs an extended residue paired with
    another extended residue whose neighbours are also paired (a parallel or antiparallel
    ladder), so a lone extended chain stays coil, as in DSSP. Helices shorter than four
    residues and strands shorter than three become coil. On steered-MD frames of a
    302-residue protein, from unfolded to native, this agreed with DSSP for 86% of residues.
    """
    ca = np.asarray(ca, dtype=np.float64)
    n = len(ca)
    state = np.array(["C"] * n)
    # The helix test reads a window of five residues.
    if n < 5:
        return "".join(state)
    # d[i, j] is the Cα(i)–Cα(j) distance in Å. b[i] is the virtual bond Cα(i) → Cα(i+1),
    # and normals[i] is normal to the plane through Cα(i), Cα(i+1) and Cα(i+2).
    d = np.linalg.norm(ca[:, None] - ca[None], axis=-1)
    helix = np.zeros(n, bool)
    b = ca[1:] - ca[:-1]
    normals = np.cross(b[:-1], b[1:])
    # Helix: the reference values are those of a Pauling α helix (φ −57°, ψ −47°). Cα(i)
    # lies 5.5 Å from Cα(i+2), 5.3 Å from Cα(i+3) and 6.4 Å from Cα(i+4), and the test
    # allows 0.6, 0.7 and 0.8 Å. The virtual dihedral Cα(i)–Cα(i+1)–Cα(i+2)–Cα(i+3) is
    # about +50° (25° allowed); a left-handed helix has the opposite sign. A matching
    # window marks all five residues.
    for i in range(n - 4):
        # Signed angle between the planes (i, i+1, i+2) and (i+1, i+2, i+3), about b[i+1].
        b1 = b[i + 1] / np.linalg.norm(b[i + 1])
        dihedral = np.degrees(
            np.arctan2(np.dot(np.cross(normals[i], normals[i + 1]), b1), np.dot(normals[i], normals[i + 1]))
        )
        if (
            abs(d[i, i + 2] - 5.5) < 0.6
            and abs(d[i, i + 3] - 5.3) < 0.7
            and abs(d[i, i + 4] - 6.4) < 0.8
            and abs(dihedral - 50) < 25
        ):
            helix[i : i + 5] = True
    # Extended residues: the virtual bond angle at Cα(i) opens past 95° (about 92° in an
    # α helix, 120–130° in a β strand), and Cα(i−2)–Cα(i+2) spans more than 9.5 Å, which
    # rules out tight turns. The span test is skipped within two residues of either end,
    # where i−2 or i+2 does not exist (a negative index would wrap around).
    extended = np.zeros(n, bool)
    for i in range(1, n - 1):
        u, v = ca[i - 1] - ca[i], ca[i + 1] - ca[i]
        angle = np.degrees(np.arccos(np.clip(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
        extended[i] = angle > 95 and (i < 2 or i > n - 3 or d[i - 2, i + 2] > 9.5)
    # Strands, the Cα version of a DSSP bridge: pair extended residues i and j at least
    # three apart with Cα atoms within 5.6 Å, whose neighbours are also close (under
    # 6.6 Å): i±1 with j±1 for a parallel ladder, i±1 with j∓1 for an antiparallel one.
    # j must have a neighbour on each side.
    strand = np.zeros(n, bool)
    for i in np.flatnonzero(extended):
        for j in np.flatnonzero(extended & (d[i] <= 5.6)):
            if abs(i - j) < 3 or j < 1 or j > n - 2:
                continue
            parallel = d[i - 1, j - 1] < 6.6 and d[i + 1, j + 1] < 6.6
            antiparallel = d[i - 1, j + 1] < 6.6 and d[i + 1, j - 1] < 6.6
            if parallel or antiparallel:
                strand[[i, j]] = True
    # Helix is written last, so it wins where both tests pass, as α helices do in DSSP.
    state[strand] = "E"
    state[helix] = "H"
    # Turn runs shorter than four helix or three strand residues back into coil.
    for code, shortest in (("H", 4), ("E", 3)):
        k = 0
        while k < n:
            if state[k] != code:
                k += 1
                continue
            end = k
            while end < n and state[end] == code:
                end += 1
            if end - k < shortest:
                state[k:end] = "C"
            k = end
    return "".join(state)


def assign(topology, xyz):
    """DSSP assignment reduced to H (helix), E (strand) or C (coil) per topology residue.

    Amino-acid chains with Cα atoms but no complete N, Cα, C, O backbone are assigned
    from their Cα geometry with ``trace_states``.
    """
    result = ["C"] * len(topology.residues)
    table = _backbone(topology)
    xyz = np.asarray(xyz, dtype=np.float64)
    # Cα-only chains: every residue of the traced chain is an amino acid with a Cα and
    # none has a complete backbone. topology.chains splits chains at gaps, so each one is
    # a continuous Cα trace. A chain that mixes complete and Cα-only residues goes through
    # DSSP below, and its Cα-only residues stay coil.
    complete = set(table[:, 0].tolist())
    for chain in topology.chains:
        residues = [int(i) for i in chain]
        if all(
            not topology.residues[i].is_nucleic and topology.residues[i].ca >= 0 and i not in complete
            for i in residues
        ):
            ca = xyz[[topology.residues[i].ca for i in residues]]
            for i, code in zip(residues, trace_states(ca)):
                result[i] = code
    # Too few complete residues for any DSSP pattern.
    if len(table) < 3:
        return "".join(result)
    # From here on, k, i and j index rows of the backbone table (complete residues in
    # topology order), not topology residues. n, ca, c and o are (rows, 3) coordinates in Å.
    res = table[:, 0]
    n, ca, c, o = (xyz[table[:, k]] for k in range(1, 5))
    count = len(res)
    chains = np.array([topology.residues[i].chain for i in res])
    # A peptide link joins consecutive residues of one chain (C–N under 2.5 Å, as in DSSP).
    linked = (chains[1:] == chains[:-1]) & (np.linalg.norm(c[:-1] - n[1:], axis=1) < 2.5)
    link = np.r_[linked, False]  # link[k]: residue k is bonded to k + 1

    # DSSP's NoChainBreak: rows a..b exist and are linked one to the next. An empty or
    # reversed range (a >= b) counts as unbroken.
    def unbroken(a, b):
        return 0 <= a and b < count and (a >= b or bool(np.all(link[a:b])))

    # DSSP places the amide H 1 Å from N, antiparallel to the previous C=O.
    h = np.full((count, 3), np.nan)
    carbonyl = c[:-1] - o[:-1]
    h[1:] = n[1:] + carbonyl / np.linalg.norm(carbonyl, axis=1)[:, None]
    # No amide H on the first residue of each linked stretch (no previous C=O to place it
    # from) or on proline, whose N bonds to the ring's Cδ instead.
    proline = np.array([topology.residues[i].name == "PRO" for i in res])
    h[np.r_[True, ~linked] | proline] = np.nan
    hbonds = _hydrogen_bonds(n, ca, c, o, h)

    # bond(i, j): the C=O of row i accepts a hydrogen bond from the N–H of row j.
    def bond(acceptor, donor):
        return (acceptor, donor) in hbonds

    # n-turns: starts[n][k] when the C=O of row k bonds to the N–H of row k + n
    # (n = 3, 4, 5), with no break between them.
    starts = {
        size: np.array([unbroken(k, k + size) and bond(k, k + size) for k in range(count)])
        for size in (3, 4, 5)
    }
    # β-bridge candidates. Every bridge pattern below uses a hydrogen bond whose ends lie
    # within one residue of i and of j, so the pairs around each bond cover every bridge
    # without testing all pairs of residues.
    candidates = set()
    for acceptor, donor in hbonds:
        for a in (acceptor - 1, acceptor, acceptor + 1):
            for d in (donor - 1, donor, donor + 1):
                candidates.add((min(a, d), max(a, d)))
    # Kabsch–Sander bridge patterns between rows i and j at least three apart, each with
    # an unbroken neighbour on both sides:
    #   parallel:     bond(i−1, j) and bond(j, i+1), or bond(j−1, i) and bond(i, j+1)
    #   antiparallel: bond(i, j) and bond(j, i), or bond(i−1, j+1) and bond(j−1, i+1)
    bridges = []
    for i, j in sorted(candidates):
        if j - i < 3 or i < 1 or j + 1 >= count or not unbroken(i - 1, i + 1) or not unbroken(j - 1, j + 1):
            continue
        if (bond(j, i + 1) and bond(i - 1, j)) or (bond(i, j + 1) and bond(j - 1, i)):
            bridges.append((i, j, True))
        elif (bond(j - 1, i + 1) and bond(i - 1, j + 1)) or (bond(i, j) and bond(j, i)):
            bridges.append((i, j, False))
    # Internal DSSP states: E ladder, B isolated bridge, H α, G 3₁₀, I π helix, C coil.
    # Each strand of a ladder is marked from its first to its last paired residue, which
    # fills in bulges; an isolated bridge never overwrites a ladder residue.
    state = ["C"] * count
    for ladder in _ladders(bridges, chains):
        code = "E" if ladder["size"] > 1 else "B"
        for strand in (ladder["i"], ladder["j"]):
            for k in range(strand[0], strand[-1] + 1):
                if state[k] != "E":
                    state[k] = code
    # Minimal helices: two consecutive n-turns, starting at k − 1 and k, make residues
    # k..k+n−1 helical. α helices take precedence over strands. A 3₁₀ helix needs an
    # otherwise empty segment; π helices may also replace α helices, as in DSSP 2.2 and later.
    for size, code, allowed in ((4, "H", "CEBHGI"), (3, "G", "CG"), (5, "I", "CIH")):
        for k in range(1, count - size):
            if (
                starts[size][k]
                and starts[size][k - 1]
                and all(state[m] in allowed for m in range(k, k + size))
            ):
                state[k : k + size] = [code] * size
    # Reduce to the cartoon's three states: α, 3₁₀ and π draw as helix, ladders as strand,
    # and isolated bridges with everything else as coil.
    state = ["H" if code in "HGI" else "E" if code == "E" else "C" for code in state]
    for ri, code in zip(res, state):
        result[ri] = code
    return "".join(result)
