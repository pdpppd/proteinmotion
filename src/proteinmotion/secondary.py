"""DSSP secondary structure from backbone coordinates, and blendable cartoon states.

``assign`` follows the hydrogen-bond energy and pattern rules of Kabsch and Sander
(Biopolymers 22, 2577–2637, 1983) as implemented in DSSP 2: C=O···H–N electrostatic
energies, n-turns, minimal helices, and β-bridges grouped into ladders with bulges.
Results are reduced to the three states a cartoon draws: helix (H: α, 3₁₀ and π
helices), strand (E: ladders of two or more bridges) and coil (C: everything else,
including isolated bridges, turns and bends).

Chains with Cα atoms but no backbone N, C and O, such as coarse-grained models and
predicted Cα traces, are assigned from Cα geometry instead (``trace_states``).
"""

import numpy as np
from scipy.spatial import cKDTree

STATES = "CHE"
_ENERGY = 0.084 * 332.0  # q1·q2·f in kcal·Å/mol
_HBOND = -0.5  # kcal/mol; weaker pairs are not hydrogen bonds
_FLOOR = -9.9  # kcal/mol; DSSP's lower bound for overlapping atoms


def weights(codes):
    """Coil, helix and strand weights, one row per residue, for an H/E/C string."""
    if any(c not in STATES for c in codes):
        raise ValueError("Secondary structure codes must be H, E or C")
    result = np.zeros((len(codes), 3), np.float32)
    result[np.arange(len(codes)), [STATES.index(c) for c in codes]] = 1
    result.flags.writeable = False
    return result


def codes(weights):
    """The dominant H/E/C state of each residue."""
    return "".join(STATES[i] for i in np.argmax(np.asarray(weights), axis=1))


def _backbone(topology):
    """Topology residue indices with N, CA, C and O atom indices for complete amino acids."""
    rows = []
    for ri, (residue, names) in enumerate(zip(topology.residues, topology.residue_atoms)):
        if residue.is_nucleic or residue.ca < 0:
            continue
        ids = [names.get(name, -1) for name in ("N", "CA", "C", "O")]
        if min(ids) >= 0:
            rows.append((ri, *ids))
    return np.array(rows, dtype=int).reshape(-1, 5)


def _hydrogen_bonds(n, ca, c, o, h):
    """{(acceptor, donor)} for C=O(acceptor)···H–N(donor), keeping each donor's two best."""
    pairs = cKDTree(ca).query_pairs(9.0, output_type="ndarray")
    if not len(pairs):
        return set()
    lo, hi = pairs.min(1), pairs.max(1)
    # As in DSSP, residue i+1 never donates to the C=O of residue i.
    forward = hi - lo > 1
    donor = np.concatenate((lo, hi[forward]))
    acceptor = np.concatenate((hi, lo[forward]))
    keep = np.isfinite(h[donor, 0])
    donor, acceptor = donor[keep], acceptor[keep]
    if not len(donor):
        return set()
    r_on = np.linalg.norm(o[acceptor] - n[donor], axis=1)
    r_ch = np.linalg.norm(c[acceptor] - h[donor], axis=1)
    r_oh = np.linalg.norm(o[acceptor] - h[donor], axis=1)
    r_cn = np.linalg.norm(c[acceptor] - n[donor], axis=1)
    close = np.minimum.reduce((r_on, r_ch, r_oh, r_cn)) < 0.5
    with np.errstate(divide="ignore"):
        energy = _ENERGY * (1 / r_on + 1 / r_ch - 1 / r_oh - 1 / r_cn)
    energy = np.where(close, _FLOOR, np.maximum(energy, _FLOOR))
    bonded = energy < _HBOND
    donor, acceptor, energy = donor[bonded], acceptor[bonded], energy[bonded]
    order = np.lexsort((energy, donor))
    result, previous, rank = set(), -1, 0
    for k in order:
        rank = rank + 1 if donor[k] == previous else 0
        previous = donor[k]
        if rank < 2:
            result.add((int(acceptor[k]), int(donor[k])))
    return result


def _ladders(bridges, chains):
    """Group sorted (i, j, parallel) bridges into DSSP ladders, joined across bulges."""
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

    def gap(a, b):  # DSSP compares unsigned residue numbers
        return a - b if a >= b else 1 << 30

    k = 0
    while k < len(ladders):
        m = k + 1
        while m < len(ladders):
            a, b = ladders[k], ladders[m]
            ibi, iei, jbi, jei = a["i"][0], a["i"][-1], a["j"][0], a["j"][-1]
            ibj, iej, jbj, jej = b["i"][0], b["i"][-1], b["j"][0], b["j"][-1]
            joinable = (
                a["parallel"] == b["parallel"]
                and chains[min(ibi, ibj)] == chains[max(iei, iej)]
                and chains[min(jbi, jbj)] == chains[max(jei, jej)]
                and gap(ibj, iei) < 6
                and not (iei >= ibj and ibi <= iej)
            )
            if joinable:
                if a["parallel"]:
                    bulge = (gap(jbj, jei) < 6 and gap(ibj, iei) < 3) or gap(jbj, jei) < 3
                else:
                    bulge = (gap(jbi, jej) < 6 and gap(ibj, iei) < 3) or gap(jbi, jej) < 3
                if bulge:
                    a["i"] = sorted(set(a["i"]) | set(b["i"]))
                    a["j"] = sorted(set(a["j"]) | set(b["j"]))
                    a["size"] += b["size"]
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
    if n < 5:
        return "".join(state)
    d = np.linalg.norm(ca[:, None] - ca[None], axis=-1)
    helix = np.zeros(n, bool)
    b = ca[1:] - ca[:-1]
    normals = np.cross(b[:-1], b[1:])
    for i in range(n - 4):
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
    extended = np.zeros(n, bool)
    for i in range(1, n - 1):
        u, v = ca[i - 1] - ca[i], ca[i + 1] - ca[i]
        angle = np.degrees(np.arccos(np.clip(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
        extended[i] = angle > 95 and (i < 2 or i > n - 3 or d[i - 2, i + 2] > 9.5)
    strand = np.zeros(n, bool)
    for i in np.flatnonzero(extended):
        for j in np.flatnonzero(extended & (d[i] <= 5.6)):
            if abs(i - j) < 3 or j < 1 or j > n - 2:
                continue
            parallel = d[i - 1, j - 1] < 6.6 and d[i + 1, j + 1] < 6.6
            antiparallel = d[i - 1, j + 1] < 6.6 and d[i + 1, j - 1] < 6.6
            if parallel or antiparallel:
                strand[[i, j]] = True
    state[strand] = "E"
    state[helix] = "H"
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
    if len(table) < 3:
        return "".join(result)
    res = table[:, 0]
    n, ca, c, o = (xyz[table[:, k]] for k in range(1, 5))
    count = len(res)
    chains = np.array([topology.residues[i].chain for i in res])
    # A peptide link joins consecutive residues of one chain (C–N under 2.5 Å, as in DSSP).
    linked = (chains[1:] == chains[:-1]) & (np.linalg.norm(c[:-1] - n[1:], axis=1) < 2.5)
    link = np.r_[linked, False]  # link[k]: residue k is bonded to k + 1

    def unbroken(a, b):
        return 0 <= a and b < count and (a >= b or bool(np.all(link[a:b])))

    # DSSP places the amide H 1 Å from N, antiparallel to the previous C=O.
    h = np.full((count, 3), np.nan)
    carbonyl = c[:-1] - o[:-1]
    h[1:] = n[1:] + carbonyl / np.linalg.norm(carbonyl, axis=1)[:, None]
    proline = np.array([topology.residues[i].name == "PRO" for i in res])
    h[np.r_[True, ~linked] | proline] = np.nan
    hbonds = _hydrogen_bonds(n, ca, c, o, h)

    def bond(acceptor, donor):
        return (acceptor, donor) in hbonds

    starts = {
        size: np.array([unbroken(k, k + size) and bond(k, k + size) for k in range(count)])
        for size in (3, 4, 5)
    }
    candidates = set()
    for acceptor, donor in hbonds:
        for a in (acceptor - 1, acceptor, acceptor + 1):
            for d in (donor - 1, donor, donor + 1):
                candidates.add((min(a, d), max(a, d)))
    bridges = []
    for i, j in sorted(candidates):
        if j - i < 3 or i < 1 or j + 1 >= count or not unbroken(i - 1, i + 1) or not unbroken(j - 1, j + 1):
            continue
        if (bond(j, i + 1) and bond(i - 1, j)) or (bond(i, j + 1) and bond(j - 1, i)):
            bridges.append((i, j, True))
        elif (bond(j - 1, i + 1) and bond(i - 1, j + 1)) or (bond(i, j) and bond(j, i)):
            bridges.append((i, j, False))
    # Internal DSSP states: E ladder, B isolated bridge, H α, G 3₁₀, I π helix, C coil.
    state = ["C"] * count
    for ladder in _ladders(bridges, chains):
        code = "E" if ladder["size"] > 1 else "B"
        for strand in (ladder["i"], ladder["j"]):
            for k in range(strand[0], strand[-1] + 1):
                if state[k] != "E":
                    state[k] = code
    # α helices take precedence over strands. A 3₁₀ helix needs an otherwise empty
    # segment; π helices may also replace α helices, as in DSSP 2.2 and later.
    for size, code, allowed in ((4, "H", "CEBHGI"), (3, "G", "CG"), (5, "I", "CIH")):
        for k in range(1, count - size):
            if (
                starts[size][k]
                and starts[size][k - 1]
                and all(state[m] in allowed for m in range(k, k + size))
            ):
                state[k : k + size] = [code] * size
    state = ["H" if code in "HGI" else "E" if code == "E" else "C" for code in state]
    for ri, code in zip(res, state):
        result[ri] = code
    return "".join(result)
