"""Ideal peptides built from a sequence and backbone torsion angles.

Geometry uses Engh & Huber backbone values and standard side-chain internal
coordinates. Every atom is placed from three earlier atoms by a bond length, a
bond angle and a torsion, so φ, ψ, ω and χ angles are exact by construction.
"""

import numpy as np

from .structure import Atom, Topology, coordinates, infer_bonds, make_chains, residue_record

ONE_LETTER = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
    "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
    "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}  # fmt: skip

# Backbone bond lengths (Å) and angles (degrees).
N_CA, CA_C, C_N, C_O, CA_CB, N_H = 1.458, 1.525, 1.329, 1.231, 1.530, 1.01
N_CA_C, CA_C_N, C_N_CA, CA_C_O, N_CA_CB = 111.2, 116.2, 121.7, 120.5, 110.5
# The improper C–N–Cα–Cβ torsion of an L-amino acid (Cβ placed from C, N and Cα).
# Proline's ring narrows its N–Cα–Cβ angle.
CB_TORSION, PRO_N_CA_CB = -122.6, 103.5

# Side chains after Cβ: (atom, element, (a, b, c), length, angle, torsion). The torsion
# is a number, a χ name ("chi1"…) or a χ name plus a constant offset, ("chi2", 180).
# χ values below are common rotamers; bonds within rings close automatically.
SIDE_CHAINS = {
    "ARG": [
        ("CG", "C", ("N", "CA", "CB"), 1.52, 113.8, "chi1"),
        ("CD", "C", ("CA", "CB", "CG"), 1.52, 111.8, "chi2"),
        ("NE", "N", ("CB", "CG", "CD"), 1.46, 111.7, "chi3"),
        ("CZ", "C", ("CG", "CD", "NE"), 1.33, 124.8, "chi4"),
        ("NH1", "N", ("CD", "NE", "CZ"), 1.33, 120.6, 0.0),
        ("NH2", "N", ("CD", "NE", "CZ"), 1.33, 119.6, 180.0),
    ],
    "ASN": [
        ("CG", "C", ("N", "CA", "CB"), 1.52, 112.6, "chi1"),
        ("OD1", "O", ("CA", "CB", "CG"), 1.23, 120.9, "chi2"),
        ("ND2", "N", ("CA", "CB", "CG"), 1.33, 116.5, ("chi2", 180.0)),
    ],
    "ASP": [
        ("CG", "C", ("N", "CA", "CB"), 1.52, 113.1, "chi1"),
        ("OD1", "O", ("CA", "CB", "CG"), 1.25, 119.2, "chi2"),
        ("OD2", "O", ("CA", "CB", "CG"), 1.25, 118.2, ("chi2", 180.0)),
    ],
    "CYS": [("SG", "S", ("N", "CA", "CB"), 1.81, 113.8, "chi1")],
    "GLN": [
        ("CG", "C", ("N", "CA", "CB"), 1.52, 113.8, "chi1"),
        ("CD", "C", ("CA", "CB", "CG"), 1.52, 112.8, "chi2"),
        ("OE1", "O", ("CB", "CG", "CD"), 1.24, 120.9, "chi3"),
        ("NE2", "N", ("CB", "CG", "CD"), 1.33, 116.5, ("chi3", 180.0)),
    ],
    "GLU": [
        ("CG", "C", ("N", "CA", "CB"), 1.52, 113.8, "chi1"),
        ("CD", "C", ("CA", "CB", "CG"), 1.52, 113.3, "chi2"),
        ("OE1", "O", ("CB", "CG", "CD"), 1.25, 119.0, "chi3"),
        ("OE2", "O", ("CB", "CG", "CD"), 1.25, 118.1, ("chi3", 180.0)),
    ],
    "HIS": [
        ("CG", "C", ("N", "CA", "CB"), 1.50, 113.7, "chi1"),
        ("ND1", "N", ("CA", "CB", "CG"), 1.38, 122.7, "chi2"),
        ("CD2", "C", ("CA", "CB", "CG"), 1.36, 131.0, ("chi2", 180.0)),
        ("CE1", "C", ("CB", "CG", "ND1"), 1.32, 109.0, 180.0),
        ("NE2", "N", ("CB", "CG", "CD2"), 1.37, 107.0, 180.0),
    ],
    "ILE": [
        ("CG1", "C", ("N", "CA", "CB"), 1.53, 110.4, "chi1"),
        ("CG2", "C", ("N", "CA", "CB"), 1.52, 110.5, ("chi1", -122.6)),
        ("CD1", "C", ("CA", "CB", "CG1"), 1.52, 113.9, "chi2"),
    ],
    "LEU": [
        ("CG", "C", ("N", "CA", "CB"), 1.53, 116.1, "chi1"),
        ("CD1", "C", ("CA", "CB", "CG"), 1.52, 110.3, "chi2"),
        ("CD2", "C", ("CA", "CB", "CG"), 1.52, 110.6, ("chi2", 122.6)),
    ],
    "LYS": [
        ("CG", "C", ("N", "CA", "CB"), 1.52, 113.8, "chi1"),
        ("CD", "C", ("CA", "CB", "CG"), 1.52, 111.8, "chi2"),
        ("CE", "C", ("CB", "CG", "CD"), 1.52, 111.8, "chi3"),
        ("NZ", "N", ("CG", "CD", "CE"), 1.49, 111.7, "chi4"),
    ],
    "MET": [
        ("CG", "C", ("N", "CA", "CB"), 1.52, 113.7, "chi1"),
        ("SD", "S", ("CA", "CB", "CG"), 1.81, 112.7, "chi2"),
        ("CE", "C", ("CB", "CG", "SD"), 1.79, 100.6, "chi3"),
    ],
    "PHE": [
        ("CG", "C", ("N", "CA", "CB"), 1.50, 113.9, "chi1"),
        ("CD1", "C", ("CA", "CB", "CG"), 1.39, 120.7, "chi2"),
        ("CD2", "C", ("CA", "CB", "CG"), 1.39, 120.7, ("chi2", 180.0)),
        ("CE1", "C", ("CB", "CG", "CD1"), 1.39, 120.7, 180.0),
        ("CE2", "C", ("CB", "CG", "CD2"), 1.39, 120.7, 180.0),
        ("CZ", "C", ("CG", "CD1", "CE1"), 1.39, 120.0, 0.0),
    ],
    "PRO": [
        ("CG", "C", ("N", "CA", "CB"), 1.50, 104.0, "chi1"),
        ("CD", "C", ("CA", "CB", "CG"), 1.51, 105.0, "chi2"),
    ],
    "SER": [("OG", "O", ("N", "CA", "CB"), 1.42, 111.1, "chi1")],
    "THR": [
        ("OG1", "O", ("N", "CA", "CB"), 1.43, 109.2, "chi1"),
        ("CG2", "C", ("N", "CA", "CB"), 1.52, 111.1, ("chi1", -120.0)),
    ],
    "TRP": [
        ("CG", "C", ("N", "CA", "CB"), 1.50, 114.1, "chi1"),
        ("CD1", "C", ("CA", "CB", "CG"), 1.37, 127.0, "chi2"),
        ("CD2", "C", ("CA", "CB", "CG"), 1.43, 126.6, ("chi2", 180.0)),
        ("NE1", "N", ("CB", "CG", "CD1"), 1.38, 110.2, 180.0),
        ("CE2", "C", ("CB", "CG", "CD2"), 1.41, 107.2, 180.0),
        ("CE3", "C", ("CB", "CG", "CD2"), 1.40, 133.9, 0.0),
        ("CZ2", "C", ("CG", "CD2", "CE2"), 1.40, 122.4, 180.0),
        ("CZ3", "C", ("CG", "CD2", "CE3"), 1.39, 118.7, 180.0),
        ("CH2", "C", ("CD2", "CE2", "CZ2"), 1.37, 117.5, 0.0),
    ],
    "TYR": [
        ("CG", "C", ("N", "CA", "CB"), 1.51, 113.8, "chi1"),
        ("CD1", "C", ("CA", "CB", "CG"), 1.39, 121.0, "chi2"),
        ("CD2", "C", ("CA", "CB", "CG"), 1.39, 121.0, ("chi2", 180.0)),
        ("CE1", "C", ("CB", "CG", "CD1"), 1.39, 121.3, 180.0),
        ("CE2", "C", ("CB", "CG", "CD2"), 1.39, 121.3, 180.0),
        ("CZ", "C", ("CG", "CD1", "CE1"), 1.38, 119.8, 0.0),
        ("OH", "O", ("CD1", "CE1", "CZ"), 1.38, 119.8, 180.0),
    ],
    "VAL": [
        ("CG1", "C", ("N", "CA", "CB"), 1.52, 110.5, "chi1"),
        ("CG2", "C", ("N", "CA", "CB"), 1.52, 110.5, ("chi1", 122.6)),
    ],
}
# Common rotamers (degrees): mostly χ1 near −65° (m); Ser/Thr/Val use their usual ones.
# A side chain that would clash with the backbone turns χ1 to the clearest of these.
ROTAMERS = {
    "ARG": (-65, 180, 180, 180),
    "ASN": (-65, -40),
    "ASP": (-70, -15),
    "CYS": (-65,),
    "GLN": (-65, 180, -25),
    "GLU": (-65, 180, -40),
    "HIS": (-65, -75),
    "ILE": (-65, 170),
    "LEU": (-65, 175),
    "LYS": (-65, 180, 180, 180),
    "MET": (-65, 180, 70),
    "PHE": (-65, 95),
    "PRO": (29.5, -34.75),  # Cγ-endo pucker; closes the ring with C–N = 1.47 Å
    "SER": (65,),
    "THR": (60,),
    "TRP": (-65, 95),
    "TYR": (-65, 95),
    "VAL": (175,),
}


def place(a, b, c, length, angle, torsion):
    """The atom bonded to c with bond length, angle b–c–d and torsion a–b–c–d in degrees."""
    bc = c - b
    bc /= np.linalg.norm(bc)
    normal = np.cross(b - a, bc)
    normal /= np.linalg.norm(normal)
    theta, tau = np.radians(angle), np.radians(torsion)
    local = np.array([-np.cos(theta), np.sin(theta) * np.cos(tau), np.sin(theta) * np.sin(tau)])
    return c + length * (local[0] * bc + local[1] * np.cross(normal, bc) + local[2] * normal)


def _side_chain(name, own, chi):
    """Side-chain atoms beyond Cβ for the given χ angles, placed from the backbone."""
    placed = dict(own)
    result = {}
    for atom, _, (a, b, c), length, bond_angle, torsion in SIDE_CHAINS.get(name, ()):
        if isinstance(torsion, str):
            torsion = chi[int(torsion[3:]) - 1]
        elif isinstance(torsion, tuple):
            torsion = chi[int(torsion[0][3:]) - 1] + torsion[1]
        placed[atom] = result[atom] = place(placed[a], placed[b], placed[c], length, bond_angle, torsion)
    return result


def _sequence(sequence):
    if isinstance(sequence, str):
        letters = [c for c in sequence.upper() if not c.isspace() and c != "-"]
        unknown = sorted({c for c in letters if c not in ONE_LETTER})
        if unknown:
            raise ValueError(f"Unknown one-letter codes {unknown}; use the 20 standard amino acids")
        return [ONE_LETTER[c] for c in letters]
    names = [str(name).upper() for name in sequence]
    unknown = sorted({n for n in names if n not in SIDE_CHAINS and n not in ("ALA", "GLY")})
    if unknown:
        raise ValueError(f"Unknown residue names {unknown}")
    return names


def _per_residue(value, count, name):
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(count, float(array))
    if array.shape != (count,) or not np.isfinite(array).all():
        raise ValueError(f"{name} needs one finite value or one per residue ({count})")
    return array


def build_peptide(
    sequence, conformation="extended", *, phi=None, psi=None, omega=180.0, chain="A", first=1, hydrogens=False
):
    """(Topology, coordinates) of an ideal peptide; see Protein.build."""
    from .torsions import CONFORMATIONS

    names = _sequence(sequence)
    count = len(names)
    if count < 1:
        raise ValueError("The sequence needs at least one residue")
    key = str(conformation).lower().replace("-", "").replace("_", "").replace(" ", "")
    if key not in CONFORMATIONS:
        raise ValueError(f"Unknown conformation {conformation!r}; choose from {', '.join(CONFORMATIONS)}")
    phi = _per_residue(CONFORMATIONS[key][0] if phi is None else phi, count, "phi")
    psi = _per_residue(CONFORMATIONS[key][1] if psi is None else psi, count, "psi")
    omega = _per_residue(omega, count, "omega")
    if not isinstance(chain, str) or not chain:
        raise ValueError("chain must be a nonempty string")
    # Proline's ring fixes its φ near −65°.
    phi = np.where(np.array(names) == "PRO", -65.0, phi)

    # Backbone first, so side chains can avoid it.
    frames = []
    n = np.array([0.0, 0.0, 0.0])
    ca = np.array([N_CA, 0.0, 0.0])
    angle = np.radians(180 - N_CA_C)
    c = ca + CA_C * np.array([np.cos(angle), np.sin(angle), 0.0])
    for i, name in enumerate(names):
        if frames:
            p_n, p_ca, p_c = frames[-1]["N"], frames[-1]["CA"], frames[-1]["C"]
            n = place(p_n, p_ca, p_c, C_N, CA_C_N, psi[i - 1])
            ca = place(p_ca, p_c, n, N_CA, C_N_CA, omega[i - 1])
            c = place(p_c, n, ca, CA_C, N_CA_C, phi[i])
        own = {"N": n, "CA": ca, "C": c, "O": place(n, ca, c, C_O, CA_C_O, psi[i] + 180.0)}
        if name != "GLY":
            own["CB"] = place(c, n, ca, CA_CB, PRO_N_CA_CB if name == "PRO" else N_CA_CB, CB_TORSION)
        if i == count - 1:
            own["OXT"] = place(n, ca, c, 1.25, 117.0, psi[i])
        if hydrogens and frames and name != "PRO":
            bisector = (n - frames[-1]["C"]) / C_N + (n - ca) / N_CA
            own["H"] = n + N_H * bisector / np.linalg.norm(bisector)
        frames.append(own)
    heavy = [np.array([v for k, v in f.items() if k != "H"]) for f in frames]
    for i, name in enumerate(names):
        own = frames[i]
        others = np.concatenate([h for k, h in enumerate(heavy) if k != i])
        default = ROTAMERS.get(name, ())
        best = None
        for chi1 in dict.fromkeys((default[:1] or (0,)) + (-65.0, 180.0, 60.0)):
            chi = (chi1, *default[1:]) if name != "PRO" else default
            atoms_i = _side_chain(name, own, chi)
            if not atoms_i:
                break
            clearance = np.min(
                np.linalg.norm(np.array(list(atoms_i.values()))[:, None] - others[None], axis=2)
            )
            if best is None or clearance > best[0] + 0.05:
                best = (clearance, atoms_i)
            if clearance >= 2.9 or name == "PRO":
                best = (clearance, atoms_i)
                break
        if best is not None:
            own.update(best[1])
            heavy[i] = np.array([v for k, v in own.items() if k != "H"])

    atoms, xyz, residues = [], [], []
    for i, (name, own) in enumerate(zip(names, frames)):
        elements = {"N": "N", "CA": "C", "C": "C", "O": "O", "CB": "C", "OXT": "O", "H": "H"}
        elements.update({atom: element for atom, element, *_ in SIDE_CHAINS.get(name, ())})
        order = [k for k in ("N", "CA", "C", "O", "CB") if k in own]
        order += [k for k in own if k not in order and k not in ("OXT", "H")]
        order += [k for k in ("OXT", "H") if k in own]
        index = {}
        for atom_name in order:
            index[atom_name] = len(atoms)
            atoms.append(Atom(chain, first + i, "", name, atom_name, elements[atom_name], i))
            xyz.append(own[atom_name])
        residues.append(residue_record(chain, first + i, "", name, index, atoms))
    xyz = np.asarray(xyz, dtype=np.float64)
    # Center the chain and lay it along x, N terminus on the left.
    trace = xyz[[r.ca for r in residues]]
    center = trace.mean(0)
    if count > 2:
        _, _, axes = np.linalg.svd(trace - center)
        x = axes[0] if np.dot(axes[0], trace[-1] - trace[0]) >= 0 else -axes[0]
        y = axes[1]
        z = np.cross(x, y)
        xyz = (xyz - center) @ np.array([x, y, z]).T
    else:
        xyz = xyz - center
    xyz = coordinates(xyz)
    atoms = tuple(atoms)
    topology = Topology(atoms, tuple(residues), infer_bonds(atoms, xyz), make_chains(residues, xyz, atoms))
    from dataclasses import replace

    from .secondary import assign

    states = assign(topology, xyz)
    residues = tuple(replace(r, secondary=s) for r, s in zip(topology.residues, states))
    return Topology(atoms, residues, topology.bonds, topology.chains), xyz
