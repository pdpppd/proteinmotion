"""Validated molecular topology and PDB/mmCIF loading via Gemmi."""

import warnings
from dataclasses import dataclass
from pathlib import Path

import gemmi
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Atom:
    chain: str
    resid: int
    icode: str
    resname: str
    name: str
    element: str
    residue_index: int
    bfactor: float = float("nan")

    @property
    def key(self):
        return (self.chain, self.resid, self.icode, self.resname, self.name)


@dataclass(frozen=True)
class Residue:
    chain: str
    resid: int
    icode: str
    name: str
    ca: int
    oxygen: int
    secondary: str = "C"


@dataclass
class Topology:
    atoms: tuple[Atom, ...]
    residues: tuple[Residue, ...]
    bonds: np.ndarray
    chains: tuple[np.ndarray, ...]

    @property
    def keys(self):
        return tuple(a.key for a in self.atoms)


def coordinates(value, n_atoms=None):
    a = np.array(value, dtype=np.float32, copy=True, order="C")
    if a.ndim != 2 or a.shape[1] != 3 or not len(a) or not np.isfinite(a).all():
        raise ValueError("Coordinates must be a nonempty, finite (atoms, 3) array")
    if n_atoms is not None and len(a) != n_atoms:
        raise ValueError(f"Expected {n_atoms} atoms, got {len(a)}")
    a.flags.writeable = False
    return a


def infer_bonds(atoms, xyz):
    """Spatially indexed covalent-radius heuristic; not a chemistry bond-order model."""
    pairs = cKDTree(xyz).query_pairs(2.65, output_type="ndarray")
    if not len(pairs):
        return np.empty((0, 2), np.uint32)
    radii = np.array([gemmi.Element(a.element).covalent_r for a in atoms])
    d = np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1)
    limits = radii[pairs[:, 0]] + radii[pairs[:, 1]] + 0.40
    mask = (d > 0.35) & (d < limits)
    # Avoid close contacts between independent chains, except disulfide bridges.
    mask &= np.array(
        [atoms[i].chain == atoms[j].chain or (atoms[i].element == atoms[j].element == "S") for i, j in pairs]
    )
    return pairs[mask].astype(np.uint32)


def make_chains(residues, xyz):
    runs, run = [], []
    for i, r in enumerate(residues):
        if r.ca < 0:
            if len(run) > 1:
                runs.append(np.array(run, np.uint32))
            run = []
            continue
        if run:
            prev = residues[run[-1]]
            # Do not draw ribbons across missing residues or chain discontinuities.
            if r.chain != prev.chain or np.linalg.norm(xyz[r.ca] - xyz[prev.ca]) > 4.8:
                if len(run) > 1:
                    runs.append(np.array(run, np.uint32))
                run = []
        run.append(i)
    if len(run) > 1:
        runs.append(np.array(run, np.uint32))
    return tuple(runs)


def load_structure(path, *, chains=None, include_water=False, include_hydrogens=False):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    st = gemmi.read_structure(str(path))
    if not len(st):
        raise ValueError("Structure contains no models")
    allowed = None if chains is None else ({chains} if isinstance(chains, str) else set(chains))
    ranges = []
    for h in st.helices:
        ranges.append((h.start, h.end, "H"))
    for sheet in st.sheets:
        for strand in sheet.strands:
            ranges.append((strand.start, strand.end, "E"))

    def secondary(chain, res):
        key = (res.seqid.num, res.seqid.icode.strip())
        for start, end, kind in ranges:
            if start.chain_name == chain == end.chain_name:
                lo = (start.res_id.seqid.num, start.res_id.seqid.icode.strip())
                hi = (end.res_id.seqid.num, end.res_id.seqid.icode.strip())
                if lo <= key <= hi:
                    return kind
        return "C"

    def parse(model):
        atoms, residues, xyz = [], [], []
        for chain in model:
            if allowed is not None and chain.name not in allowed:
                continue
            for res in chain:
                if res.is_water() and not include_water:
                    continue
                best = {}
                for a in res:
                    if a.element.is_hydrogen and not include_hydrogens:
                        continue
                    # Deterministic highest-occupancy altloc, then blank/A preference.
                    rank = (a.occ, a.altloc in ("\x00", " "), a.altloc == "A")
                    if a.name not in best or rank > best[a.name][0]:
                        best[a.name] = (rank, a)
                if not best:
                    continue
                names = {}
                for _, a in best.values():
                    names[a.name] = len(atoms)
                    atoms.append(
                        Atom(
                            chain.name,
                            res.seqid.num,
                            res.seqid.icode.strip(),
                            res.name,
                            a.name,
                            a.element.name,
                            len(residues),
                            float(a.b_iso),
                        )
                    )
                    xyz.append([a.pos.x, a.pos.y, a.pos.z])
                # Calcium ions named CA are not alpha carbons.
                ca = names.get("CA", -1)
                if ca >= 0 and atoms[ca].element != "C":
                    ca = -1
                residues.append(
                    Residue(
                        chain.name,
                        res.seqid.num,
                        res.seqid.icode.strip(),
                        res.name,
                        ca,
                        names.get("O", -1),
                        secondary(chain.name, res),
                    )
                )
        return tuple(atoms), tuple(residues), coordinates(xyz)

    atoms, residues, first = parse(st[0])
    keys = tuple(a.key for a in atoms)
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate atom identities; split ambiguous chain/residue identifiers first")
    frames = [first]
    for m in list(st)[1:]:
        aa, _, xx = parse(m)
        lookup = {a.key: i for i, a in enumerate(aa)}
        if len(lookup) != len(aa) or set(lookup) != set(keys):
            raise ValueError("All models must contain exactly the same selected atom identities")
        frames.append(coordinates(xx[[lookup[k] for k in keys]]))
    if not ranges:
        warnings.warn(
            "No helix/sheet annotations found; cartoon uses coils. Supply secondary structure "
            "with with_secondary_structure() or a file containing assignments.",
            stacklevel=2,
        )
    topo = Topology(atoms, residues, infer_bonds(atoms, first), make_chains(residues, first))
    return topo, frames
