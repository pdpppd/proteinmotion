"""Validated molecular topology and PDB/mmCIF loading via Gemmi."""

import warnings
from dataclasses import dataclass
from functools import cached_property
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
    kind: str = "protein"
    backbone: int = -1
    guide: int = -1
    base: str = ""

    @property
    def is_nucleic(self):
        return self.kind in ("dna", "rna", "nucleic")

    @property
    def trace_atom(self):
        """Cα for amino acids; C4′ (or P in coarse models) for nucleotides."""
        return self.backbone if self.is_nucleic else self.ca

    @property
    def guide_atom(self):
        return self.guide if self.is_nucleic else self.oxygen

    @property
    def morph_atom(self):
        """C1′ for nucleotides, Cα for amino acids; -1 when that atom is missing."""
        return self.guide if self.is_nucleic else self.ca

    @property
    def morph_atom_name(self):
        return "C1'" if self.is_nucleic else "CA"


WATER_NAMES = frozenset(
    {"HOH", "WAT", "H2O", "DOD", "D2O", "SOL", "TIP", "TIP3", "TIP4", "TIP5", "SPC", "T3P"}
)


@dataclass
class Topology:
    atoms: tuple[Atom, ...]
    residues: tuple[Residue, ...]
    bonds: np.ndarray
    chains: tuple[np.ndarray, ...]

    @property
    def keys(self):
        return tuple(a.key for a in self.atoms)

    @cached_property
    def residue_categories(self):
        """Polymer residues lie on a traced chain; the rest are water, single-atom ions or ligands."""
        traced = np.zeros(len(self.residues), bool)
        for chain in self.chains:
            traced[chain] = True
        heavy = [[] for _ in self.residues]
        for atom in self.atoms:
            if atom.element not in ("H", "D"):
                heavy[atom.residue_index].append(atom.element)
        result = []
        for i, r in enumerate(self.residues):
            if traced[i]:
                result.append("polymer")
            elif r.name.upper() in WATER_NAMES:
                result.append("water")
            elif len(heavy[i]) == 1 and heavy[i][0] not in ("C", "N", "O"):
                result.append("ion")
            else:
                result.append("ligand")
        return tuple(result)

    @cached_property
    def untraced_atoms(self):
        """Atoms a cartoon or ribbon cannot draw; these default to ball-and-stick detail."""
        categories = self.residue_categories
        return np.array([categories[a.residue_index] != "polymer" for a in self.atoms], bool)


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


def residue_record(chain, resid, icode, name, names, atoms, secondary="C", parent=None):
    """Shared structure/MD classification; keep Cα distinct from nucleotide anchors."""
    names = {n.replace("*", "'"): i for n, i in names.items()}
    info = gemmi.find_tabulated_residue(parent or name)
    nucleic = info.is_nucleic_acid() or ("C1'" in names and "C4'" in names)
    ca = names.get("CA", -1)
    if ca >= 0 and atoms[ca].element != "C":
        ca = -1
    if nucleic:
        kind = (
            "dna"
            if info.kind == gemmi.ResidueKind.DNA
            else "rna"
            if info.kind == gemmi.ResidueKind.RNA
            else "nucleic"
        )
        base = info.one_letter_code.strip().upper()
        return Residue(
            chain,
            resid,
            icode,
            name,
            -1,
            -1,
            "C",
            kind,
            names.get("C4'", names.get("P", -1)),
            names.get("C1'", -1),
            base if base in ("A", "C", "G", "T", "U", "I") else "N",
        )
    return Residue(
        chain,
        resid,
        icode,
        name,
        ca,
        names.get("O", -1),
        secondary,
        "protein" if info.is_amino_acid() or ca >= 0 else "other",
    )


def make_chains(residues, xyz, atoms=()):
    runs, run = [], []
    links = {}
    for i, atom in enumerate(atoms):
        if atom.name.replace("*", "'") in ("P", "O3'"):
            links[atom.residue_index, atom.name.replace("*", "'")] = i
    for i, r in enumerate(residues):
        if r.trace_atom < 0:
            if len(run) > 1:
                runs.append(np.array(run, np.uint32))
            run = []
            continue
        if run:
            prev = residues[run[-1]]
            # Do not draw ribbons across missing residues or chain discontinuities.
            broken = (
                r.chain != prev.chain
                or r.is_nucleic != prev.is_nucleic
                or np.linalg.norm(xyz[r.trace_atom] - xyz[prev.trace_atom]) > (9.0 if r.is_nucleic else 4.8)
            )
            if r.is_nucleic and prev.is_nucleic:
                a, b = links.get((run[-1], "O3'")), links.get((i, "P"))
                # A deposited phosphodiester link takes precedence over numbering gaps.
                if a is not None and b is not None:
                    broken |= np.linalg.norm(xyz[a] - xyz[b]) > 2.4
                else:
                    broken |= r.resid - prev.resid not in (0, 1)
            if broken:
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
    parents = {
        (m.chain_name, m.res_id.seqid.num, m.res_id.seqid.icode.strip()): m.parent_comp_id
        for m in st.mod_residues
    }
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
                residues.append(
                    residue_record(
                        chain.name,
                        res.seqid.num,
                        res.seqid.icode.strip(),
                        res.name,
                        names,
                        atoms,
                        secondary(chain.name, res),
                        parents.get((chain.name, res.seqid.num, res.seqid.icode.strip())),
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
    if not ranges and any(r.ca >= 0 for r in residues):
        warnings.warn(
            "No helix/sheet annotations found; cartoon uses coils. Supply secondary structure "
            "with with_secondary_structure() or a file containing assignments.",
            stacklevel=2,
        )
    topo = Topology(atoms, residues, infer_bonds(atoms, first), make_chains(residues, first, atoms))
    return topo, frames
