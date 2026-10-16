"""Validated molecular topology and PDB/mmCIF loading via Gemmi.

A Topology lists the atoms (Atom), the residues (Residue), the covalent bonds and the
traced chains of a structure. It never changes while coordinates animate; Protein keeps
the coordinates separately. load_structure() reads a file into a Topology plus one
coordinate frame per model and is what Protein.from_file() calls.

residue_record(), infer_bonds() and make_chains() are shared with the peptide builder
(builder.py) and the MDAnalysis reader (trajectory.py), so residues from every source
are classified, bonded and split into chains by the same rules.
"""

import warnings
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import gemmi
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Atom:
    """One atom's identity, as in the file.

    ``chain``, ``resid`` and ``icode`` are the author chain ID, residue number and
    insertion code ("" when there is none). ``name`` is the atom name as written
    (``"CA"``, ``"C1'"``) and ``element`` the symbol as Gemmi spells it (``"C"``, ``"Fe"``).
    ``residue_index`` points into Topology.residues. ``bfactor`` is the isotropic B-factor
    in Å² (pLDDT in AlphaFold models), or NaN when the source has none.
    """

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
        """Identity used to match the same atom across models, trajectories and topologies."""
        return (self.chain, self.resid, self.icode, self.resname, self.name)


@dataclass(frozen=True)
class Residue:
    """One residue and the atom indices that cartoons and morphs need from it.

    ``ca`` and ``oxygen`` are the Cα and backbone carbonyl O of an amino acid, or -1.
    ``backbone`` and ``guide`` are the C4′ (P in phosphate-only models) and C1′ of a
    nucleotide, or -1. ``secondary`` is ``"H"``, ``"E"`` or ``"C"``. ``kind`` is
    ``"protein"``, ``"dna"``, ``"rna"``, ``"nucleic"`` (a nucleotide Gemmi cannot type)
    or ``"other"``. ``base`` is the one-letter base of a nucleotide (A, C, G, T, U, I,
    or N when unknown) and "" for everything else.
    """

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
        """Atom that sets the direction of the cartoon's width: carbonyl O, or C1′ for nucleotides."""
        return self.guide if self.is_nucleic else self.oxygen

    @property
    def morph_atom(self):
        """C1′ for nucleotides, Cα for amino acids; -1 when that atom is missing."""
        return self.guide if self.is_nucleic else self.ca

    @property
    def morph_atom_name(self):
        return "C1'" if self.is_nucleic else "CA"


# Residue names that mean water in PDB files and common MD force fields.
WATER_NAMES = frozenset(
    {"HOH", "WAT", "H2O", "DOD", "D2O", "SOL", "TIP", "TIP3", "TIP4", "TIP5", "SPC", "T3P"}
)


@dataclass
class Topology:
    """Atoms, residues, covalent bonds and traced chains of a structure.

    ``bonds`` is a (bonds, 2) uint32 array of atom-index pairs. ``chains`` holds one
    uint32 array of residue indices per continuous traced run; each run draws as one
    cartoon tube.
    """

    # Other modules cache derived arrays in ``topology.__dict__`` (element colors, torsion
    # tables), so a Topology is never edited. Code that changes residues builds a new one.
    atoms: tuple[Atom, ...]
    residues: tuple[Residue, ...]
    bonds: np.ndarray
    chains: tuple[np.ndarray, ...]

    @property
    def keys(self):
        """Atom identities in order; equal keys mean two topologies list the same atoms."""
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
            # A lone heavy atom is an ion (Na, Zn, Cl), unless it is C, N or O, which makes
            # it a small molecule instead.
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

    @cached_property
    def residue_atoms(self):
        """Atom indices by name for each residue; legacy ``*`` primes read as ``'``."""
        names = tuple({} for _ in self.residues)
        for i, atom in enumerate(self.atoms):
            # The first atom with a name wins if a residue repeats it.
            names[atom.residue_index].setdefault(atom.name.replace("*", "'"), i)
        return names


def coordinates(value, n_atoms=None):
    """A validated, read-only, C-ordered float32 copy of (atoms, 3) coordinates in Å."""
    a = np.array(value, dtype=np.float32, copy=True, order="C")
    if a.ndim != 2 or a.shape[1] != 3 or not len(a) or not np.isfinite(a).all():
        raise ValueError("Coordinates must be a nonempty, finite (atoms, 3) array")
    if n_atoms is not None and len(a) != n_atoms:
        raise ValueError(f"Expected {n_atoms} atoms, got {len(a)}")
    # Read-only arrays can be shared by snapshots, frame caches and GPU uploads without
    # copies, and caches keyed on id() stay valid because nobody edits them in place.
    a.flags.writeable = False
    return a


def infer_bonds(atoms, xyz):
    """Spatially indexed covalent-radius heuristic; not a chemistry bond-order model."""
    # Only pairs closer than 2.65 Å are candidates. This caps the longest bond found.
    pairs = cKDTree(xyz).query_pairs(2.65, output_type="ndarray")
    if not len(pairs):
        return np.empty((0, 2), np.uint32)
    radii = np.array([gemmi.Element(a.element).covalent_r for a in atoms])
    d = np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1)
    # Bonded atoms sit closer than the sum of their covalent radii plus 0.40 Å of slack
    # for coordinate error. Pairs closer than 0.35 Å are overlapping atoms, not bonds.
    limits = radii[pairs[:, 0]] + radii[pairs[:, 1]] + 0.40
    mask = (d > 0.35) & (d < limits)
    # Avoid close contacts between independent chains, except disulfide bridges.
    mask &= np.array(
        [atoms[i].chain == atoms[j].chain or (atoms[i].element == atoms[j].element == "S") for i, j in pairs]
    )
    return pairs[mask].astype(np.uint32)


def residue_record(chain, resid, icode, name, names, atoms, secondary="C", parent=None):
    """Shared structure/MD classification; keep Cα distinct from nucleotide anchors.

    ``names`` maps this residue's atom names to atom indices in ``atoms``. ``parent`` is the
    standard component a modified residue derives from, if the file records one.
    """
    # Old files spell primes as "*" (C1*); read them as C1'.
    names = {n.replace("*", "'"): i for n, i in names.items()}
    # Gemmi's table of common components types amino acids, nucleotides and frequent
    # modified residues. Others are looked up by their parent when the file names one, and
    # an unknown residue with both C1′ and C4′ still counts as a nucleotide.
    info = gemmi.find_tabulated_residue(parent or name)
    nucleic = info.is_nucleic_acid() or ("C1'" in names and "C4'" in names)
    ca = names.get("CA", -1)
    # An atom named CA that is not carbon is a calcium ion, not Cα.
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
        # Nucleotides have no Cα and no helix/strand state. The tube follows C4′, or P in
        # coarse models that keep only phosphates, and C1′ orients it toward the base.
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


# Longest step between consecutive trace atoms inside one continuous chain, in Å.
# Consecutive Cα atoms are 3.8 Å apart and consecutive C4′ atoms about 5 to 7 Å; a longer
# step means residues are missing from the model.
PROTEIN_TRACE_GAP, NUCLEIC_TRACE_GAP = 4.8, 9.0


def make_chains(residues, xyz, atoms=()):
    """Split residues into continuous traced runs, one uint32 array of residue indices each.

    A run ends at a residue without a trace atom, a change of chain ID or polymer type,
    or a gap in the backbone. Pass ``atoms`` so nucleotides can be joined by their
    O3′–P links.
    """
    runs, run = [], []

    def close(run):
        # One residue cannot make a tube, so a run needs at least two.
        if len(run) > 1:
            runs.append(np.array(run, np.uint32))

    # Atom indices of each residue's P and O3′, to test phosphodiester links.
    links = {}
    for i, atom in enumerate(atoms):
        if atom.name.replace("*", "'") in ("P", "O3'"):
            links[atom.residue_index, atom.name.replace("*", "'")] = i
    for i, r in enumerate(residues):
        if r.trace_atom < 0:
            close(run)
            run = []
            continue
        if run:
            prev = residues[run[-1]]
            # Do not draw ribbons across missing residues or chain discontinuities.
            gap = NUCLEIC_TRACE_GAP if r.is_nucleic else PROTEIN_TRACE_GAP
            broken = (
                r.chain != prev.chain
                or r.is_nucleic != prev.is_nucleic
                or np.linalg.norm(xyz[r.trace_atom] - xyz[prev.trace_atom]) > gap
            )
            if r.is_nucleic and prev.is_nucleic:
                o3_prime, phosphorus = links.get((run[-1], "O3'")), links.get((i, "P"))
                # A deposited phosphodiester link takes precedence over numbering gaps.
                # The O3′–P bond is 1.6 Å; 2.4 Å allows for poor geometry.
                if o3_prime is not None and phosphorus is not None:
                    broken |= np.linalg.norm(xyz[o3_prime] - xyz[phosphorus]) > 2.4
                else:
                    # Without both atoms, trust the numbering. A step of 0 is an insertion code.
                    broken |= r.resid - prev.resid not in (0, 1)
            if broken:
                close(run)
                run = []
        run.append(i)
    close(run)
    return tuple(runs)


def load_structure(
    path, *, chains=None, include_water=False, include_hydrogens=False, secondary="auto", assembly=None
):
    """Topology and coordinate frames from PDB/mmCIF.

    ``secondary`` is "auto" (the file's helix/sheet records, or DSSP when it has
    none), "file" (records only) or "dssp" (always computed from the first model).
    ``assembly`` builds a biological assembly (for example "1") from the deposited
    chains; generated copies get new single-letter chain IDs.
    """
    if secondary not in ("auto", "file", "dssp"):
        raise ValueError("secondary must be 'auto', 'file' or 'dssp'")
    st = _read_structure(path, assembly)
    allowed = None if chains is None else ({chains} if isinstance(chains, str) else set(chains))
    # Modified residues (MODRES records) map to the standard component they derive from,
    # so one that Gemmi's table does not know is classified like its parent. Keys are
    # (chain, number, insertion code).
    parents = {
        (m.chain_name, m.res_id.seqid.num, m.res_id.seqid.icode.strip()): m.parent_comp_id
        for m in st.mod_residues
    }
    ranges = _secondary_ranges(st)
    options = (allowed, include_water, include_hydrogens, parents, ranges)

    # The first model defines the topology. Every atom identity must be unique in it.
    atoms, residues, first = _parse_model(st[0], *options)
    keys = tuple(a.key for a in atoms)
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate atom identities; split ambiguous chain/residue identifiers first")
    # Later models (NMR ensembles, multi-model files) become trajectory frames. Each must
    # hold exactly the same atoms, and is reordered by identity to match the first model.
    frames = [first]
    for m in list(st)[1:]:
        aa, _, xx = _parse_model(m, *options)
        lookup = {a.key: i for i, a in enumerate(aa)}
        if len(lookup) != len(aa) or set(lookup) != set(keys):
            raise ValueError("All models must contain exactly the same selected atom identities")
        frames.append(coordinates(xx[[lookup[k] for k in keys]]))
    topo = Topology(atoms, residues, infer_bonds(atoms, first), make_chains(residues, first, atoms))
    if secondary == "dssp" or (secondary == "auto" and not ranges):
        topo = with_dssp(topo, first)
    elif not ranges and any(r.ca >= 0 for r in residues):
        warnings.warn(
            "No helix/sheet annotations found; cartoon uses coils. Use secondary='auto' to assign "
            "them with DSSP, or supply them with with_secondary_structure().",
            stacklevel=2,
        )
    return topo, frames


def _read_structure(path, assembly):
    """The Gemmi structure in ``path``, expanded to a biological assembly when one is named."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    st = gemmi.read_structure(str(path))
    if not len(st):
        raise ValueError("Structure contains no models")
    if assembly is not None:
        st.setup_entities()
        names = [a.name for a in st.assemblies]
        if str(assembly) not in names:
            raise ValueError(f"Assembly {assembly!r} not found; the file defines {names or 'none'}")
        st.transform_to_assembly(str(assembly), gemmi.HowToNameCopiedChain.Short)
    return st


def _secondary_ranges(st):
    """The file's helix and strand records as (first residue, last residue, "H" or "E")."""
    ranges = []
    for h in st.helices:
        ranges.append((h.start, h.end, "H"))
    for sheet in st.sheets:
        for strand in sheet.strands:
            ranges.append((strand.start, strand.end, "E"))
    return ranges


def _file_secondary(ranges, chain, res):
    """H or E when a helix or strand record covers this residue, otherwise C."""
    # Compare (number, insertion code) pairs, so 52A falls between 52 and 53.
    key = (res.seqid.num, res.seqid.icode.strip())
    for start, end, kind in ranges:
        if start.chain_name == chain == end.chain_name:
            lo = (start.res_id.seqid.num, start.res_id.seqid.icode.strip())
            hi = (end.res_id.seqid.num, end.res_id.seqid.icode.strip())
            if lo <= key <= hi:
                return kind
    return "C"


def _parse_model(model, allowed, include_water, include_hydrogens, parents, ranges):
    """Atoms, residues and (atoms, 3) coordinates of one Gemmi model, in file order."""
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
                # Keep one position per atom name. Deterministic highest-occupancy altloc,
                # then blank/A preference. Gemmi reports a blank altloc as "\x00".
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
                    _file_secondary(ranges, chain.name, res),
                    parents.get((chain.name, res.seqid.num, res.seqid.icode.strip())),
                )
            )
    return tuple(atoms), tuple(residues), coordinates(xyz)


def with_dssp(topology, xyz):
    """The topology with residue secondary structure assigned by DSSP from coordinates."""
    from dataclasses import replace

    from .secondary import assign

    states = assign(topology, xyz)
    residues = tuple(replace(r, secondary=c) for r, c in zip(topology.residues, states))
    return Topology(topology.atoms, residues, topology.bonds, topology.chains)
