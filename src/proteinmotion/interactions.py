"""Geometric hydrogen bonds and explicitly parameterized screened Coulomb estimates.

These are visualization analyses, not a force field, protonation assignment, or PB solver.
Distances use molecular coordinates, independent of scene transforms.

``Protein.hydrogen_bonds()`` and ``Protein.electrostatics()`` build the two analyses defined
here. Both expose ``.pairs``, a tuple of ``Interaction`` records that is recomputed only when
the protein's coordinates or the analysis settings change, and ``.highlight()``, which returns
an ``InteractionHighlight``: a pool of ``Distance`` rulers (distances.py) that the renderer
draws as depth-tested cylinders or the overlay draws as 2D lines, one per visible pair.
"""

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from .annotations import Annotation, AnnotationLayout
from .distances import Distance
from .math3d import color as parse_color
from .regions import Region

# Coulomb's constant 1/(4πε₀) in kcal·Å/(mol·e²): E = 332.06 q₁q₂ / r for charges in e and r in Å.
# It is the GROMACS value of 138.935458 kJ·nm/(mol·e²) converted to these units.
COULOMB_KCAL_ANGSTROM = 332.0637133


@dataclass(frozen=True)
class Interaction:
    """One detected contact between atoms ``a`` and ``b`` (zero-based topology atom indices).

    For a hydrogen bond, ``a`` is the donor and ``b`` the acceptor, ``distance`` is the
    donor–acceptor distance in Å, ``angle`` is the D–H–A angle in degrees, and ``hydrogen``
    is the explicit H atom index, or None when ``inferred_hydrogen`` marks a virtual H. For an
    electrostatic contact, ``a < b``, ``kind`` is "attractive" or "repulsive", and ``energy``
    is in kcal/mol.
    """

    a: int
    b: int
    distance: float
    kind: str
    angle: float | None = None
    energy: float | None = None
    hydrogen: int | None = None
    inferred_hydrogen: bool = False

    def as_dict(self):
        return asdict(self)


def _indices(value, protein, defaults):
    """Atom indices for a donor or acceptor selection.

    ``None`` gives ``defaults``; a Region gives its atoms; anything else must be a 1D array of
    valid integer atom indices, returned sorted and without duplicates.
    """
    if value is None:
        return np.asarray(defaults, dtype=int)
    if isinstance(value, Region):
        if value.protein is not protein:
            raise ValueError("Interaction selections must belong to the analyzed protein")
        return value.atom_indices
    ids = np.asarray(value)
    if (
        ids.ndim != 1
        or ids.dtype.kind not in "iu"
        or (len(ids) and (ids.min() < 0 or ids.max() >= len(protein.topology.atoms)))
    ):
        raise ValueError("Supply a Region or valid integer atom indices")
    return np.unique(ids)


class _Analysis:
    """Cached analysis shared by HydrogenBonds and Electrostatics.

    Subclasses set ``self.protein`` and implement ``_compute()``, which reads ``self._xyz``
    and returns Interaction records. ``pairs`` reruns it only after the coordinates or the
    values in ``_settings_key()`` change, so a scene can read it every frame.
    """

    def _coordinates(self):
        """Current model coordinates, shape (atoms, 3) in Å, before any scene transform."""
        protein = self.protein
        # positions blends the two coordinate sets named by _key_a and _key_b, by the blend
        # fraction _mix and the per-atom timing in _controls. Together these fix the result.
        key = (protein._key_a, protein._key_b, protein._mix, id(protein._controls), self._settings_key())
        if key != getattr(self, "_coordinate_key", None):
            self._coordinate_key = key
            # Hold the arrays alive so Python cannot reuse their ids for different data.
            self._coordinate_refs = (protein._a, protein._b, protein._controls)
            self._xyz = protein.positions
            self._result = None
        return self._xyz

    def _settings_key(self):
        """Analysis parameters that invalidate cached pairs when changed."""
        return ()

    @property
    def pairs(self):
        self._coordinates()
        if self._result is None:
            self._result = tuple(self._compute())
        return self._result

    def highlight(self, **kwargs):
        return InteractionHighlight(self, **kwargs)


# Default side-chain donor and acceptor atoms by residue and atom name. HydrogenBonds adds
# the backbone amide N and carbonyl O separately. Ser, Thr and Tyr hydroxyls appear in both
# tables. Histidine is left out: which ring N carries the H depends on its tautomer, which
# a structure without hydrogens does not record.
DONORS = {
    "ARG": {"NE", "NH1", "NH2"},
    "LYS": {"NZ"},
    "ASN": {"ND2"},
    "GLN": {"NE2"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
    "TRP": {"NE1"},
    "CYS": {"SG"},
}
ACCEPTORS = {
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
    "ASN": {"OD1"},
    "GLN": {"OE1"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
}
# Base donors (amino and imino N) and acceptors (ring N and carbonyl O) on the Watson–Crick,
# Hoogsteen and minor-groove edges, keyed by one-letter base. Sugar and phosphate oxygens
# are not included.
BASE_DONORS = {"A": {"N6"}, "C": {"N4"}, "G": {"N1", "N2"}, "T": {"N3"}, "U": {"N3"}, "I": {"N1"}}
BASE_ACCEPTORS = {
    "A": {"N1", "N3", "N7"},
    "C": {"N3", "O2"},
    "G": {"O6", "N3", "N7"},
    "T": {"O2", "O4"},
    "U": {"O2", "O4"},
    "I": {"O6", "N3", "N7"},
}
# RNA and DNA residue names whose bases the two tables above describe.
_STANDARD_NUCLEOTIDES = ("A", "C", "G", "T", "U", "I", "DA", "DC", "DG", "DT", "DU", "DI")


def _nucleotide_sites(residue, donor=False):
    """Default donor (``donor=True``) or acceptor atom names for a residue's base."""
    # Modified bases can change protonation/donor identity even with the same parent.
    if not residue.is_nucleic or residue.name not in _STANDARD_NUCLEOTIDES:
        return set()
    return (BASE_DONORS if donor else BASE_ACCEPTORS).get(residue.base, set())


class HydrogenBonds(_Analysis):
    """D–A distance and D–H–A angle criteria, with explicit or virtual backbone H.

    Defaults cover standard amino acids and DNA/RNA bases. Nucleotide donors need
    explicit H; virtual hydrogens are limited to the protein backbone.
    Histidine tautomerism and nonstandard
    residues require explicit donor/acceptor selections. Missing side-chain H are
    never invented. Load explicit H with Protein.from_file(..., include_hydrogens=True).
    """

    def __init__(
        self,
        protein,
        *,
        donors=None,
        acceptors=None,
        max_distance=3.5,
        min_angle=150,
        hydrogens="auto",
        donor_h_cutoff=1.3,
    ):
        # Pairs closer than 1.5 Å are always rejected as bonded or overlapping atoms, so a
        # smaller search radius could never find anything.
        if not np.isfinite(max_distance) or max_distance <= 1.5:
            raise ValueError("max_distance must be finite and > 1.5 Å")
        if not np.isfinite(min_angle) or not 0 <= min_angle <= 180:
            raise ValueError("min_angle must be in [0, 180] degrees")
        if hydrogens not in ("auto", "explicit", "backbone"):
            raise ValueError("hydrogens must be auto, explicit, or backbone")
        if not np.isfinite(donor_h_cutoff) or donor_h_cutoff <= 0:
            raise ValueError("donor_h_cutoff must be finite and positive")
        self.protein = protein
        self.max_distance = max_distance
        self.min_angle = min_angle
        self.hydrogens = hydrogens
        self.donor_h_cutoff = donor_h_cutoff
        atoms = protein.topology.atoms
        residues = protein.topology.residues
        # Default donors: the backbone amide N of every amino acid except proline, whose N
        # has no H, plus the side-chain and base donors in the tables above. A residue with
        # a Cα (ca >= 0) counts as an amino acid.
        default_donors = [
            i
            for i, atom in enumerate(atoms)
            if (atom.name == "N" and atom.resname != "PRO" and residues[atom.residue_index].ca >= 0)
            or atom.name in DONORS.get(atom.resname, set())
            or atom.name in _nucleotide_sites(residues[atom.residue_index], donor=True)
        ]
        # Default acceptors: the backbone carbonyl O and C-terminal OXT of amino acids, plus
        # the side-chain and base acceptors.
        default_acceptors = [
            i
            for i, atom in enumerate(atoms)
            if (atom.name in ("O", "OXT") and residues[atom.residue_index].ca >= 0)
            or atom.name in ACCEPTORS.get(atom.resname, set())
            or atom.name in _nucleotide_sites(residues[atom.residue_index])
        ]
        self.donors = _indices(donors, protein, default_donors)
        self.acceptors = _indices(acceptors, protein, default_acceptors)
        # Explicit hydrogens (and deuteriums) present in the loaded coordinates.
        self._hydrogen_ids = np.array(
            [i for i, atom in enumerate(atoms) if atom.element in ("H", "D")], dtype=int
        )
        # {atom name: atom index} per residue, to find C(i−1) and Cα for a virtual amide H.
        self._names = [{} for _ in residues]
        for i, atom in enumerate(atoms):
            self._names[atom.residue_index][atom.name] = i

    def diagnostics(self):
        """Report usable donors and missing hydrogens without inventing chemistry."""
        xyz, atoms = self._coordinates(), self.protein.topology.atoms
        missing, explicit, virtual = [], [], []
        for donor in self.donors:
            atom = atoms[donor]
            pool = [h for h in self._hydrogen_ids if atoms[h].residue_index == atom.residue_index]
            candidates = self._explicit_hydrogens(donor, xyz, pool)
            identity = (atom.chain, atom.resid, atom.icode, atom.name)
            # Classify each donor the way _compute would use it: explicit H first (unless
            # only backbone H are wanted), then a virtual amide H, else missing.
            if candidates and self.hydrogens != "backbone":
                explicit.append(identity)
            elif self.hydrogens != "explicit" and self._virtual_hydrogen(int(donor), xyz) is not None:
                virtual.append(identity)
            else:
                missing.append(identity)
        return dict(
            donors=len(self.donors),
            acceptors=len(self.acceptors),
            explicit_donors=tuple(explicit),
            virtual_backbone_donors=tuple(virtual),
            missing_hydrogen_donors=tuple(missing),
            pairs=len(self.pairs),
            hint="Load with include_hydrogens=True; nonstandard residues need explicit donors/acceptors",
        )

    def _settings_key(self):
        # Donor and acceptor arrays enter by id, so assigning new arrays invalidates the cache.
        return (
            self.max_distance,
            self.min_angle,
            self.hydrogens,
            self.donor_h_cutoff,
            id(self.donors),
            id(self.acceptors),
        )

    def _explicit_hydrogens(self, donor, xyz, pool):
        """The H atoms from ``pool`` (atom indices) that belong to atom ``donor``.

        An H belongs to a donor when it lies within (0.4, donor_h_cutoff] Å of it and that
        donor is its nearest donor in the same residue, so one H never counts for two donors.
        """
        atoms = self.protein.topology.atoms
        found = []
        for h in pool:
            # The 0.4 Å floor skips an H that sits on top of the donor.
            if 0.4 < np.linalg.norm(xyz[h] - xyz[donor]) <= self.donor_h_cutoff:
                # Assign H to its nearest candidate donor to avoid double counting.
                same = [d for d in self.donors if atoms[d].residue_index == atoms[donor].residue_index]
                if donor == same[int(np.argmin(np.linalg.norm(xyz[same] - xyz[h], axis=1)))]:
                    found.append(h)
        return found

    def _donor_hydrogens(self, donor, xyz, by_residue):
        """(H atom index or None, H position in Å, inferred) for each H that ``donor`` can donate.

        ``by_residue`` maps a residue index to its explicit H atom indices.
        """
        atoms = self.protein.topology.atoms
        hydrogens = []
        if self.hydrogens != "backbone":
            pool = by_residue.get(atoms[donor].residue_index, [])
            hydrogens = [(h, xyz[h], False) for h in self._explicit_hydrogens(donor, xyz, pool)]
        # "auto" infers a backbone amide H only for donors without an explicit one.
        if not hydrogens and self.hydrogens != "explicit":
            virtual = self._virtual_hydrogen(int(donor), xyz)
            if virtual is not None:
                hydrogens.append((None, virtual, True))
        return hydrogens

    def _virtual_hydrogen(self, index, xyz):
        """Position in Å of the inferred amide H on backbone N ``index``, or None if none fits.

        The H goes 1.01 Å (the N–H bond length) from N along the outward bisector of the
        C(i−1)→N and Cα→N bonds, in the plane of those three atoms.
        """
        atom = self.protein.topology.atoms[index]
        ri = atom.residue_index
        # Proline N has no H. The first residue of the topology or of a chain has no
        # preceding carbonyl C to place the H against.
        if atom.name != "N" or atom.resname == "PRO" or ri == 0:
            return None
        previous = self.protein.topology.residues[ri - 1]
        if previous.chain != atom.chain:
            return None
        carbon, alpha = self._names[ri - 1].get("C"), self._names[ri].get("CA")
        # A peptide C–N bond is about 1.33 Å. Outside 0.8–1.8 Å the residues are not joined,
        # for example across a gap of unmodelled residues.
        if carbon is None or alpha is None or not 0.8 < np.linalg.norm(xyz[index] - xyz[carbon]) < 1.8:
            return None
        # Rows: C(i−1)→N and Cα→N. The sum of their unit vectors points away from both atoms.
        v = xyz[index] - xyz[[carbon, alpha]]
        norms = np.linalg.norm(v, axis=1)
        if np.any(norms < 1e-6):
            return None
        direction = (v / norms[:, None]).sum(0)
        if np.linalg.norm(direction) < 1e-6:
            return None
        return xyz[index] + 1.01 * direction / np.linalg.norm(direction)

    def hydrogen_position(self, pair):
        """Current explicit or inferred H position for a detected hydrogen bond, in Å."""
        if pair.kind != "hydrogen_bond":
            raise ValueError("Expected a hydrogen-bond Interaction")
        xyz = self._coordinates()
        point = xyz[pair.hydrogen] if pair.hydrogen is not None else self._virtual_hydrogen(pair.a, xyz)
        if point is None:
            raise ValueError("This conformation has no valid hydrogen for the supplied pair")
        return np.array(point, copy=True)

    def _compute(self):
        # D–A distance and D–H–A angle are defined as in MDAnalysis hydrogen-bond analysis.
        # A pair counts when donor and acceptor lie in different residues,
        # 1.5 Å < D–A ≤ max_distance and D–H–A ≥ min_angle.
        xyz, atoms = self._xyz, self.protein.topology.atoms
        if not len(self.acceptors) or not len(self.donors):
            return []
        # A k-d tree over acceptors finds those within max_distance of each donor.
        tree = cKDTree(xyz[self.acceptors])
        by_residue = {}
        for h in self._hydrogen_ids:
            by_residue.setdefault(atoms[h].residue_index, []).append(int(h))
        # (donor, acceptor) -> Interaction. When several H qualify, the most linear one wins.
        found = {}
        for donor in self.donors:
            hydrogens = self._donor_hydrogens(donor, xyz, by_residue)
            for local in tree.query_ball_point(xyz[donor], self.max_distance):
                acceptor = int(self.acceptors[local])
                if atoms[donor].residue_index == atoms[acceptor].residue_index:
                    continue
                distance = float(np.linalg.norm(xyz[donor] - xyz[acceptor]))
                # Closer than 1.5 Å the atoms are bonded or overlapping.
                if distance <= 1.5:
                    continue
                for hydrogen, point, inferred in hydrogens:
                    # The D–H–A angle at the hydrogen, between H→D and H→A; 180° is linear.
                    a, b = xyz[donor] - point, xyz[acceptor] - point
                    denom = np.linalg.norm(a) * np.linalg.norm(b)
                    # An H on top of the donor or acceptor has no defined angle.
                    if denom < 1e-8:
                        continue
                    angle = float(np.degrees(np.arccos(np.clip(np.dot(a, b) / denom, -1, 1))))
                    if angle >= self.min_angle:
                        key = (int(donor), acceptor)
                        record = Interaction(
                            *key,
                            distance,
                            "hydrogen_bond",
                            angle=angle,
                            hydrogen=hydrogen,
                            inferred_hydrogen=inferred,
                        )
                        if key not in found or angle > found[key].angle:
                            found[key] = record
        return sorted(found.values(), key=lambda r: (r.a, r.b))


def formal_charges(protein):
    """Illustrative charged-side-chain templates; no pKa, termini or tautomer prediction."""
    charges = np.zeros(len(protein.topology.atoms), float)
    # Residue name -> (net charge in e, atoms that share it). These are the ionized states
    # at neutral pH; histidine and the chain termini are left neutral.
    templates = {
        "ASP": (-1, {"OD1", "OD2"}),
        "GLU": (-1, {"OE1", "OE2"}),
        "LYS": (1, {"NZ"}),
        "ARG": (1, {"NH1", "NH2"}),
    }
    groups = {}
    for i, atom in enumerate(protein.topology.atoms):
        template = templates.get(atom.resname)
        if template is not None and atom.name in template[1]:
            groups.setdefault(atom.residue_index, []).append(i)
    # Split each residue's charge evenly over the template atoms it has, for example
    # −0.5 e on each carboxylate O. If one atom is missing, the other carries the whole charge.
    for ri, ids in groups.items():
        charges[ids] = templates[protein.topology.residues[ri].name][0] / len(ids)
    return charges


def charges_from_pqr(protein, path, *, allow_extra=False):
    """Read PQR charges by exact chain/residue/insertion/name identity; reject mismatches."""
    # Atom identity (chain, resid, icode, resname, name), the same tuple as Atom.key -> charge in e.
    lookup = {}
    for line in Path(path).read_text().splitlines():
        # PQR atom records are "ATOM serial name resname [chain] resSeq x y z charge radius".
        # The chain column is optional, giving 11 fields with it and 10 without.
        fields = line.split()
        if not fields or fields[0] not in ("ATOM", "HETATM"):
            continue
        if len(fields) not in (10, 11):
            raise ValueError("Expected standard whitespace-delimited PQR atom records")
        chain = fields[4] if len(fields) == 11 else ""
        number = fields[5] if len(fields) == 11 else fields[4]
        # A residue number may carry a one-letter insertion code, as in "52A".
        match = re.fullmatch(r"(-?\d+)([A-Za-z]?)", number)
        if match is None:
            raise ValueError(f"Invalid PQR residue number: {number}")
        key = (chain, int(match[1]), match[2], fields[3], fields[2])
        if key in lookup:
            raise ValueError(f"Duplicate PQR atom identity: {key}")
        # The charge column comes just before the radius.
        charge = float(fields[-2])
        if not np.isfinite(charge):
            raise ValueError("PQR charges must be finite")
        lookup[key] = charge
    # Every loaded atom needs a charge. Extra PQR atoms, often hydrogens the structure was
    # loaded without, would have their charges dropped, so they are an error unless allowed.
    missing = [a.key for a in protein.topology.atoms if a.key not in lookup]
    if missing:
        raise ValueError(f"PQR lacks {len(missing)} selected atom identities; first: {missing[0]}")
    extra = set(lookup) - set(protein.topology.keys)
    if extra and not allow_extra:
        raise ValueError(
            f"PQR contains {len(extra)} extra atoms; load matching hydrogens/hetero atoms or explicitly set allow_extra=True"
        )
    return np.array([lookup[a.key] for a in protein.topology.atoms], float)


class Electrostatics(_Analysis):
    """Screened pair energies in kcal/mol; potential in kcal/mol per elementary charge."""

    def __init__(
        self,
        protein,
        charges="formal",
        *,
        dielectric=80,
        screening_length=8,
        cutoff=12,
        min_energy=0.05,
        exclude_same_residue=True,
        exclude_bonded=True,
    ):
        self.protein = protein
        if isinstance(charges, str):
            if charges != "formal":
                raise ValueError("Use charges=formal, an atom-ordered array, or from_pqr()")
            values = formal_charges(protein)
            self.charge_source = "illustrative formal side-chain charges"
        else:
            values = np.array(charges, dtype=float, copy=True)
            self.charge_source = "user-supplied atom charges"
        if values.shape != (len(protein.topology.atoms),) or not np.isfinite(values).all():
            raise ValueError("Supply one finite charge (in e) per selected atom")
        for name, value in (("dielectric", dielectric), ("cutoff", cutoff)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if screening_length is not None and (not np.isfinite(screening_length) or screening_length <= 0):
            raise ValueError("screening_length must be positive, or None for unscreened Coulomb")
        if not np.isfinite(min_energy) or min_energy < 0:
            raise ValueError("min_energy must be finite and nonnegative")
        # Freeze the charges: the cache key holds their id, so an in-place edit would leave
        # stale pairs.
        values.flags.writeable = False
        self.charges = values
        self.dielectric = float(dielectric)
        self.screening_length = screening_length
        self.cutoff = float(cutoff)
        self.min_energy = float(min_energy)
        self.exclude_same_residue, self.exclude_bonded = exclude_same_residue, exclude_bonded
        # Covalent bonds as sorted (i, j) atom pairs, for exclude_bonded.
        self._bonds = {tuple(sorted(map(int, pair))) for pair in protein.topology.bonds}

    @classmethod
    def from_pqr(cls, protein, path, *, allow_extra=False, **kwargs):
        """Use charges read from a PQR file; other options are as for Electrostatics."""
        result = cls(protein, charges_from_pqr(protein, path, allow_extra=allow_extra), **kwargs)
        result.charge_source = f"PQR: {Path(path).name}"
        return result

    def _settings_key(self):
        # Charges enter by id; replacing the array invalidates the cache.
        return (
            id(self.charges),
            self.dielectric,
            self.screening_length,
            self.cutoff,
            self.min_energy,
            self.exclude_same_residue,
            self.exclude_bonded,
        )

    def pair_energy(self, a, b):
        """Screened Coulomb energy of atoms ``a`` and ``b`` in kcal/mol, ignoring cutoff and min_energy."""
        if not (0 <= a < len(self.charges) and 0 <= b < len(self.charges)) or a == b:
            raise ValueError("Choose two distinct valid atom indices")
        xyz = self._coordinates()
        r = float(np.linalg.norm(xyz[a] - xyz[b]))
        if r < 1e-8:
            raise ValueError("Coincident charges have undefined point-charge energy")
        # E = 332.06 qa qb exp(−r/λ) / (ε r): Coulomb's law in a uniform dielectric ε with
        # Debye–Hückel (Yukawa) screening length λ in Å; λ = None gives plain Coulomb.
        screening = np.exp(-r / self.screening_length) if self.screening_length is not None else 1
        return float(
            COULOMB_KCAL_ANGSTROM * self.charges[a] * self.charges[b] * screening / (self.dielectric * r)
        )

    def _compute(self):
        # Only charged atoms can contribute.
        ids = np.flatnonzero(self.charges)
        if not len(ids):
            return []
        xyz, atoms = self._xyz, self.protein.topology.atoms
        # Every charged pair within cutoff Å, as rows of indices into ids.
        pairs = cKDTree(xyz[ids]).query_pairs(self.cutoff, output_type="ndarray")
        result = []
        for x, y in pairs:
            a, b = sorted((int(ids[x]), int(ids[y])))
            # By default skip pairs inside one residue, such as the two carboxylate O that split
            # the formal charge of an Asp, and pairs joined by a covalent bond.
            if self.exclude_same_residue and atoms[a].residue_index == atoms[b].residue_index:
                continue
            if self.exclude_bonded and (a, b) in self._bonds:
                continue
            r = float(np.linalg.norm(xyz[a] - xyz[b]))
            if r < 1e-8:
                continue
            energy = self.pair_energy(a, b)
            if abs(energy) >= self.min_energy:
                result.append(
                    Interaction(a, b, r, "attractive" if energy < 0 else "repulsive", energy=energy)
                )
        # Strongest first; atom indices break ties so the order is deterministic.
        return sorted(result, key=lambda r: (-abs(r.energy), r.a, r.b))

    def potential(self, points, *, softening=1.0, chunk_size=2048):
        """Sum all charges (no pair cutoff), with explicit Plummer softening in ångströms."""
        points = np.asarray(points, float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("points must be a finite (n,3) array in model coordinates")
        if not np.isfinite(softening) or softening < 0 or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("softening must be nonnegative and chunk_size a positive integer")
        xyz = self._coordinates()
        ids = np.flatnonzero(self.charges)
        output = np.zeros(len(points))
        # Work through the points in chunks so the (points × charges) distance matrix stays small.
        for start in range(0, len(points), chunk_size):
            # Plummer softening: sqrt(r² + s²) replaces r in both the 1/r and screening terms.
            distances = np.sqrt(cdist(points[start : start + chunk_size], xyz[ids]) ** 2 + softening**2)
            if np.any(distances < 1e-8):
                raise ValueError("Potential is singular at a charge; use positive softening")
            kernel = 1 / distances
            if self.screening_length is not None:
                kernel *= np.exp(-distances / self.screening_length)
            output[start : start + chunk_size] = (
                COULOMB_KCAL_ANGSTROM / self.dielectric * (kernel @ self.charges[ids])
            )
        return output


class _HydrogenAnchor(Region):
    """An inferred H anchor that inherits the donor's transform and opacity."""

    def __init__(self, analysis, pair):
        super().__init__(analysis.protein, [pair.a])
        self.analysis, self.pair = analysis, pair

    @property
    def positions(self):
        # Shape (1, 3) in model Å, like a one-atom Region. world_positions applies the
        # protein's transform to it, and opacity still comes from the donor atom.
        self._validate()
        return self.analysis.hydrogen_position(self.pair)[None, :]


class InteractionHighlight(Annotation):
    """Live styled interaction lines; a fixed slot pool bounds GPU allocations."""

    # Each visible pair is drawn by one Distance ruler, a "slot". Slots are created on first
    # use and then reused, so at most max_pairs rulers ever exist and the GPU buffers behind
    # them (a cylinder mesh per slot in 3D mode) stay bounded. When the pairs change,
    # _refresh() points the first slots at the new pairs and switches the rest off.

    def __init__(
        self,
        analysis,
        *,
        mode="3d",
        color=None,
        attractive_color="#58b8fa",
        repulsive_color="#ef7484",
        show_distances=False,
        max_pairs=100,
        region=None,
        between=None,
        endpoints="donor_acceptor",
        **line_options,
    ):
        super().__init__()
        if not isinstance(max_pairs, int) or not 1 <= max_pairs <= 2000:
            raise ValueError("max_pairs must be between 1 and 2000")
        if region is not None and (not isinstance(region, Region) or region.protein is not analysis.protein):
            raise ValueError("region must belong to the analyzed protein")
        if endpoints not in ("donor_acceptor", "hydrogen_acceptor"):
            raise ValueError("endpoints must be donor_acceptor or hydrogen_acceptor")
        if endpoints == "hydrogen_acceptor" and not isinstance(analysis, HydrogenBonds):
            raise ValueError("hydrogen_acceptor endpoints require HydrogenBonds")
        if between is not None:
            if len(between) != 2 or any(
                not isinstance(r, Region) or r.protein is not analysis.protein for r in between
            ):
                raise ValueError("between needs two Regions from the analyzed protein")
        self.between = between
        self._warned_truncation = False
        self.endpoints = endpoints
        self.analysis, self.max_pairs, self.region = analysis, max_pairs, region
        self.color = None if color is None else parse_color(color)
        self.attractive_color, self.repulsive_color = (
            parse_color(attractive_color),
            parse_color(repulsive_color),
        )
        # Options for every slot's Distance. Each endpoint is a single atom, and "centroid"
        # keeps it as given instead of swapping in the residue's backbone atom.
        self._options = dict(mode=mode, show_distance=show_distances, anchor="centroid", **line_options)
        self._slots = []
        self.visible_pairs = ()
        self.total_pairs = 0
        self._refresh_key = None

    def _refresh(self):
        """Point the slots at the current pairs; a no-op until pairs or write state change."""
        records = self.analysis.pairs
        # The pairs tuple is rebuilt whenever the analysis recomputes, so its id tracks the
        # results. The Write state is part of the key because slots copy it below, and the
        # selections are read live because a Region may be an updating query.
        key = (
            id(records),
            self.endpoints,
            self._write,
            self._lag_ratio,
            self._stroke_width,
            self._reverse,
            self.max_pairs,
            None if self.region is None else tuple(self.region.atom_indices),
            None if self.between is None else tuple(tuple(r.atom_indices) for r in self.between),
        )
        if key == self._refresh_key:
            return
        self._refresh_key = key
        # Hold the tuple alive so Python cannot reuse its id for a later result.
        self._record_reference = records
        records = self._selected(records)
        self.total_pairs = len(records)
        if self.total_pairs > self.max_pairs and not self._warned_truncation:
            import warnings

            warnings.warn(
                f"Showing {self.max_pairs} of {self.total_pairs} interactions; "
                "increase max_pairs or restrict between/region",
                UserWarning,
                stacklevel=2,
            )
            self._warned_truncation = True
        # The analyses sort their pairs (hydrogen bonds by atom index, electrostatics
        # strongest first), so truncation keeps the first max_pairs in that order.
        self.visible_pairs = tuple(records[: self.max_pairs])
        for index, record in enumerate(self.visible_pairs):
            self._assign(index, record)
        for slot in self._slots[len(self.visible_pairs) :]:
            slot.active = False

    def _selected(self, records):
        """The records with an endpoint in ``region`` and, with ``between``, one in each Region."""
        if self.region is not None:
            allowed = set(self.region.atom_indices)
            records = [r for r in records if r.a in allowed or r.b in allowed]
        if self.between is not None:
            left, right = (set(r.atom_indices) for r in self.between)
            records = [r for r in records if (r.a in left and r.b in right) or (r.b in left and r.a in right)]
        return records

    def _assign(self, index, record):
        """Draw ``record`` with slot ``index``, creating that slot the first time it is needed."""
        p = self.analysis.protein
        a, b = Region(p, [record.a]), Region(p, [record.b])
        if self.endpoints == "hydrogen_acceptor":
            # Start at the explicit H, or at the inferred H position when there is none.
            a = (
                Region(p, [record.hydrogen])
                if record.hydrogen is not None
                else _HydrogenAnchor(self.analysis, record)
            )
        # A uniform color overrides the attractive/repulsive pair; hydrogen bonds use
        # the attractive color.
        color = (
            self.color
            if self.color is not None
            else self.repulsive_color
            if record.kind == "repulsive"
            else self.attractive_color
        )
        if index == len(self._slots):
            self._slots.append(Distance(a, b, color=color, **self._options))
        slot = self._slots[index]
        slot.start, slot.end, slot.color = a, b, color
        # A reused slot keeps its old caption color, so set it again: the line color unless
        # label_color was given.
        label_color = self._options.get("label_color")
        slot.title.color = parse_color(color if label_color is None else label_color)
        # Write animates this group, not the slots, so copy its progress to each ruler.
        slot._write, slot._lag_ratio, slot._stroke_width, slot._reverse = (
            self._write,
            self._lag_ratio,
            self._stroke_width,
            self._reverse,
        )
        slot.active = True

    @property
    def glyph_count(self):
        self._refresh()
        return sum(s.glyph_count for s in self._slots if s.active)

    @property
    def text_progress(self):
        # Captions start at 15% of a Write, while the rulers are still drawing.
        return float(np.clip((self._write - 0.15) / 0.85, 0, 1))

    def _geometry_objects(self, camera, width, height):
        self._refresh()
        # 3D rulers are Protein objects that the renderer draws on their own, so pass this
        # group's opacity down to them.
        return [
            obj
            for slot in self._slots
            if slot.active
            for obj in slot._geometry_objects(camera, width, height, parent_opacity=self.opacity)
        ]

    def layout(self, camera, width, height):
        self._refresh()
        # Merge the slots' captions and 2D lines into one layout. The overlay applies only
        # this group's opacity, so fold in each slot's opacity here. Offsetting glyph indices
        # makes a Write run through all captions as if they were one string.
        text, leaders, offset = [], [], 0
        for slot in self._slots:
            if not slot.active:
                continue
            layout = slot.layout(camera, width, height)
            for placement in layout.text:
                placement.glyph_offset += offset
                placement.opacity *= slot.opacity
            for leader in layout.leaders:
                leader.opacity *= slot.opacity
            text.extend(layout.text)
            leaders.extend(layout.leaders)
            offset += slot.glyph_count
        return AnnotationLayout(text, leaders)
