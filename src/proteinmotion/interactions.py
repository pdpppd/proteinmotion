"""Geometric hydrogen bonds and explicitly parameterized screened Coulomb estimates.

These are visualization analyses, not a force field, protonation assignment, or PB solver.
Distances use molecular coordinates, independent of scene transforms.
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

COULOMB_KCAL_ANGSTROM = 332.0637133


@dataclass(frozen=True)
class Interaction:
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
    def _coordinates(self):
        p = self.protein
        key = (p._key_a, p._key_b, p._mix, id(p._controls), self._settings_key())
        if key != getattr(self, "_coordinate_key", None):
            self._coordinate_key = key
            self._coordinate_refs = (p._a, p._b, p._controls)
            self._xyz = p.positions
            self._result = None
        return self._xyz

    def _settings_key(self):
        return ()

    @property
    def pairs(self):
        self._coordinates()
        if self._result is None:
            self._result = tuple(self._compute())
        return self._result

    def highlight(self, **kwargs):
        return InteractionHighlight(self, **kwargs)


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


class HydrogenBonds(_Analysis):
    """D–A distance and D–H–A angle criteria, with explicit or virtual backbone H.

    Defaults cover standard amino acids. Histidine tautomerism and nonstandard
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
        if not np.isfinite(max_distance) or max_distance <= 1.5:
            raise ValueError("max_distance must be finite and > 1.5 Å")
        if not np.isfinite(min_angle) or not 0 <= min_angle <= 180:
            raise ValueError("min_angle must be in [0, 180] degrees")
        if hydrogens not in ("auto", "explicit", "backbone"):
            raise ValueError("hydrogens must be auto, explicit, or backbone")
        if not np.isfinite(donor_h_cutoff) or donor_h_cutoff <= 0:
            raise ValueError("donor_h_cutoff must be finite and positive")
        self.protein = protein
        self.max_distance, self.min_angle, self.hydrogens, self.donor_h_cutoff = (
            max_distance,
            min_angle,
            hydrogens,
            donor_h_cutoff,
        )
        atoms = protein.topology.atoms
        defaults_d = [
            i
            for i, a in enumerate(atoms)
            if (a.name == "N" and a.resname != "PRO" and protein.topology.residues[a.residue_index].ca >= 0)
            or a.name in DONORS.get(a.resname, set())
        ]
        defaults_a = [
            i
            for i, a in enumerate(atoms)
            if (a.name in ("O", "OXT") and protein.topology.residues[a.residue_index].ca >= 0)
            or a.name in ACCEPTORS.get(a.resname, set())
        ]
        self.donors, self.acceptors = (
            _indices(donors, protein, defaults_d),
            _indices(acceptors, protein, defaults_a),
        )
        self._hydrogen_ids = np.array([i for i, a in enumerate(atoms) if a.element in ("H", "D")], dtype=int)
        self._names = [{} for _ in protein.topology.residues]
        for i, a in enumerate(atoms):
            self._names[a.residue_index][a.name] = i

    def _settings_key(self):
        return (
            self.max_distance,
            self.min_angle,
            self.hydrogens,
            self.donor_h_cutoff,
            id(self.donors),
            id(self.acceptors),
        )

    def _virtual_hydrogen(self, index, xyz):
        atom = self.protein.topology.atoms[index]
        ri = atom.residue_index
        if atom.name != "N" or atom.resname == "PRO" or ri == 0:
            return None
        previous = self.protein.topology.residues[ri - 1]
        if previous.chain != atom.chain:
            return None
        carbon, alpha = self._names[ri - 1].get("C"), self._names[ri].get("CA")
        if carbon is None or alpha is None or not 0.8 < np.linalg.norm(xyz[index] - xyz[carbon]) < 1.8:
            return None
        v = xyz[index] - xyz[[carbon, alpha]]
        norms = np.linalg.norm(v, axis=1)
        if np.any(norms < 1e-6):
            return None
        direction = (v / norms[:, None]).sum(0)
        if np.linalg.norm(direction) < 1e-6:
            return None
        return xyz[index] + 1.01 * direction / np.linalg.norm(direction)

    def _compute(self):
        xyz, atoms = self._xyz, self.protein.topology.atoms
        if not len(self.acceptors) or not len(self.donors):
            return []
        tree = cKDTree(xyz[self.acceptors])
        by_residue = {}
        for h in self._hydrogen_ids:
            by_residue.setdefault(atoms[h].residue_index, []).append(int(h))
        found = {}
        for donor in self.donors:
            hydrogens = []
            if self.hydrogens != "backbone":
                for h in by_residue.get(atoms[donor].residue_index, []):
                    if 0.4 < np.linalg.norm(xyz[h] - xyz[donor]) <= self.donor_h_cutoff:
                        # Assign H to its nearest candidate donor to avoid double counting.
                        same = [
                            d for d in self.donors if atoms[d].residue_index == atoms[donor].residue_index
                        ]
                        if donor == same[int(np.argmin(np.linalg.norm(xyz[same] - xyz[h], axis=1)))]:
                            hydrogens.append((h, xyz[h], False))
            if not hydrogens and self.hydrogens != "explicit":
                virtual = self._virtual_hydrogen(int(donor), xyz)
                if virtual is not None:
                    hydrogens.append((None, virtual, True))
            for local in tree.query_ball_point(xyz[donor], self.max_distance):
                acceptor = int(self.acceptors[local])
                if atoms[donor].residue_index == atoms[acceptor].residue_index:
                    continue
                distance = float(np.linalg.norm(xyz[donor] - xyz[acceptor]))
                if distance <= 1.5:
                    continue
                for hydrogen, point, inferred in hydrogens:
                    a, b = xyz[donor] - point, xyz[acceptor] - point
                    denom = np.linalg.norm(a) * np.linalg.norm(b)
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
    for ri, ids in groups.items():
        charges[ids] = templates[protein.topology.residues[ri].name][0] / len(ids)
    return charges


def charges_from_pqr(protein, path, *, allow_extra=False):
    """Read PQR charges by exact chain/residue/insertion/name identity; reject mismatches."""
    lookup = {}
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if not fields or fields[0] not in ("ATOM", "HETATM"):
            continue
        if len(fields) not in (10, 11):
            raise ValueError("Expected standard whitespace-delimited PQR atom records")
        chain = fields[4] if len(fields) == 11 else ""
        number = fields[5] if len(fields) == 11 else fields[4]
        match = re.fullmatch(r"(-?\d+)([A-Za-z]?)", number)
        if match is None:
            raise ValueError(f"Invalid PQR residue number: {number}")
        key = (chain, int(match[1]), match[2], fields[3], fields[2])
        if key in lookup:
            raise ValueError(f"Duplicate PQR atom identity: {key}")
        charge = float(fields[-2])
        if not np.isfinite(charge):
            raise ValueError("PQR charges must be finite")
        lookup[key] = charge
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
        values.flags.writeable = False
        self.charges = values
        self.dielectric, self.screening_length, self.cutoff, self.min_energy = (
            float(dielectric),
            screening_length,
            float(cutoff),
            float(min_energy),
        )
        self.exclude_same_residue, self.exclude_bonded = exclude_same_residue, exclude_bonded
        self._bonds = {tuple(sorted(map(int, pair))) for pair in protein.topology.bonds}

    @classmethod
    def from_pqr(cls, protein, path, *, allow_extra=False, **kwargs):
        result = cls(protein, charges_from_pqr(protein, path, allow_extra=allow_extra), **kwargs)
        result.charge_source = f"PQR: {Path(path).name}"
        return result

    def _settings_key(self):
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
        if not (0 <= a < len(self.charges) and 0 <= b < len(self.charges)) or a == b:
            raise ValueError("Choose two distinct valid atom indices")
        xyz = self._coordinates()
        r = float(np.linalg.norm(xyz[a] - xyz[b]))
        if r < 1e-8:
            raise ValueError("Coincident charges have undefined point-charge energy")
        screening = np.exp(-r / self.screening_length) if self.screening_length is not None else 1
        return float(
            COULOMB_KCAL_ANGSTROM * self.charges[a] * self.charges[b] * screening / (self.dielectric * r)
        )

    def _compute(self):
        ids = np.flatnonzero(self.charges)
        if not len(ids):
            return []
        xyz, atoms = self._xyz, self.protein.topology.atoms
        pairs = cKDTree(xyz[ids]).query_pairs(self.cutoff, output_type="ndarray")
        result = []
        for x, y in pairs:
            a, b = sorted((int(ids[x]), int(ids[y])))
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
        for start in range(0, len(points), chunk_size):
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


class InteractionHighlight(Annotation):
    """Live styled interaction lines; a fixed slot pool bounds GPU allocations."""

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
        **line_options,
    ):
        super().__init__()
        if not isinstance(max_pairs, int) or not 1 <= max_pairs <= 2000:
            raise ValueError("max_pairs must be between 1 and 2000")
        if region is not None and (not isinstance(region, Region) or region.protein is not analysis.protein):
            raise ValueError("region must belong to the analyzed protein")
        self.analysis, self.max_pairs, self.region = analysis, max_pairs, region
        self.color = None if color is None else parse_color(color)
        self.attractive_color, self.repulsive_color = (
            parse_color(attractive_color),
            parse_color(repulsive_color),
        )
        self._options = dict(mode=mode, show_distance=show_distances, anchor="centroid", **line_options)
        self._slots = []
        self.visible_pairs = ()
        self.total_pairs = 0
        self._refresh_key = None

    def _refresh(self):
        records = self.analysis.pairs
        key = (id(records), self._write, self._lag_ratio, self._stroke_width, self._reverse)
        if key == self._refresh_key:
            return
        self._refresh_key = key
        self._record_reference = records
        if self.region is not None:
            allowed = set(self.region.atom_indices)
            records = [r for r in records if r.a in allowed or r.b in allowed]
        self.total_pairs = len(records)
        self.visible_pairs = tuple(records[: self.max_pairs])
        p = self.analysis.protein
        for index, record in enumerate(self.visible_pairs):
            a, b = Region(p, [record.a]), Region(p, [record.b])
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
            label_color = self._options.get("label_color")
            slot.title.color = parse_color(color if label_color is None else label_color)
            slot._write, slot._lag_ratio, slot._stroke_width, slot._reverse = (
                self._write,
                self._lag_ratio,
                self._stroke_width,
                self._reverse,
            )
            slot.active = True
        for slot in self._slots[len(self.visible_pairs) :]:
            slot.active = False

    @property
    def glyph_count(self):
        self._refresh()
        return sum(s.glyph_count for s in self._slots if s.active)

    @property
    def text_progress(self):
        return float(np.clip((self._write - 0.15) / 0.85, 0, 1))

    def _geometry_objects(self, camera, width, height):
        self._refresh()
        return [
            obj
            for slot in self._slots
            if slot.active
            for obj in slot._geometry_objects(camera, width, height, parent_opacity=self.opacity)
        ]

    def layout(self, camera, width, height):
        self._refresh()
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
