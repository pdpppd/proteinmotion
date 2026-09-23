"""Live atom selections and depth-tested 3D region annotations."""

import gemmi
import numpy as np
from scipy.spatial import cKDTree

from .math3d import color as parse_color
from .protein import Protein
from .structure import Atom, Residue, Topology


class Region:
    """An atom-index selection that reads its parent's current coordinates."""

    def __init__(self, protein, atom_indices):
        ids = np.asarray(atom_indices)
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids):
            raise ValueError("A region needs a nonempty list of integer atom indices")
        if ids.min() < 0 or ids.max() >= len(protein.topology.atoms):
            raise ValueError("Region atom index is out of bounds")
        self.protein = protein
        self.atom_indices = np.unique(ids)
        self.atom_indices.flags.writeable = False
        self.residue_indices = np.unique([protein.topology.atoms[i].residue_index for i in self.atom_indices])
        self.residue_indices.flags.writeable = False
        self._topology = protein.topology
        self._keys = protein.topology.keys

    @classmethod
    def select(
        cls,
        protein,
        *,
        chain=None,
        residues=None,
        atoms=None,
        resname=None,
        ligands=False,
        ions=False,
        water=False,
        within=None,
        of=None,
    ):
        """Residue numbers are PDB/auth numbers. A two-item tuple is an inclusive range.

        Criteria combine with AND, except the ligands/ions/water categories, which
        combine with each other. ``within`` measures the current coordinates once.
        """
        chains = None if chain is None else {chain} if isinstance(chain, str) else set(chain)
        names = None if atoms is None else {atoms} if isinstance(atoms, str) else set(atoms)
        if names is not None:
            names = {n.replace("*", "'") for n in names}
        resnames = None if resname is None else {resname} if isinstance(resname, str) else set(resname)
        if resnames is not None:
            resnames = {n.upper() for n in resnames}
        categories = {c for c, on in (("ligand", ligands), ("ion", ions), ("water", water)) if on}
        residue_categories = protein.topology.residue_categories if categories else None
        near = None
        if (within is None) != (of is None):
            raise ValueError("Use within and of together, e.g. within=5.0, of=ligand")
        if within is not None:
            near = _residues_near(protein, within, of)
        numbers = None
        if residues is not None:
            if isinstance(residues, (int, np.integer)):
                numbers = {int(residues)}
            else:
                values = list(residues)
                if not values or any(not isinstance(i, (int, np.integer)) for i in values):
                    raise ValueError("residues must contain integer PDB residue numbers")
                if isinstance(residues, tuple):
                    if len(values) != 2 or values[0] > values[1]:
                        raise ValueError("Use (first, last) for an inclusive residue range")
                    numbers = set(range(values[0], values[1] + 1))
                else:
                    numbers = set(values)
        ids = [
            i
            for i, atom in enumerate(protein.topology.atoms)
            if (chains is None or atom.chain in chains)
            and (numbers is None or atom.resid in numbers)
            and (names is None or atom.name.replace("*", "'") in names)
            and (resnames is None or atom.resname.upper() in resnames)
            and (residue_categories is None or residue_categories[atom.residue_index] in categories)
            and (near is None or near[atom.residue_index])
        ]
        if not ids:
            raise ValueError(
                "Selection contains no atoms; check chain, residue numbers, names, categories and distance"
            )
        return cls(protein, np.array(ids, dtype=int))

    def _validate(self):
        if self.protein.topology is not self._topology:
            if self.protein.topology.keys != self._keys:
                raise ValueError("Region parent atom identities changed after selection")
            self._topology = self.protein.topology

    @property
    def positions(self):
        self._validate()
        return self.protein.positions[self.atom_indices]

    @property
    def model_matrix(self):
        return self.protein.model_matrix

    @property
    def position(self):
        return self.protein.position

    @property
    def world_positions(self):
        m = self.model_matrix
        return self.positions @ m[:3, :3].T + m[:3, 3]

    def __or__(self, other):
        if not isinstance(other, Region) or other.protein is not self.protein:
            raise ValueError("Combine regions from the same Protein object")
        return Region(self.protein, np.union1d(self.atom_indices, other.atom_indices))

    def highlight(self, *, style="sphere", color="#f2ba67", opacity=None, padding=None, line_width=0.12):
        return RegionHighlight(
            self, style=style, color=color, opacity=opacity, padding=padding, line_width=line_width
        )

    def set_color(self, color):
        from .styling import set_color

        return set_color(self, color)

    def set_opacity(self, opacity):
        from .styling import set_opacity

        return set_opacity(self, opacity)

    def show_atoms(self):
        """Draw these atoms as ball-and-stick over a cartoon, ribbon or surface."""
        from .styling import set_detail

        return set_detail(self, 1.0)

    def hide_atoms(self):
        """Stop drawing these atoms as ball-and-stick detail."""
        from .styling import set_detail

        return set_detail(self, 0.0)

    def side_chains(self):
        """A region with the side chains of this region's amino acids, joined at Cα."""
        from .styling import side_chains

        return side_chains(self)

    @property
    def animate(self):
        from .styling import RegionAnimate

        return RegionAnimate(self)

    def distance_to(self, other, **kwargs):
        from .distances import Distance

        return Distance(self, other, **kwargs)

    def callout(self, text, **kwargs):
        """Create a screen-fixed callout whose leader follows this region."""
        from .annotations import Callout

        return Callout(self, text, **kwargs)

    def label(self, text=None, **kwargs):
        """Label a single residue, using its name and PDB residue number by default."""
        from .annotations import ResidueLabel

        return ResidueLabel(self, text, **kwargs)


def _residues_near(protein, within, of):
    """Residues with any atom within ``within`` Å of ``of``, compared in world space."""
    if not np.isfinite(within) or within <= 0:
        raise ValueError("within must be a finite, positive distance in Å")
    if isinstance(of, Region):
        of._validate()
        reference, source, own = of.world_positions, of.protein, of.residue_indices
    elif isinstance(of, Protein):
        m = of.model_matrix
        reference = of.positions @ m[:3, :3].T + m[:3, 3]
        source, own = of, np.arange(len(of.topology.residues))
    else:
        raise TypeError("of must be a Region or Protein")
    m = protein.model_matrix
    points = protein.positions @ m[:3, :3].T + m[:3, 3]
    distance, _ = cKDTree(reference).query(points, distance_upper_bound=within * protein.size)
    owners = np.array([a.residue_index for a in protein.topology.atoms])
    near = np.zeros(len(protein.topology.residues), bool)
    near[owners[np.isfinite(distance)]] = True
    if source is protein:
        near[own] = False
    return near


class RegionHighlight(Protein):
    """A sphere, wire box or atom halo following a region through motion/deformation.

    These are annotation geometries, not molecular surfaces. The parent owns their
    positions and transform; animate the highlight's opacity, not its transform.
    """

    def __init__(
        self, region, *, style="sphere", color="#f2ba67", opacity=None, padding=None, line_width=0.12
    ):
        if not isinstance(region, Region):
            raise TypeError("Highlight target must be a Region")
        if style not in ("sphere", "box", "atoms"):
            raise ValueError("Highlight style must be 'sphere', 'box', or 'atoms'")
        opacity = (0.9 if style == "box" else 0.18) if opacity is None else opacity
        padding = (0.3 if style == "atoms" else 2.0) if padding is None else padding
        if not np.isfinite(padding) or padding < 0:
            raise ValueError("Highlight padding must be finite and nonnegative")
        if not np.isfinite(line_width) or line_width <= 0:
            raise ValueError("line_width must be finite and positive")
        self.region, self.style, self.padding = region, style, float(padding)
        self.highlight_color = parse_color(color)
        count = 8 if style == "box" else len(region.atom_indices) if style == "atoms" else 1
        atoms = tuple(Atom("highlight", i + 1, "", "MARKER", "CA", "C", i) for i in range(count))
        residues = tuple(Residue("highlight", i + 1, "", "MARKER", i, -1) for i in range(count))
        bonds = (
            [(i, j) for i in range(8) for j in range(i + 1, 8) if (i ^ j) in (1, 2, 4)]
            if style == "box"
            else []
        )
        topology = Topology(atoms, residues, np.array(bonds, np.uint32).reshape(-1, 2), ())
        super().__init__(topology, np.zeros((count, 3)))
        self.ball_and_stick(atom_scale=1.0, bond_radius=line_width)
        self.set_opacity(opacity)
        self._sync()

    def _sync(self):
        points = self.region.positions
        parent = self.region.protein
        lo, hi = points.min(0) - self.padding, points.max(0) + self.padding
        center = (lo + hi) / 2
        if self.style == "sphere":
            xyz = center[None, :]
            radii = np.array([max(0.05, np.linalg.norm(points - center, axis=1).max() + self.padding)])
        elif self.style == "box":
            xyz = np.array([[hi[k] if i & (1 << k) else lo[k] for k in range(3)] for i in range(8)])
            radii = np.full(8, self.bond_radius)
        else:
            xyz = points
            radii = (
                np.array(
                    [
                        max(float(gemmi.Element(parent.topology.atoms[i].element).vdw_r), 1.0)
                        for i in self.region.atom_indices
                    ]
                )
                * parent.atom_scale
                + self.padding
            )
        xyz = np.asarray(xyz, np.float32)
        if not np.array_equal(xyz, self._a) or self._a is not self._b:
            self.set_positions(xyz)
        self.position = parent.position.copy()
        self.orientation = parent.orientation.copy()
        self.size = parent.size
        metadata = np.empty((len(xyz), 4), np.float32)
        metadata[:, :3], metadata[:, 3] = self.highlight_color, radii
        if self._metadata_override is None or not np.array_equal(metadata, self._metadata_override):
            metadata.flags.writeable = False
            self._metadata_override = metadata

    def copy(self):
        raise ValueError("Create another region.highlight() for an independently styled annotation")
