"""Live atom selections and depth-tested 3D region annotations.

Protein.select() returns a Region: a set of atom indices into one Protein that reads the
protein's current coordinates whenever it is used, so it follows trajectories, morphs and
transforms. Regions are the targets of styling (colors, opacity, detail atoms), labels,
callouts, distances, torsion tools and camera focus. RegionHighlight draws a sphere, wire
box or atom halos around a region; it is a small synthetic Protein that copies the
region's geometry and its parent's transform every frame.
"""

import numpy as np
from scipy.spatial import cKDTree

from .geometry import _element_data
from .math3d import color as parse_color
from .protein import Protein
from .structure import Atom, Residue, Topology


class Region:
    """An atom-index selection that reads its parent's current coordinates."""

    def __init__(self, protein, atom_indices, *, allow_empty=False):
        ids = np.asarray(atom_indices)
        if ids.size == 0 and allow_empty:
            ids = np.array([], dtype=int)
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or (not len(ids) and not allow_empty):
            raise ValueError("A region needs a nonempty list of integer atom indices")
        if len(ids) and (ids.min() < 0 or ids.max() >= len(protein.topology.atoms)):
            raise ValueError("Region atom index is out of bounds")
        self.protein = protein
        # Sorted, unique and read-only, so callers may share the array.
        self._atom_indices = np.unique(ids)
        self._atom_indices.flags.writeable = False
        # None for a static region. An updating region keeps its select() criteria here and
        # re-runs them on every read of atom_indices.
        self._query = None
        # The parent's topology and atom identities at selection time; see _validate().
        self._topology = protein.topology
        self._keys = protein.topology.keys

    @property
    def atom_indices(self):
        """Sorted atom indices of the selection; an updating region re-evaluates them on each read."""
        if self._query is not None:
            selected = Region.select(self.protein, **self._query, required=False)
            return selected.atom_indices if selected is not None else np.array([], dtype=int)
        return self._atom_indices

    @property
    def residue_indices(self):
        """Sorted topology residue indices that contain a selected atom."""
        return np.unique([self.protein.topology.atoms[i].residue_index for i in self.atom_indices]).astype(
            int
        )

    @property
    def residue_ids(self):
        """Author identities (chain, residue number, insertion code)."""
        return tuple(
            (r.chain, r.resid, r.icode)
            for r in (self.protein.topology.residues[i] for i in self.residue_indices)
        )

    def __len__(self):
        return len(self.atom_indices)

    def __bool__(self):
        return len(self) > 0

    def __repr__(self):
        return f"Region({len(self)} atoms, {len(self.residue_indices)} residues, {'updating' if self._query else 'static'})"

    def freeze(self):
        """Capture the current membership of an updating region."""
        return Region(self.protein, self.atom_indices, allow_empty=True)

    def filter(self, **criteria):
        """Filter within this region using the protein selection vocabulary."""
        selected = Region.select(self.protein, **criteria, required=False)
        return self & selected if selected is not None else Region(self.protein, [], allow_empty=True)

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
        residue_range=None,
        element=None,
        secondary=None,
        polymer=None,
        icode=None,
        required=True,
        updating=False,
    ):
        """Residue numbers are PDB/auth numbers. A two-item tuple is an inclusive range.

        Criteria combine with AND, except the ligands/ions/water categories, which
        combine with each other. ``within`` measures the current coordinates once.

        The criteria:

        - ``chain``: author chain IDs.
        - ``residues``: a residue number, a list of numbers, or a (first, last) tuple.
          ``residue_range`` takes the same inclusive (first, last) bounds.
        - ``atoms`` and ``resname``: atom and residue names; ``resname`` ignores case.
        - ``element``: element symbols, in any case.
        - ``icode``: insertion codes; "" means none.
        - ``secondary``: "H", "E" or "C", as the cartoon currently draws each residue.
        - ``polymer``: True keeps residues on traced chains, False everything else.
        - ``ligands``, ``ions``, ``water``: non-polymer residue categories.
        - ``within=r, of=target``: whole residues with an atom within r Å of a Region or
          Protein, measured in world space, without the target's own residues when it
          belongs to this protein.
        - ``required=False`` returns None instead of raising when nothing matches.
        - ``updating=True`` re-runs the query whenever the region is read, so membership
          follows motion and changes of secondary structure.

        ``chain``, ``atoms``, ``resname``, ``element``, ``icode`` and ``secondary`` take one
        value or a collection of values.
        """
        if updating:
            # Select once now for the initial membership, and keep the criteria so that
            # atom_indices can run them again on every read.
            options = dict(
                chain=chain,
                residues=residues,
                atoms=atoms,
                resname=resname,
                ligands=ligands,
                ions=ions,
                water=water,
                within=within,
                of=of,
                residue_range=residue_range,
                element=element,
                secondary=secondary,
                polymer=polymer,
                icode=icode,
            )
            initial = cls.select(protein, **options, required=False)
            result = cls(protein, [] if initial is None else initial.atom_indices, allow_empty=True)
            result._query = options
            return result
        if residue_range is not None:
            if residues is not None:
                raise ValueError("Choose residues (individual IDs) or residue_range (inclusive bounds)")
            residues = tuple(residue_range)

        # Turn each criterion into a set of accepted values, or None when it is not used.
        # Names compare in upper case where the file's case is not meaningful.
        elements, structures, codes = _choices(element, True), _choices(secondary, True), _choices(icode)
        chains = _choices(chain)
        names = _choices(atoms)
        if names is not None:
            # Legacy "*" primes (C1*) match the modern spelling (C1').
            names = {n.replace("*", "'") for n in names}
        resnames = _choices(resname, True)
        categories = {c for c, on in (("ligand", ligands), ("ion", ions), ("water", water)) if on}
        residue_categories = protein.topology.residue_categories
        states = protein.secondary_structure if structures is not None else None
        # ``near`` is a per-residue mask of residues close to ``of``.
        near = None
        if (within is None) != (of is None):
            raise ValueError("Use within and of together, e.g. within=5.0, of=ligand")
        if within is not None:
            near = _residues_near(protein, within, of)
        numbers = _residue_numbers(residues)
        # One pass over the atoms. An unused criterion is None (or an empty category set)
        # and passes every atom.
        ids = [
            i
            for i, atom in enumerate(protein.topology.atoms)
            if (chains is None or atom.chain in chains)
            and (numbers is None or atom.resid in numbers)
            and (names is None or atom.name.replace("*", "'") in names)
            and (resnames is None or atom.resname.upper() in resnames)
            and (not categories or residue_categories[atom.residue_index] in categories)
            and (polymer is None or (residue_categories[atom.residue_index] == "polymer") == polymer)
            and (elements is None or atom.element.upper() in elements)
            and (structures is None or states[atom.residue_index] in structures)
            and (codes is None or atom.icode in codes)
            and (near is None or near[atom.residue_index])
        ]
        if not ids:
            if not required:
                return None
            chains_available = sorted({a.chain for a in protein.topology.atoms})
            raise ValueError(
                f"Selection contains no atoms; available chains: {chains_available}. "
                "Inspect protein.summary() or use required=False for optional selections"
            )
        return cls(protein, np.array(ids, dtype=int))

    def _validate(self):
        """Raise if the parent's atoms changed since selection, which would make the indices stale."""
        # A new Topology object with the same atoms is fine: with_secondary_structure() and
        # similar calls replace the topology without changing which atoms it lists.
        if self.protein.topology is not self._topology:
            if self.protein.topology.keys != self._keys:
                raise ValueError("Region parent atom identities changed after selection")
            self._topology = self.protein.topology

    @property
    def positions(self):
        """Current coordinates (atoms, 3) in Å of the selected atoms, in the parent's frame."""
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
        """Current coordinates of the selected atoms after the parent's transform."""
        m = self.model_matrix
        return self.positions @ m[:3, :3].T + m[:3, 3]

    def _combine(self, other, operation):
        # Set operations give a static region, even when either operand is updating.
        if not isinstance(other, Region) or other.protein is not self.protein:
            raise ValueError("Combine regions from the same Protein object")
        return Region(self.protein, operation(self.atom_indices, other.atom_indices), allow_empty=True)

    def __or__(self, other):
        return self._combine(other, np.union1d)

    def __and__(self, other):
        return self._combine(other, np.intersect1d)

    def __sub__(self, other):
        return self._combine(other, np.setdiff1d)

    def __invert__(self):
        return Region(
            self.protein,
            np.setdiff1d(np.arange(len(self.protein.topology.atoms)), self.atom_indices),
            allow_empty=True,
        )

    def highlight(self, *, style="sphere", color="#f2ba67", opacity=None, padding=None, line_width=0.12):
        """Create a sphere, box, or atom highlight around the selection; see RegionHighlight."""
        return RegionHighlight(
            self, style=style, color=color, opacity=opacity, padding=padding, line_width=line_width
        )

    def set_color(self, color):
        """Tint these atoms with a ``"#RRGGBB"`` color, or None to restore the color scheme."""
        from .styling import set_color

        return set_color(self, color)

    def set_opacity(self, opacity):
        """Set the opacity of these atoms, in [0, 1]."""
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
        """Animate residue color and opacity fluently, as in ``region.animate.set_color(...)``."""
        from .styling import RegionAnimate

        return RegionAnimate(self)

    def distance_to(self, other, **kwargs):
        """A live Distance ruler from this region to another; options go to Distance."""
        from .distances import Distance

        return Distance(self, other, **kwargs)

    def torsions(self):
        """Measure φ, ψ, ω and χ1–χ5 in degrees for this region's residues."""
        from .torsions import measure

        self._validate()
        return measure(self.protein, self.residue_indices)

    def set_torsions(self, *, conformation=None, anchor="center", secondary="auto", **angles):
        """Set this region's torsions in degrees immediately; see SetTorsions."""
        from .torsions import set_torsions

        return set_torsions(self, conformation=conformation, anchor=anchor, secondary=secondary, **angles)

    def torsion_marker(self, torsion="phi", **kwargs):
        """A live protractor showing one torsion of this single-residue region."""
        from .torsions import TorsionMarker

        return TorsionMarker(self, torsion, **kwargs)

    def callout(self, text, **kwargs):
        """Create a screen-fixed callout whose leader follows this region."""
        from .annotations import Callout

        return Callout(self, text, **kwargs)

    def label(self, text=None, **kwargs):
        """Label a single residue, using its name and PDB residue number by default."""
        from .annotations import ResidueLabel

        return ResidueLabel(self, text, **kwargs)


def _choices(value, upper=False):
    """A selection criterion as a set of accepted values: None stays None, a string is one value."""
    values = None if value is None else {value} if isinstance(value, str) else set(value)
    return {v.upper() for v in values} if values is not None and upper else values


def _residue_numbers(residues):
    """Author residue numbers accepted by ``residues``, or None when it is not used.

    An integer is one number, a list is individual numbers, and a (first, last) tuple
    is an inclusive range.
    """
    if residues is None:
        return None
    if isinstance(residues, (int, np.integer)):
        return {int(residues)}
    values = list(residues)
    if not values or any(not isinstance(i, (int, np.integer)) for i in values):
        raise ValueError("residues must contain integer PDB residue numbers")
    if isinstance(residues, tuple):
        if len(values) != 2 or values[0] > values[1]:
            raise ValueError("Use (first, last) for an inclusive residue range")
        return set(range(values[0], values[1] + 1))
    return set(values)


def _residues_near(protein, within, of):
    """Residues with any atom within ``within`` Å of ``of``, compared in world space."""
    if not np.isfinite(within) or within <= 0:
        raise ValueError("within must be a finite, positive distance in Å")
    # World space lets a region of one protein measure contacts with another protein that
    # has its own transform.
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
    # ``within`` is in the protein's own Å, and world distances are scaled by its size.
    # Atoms with no reference point inside the bound get an infinite distance.
    distance, _ = cKDTree(reference).query(points, distance_upper_bound=within * protein.size)
    owners = np.array([a.residue_index for a in protein.topology.atoms])
    near = np.zeros(len(protein.topology.residues), bool)
    near[owners[np.isfinite(distance)]] = True
    # A region is always near itself; leave its own residues out when it is part of this protein.
    if source is protein:
        near[own] = False
    return near


class RegionHighlight(Protein):
    """A sphere, wire box or atom halo following a region through motion/deformation.

    These are annotation geometries, not molecular surfaces. The parent owns their
    positions and transform; animate the highlight's opacity, not its transform.
    """

    # The highlight is a tiny Protein drawn as ball-and-stick. Its "atoms" are markers: one
    # sphere for "sphere", the 8 corners of a box joined by its 12 edges for "box", or one
    # halo per region atom for "atoms". _metadata_override gives each marker its color and
    # radius in Å, and _sync() refreshes everything from the region before each frame.

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
        # One marker atom per residue. No chains are traced, so nothing draws as a cartoon.
        atoms = tuple(Atom("highlight", i + 1, "", "MARKER", "CA", "C", i) for i in range(count))
        residues = tuple(Residue("highlight", i + 1, "", "MARKER", i, -1) for i in range(count))
        # Box corner i has bit k set when it sits at the high end of axis k (see _sync).
        # Corners that differ in exactly one bit share an edge.
        bonds = (
            [(i, j) for i in range(8) for j in range(i + 1, 8) if (i ^ j) in (1, 2, 4)]
            if style == "box"
            else []
        )
        topology = Topology(atoms, residues, np.array(bonds, np.uint32).reshape(-1, 2), ())
        super().__init__(topology, np.zeros((count, 3)))
        # atom_scale 1 makes the sphere radii in the metadata exact radii in Å.
        self.ball_and_stick(atom_scale=1.0, bond_radius=line_width)
        self.set_opacity(opacity)
        self._sync()

    def _sync(self):
        """Refit the markers to the region's current atoms and copy the parent's transform."""
        points = self.region.positions
        parent = self.region.protein
        lo, hi = points.min(0) - self.padding, points.max(0) + self.padding
        center = (lo + hi) / 2
        if self.style == "sphere":
            # Centered on the bounding box, reaching the farthest atom plus the padding.
            xyz = center[None, :]
            radii = np.array([max(0.05, np.linalg.norm(points - center, axis=1).max() + self.padding)])
        elif self.style == "box":
            # Corner i takes the high bound on axis k when bit k of i is set.
            xyz = np.array([[hi[k] if i & (1 << k) else lo[k] for k in range(3)] for i in range(8)])
            # Corner spheres as thick as the edges round off the joints.
            radii = np.full(8, self.bond_radius)
        else:
            # Each halo is the drawn sphere grown by the padding. The sphere radius comes from
            # the table the renderer uses (van der Waals, at least 1 Å, ions × ION_SCALE), at
            # the parent's atom_scale. The sum is taken in float64 and rounded once, in the
            # float32 metadata below.
            xyz = points
            spheres = _element_data(parent.topology)[self.region.atom_indices, 3].astype(np.float64)
            radii = spheres * parent.atom_scale + self.padding
        xyz = np.asarray(xyz, np.float32)
        # Replace coordinates and metadata only when they change, so the renderer keeps
        # its uploaded copies (it compares array identity).
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
