"""Protein scene objects. Topology stays fixed while coordinates and transforms animate.

Protein is the object a scene draws: a Topology (structure.py), two coordinate keyframes
with a blend fraction between them, a rigid transform, and display settings. Animations
(animation.py, styling.py, backbone.py, conformations.py, torsions.py) write its fields;
the native renderer (renderer.py, studio.py) and the EEVEE exporter read them each frame.
Many of those fields are private per-atom or per-residue arrays shared with the shaders;
__init__ documents their layout. NucleicAcid is the same object with base colors.
"""

from dataclasses import replace

import numpy as np

from .math3d import rotation
from .structure import Topology, coordinates, load_structure
from .trajectory import Trajectory


class Protein:
    """A molecular structure with fixed atom identities, coordinates, and display settings.

    Use from_file() or fetch() for PDB/mmCIF, build() for an ideal peptide, or
    from_trajectory() for coordinate ensembles. Distances and radii are in ångströms and
    rotation angles in radians. Immediate setters return the protein for chaining.
    """

    # The private arrays set in __init__ are replaced as a whole, never edited in place, and
    # most are marked read-only. The renderers detect most changes by object identity (``is``
    # or id()) and upload only what changed, and snapshot() shares the arrays without copies.

    def __init__(self, topology, xyz, *, trajectory=None):
        self.topology = topology
        xyz = coordinates(xyz, len(topology.atoms))
        # Coordinates are two keyframes, _a and _b, each a read-only (atoms, 3) float32 array
        # in Å. The drawn positions blend from _a to _b by each atom's progress (see
        # atom_progress). The GPU holds both keyframes and blends them in the vertex shader,
        # so a tween between two states costs no upload per frame.
        self._a = self._b = xyz
        # Cache keys for the keyframes. The renderer uploads a keyframe only when its key is
        # new. The default is the array's id(); PlayTrajectory passes (id(trajectory), frame).
        self._key_a = self._key_b = (id(xyz), 0)
        # Blend clock from _a (0) to _b (1). The shaders read it as object.params.x.
        self._mix = 0.0
        self.trajectory_frame = 0.0
        # The Trajectory that PlayTrajectory last played into this protein, or None.
        # Time-series plots check it so they never show samples of a different trajectory.
        self._trajectory_source = None
        self.trajectory = trajectory or Trajectory([xyz], topology=topology)
        self.position = np.zeros(3)
        self.orientation = np.eye(3)
        self.size = 1.0
        self.opacity = 1.0
        # Blend weights of the three representations; a Representation animation crossfades
        # them. surface() sets all three to 0 and shows the surface through surface_opacity.
        self.representation = np.array([1.0, 0.0, 0.0])  # cartoon, ribbon, ball-and-stick
        self.color_scheme = "secondary"
        self.atom_scale = 0.30
        self.bond_radius = 0.14
        self.ribbon_width = 1.05
        # Per-atom motion and visibility windows: (atoms, 8) float32, uploaded as the WGSL
        # struct Control { motion, visibility } in shaders/molecule.wgsl. Times are on the
        # _mix clock.
        #   0  motion start          1  motion span
        #   2  motion mode: 0 follows _mix, 1 linear over the atom's own window,
        #      2 smootherstep over that window
        #   3  unused (pads the vec4)
        #   4  opacity at fade start 5  opacity at fade end
        #   6  fade start            7  fade span (smootherstep)
        # The defaults leave every atom on the shared clock and fully opaque. BackboneMorph
        # and StructureMorph swap in their own arrays to stagger residues and cross-fade
        # the two molecules.
        self._controls = np.zeros((len(topology.atoms), 8), np.float32)
        self._controls[:, 1] = 1
        self._controls[:, 4:6] = 1
        self._controls[:, 7] = 1
        self._controls.flags.writeable = False
        # None, or (atoms, 4) float32 of RGB in [0, 1] plus a sphere radius in Å (before
        # atom_scale) that replaces the color scheme and element radii. Annotations that draw
        # as a Protein (RegionHighlight, the Distance ruler, the Thread wire) use it for their
        # marker colors and sizes.
        self._metadata_override = None
        from .styling import initial_appearance

        # Per-atom appearance tracks: (atoms, 24) float32, uploaded as the WGSL struct
        # Appearance (six vec4). Each track fades from a "before" to an "after" value over a
        # start and span on its own clock; styling.py writes and evaluates them.
        #   0:4   tint before (RGB, strength)   4:8   tint after
        #   8     tint start   9  tint span     10    tint eased flag
        #   11    opacity eased flag
        #   12    opacity before                13    opacity after
        #   14    opacity start                 15    opacity span
        #   16    detail before                 17    detail after
        #   18    detail start                  19    detail span
        #   20    detail eased flag             21:24 unused (pads the last vec4)
        # "Detail" is the ball-and-stick weight an atom keeps over a cartoon, ribbon or surface.
        # Ligands, ions and other atoms outside traced chains draw as ball-and-stick
        # detail over cartoons and ribbons; side chains can be added per residue.
        self._appearance = initial_appearance(len(topology.atoms), topology.untraced_atoms)
        # Clocks of the tint, opacity and detail tracks, in scene seconds while an appearance
        # animation runs. 1.0 against the default start 0 and span 1 shows every "after" value.
        self._color_mix = self._opacity_mix = self._detail_mix = 1.0
        # (residues,) float32 factor on cartoon width and thickness; 1 is the standard profile.
        # color_by(thickness=...) and ColorByProperty set it.
        self._cartoon_scale = np.ones(len(topology.residues), np.float32)
        self._cartoon_scale.flags.writeable = False
        # None until surface() or a Representation animation sets a SurfaceOptions (kind,
        # probe radius, grid spacing, update mode, voxel cap and reference coordinates).
        self._surface_options = None
        self.surface_opacity = 0.0
        # Blend weights of the four nucleotide base shapes; all zero hides the bases.
        self.base_style = np.array([1.0, 0.0, 0.0, 0.0])  # slabs, rings, sticks, ladder
        self.backbone_radius = 0.38
        self.base_thickness = 0.36
        self.base_radius = 0.16
        # Stable reference geometry for base planes, independent of seeking order.
        # nucleic.BaseGeometry shapes each base once from these coordinates, then poses it
        # every frame from three of its ring atoms.
        self._base_reference = xyz
        # Coil/helix/strand weights per residue. Cartoons blend between these profiles
        # when torsion changes or morphs alter the secondary structure.
        # (residues, 3) read-only float32 with columns coil, helix, strand (secondary.STATES);
        # each row is one-hot except during a blend.
        from .secondary import weights

        self._secondary = weights("".join(r.secondary for r in topology.residues))
        # The (φ, ψ) paths of a running torsion animation, read by Ramachandran plots.
        # None, or (residue indices, start (φ, ψ), change in (φ, ψ), per-residue progress,
        # clip progress), with angles in degrees.
        self._torsion_motion = None

    def select(
        self,
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
        """Select atoms by chain, PDB residue number, atom and residue name, or category.

        ``ligands``, ``ions`` and ``water`` choose non-polymer residues; set several to
        combine them. ``within=r, of=region`` keeps whole residues with an atom within
        r Å of that region, excluding the region's own residues. The returned Region
        follows this protein. Region.select() lists every criterion.
        """
        from .regions import Region

        return Region.select(
            self,
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
            required=required,
            updating=updating,
        )

    @classmethod
    def build(
        cls,
        sequence,
        conformation="extended",
        *,
        phi=None,
        psi=None,
        omega=180.0,
        chain="A",
        first=1,
        hydrogens=False,
    ):
        """Build an ideal peptide from a one-letter sequence, with every heavy atom.

        ``conformation`` is helix, strand, ppii, 310, pi, left or extended. ``phi``,
        ``psi`` and ``omega`` (degrees; one value or one per residue) override it.
        Side chains start in common rotamers. ``hydrogens=True`` adds amide H atoms.
        Coordinates are centered with the chain running along x from the N terminus.
        """
        from .builder import build_peptide

        topology, xyz = build_peptide(
            sequence,
            conformation,
            phi=phi,
            psi=psi,
            omega=omega,
            chain=chain,
            first=first,
            hydrogens=hydrogens,
        )
        return cls(topology, xyz)

    @property
    def secondary_structure(self):
        """Current H/E/C state of each residue, as the cartoon draws it."""
        from .secondary import codes

        return codes(self._secondary)

    def set_secondary_structure(self, assignments):
        """Set one H/E/C state per residue for drawing; can change between animations."""
        from .secondary import weights

        if len(assignments) != len(self.topology.residues):
            raise ValueError("Supply one H/E/C code per residue")
        self._secondary = weights(assignments)
        return self

    def assign_secondary_structure(self):
        """Assign helix, strand and coil with DSSP from the current coordinates."""
        from .secondary import assign

        return self.set_secondary_structure(assign(self.topology, self.positions))

    def torsions(self, *, chain=None, residues=None):
        """Measure φ, ψ, ω and χ1–χ5 in degrees for all residues, or a selection."""
        from .torsions import measure

        if chain is None and residues is None:
            return measure(self)
        return measure(self, self.select(chain=chain, residues=residues).residue_indices)

    def set_torsions(self, *, conformation=None, anchor="center", secondary="auto", **angles):
        """Set torsions in degrees immediately, for every residue; see SetTorsions."""
        from .torsions import set_torsions

        set_torsions(self, conformation=conformation, anchor=anchor, secondary=secondary, **angles)
        return self

    def summary(self):
        """Describe available selections without reading topology internals."""
        from collections import Counter

        return dict(
            atoms=len(self.topology.atoms),
            residues=len(self.topology.residues),
            chains=sorted({a.chain for a in self.topology.atoms}),
            categories=dict(Counter(self.topology.residue_categories)),
            residue_names=sorted({r.name for r in self.topology.residues}),
            frames=len(self.trajectory),
        )

    def set_style(self, style):
        """Apply a reusable MolecularStyle."""
        return style.apply(self)

    def reset_style(self):
        """Clear atom overrides and cartoon thickness, restoring the base palette."""
        from .styling import initial_appearance

        self._appearance = initial_appearance(len(self.topology.atoms), self.topology.untraced_atoms)
        self._color_mix = self._opacity_mix = self._detail_mix = 1.0
        self._cartoon_scale = np.ones(len(self.topology.residues), np.float32)
        self._cartoon_scale.flags.writeable = False
        self.opacity = 1.0
        return self

    def show_atoms(self):
        """Draw every atom as ball-and-stick over the current cartoon, ribbon or surface."""
        from .styling import set_detail

        return set_detail(self, 1.0)

    def hide_atoms(self):
        """Remove all ball-and-stick detail, including the default ligands and ions."""
        from .styling import set_detail

        return set_detail(self, 0.0)

    def label_residues(
        self,
        *,
        chain=None,
        residues=None,
        font_size=26,
        color="#edf3fc",
        format="three_letter",
        include_chain=True,
        avoid_overlap=True,
        offsets=None,
        **kwargs,
    ):
        """Label residues at backbone anchors, or centroids for ligands and ions."""
        from .annotations import ResidueLabels

        return ResidueLabels(
            self.select(chain=chain, residues=residues),
            font_size=font_size,
            color=color,
            format=format,
            include_chain=include_chain,
            avoid_overlap=avoid_overlap,
            offsets=offsets,
            **kwargs,
        )

    def hydrogen_bonds(
        self,
        *,
        donors=None,
        acceptors=None,
        max_distance=3.5,
        min_angle=150,
        hydrogens="auto",
        donor_h_cutoff=1.3,
    ):
        """Find hydrogen bonds in this protein; options go to HydrogenBonds."""
        from .interactions import HydrogenBonds

        return HydrogenBonds(
            self,
            donors=donors,
            acceptors=acceptors,
            max_distance=max_distance,
            min_angle=min_angle,
            hydrogens=hydrogens,
            donor_h_cutoff=donor_h_cutoff,
        )

    def electrostatics(
        self,
        charges="formal",
        *,
        dielectric=80,
        screening_length=8,
        cutoff=12,
        min_energy=0.05,
        exclude_same_residue=True,
        exclude_bonded=True,
    ):
        """Screened Coulomb interactions between charged atoms; options go to Electrostatics."""
        from .interactions import Electrostatics

        return Electrostatics(
            self,
            charges,
            dielectric=dielectric,
            screening_length=screening_length,
            cutoff=cutoff,
            min_energy=min_energy,
            exclude_same_residue=exclude_same_residue,
            exclude_bonded=exclude_bonded,
        )

    @classmethod
    def from_file(
        cls,
        path,
        *,
        chains=None,
        include_water=False,
        include_hydrogens=False,
        secondary="auto",
        assembly=None,
    ):
        """Load PDB/mmCIF models. Cartoons use the file's helix and sheet records, or
        DSSP when the file has none (predicted models, MD frames); ``secondary="dssp"``
        always computes them and ``secondary="file"`` never does. ``assembly="1"``
        builds that biological assembly from the deposited chains."""
        topo, frames = load_structure(
            path,
            chains=chains,
            include_water=include_water,
            include_hydrogens=include_hydrogens,
            secondary=secondary,
            assembly=assembly,
        )
        return cls(topo, frames[0], trajectory=Trajectory(frames, topology=topo))

    @classmethod
    def fetch(cls, identifier, *, cache=None, **options):
        """Download a PDB entry (``"4HHB"``) or AlphaFold DB model (``"AF-P0DP23-F1"``) and load it.

        Files are kept in ``cache`` (default ``~/.cache/proteinmotion``) and reused.
        Other options, such as ``assembly="1"`` or ``chains="A"``, go to from_file().
        """
        from .download import fetch_structure

        return cls.from_file(fetch_structure(identifier, cache=cache), **options)

    @classmethod
    def from_trajectory(cls, trajectory):
        """Create a protein from the trajectory's topology and first frame."""
        if trajectory.topology is None:
            raise ValueError("Trajectory needs a topology; use Protein(topology, frame) for raw arrays")
        return cls(trajectory.topology, trajectory.frame(0), trajectory=trajectory)

    @property
    def positions(self):
        """Current atom coordinates in Å, (atoms, 3), in the protein's own frame (before the transform)."""
        alpha = self.atom_progress[:, None]
        return (1 - alpha) * self._a + alpha * self._b

    @property
    def atom_progress(self):
        """Current blend fraction from the first keyframe to the second, one value per atom."""
        # CPU twin of atom_progress() in shaders/molecule.wgsl; see _controls in __init__.
        controls = self._controls
        # Fast path: no atom has a window of its own, so all follow the shared clock.
        if not np.any(controls[:, 2]):
            return np.full(len(controls), self._mix, np.float32)
        # Local progress through each atom's [start, start + span] window. The 1e-8 floor
        # keeps a zero span from dividing by zero.
        t = np.clip((self._mix - controls[:, 0]) / np.maximum(controls[:, 1], 1e-8), 0, 1)
        # Smootherstep, 6t⁵ − 15t⁴ + 10t³, for mode 2 atoms.
        eased = t * t * t * (10 + t * (-15 + 6 * t))
        return np.where(controls[:, 2] > 1.5, eased, np.where(controls[:, 2] > 0.5, t, self._mix))

    @property
    def atom_opacities(self):
        """Current opacity of each atom: global opacity × style opacity × morph cross-fade."""
        # CPU twin of atom_opacity() in shaders/molecule.wgsl. Columns 6:8 of _controls are
        # the fade window, 4:6 the opacities at its start and end.
        controls = self._controls
        t = np.clip((self._mix - controls[:, 6]) / np.maximum(controls[:, 7], 1e-8), 0, 1)
        t = t * t * t * (10 + t * (-15 + 6 * t))
        from .styling import current_opacities

        return self.opacity * current_opacities(self) * ((1 - t) * controls[:, 4] + t * controls[:, 5])

    def _pair(self, a, b, alpha, key_a=None, key_b=None):
        """Show the blend ``alpha`` of keyframes ``a`` and ``b``; keys default to the arrays' ids."""
        self._a, self._b, self._mix = a, b, float(alpha)
        self._key_a = (id(a), 0) if key_a is None else key_a
        self._key_b = (id(b), 0) if key_b is None else key_b

    def set_positions(self, xyz):
        """Replace the coordinates (atoms, 3) in Å; atom identities and topology stay."""
        xyz = coordinates(xyz, len(self.topology.atoms))
        self._pair(xyz, xyz, 0.0)
        return self

    def copy(self):
        """An independent copy of the current state; topology and trajectory are shared."""
        clone = type(self)(self.topology, self.positions, trajectory=self.trajectory)
        clone.restore(self.snapshot())
        return clone

    def center(self):
        """Move the centroid of the atoms to the world origin."""
        # A local point x lands at position + size * orientation @ x in the world.
        self.position = -self.size * (self.orientation @ self.positions.mean(0))
        return self

    def shift(self, vector):
        """Move the protein by a 3-vector in Å, in world coordinates."""
        v = np.asarray(vector, dtype=float)
        if v.shape != (3,) or not np.isfinite(v).all():
            raise ValueError("shift requires a finite 3-vector")
        self.position += v
        return self

    def rotate(self, angle, axis=(0, 1, 0)):
        """Turn the protein by ``angle`` radians about a world ``axis`` through its centroid."""
        # Rotate about the molecular centroid while preserving its world position.
        center = self.positions.mean(0)
        old = self.orientation.copy()
        self.orientation = rotation(angle, axis) @ self.orientation
        self.position += self.size * ((old - self.orientation) @ center)
        return self

    def scale(self, factor):
        """Multiply the drawn size by ``factor`` about the centroid."""
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("Scale factor must be positive")
        # Shift by how far the scaled centroid would move, so the centroid stays in place.
        center = self.positions.mean(0)
        self.position += self.size * (1 - factor) * (self.orientation @ center)
        self.size *= factor
        return self

    def set_opacity(self, opacity):
        """Set the global opacity in [0, 1]."""
        if not 0 <= opacity <= 1:
            raise ValueError("Opacity must be in [0, 1]")
        self.opacity = float(opacity)
        return self

    def set_color(self, color):
        """Tint every atom with a ``"#RRGGBB"`` color, or None to restore the color scheme."""
        from .styling import set_color

        return set_color(self, color)

    def color_by(self, values, *, scale=None, thickness=None):
        """Map residue values to color and optional cartoon cross-section scale factors."""
        from .properties import color_by

        return color_by(self, values, scale=scale, thickness=thickness)

    def color_residues(self, color, *, chain=None, residues=None):
        """Tint the selected residues; see select() for ``chain`` and ``residues``."""
        self.select(chain=chain, residues=residues).set_color(color)
        return self

    def set_residue_opacity(self, opacity, *, chain=None, residues=None):
        """Set the opacity of the selected residues, in [0, 1]."""
        self.select(chain=chain, residues=residues).set_opacity(opacity)
        return self

    def surface(
        self,
        *,
        kind="ses",
        probe_radius=1.4,
        resolution=None,
        grid_spacing=None,
        color="secondary",
        update="rebuild",
        max_voxels=8_000_000,
    ):
        """Show a molecular surface: ``kind`` is vdw, sas or ses, and lengths are in Å."""
        from .surface import surface_options

        if resolution is not None and grid_spacing is not None:
            raise ValueError("Use grid_spacing or its legacy alias resolution, not both")
        # ``resolution`` is the old name of ``grid_spacing``; either one sets the voxel size.
        if grid_spacing is not None:
            resolution = grid_spacing
        elif resolution is None:
            resolution = 0.7
        self._surface_options = surface_options(
            kind=kind,
            probe_radius=probe_radius,
            resolution=resolution,
            update=update,
            reference=self.positions,
            max_voxels=max_voxels,
        )
        # The surface replaces cartoon, ribbon and ball-and-stick; detail atoms such as
        # ligands still draw over it.
        self.representation = np.zeros(3)
        self.surface_opacity = 1.0
        self.color_scheme = color
        return self

    def cartoon(
        self,
        *,
        color="secondary",
        bases="slabs",
        backbone_radius=0.38,
        base_thickness=0.36,
        base_radius=0.16,
        atom_scale=None,
        bond_radius=None,
    ):
        """Protein cartoons and nucleotide backbone tubes with selectable base shapes.

        ``bases`` is slabs, rings, sticks, ladder, or none. Dimensions are in Å.
        Nucleotide slabs/rings follow the deposited base plane; ladder rods are schematic.
        """
        self.set_bases(
            bases, backbone_radius=backbone_radius, base_thickness=base_thickness, base_radius=base_radius
        )
        for name, value in (("atom_scale", atom_scale), ("bond_radius", bond_radius)):
            if value is not None:
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f"{name} must be finite and positive")
                setattr(self, name, float(value))
        self.representation = np.array([1.0, 0.0, 0.0])
        self.surface_opacity = 0.0
        self.color_scheme = color
        return self

    def ribbon(self, *, color="rainbow", width=1.05, bases="none"):
        """Show a flat ribbon of ``width`` Å through the backbone."""
        if width <= 0:
            raise ValueError("Ribbon width must be positive")
        self.representation = np.array([0.0, 1.0, 0.0])
        self.surface_opacity = 0.0
        self.color_scheme, self.ribbon_width = color, float(width)
        self.set_bases(bases)
        return self

    def set_bases(self, style, *, backbone_radius=None, base_thickness=None, base_radius=None):
        """Set nucleotide base geometry. Use BaseStyle in a scene to animate this change."""
        from .nucleic import base_weights

        weights = base_weights(style)
        dimensions = dict(
            backbone_radius=backbone_radius, base_thickness=base_thickness, base_radius=base_radius
        )
        # Validate every dimension before setting any, so a bad value changes nothing.
        for name, value in dimensions.items():
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        self.base_style = weights
        for name, value in dimensions.items():
            if value is not None:
                setattr(self, name, float(value))
        return self

    def ball_and_stick(self, *, atom_scale=0.30, bond_radius=0.14, color="element"):
        """Show every atom; spheres are ``atom_scale`` × van der Waals radius, bonds ``bond_radius`` Å."""
        if atom_scale <= 0 or bond_radius <= 0:
            raise ValueError("Atom and bond radii must be positive")
        self.representation = np.array([0.0, 0.0, 1.0])
        self.surface_opacity = 0.0
        self.atom_scale, self.bond_radius = float(atom_scale), float(bond_radius)
        self.color_scheme = color
        return self

    def with_secondary_structure(self, assignments):
        """Set one H/E/C label per topology residue. Before add().

        ``assignments="dssp"`` computes them from the coordinates. Use
        set_secondary_structure() to change states after the protein is drawn.
        """
        if isinstance(assignments, str) and assignments.lower() == "dssp":
            from .secondary import assign

            assignments = assign(self.topology, self.positions)
        if len(assignments) != len(self.topology.residues) or any(c not in "HEC" for c in assignments):
            raise ValueError("Supply one H/E/C code per residue")
        # Topologies are never edited (other modules cache data on them), so build a new one.
        residues = tuple(replace(r, secondary=c) for r, c in zip(self.topology.residues, assignments))
        self.topology = Topology(self.topology.atoms, residues, self.topology.bonds, self.topology.chains)
        return self.set_secondary_structure(assignments)

    @property
    def animate(self):
        """Animate transforms fluently: ``scene.play(protein.animate.shift(...).rotate(...))``."""
        from .animation import Animate

        return Animate(self)

    @property
    def model_matrix(self):
        """4 × 4 local-to-world matrix: uniform scale ``size``, then ``orientation``, then ``position``."""
        m = np.eye(4)
        m[:3, :3] = self.size * self.orientation
        m[:3, 3] = self.position
        return m

    def snapshot(self):
        """All animatable state as a dict for restore(); the timeline restores it to seek."""
        # position, orientation, representation and base_style may be edited in place
        # (shift() uses +=), so they are copied. The private arrays are only ever replaced,
        # so sharing them is safe and keeps snapshots cheap.
        return dict(
            a=self._a,
            b=self._b,
            key_a=self._key_a,
            key_b=self._key_b,
            mix=self._mix,
            trajectory_frame=self.trajectory_frame,
            trajectory_source=self._trajectory_source,
            position=self.position.copy(),
            orientation=self.orientation.copy(),
            size=self.size,
            opacity=self.opacity,
            representation=self.representation.copy(),
            color_scheme=self.color_scheme,
            atom_scale=self.atom_scale,
            bond_radius=self.bond_radius,
            ribbon_width=self.ribbon_width,
            controls=self._controls,
            metadata_override=self._metadata_override,
            cartoon_scale=self._cartoon_scale,
            appearance=self._appearance,
            color_mix=self._color_mix,
            opacity_mix=self._opacity_mix,
            detail_mix=self._detail_mix,
            surface_options=self._surface_options,
            surface_opacity=self.surface_opacity,
            base_style=self.base_style.copy(),
            backbone_radius=self.backbone_radius,
            base_thickness=self.base_thickness,
            base_radius=self.base_radius,
            base_reference=self._base_reference,
            secondary=self._secondary,
            torsion_motion=self._torsion_motion,
        )

    def restore(self, s):
        """Return to a state recorded by snapshot()."""
        self._pair(s["a"], s["b"], s["mix"], s["key_a"], s["key_b"])
        self.trajectory_frame = s["trajectory_frame"]
        self._trajectory_source = s["trajectory_source"]
        self._controls = s["controls"]
        self._metadata_override = s["metadata_override"]
        self._cartoon_scale = s["cartoon_scale"]
        self._appearance = s["appearance"]
        self._color_mix, self._opacity_mix = s["color_mix"], s["opacity_mix"]
        self._detail_mix = s["detail_mix"]
        self._surface_options, self.surface_opacity = s["surface_options"], s["surface_opacity"]
        self._base_reference = s["base_reference"]
        self._secondary, self._torsion_motion = s["secondary"], s["torsion_motion"]
        for k in ("position", "orientation", "representation", "base_style"):
            setattr(self, k, s[k].copy())
        for k in (
            "size",
            "opacity",
            "color_scheme",
            "atom_scale",
            "bond_radius",
            "ribbon_width",
            "backbone_radius",
            "base_thickness",
            "base_radius",
        ):
            setattr(self, k, s[k])


class NucleicAcid(Protein):
    """DNA/RNA scene object with the same selections, surfaces and animations as Protein.

    Loads PDB/mmCIF coordinates and uses base colors by default. Protein can also load
    mixed protein–nucleic-acid complexes and draws both polymer types together.
    """

    def __init__(self, topology, xyz, *, trajectory=None):
        if not any(r.is_nucleic for r in topology.residues):
            raise ValueError("NucleicAcid requires at least one DNA/RNA residue")
        super().__init__(topology, xyz, trajectory=trajectory)
        self.color_scheme = "base"

    def cartoon(self, *, color="base", **kwargs):
        """Show a C4′ backbone tube with bases; other options are those of Protein.cartoon()."""
        return super().cartoon(color=color, **kwargs)

    def surface(self, *, color="base", **kwargs):
        """Show a molecular surface in base colors; other options are those of Protein.surface()."""
        return super().surface(color=color, **kwargs)
