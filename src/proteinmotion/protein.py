"""Protein scene objects. Topology stays fixed while coordinates and transforms animate."""

from dataclasses import replace

import numpy as np

from .math3d import rotation
from .structure import Topology, coordinates, load_structure
from .trajectory import Trajectory


class Protein:
    def __init__(self, topology, xyz, *, trajectory=None):
        self.topology = topology
        xyz = coordinates(xyz, len(topology.atoms))
        self._a = self._b = xyz
        self._key_a = self._key_b = (id(xyz), 0)
        self._mix = 0.0
        self.trajectory_frame = 0.0
        self._trajectory_source = None
        self.trajectory = trajectory or Trajectory([xyz], topology=topology)
        self.position = np.zeros(3)
        self.orientation = np.eye(3)
        self.size = 1.0
        self.opacity = 1.0
        self.representation = np.array([1.0, 0.0, 0.0])  # cartoon, ribbon, ball-and-stick
        self.color_scheme = "secondary"
        self.atom_scale = 0.30
        self.bond_radius = 0.14
        self.ribbon_width = 1.05
        self._controls = np.zeros((len(topology.atoms), 8), np.float32)
        self._controls[:, 1] = 1
        self._controls[:, 4:6] = 1
        self._controls[:, 7] = 1
        self._controls.flags.writeable = False
        self._metadata_override = None
        from .styling import initial_appearance

        self._appearance = initial_appearance(len(topology.atoms))
        self._color_mix = self._opacity_mix = 1.0
        self._cartoon_scale = np.ones(len(topology.residues), np.float32)
        self._cartoon_scale.flags.writeable = False
        self._surface_options = None
        self.surface_opacity = 0.0

    def select(self, *, chain=None, residues=None, atoms=None):
        """Select PDB-numbered residues/atom names; the returned Region follows this protein."""
        from .regions import Region

        return Region.select(self, chain=chain, residues=residues, atoms=atoms)

    def label_residues(self, *, chain=None, residues=None, **kwargs):
        """Create amino-acid labels attached to the selected residues' Cα atoms."""
        from .annotations import ResidueLabels

        return ResidueLabels(self.select(chain=chain, residues=residues), **kwargs)

    def hydrogen_bonds(self, **kwargs):
        from .interactions import HydrogenBonds

        return HydrogenBonds(self, **kwargs)

    def electrostatics(self, charges="formal", **kwargs):
        from .interactions import Electrostatics

        return Electrostatics(self, charges, **kwargs)

    @classmethod
    def from_file(cls, path, **kwargs):
        topo, frames = load_structure(path, **kwargs)
        return cls(topo, frames[0], trajectory=Trajectory(frames, topology=topo))

    @classmethod
    def from_trajectory(cls, trajectory):
        if trajectory.topology is None:
            raise ValueError("Trajectory needs a topology; use Protein(topology, frame) for raw arrays")
        return cls(trajectory.topology, trajectory.frame(0), trajectory=trajectory)

    @property
    def positions(self):
        alpha = self.atom_progress[:, None]
        return (1 - alpha) * self._a + alpha * self._b

    @property
    def atom_progress(self):
        controls = self._controls
        if not np.any(controls[:, 2]):
            return np.full(len(controls), self._mix, np.float32)
        t = np.clip((self._mix - controls[:, 0]) / np.maximum(controls[:, 1], 1e-8), 0, 1)
        eased = t * t * t * (10 + t * (-15 + 6 * t))
        return np.where(controls[:, 2] > 1.5, eased, np.where(controls[:, 2] > 0.5, t, self._mix))

    @property
    def atom_opacities(self):
        c = self._controls
        t = np.clip((self._mix - c[:, 6]) / np.maximum(c[:, 7], 1e-8), 0, 1)
        t = t * t * t * (10 + t * (-15 + 6 * t))
        from .styling import current_opacities

        return self.opacity * current_opacities(self) * ((1 - t) * c[:, 4] + t * c[:, 5])

    def _pair(self, a, b, alpha, key_a=None, key_b=None):
        self._a, self._b, self._mix = a, b, float(alpha)
        self._key_a = (id(a), 0) if key_a is None else key_a
        self._key_b = (id(b), 0) if key_b is None else key_b

    def set_positions(self, xyz):
        xyz = coordinates(xyz, len(self.topology.atoms))
        self._pair(xyz, xyz, 0.0)
        return self

    def copy(self):
        p = Protein(self.topology, self.positions, trajectory=self.trajectory)
        p.restore(self.snapshot())
        return p

    def center(self):
        self.position = -self.size * (self.orientation @ self.positions.mean(0))
        return self

    def shift(self, vector):
        v = np.asarray(vector, dtype=float)
        if v.shape != (3,) or not np.isfinite(v).all():
            raise ValueError("shift requires a finite 3-vector")
        self.position += v
        return self

    def rotate(self, angle, axis=(0, 1, 0)):
        # Rotate about the molecular centroid while preserving its world position.
        center = self.positions.mean(0)
        old = self.orientation.copy()
        self.orientation = rotation(angle, axis) @ self.orientation
        self.position += self.size * ((old - self.orientation) @ center)
        return self

    def scale(self, factor):
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("Scale factor must be positive")
        center = self.positions.mean(0)
        self.position += self.size * (1 - factor) * (self.orientation @ center)
        self.size *= factor
        return self

    def set_opacity(self, opacity):
        if not 0 <= opacity <= 1:
            raise ValueError("Opacity must be in [0, 1]")
        self.opacity = float(opacity)
        return self

    def set_color(self, color):
        from .styling import set_color

        return set_color(self, color)

    def color_by(self, values, *, scale=None, thickness=None):
        """Map residue values to color and optional cartoon cross-section scale factors."""
        from .properties import color_by

        return color_by(self, values, scale=scale, thickness=thickness)

    def color_residues(self, color, *, chain=None, residues=None):
        self.select(chain=chain, residues=residues).set_color(color)
        return self

    def set_residue_opacity(self, opacity, *, chain=None, residues=None):
        self.select(chain=chain, residues=residues).set_opacity(opacity)
        return self

    def surface(
        self, *, kind="ses", probe_radius=1.4, resolution=0.7, color="secondary", update="rebuild", **kwargs
    ):
        from .surface import surface_options

        self._surface_options = surface_options(
            kind=kind,
            probe_radius=probe_radius,
            resolution=resolution,
            update=update,
            reference=self.positions,
            **kwargs,
        )
        self.representation = np.zeros(3)
        self.surface_opacity = 1.0
        self.color_scheme = color
        return self

    def cartoon(self, *, color="secondary"):
        self.representation = np.array([1.0, 0.0, 0.0])
        self.surface_opacity = 0.0
        self.color_scheme = color
        return self

    def ribbon(self, *, color="rainbow", width=1.05):
        if width <= 0:
            raise ValueError("Ribbon width must be positive")
        self.representation = np.array([0.0, 1.0, 0.0])
        self.surface_opacity = 0.0
        self.color_scheme, self.ribbon_width = color, float(width)
        return self

    def ball_and_stick(self, *, atom_scale=0.30, bond_radius=0.14):
        if atom_scale <= 0 or bond_radius <= 0:
            raise ValueError("Atom and bond radii must be positive")
        self.representation = np.array([0.0, 0.0, 1.0])
        self.surface_opacity = 0.0
        self.atom_scale, self.bond_radius = float(atom_scale), float(bond_radius)
        return self

    def with_secondary_structure(self, assignments):
        """Set one H/E/C label per topology residue (e.g. from DSSP). Before add()."""
        if len(assignments) != len(self.topology.residues) or any(c not in "HEC" for c in assignments):
            raise ValueError("Supply one H/E/C code per residue")
        residues = tuple(replace(r, secondary=c) for r, c in zip(self.topology.residues, assignments))
        self.topology = Topology(self.topology.atoms, residues, self.topology.bonds, self.topology.chains)
        return self

    @property
    def animate(self):
        from .animation import Animate

        return Animate(self)

    @property
    def model_matrix(self):
        m = np.eye(4)
        m[:3, :3] = self.size * self.orientation
        m[:3, 3] = self.position
        return m

    def snapshot(self):
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
            surface_options=self._surface_options,
            surface_opacity=self.surface_opacity,
        )

    def restore(self, s):
        self._pair(s["a"], s["b"], s["mix"], s["key_a"], s["key_b"])
        self.trajectory_frame = s["trajectory_frame"]
        self._trajectory_source = s["trajectory_source"]
        self._controls = s["controls"]
        self._metadata_override = s["metadata_override"]
        self._cartoon_scale = s["cartoon_scale"]
        self._appearance = s["appearance"]
        self._color_mix, self._opacity_mix = s["color_mix"], s["opacity_mix"]
        self._surface_options, self.surface_opacity = s["surface_options"], s["surface_opacity"]
        for k in ("position", "orientation", "representation"):
            setattr(self, k, s[k].copy())
        for k in ("size", "opacity", "color_scheme", "atom_scale", "bond_radius", "ribbon_width"):
            setattr(self, k, s[k])
