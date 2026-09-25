import numpy as np

from .math3d import look_at, perspective


class Camera:
    def __init__(self):
        self.target = np.zeros(3)
        self.distance = 70.0
        self.theta = 0.35
        self.phi = 0.25
        self.fov = np.deg2rad(38.0)
        self.depth_cue = 0.65
        self.radius = 20.0
        self._tracking = None
        self.dof = False
        self.fstop = 5.6
        self._focus_target = None
        self._focus_point = np.zeros(3)
        # View-dependent cutaway onto a region (native renderer); see Reveal.
        self._cutaway_target = None
        self.cutaway_opening = 0.0
        self.cutaway_window = 1.35
        self.cutaway_softness = 0.22
        self.cutaway_band = 4.0
        self.cutaway_padding = 2.5
        self.cutaway_surface_keep = 0.35
        self.cutaway_rim = np.array([0.55, 0.92, 0.86])
        self.cutaway_tunnel = False
        self._cutaway_axis = None  # Tunnel direction in the target protein's local frame.
        self.cutaway_wall = 0.28
        self.cutaway_rings = 5.0

    def set_focus(self, target, *, chain=None, residues=None, atoms=None, fstop=5.6, follow=True):
        """Set EEVEE lens focus on a Protein, Region or world point; None disables DOF.

        Selectors use PDB author residue numbers, with inclusive tuple ranges.
        This changes lens focus without moving or zooming the camera.
        """
        if not np.isfinite(fstop) or fstop <= 0:
            raise ValueError("fstop must be finite and positive")
        if any(x is not None for x in (chain, residues, atoms)):
            from .protein import Protein

            if not isinstance(target, Protein):
                raise TypeError("Focus selectors require a Protein")
            target = target.select(chain=chain, residues=residues, atoms=atoms)
        if target is None:
            self.dof, self._focus_target = False, None
            return self
        point = self._target_point(target)
        self.dof, self.fstop = True, float(fstop)
        self._focus_point = point
        self._focus_target = target if follow and hasattr(target, "positions") else None
        return self

    @staticmethod
    def _target_point(target):
        if hasattr(target, "positions"):
            m = target.model_matrix
            point = target.positions.mean(0, dtype=np.float64) @ m[:3, :3].T + m[:3, 3]
        else:
            point = np.asarray(target, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("Focus needs a Protein, nonempty Region, or finite world 3-vector")
        return point.copy()

    @property
    def focus_point(self):
        """Current lens target in world ångströms, following the selected atoms."""
        if self._focus_target is not None:
            return self._target_point(self._focus_target)
        return self._focus_point.copy()

    def cutaway_geometry(self):
        """Center, kept radius, window radius and outer-surface distance of the open cutaway.

        The outer distance is measured from the target center along the tunnel axis, to
        where the tunnel's centerline first reaches open space: atoms of the target's
        protein within 6 Å of the axis, up to the first gap longer than 8 Å. None when no
        cutaway is open.
        """
        target = self._cutaway_target
        if target is None or self.cutaway_opening <= 0:
            return None
        m = target.model_matrix
        points = target.positions @ m[:3, :3].T + m[:3, 3]
        center = (points.min(0) + points.max(0)) / 2
        keep = float(np.linalg.norm(points - center, axis=1).max()) + self.cutaway_padding
        window = keep * self.cutaway_window
        protein = getattr(target, "protein", target)
        pm = protein.model_matrix
        xyz = protein.positions @ pm[:3, :3].T + pm[:3, 3] - center
        axis = self.eye - center if self._cutaway_axis is None else pm[:3, :3] @ self._cutaway_axis
        axis /= max(np.linalg.norm(axis), 1e-9)
        height = xyz @ axis
        lateral = np.linalg.norm(xyz - height[:, None] * axis, axis=1)
        # Walk out along the centerline; the tunnel ends at the first open space (a gap
        # of more than 8 Å between atoms), such as solvent or an internal chamber.
        floor = keep * self.cutaway_surface_keep
        along = np.sort(height[(lateral < min(window, 6.0)) & (height > floor)])
        outer = floor
        if len(along):
            gaps = np.flatnonzero(np.diff(along) > 8.0)
            last = along[gaps[0]] if len(gaps) else along[-1]
            outer = max(floor, float(last) + 1.8)
        return center, keep, window, outer, axis

    def clearest_view(self, region, *objects, samples=160, max_elevation=1.1, turn_penalty=0.15):
        """Return (theta, phi) with the fewest atoms between the camera and ``region``.

        Counts atoms of ``objects`` (default: the region's protein) that project inside the
        region's silhouette and lie in front of it, from the current distance. Nearby views
        win ties through ``turn_penalty``, a fractional cost per radian of camera turn.
        """
        objects = objects or (region.protein,)
        m = region.model_matrix
        target = region.positions @ m[:3, :3].T + m[:3, 3]
        center = (target.min(0) + target.max(0)) / 2
        radius = float(np.linalg.norm(target - center, axis=1).max()) + 2.0
        own = set(map(int, region.atom_indices))
        blockers = []
        for obj in objects:
            mm = obj.model_matrix
            xyz = obj.positions @ mm[:3, :3].T + mm[:3, 3]
            if obj is region.protein:
                xyz = xyz[[i for i in range(len(xyz)) if i not in own]]
            blockers.append(xyz)
        blockers = np.concatenate(blockers) - center
        golden = np.pi * (3 - np.sqrt(5))
        best, best_cost = (self.theta, self.phi), np.inf
        for i in range(samples):
            y = 1 - 2 * (i + 0.5) / samples
            phi = float(np.arcsin(y))
            if abs(phi) > max_elevation:
                continue
            theta = float(self.theta + i * golden)
            direction = np.array([np.cos(phi) * np.sin(theta), np.sin(phi), np.cos(phi) * np.cos(theta)])
            depth = blockers @ direction  # Positive values lie toward the camera.
            lateral = np.linalg.norm(blockers - depth[:, None] * direction, axis=1)
            count = float(np.count_nonzero((depth > radius * 0.5) & (lateral < radius)))
            turn = np.arccos(np.clip(np.dot(direction, (self.eye - self.target) / self.distance), -1, 1))
            cost = count * (1 + turn_penalty * turn)
            if cost < best_cost:
                best, best_cost = (theta, phi), cost
        theta, phi = best
        # Return the equivalent azimuth nearest to the current one.
        theta = self.theta + (theta - self.theta + np.pi) % (2 * np.pi) - np.pi
        return float(theta), float(phi)

    def frame(self, *proteins, margin=1.25, aspect=16 / 9):
        if not proteins or not np.isfinite(margin) or margin <= 0 or not np.isfinite(aspect) or aspect <= 0:
            raise ValueError("frame needs targets and positive finite margin/aspect values")
        points = np.concatenate(
            [p.positions @ p.model_matrix[:3, :3].T + p.model_matrix[:3, 3] for p in proteins]
        )
        self.target = (points.max(0) + points.min(0)) / 2
        self.radius = max(float(np.linalg.norm(points - self.target, axis=1).max()) + 2, 1.0)
        half_fov = min(self.fov / 2, np.arctan(np.tan(self.fov / 2) * aspect))
        self.distance = self.radius * margin / np.sin(half_fov)
        self._tracking = None
        return self

    def focus(self, target, *, margin=1.25, aspect=16 / 9, follow=True):
        """Immediately frame a Protein or Region, optionally tracking its moving center."""
        self.frame(target, margin=margin, aspect=aspect)
        self._tracking = target if follow else None
        return self

    def update_tracking(self):
        """Follow the selected center without pumping the zoom as coordinates deform."""
        if self._tracking is not None:
            target = self._tracking
            m = target.model_matrix
            points = target.positions @ m[:3, :3].T + m[:3, 3]
            self.target = (points.min(0) + points.max(0)) / 2

    def orbit(self, theta=0.0, phi=0.0):
        self.theta += theta
        self.phi = float(np.clip(self.phi + phi, -np.pi / 2 + 0.01, np.pi / 2 - 0.01))
        return self

    def zoom(self, factor):
        if factor <= 0 or not np.isfinite(factor):
            raise ValueError("Zoom factor must be positive")
        self.distance /= factor
        return self

    @property
    def eye(self):
        return self.target + self.distance * np.array(
            [np.cos(self.phi) * np.sin(self.theta), np.sin(self.phi), np.cos(self.phi) * np.cos(self.theta)]
        )

    def matrix(self, aspect):
        near = max(0.05, self.distance - self.radius * 4)
        far = self.distance + self.radius * 8 + 100
        return perspective(self.fov, aspect, near, far) @ look_at(self.eye, self.target)

    @property
    def animate(self):
        from .animation import Animate

        return Animate(self)

    def snapshot(self):
        return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in vars(self).items()}

    def restore(self, state):
        for k, v in state.items():
            setattr(self, k, v.copy() if isinstance(v, np.ndarray) else v)
