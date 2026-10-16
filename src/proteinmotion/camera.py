"""The scene camera: orbit view, framing, region tracking, lens focus and cutaway state.

Each ProteinScene owns one Camera at ``scene.camera``. Focus, FocusPull, Reveal and
``camera.animate`` in animation.py change it over time, and the renderers read its
matrix(), eye, depth cue and cutaway_geometry() for every frame.
"""

import numpy as np

from .math3d import look_at, perspective


class Camera:
    """A perspective camera that orbits a look-at point, in world ångströms.

    The eye sits ``distance`` from ``target`` at azimuth ``theta`` and elevation ``phi``,
    both in radians. frame() and focus() set the view at once; ``camera.animate`` and
    Focus move it over time. The camera also holds EEVEE lens focus and the cutaway that
    Reveal opens.
    """

    def __init__(self):
        # theta turns the eye about the world y axis and phi raises it above the xz plane;
        # theta = phi = 0 puts the eye on the +z side of the target, looking down −z.
        self.target = np.zeros(3)
        self.distance = 70.0
        self.theta = 0.35
        self.phi = 0.25
        self.fov = np.deg2rad(38.0)  # Vertical field of view.
        self.depth_cue = 0.65  # Distance fog strength; 0 disables it.
        self.radius = 20.0  # Framing radius in Å; sets the clip planes in matrix() and the fog range.
        self._tracking = None  # Protein or Region whose center the target follows.
        self.aspect = 16 / 9
        self._screen_position = (0.5, 0.5)  # Subject position on screen; (0, 0) is the top left.
        # Lens depth of field, drawn by EEVEE.
        self.dof = False
        self.fstop = 5.6
        self._focus_target = None  # Protein or Region the focus follows; None keeps _focus_point.
        self._focus_point = np.zeros(3)
        # View-dependent cutaway onto a region; see Reveal. Renderers read it through
        # cutaway_geometry() and the cutaway_* values below.
        self._cutaway_target = None
        self.cutaway_opening = 0.0  # 0 is closed and 1 fully open; the window radius scales with it.
        self.cutaway_window = 1.35  # Window radius at the target, as a multiple of the kept radius.
        self.cutaway_softness = 0.22  # Fraction of the window radius used for its fading edge.
        self.cutaway_band = 4.0  # Depth in Å over which geometry in front of the target fades back in.
        self.cutaway_padding = 2.5  # Å added to the target's radius to give the kept sphere.
        self.cutaway_surface_keep = 0.35  # Fraction of the kept radius for cartoons, bases, surfaces.
        self.cutaway_rim = np.array([0.55, 0.92, 0.86])  # RGB of the faint rim at the fading edge.
        self.cutaway_tunnel = False  # A straight tunnel instead of a cone from the eye.
        self._cutaway_axis = None  # Tunnel direction in the target protein's local frame.
        self.cutaway_wall = 0.28  # Tunnel wall opacity.
        self.cutaway_rings = 5.0  # Spacing in Å of the depth rings on the tunnel wall.

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

    def lens_focus(self, target, *, fstop=5.6, follow=True):
        """Set depth of field without changing framing (EEVEE)."""
        return self.set_focus(target, fstop=fstop, follow=follow)

    def composition_offset(self, screen_position=None, *, distance=None, aspect=None):
        """World offset placing the subject at a normalized top-left screen position."""
        xy = np.asarray(self._screen_position if screen_position is None else screen_position, float)
        if xy.shape != (2,) or not np.isfinite(xy).all() or ((xy < 0) | (xy > 1)).any():
            raise ValueError("screen_position needs two normalized coordinates in [0, 1]")
        # Half the visible height in Å at the subject's distance.
        half_height = (self.distance if distance is None else distance) * np.tan(self.fov / 2)
        aspect = self.aspect if aspect is None else aspect
        # Screen right and up for this theta and phi; look_at() builds the same axes unless
        # the view is nearly vertical.
        right = np.array([np.cos(self.theta), 0, -np.sin(self.theta)])
        up = np.array(
            [-np.sin(self.phi) * np.sin(self.theta), np.cos(self.phi), -np.sin(self.phi) * np.cos(self.theta)]
        )
        # Moving the look-at point opposite to the wanted screen offset moves the subject there.
        # Screen y grows downward, like image rows.
        return 2 * half_height * (-right * (xy[0] - 0.5) * aspect + up * (xy[1] - 0.5))

    @staticmethod
    def _target_point(target):
        """World position (3,) of a Protein or Region's mean atom, or a world point as given."""
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
        cutaway is open. The fifth value is that unit axis in world coordinates: toward the
        eye for a cone, or the tunnel direction.
        """
        target = self._cutaway_target
        if target is None or self.cutaway_opening <= 0:
            return None
        # Kept sphere: centered on the target's bounding box, reaching its farthest atom
        # plus cutaway_padding Å.
        m = target.model_matrix
        points = target.positions @ m[:3, :3].T + m[:3, 3]
        center = (points.min(0) + points.max(0)) / 2
        keep = float(np.linalg.norm(points - center, axis=1).max()) + self.cutaway_padding
        window = keep * self.cutaway_window
        # Every atom of the parent protein, relative to the center: its height along the
        # axis and its distance from the axis.
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
            # 1.8 Å, about one atomic radius, beyond the last atom center before the gap.
            outer = max(floor, float(last) + 1.8)
        return center, keep, window, outer, axis

    def clearest_view(self, region, *objects, samples=160, max_elevation=1.1, turn_penalty=0.15):
        """Return (theta, phi) with the fewest atoms between the camera and ``region``.

        Counts atoms of ``objects`` (default: the region's protein) that project inside the
        region's silhouette and lie in front of it, from the current distance. Nearby views
        win ties through ``turn_penalty``, a fractional cost per radian of camera turn. It
        also decides between views with no atoms in front, so the nearest clear view wins.
        """
        objects = objects or (region.protein,)
        # The region's silhouette is its bounding sphere, with 2 Å to spare.
        m = region.model_matrix
        target = region.positions @ m[:3, :3].T + m[:3, 3]
        center = (target.min(0) + target.max(0)) / 2
        radius = float(np.linalg.norm(target - center, axis=1).max()) + 2.0
        # Atoms that can block the view, relative to the center; the region does not block itself.
        own = set(map(int, region.atom_indices))
        blockers = []
        for obj in objects:
            mm = obj.model_matrix
            xyz = obj.positions @ mm[:3, :3].T + mm[:3, 3]
            if obj is region.protein:
                xyz = xyz[[i for i in range(len(xyz)) if i not in own]]
            blockers.append(xyz)
        blockers = np.concatenate(blockers) - center
        # Sample view directions evenly on a Fibonacci sphere: uniform steps in height y, with
        # azimuths a golden angle apart starting from the current azimuth. Views steeper than
        # max_elevation radians are skipped.
        golden = np.pi * (3 - np.sqrt(5))
        best, best_cost = (self.theta, self.phi), (np.inf, np.inf)
        for i in range(samples):
            y = 1 - 2 * (i + 0.5) / samples
            phi = float(np.arcsin(y))
            if abs(phi) > max_elevation:
                continue
            theta = float(self.theta + i * golden)
            direction = np.array([np.cos(phi) * np.sin(theta), np.sin(phi), np.cos(phi) * np.cos(theta)])
            depth = blockers @ direction  # Positive values lie toward the camera.
            lateral = np.linalg.norm(blockers - depth[:, None] * direction, axis=1)
            # Count blockers more than half a radius in front of the center and inside the silhouette.
            count = float(np.count_nonzero((depth > radius * 0.5) & (lateral < radius)))
            # Angle in radians between this view direction and the current one.
            turn = np.arccos(np.clip(np.dot(direction, (self.eye - self.target) / self.distance), -1, 1))
            # Every clear view costs 0 atoms, so the turn penalty alone decides between them.
            cost = (count * (1 + turn_penalty * turn), turn_penalty * turn)
            if cost < best_cost:
                best, best_cost = (theta, phi), cost
        theta, phi = best
        # Return the equivalent azimuth nearest to the current one.
        theta = self.theta + (theta - self.theta + np.pi) % (2 * np.pi) - np.pi
        return float(theta), float(phi)

    def frame(self, *proteins, margin=1.25, aspect=None, screen_position=(0.5, 0.5)):
        aspect = self.aspect if aspect is None else aspect
        if not proteins or not np.isfinite(margin) or margin <= 0 or not np.isfinite(aspect) or aspect <= 0:
            raise ValueError("frame needs targets and positive finite margin/aspect values")
        points = np.concatenate(
            [p.positions @ p.model_matrix[:3, :3].T + p.model_matrix[:3, 3] for p in proteins]
        )
        # Bounding sphere of every atom: box center, farthest atom plus 2 Å, at least 1 Å.
        self.target = (points.max(0) + points.min(0)) / 2
        self.radius = max(float(np.linalg.norm(points - self.target, axis=1).max()) + 2, 1.0)
        # Fit the sphere, enlarged by margin, to the narrower of the vertical and horizontal
        # half-angles: at this distance it just touches the edges of the view.
        half_fov = min(self.fov / 2, np.arctan(np.tan(self.fov / 2) * aspect))
        self.distance = self.radius * margin / np.sin(half_fov)
        self.target += self.composition_offset(screen_position, aspect=aspect)
        self.aspect, self._screen_position = aspect, tuple(screen_position)
        self._tracking = None
        return self

    def focus(self, target, *, margin=1.25, aspect=None, follow=True, screen_position=(0.5, 0.5)):
        """Immediately frame a Protein or Region, optionally tracking its moving center."""
        self.frame(target, margin=margin, aspect=aspect, screen_position=screen_position)
        self._tracking = target if follow else None
        return self

    def update_tracking(self):
        """Follow the selected center without pumping the zoom as coordinates deform."""
        if self._tracking is not None:
            target = self._tracking
            m = target.model_matrix
            points = target.positions @ m[:3, :3].T + m[:3, 3]
            self.target = (points.min(0) + points.max(0)) / 2 + self.composition_offset()

    def look_along(self, axis, protein=None):
        """Turn the view to look down ``axis``, which then points toward the viewer.

        ``axis`` is in world coordinates, or in a protein's model frame when
        ``protein`` is given (for example a DomainMotion axis or a helix axis).
        """
        direction = np.asarray(axis, dtype=float)
        if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-9:
            raise ValueError("axis must be a nonzero 3-vector")
        if protein is not None:
            direction = protein.orientation @ direction
        direction /= np.linalg.norm(direction)
        # Invert the eye formula; phi stops 0.01 rad short of straight up or down, as in orbit().
        self.phi = float(
            np.clip(np.arcsin(np.clip(direction[1], -1, 1)), -np.pi / 2 + 0.01, np.pi / 2 - 0.01)
        )
        self.theta = float(np.arctan2(direction[0], direction[2]))
        return self

    def orbit(self, theta=0.0, phi=0.0):
        self.theta += theta
        # Stop 0.01 rad short of the poles, so the view direction never becomes vertical.
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
        # Clip planes leave generous room around the framed sphere: 4 radii in front of the
        # target, and 8 radii plus 100 Å behind it. near stays at least 0.05 Å.
        near = max(0.05, self.distance - self.radius * 4)
        far = self.distance + self.radius * 8 + 100
        return perspective(self.fov, aspect, near, far) @ look_at(self.eye, self.target)

    @property
    def animate(self):
        from .animation import Animate

        return Animate(self)

    def snapshot(self):
        # Arrays are copied here and in restore(), so later changes to the camera never
        # alter a saved state.
        return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in vars(self).items()}

    def restore(self, state):
        for k, v in state.items():
            setattr(self, k, v.copy() if isinstance(v, np.ndarray) else v)
