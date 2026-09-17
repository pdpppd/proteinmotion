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

    def frame(self, *proteins, margin=1.25, aspect=16 / 9):
        if not proteins or not np.isfinite(margin) or margin <= 0 or not np.isfinite(aspect) or aspect <= 0:
            raise ValueError("frame needs targets and positive finite margin/aspect values")
        points = np.concatenate([p.positions @ p.model_matrix[:3, :3].T + p.position for p in proteins])
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
