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
