"""Deterministic animations: capture at compilation, evaluate at any time."""

import numpy as np

from .math3d import align_coordinates, rotation
from .rates import linear, smooth
from .structure import coordinates
from .trajectory import FrameCache


class Animation:
    channels = frozenset()

    def __init__(self, target, *, rate_func=smooth):
        self.target, self.rate_func = target, rate_func
        self._bound = False

    def bind(self):
        if self._bound:
            raise ValueError("Animation objects may only be played once; create a new animation")
        self._bound = True

    @property
    def targets(self):
        return (self.target,)

    def apply(self, alpha):
        raise NotImplementedError


class Rotate(Animation):
    channels = frozenset({"transform"})

    def __init__(self, protein, angle, axis=(0, 1, 0), **kwargs):
        super().__init__(protein, **kwargs)
        rotation(angle, axis)  # validate now
        self.angle, self.axis = angle, axis

    def bind(self):
        super().bind()
        p = self.target
        self.r, self.pos = p.orientation.copy(), p.position.copy()
        self.center, self.size = p.positions.mean(0), p.size

    def apply(self, alpha):
        p = self.target
        p.orientation = rotation(self.angle * alpha, self.axis) @ self.r
        p.position = self.pos + self.size * ((self.r - p.orientation) @ self.center)


class Morph(Animation):
    channels = frozenset({"geometry"})

    def __init__(self, protein, target, *, align=True, atom_map=None, **kwargs):
        super().__init__(protein, **kwargs)
        self.destination, self.align, self.atom_map = target, align, atom_map

    def bind(self):
        super().bind()
        p, target = self.target, self.destination
        self.start = coordinates(p.positions)
        if hasattr(target, "topology"):
            target_xyz = target.positions
            if self.atom_map is None:
                lookup = {k: i for i, k in enumerate(target.topology.keys)}
                if len(lookup) != len(target.topology.keys) or set(lookup) != set(p.topology.keys):
                    raise ValueError(
                        "Morph requires matching atom identities; supply an explicit atom_map "
                        "for a different topology"
                    )
                mapping = [lookup[k] for k in p.topology.keys]
            else:
                mapping = np.asarray(self.atom_map)
            if len(mapping) != len(self.start) or len(set(mapping)) != len(mapping):
                raise ValueError("atom_map must map every source atom to a unique target atom")
            mapping = np.asarray(mapping)
            if mapping.dtype.kind not in "iu" or (mapping < 0).any() or (mapping >= len(target_xyz)).any():
                raise ValueError("atom_map contains invalid target indices")
            end = target_xyz[mapping]
        else:
            if self.atom_map is not None:
                raise ValueError("atom_map is only supported with a Protein target")
            end = target
        self.end = coordinates(end, len(self.start))
        if self.align:
            ca = [r.ca for r in p.topology.residues if r.ca >= 0]
            self.end = coordinates(align_coordinates(self.end, self.start, ca if len(ca) >= 3 else None))

    def apply(self, alpha):
        self.target._pair(self.start, self.end, alpha)


class Deform(Morph):
    """Smoothly deform to function(xyz). This is illustrative, not a dynamics simulation."""

    def __init__(self, protein, function, **kwargs):
        super().__init__(protein, None, align=False, **kwargs)
        self.function = function

    def bind(self):
        self.destination = self.function(self.target.positions.copy())
        super().bind()


class PlayTrajectory(Animation):
    channels = frozenset({"geometry"})

    def __init__(self, protein, trajectory=None, *, start=0, end=None, rate_func=linear, state_easing=linear):
        super().__init__(protein, rate_func=rate_func)
        self.trajectory = trajectory if trajectory is not None else protein.trajectory
        self.start = start
        self.end = len(self.trajectory) - 1 if end is None else end
        if not 0 <= self.start <= len(self.trajectory) - 1 or not 0 <= self.end <= len(self.trajectory) - 1:
            raise ValueError("Trajectory frame range is out of bounds")
        if self.trajectory.n_atoms != len(protein.topology.atoms):
            raise ValueError("Trajectory atom count does not match protein")
        if self.trajectory.topology is not None and self.trajectory.topology.keys != protein.topology.keys:
            raise ValueError("Trajectory atom identities/order do not match protein")
        self.cache = FrameCache(self.trajectory)
        if not callable(state_easing):
            raise TypeError("state_easing must be a function such as linear or smooth")
        self.state_easing = state_easing

    def apply(self, alpha):
        f = self.start + (self.end - self.start) * alpha
        lo = int(np.floor(f))
        hi = min(lo + 1, len(self.trajectory) - 1)
        a, b = self.cache.get(lo), self.cache.get(hi)
        blend = float(self.state_easing(f - lo))
        if not np.isfinite(blend) or not 0 <= blend <= 1:
            raise ValueError("state_easing must return a finite value in [0, 1]")
        self.target._pair(a, b, blend, (id(self.trajectory), lo), (id(self.trajectory), hi))


class Focus(Animation):
    """Ease a camera to a protein/region; optionally follow its center afterward."""

    channels = frozenset({"camera"})
    late = True  # Evaluate after coordinate/transform writers in the same play() call.

    def __init__(self, camera, region, *, margin=1.25, aspect=16 / 9, follow=True, **kwargs):
        super().__init__(camera, **kwargs)
        self.region, self.margin, self.aspect, self.follow = region, margin, aspect, follow

    def bind(self):
        from .camera import Camera

        super().bind()
        if not isinstance(self.target, Camera):
            raise TypeError("Focus needs a Camera as its first argument")
        self.target.update_tracking()
        self.start = self.target.snapshot()
        fitted = Camera()
        fitted.fov = self.target.fov
        fitted.frame(self.region, margin=self.margin, aspect=self.aspect)
        self.end = fitted.snapshot()

    def apply(self, alpha):
        camera = self.target
        endpoint = self.end["target"]
        if self.follow:
            m = self.region.model_matrix
            points = self.region.positions @ m[:3, :3].T + m[:3, 3]
            endpoint = (points.min(0) + points.max(0)) / 2
        camera.target = (1 - alpha) * self.start["target"] + alpha * endpoint
        camera.distance = self.start["distance"] * (self.end["distance"] / self.start["distance"]) ** alpha
        camera.radius = (1 - alpha) * self.start["radius"] + alpha * self.end["radius"]
        camera._tracking = self.region if self.follow and alpha >= 1 else None


class FadeIn(Animation):
    channels = frozenset({"opacity"})

    def bind(self):
        super().bind()
        self.end = self.target.opacity

    def apply(self, alpha):
        self.target.opacity = self.end * alpha


class FadeOut(Animation):
    channels = frozenset({"opacity"})

    def bind(self):
        super().bind()
        self.start = self.target.opacity

    def apply(self, alpha):
        self.target.opacity = self.start * (1 - alpha)


class Representation(Animation):
    channels = frozenset({"representation"})

    def __init__(self, protein, representation, **kwargs):
        super().__init__(protein, **kwargs)
        names = {"cartoon": 0, "ribbon": 1, "ball_and_stick": 2, "surface": 3}
        if representation not in names:
            raise ValueError(f"Choose a representation from {list(names)}")
        self.end = np.eye(4)[names[representation]][:3]
        self.surface_end = float(representation == "surface")

    def bind(self):
        super().bind()
        self.start = self.target.representation.copy()
        self.surface_start = self.target.surface_opacity
        if self.surface_end and self.target._surface_options is None:
            from .surface import surface_options

            self.target._surface_options = surface_options(reference=self.target.positions)

        self.surface_options = self.target._surface_options

    def apply(self, alpha):
        self.target.representation = (1 - alpha) * self.start + alpha * self.end
        self.target.surface_opacity = (1 - alpha) * self.surface_start + alpha * self.surface_end
        if self.surface_end:
            self.target._surface_options = self.surface_options


class Animate(Animation):
    """Fluent .animate.shift(...).rotate(...).scale(...) or camera.animate.orbit(...)."""

    def __init__(self, target):
        super().__init__(target)
        self.operations = []
        self.channels = frozenset()

    def _op(self, name, *args):
        camera = hasattr(self.target, "theta")
        valid = {"orbit", "zoom"} if camera else {"shift", "rotate", "scale", "set_opacity"}
        if name not in valid:
            raise ValueError(f"{name} is not supported for this object")
        channel = "camera" if camera else ("opacity" if name == "set_opacity" else "transform")
        self.channels = self.channels | {channel}
        self.operations.append((name, args))
        return self

    def shift(self, vector):
        return self._op("shift", np.asarray(vector, dtype=float))

    def rotate(self, angle, axis=(0, 1, 0)):
        rotation(angle, axis)
        return self._op("rotate", angle, axis)

    def scale(self, factor):
        if factor <= 0:
            raise ValueError("Scale must be positive")
        return self._op("scale", factor)

    def set_opacity(self, opacity):
        if not 0 <= opacity <= 1:
            raise ValueError("Opacity must be in [0, 1]")
        return self._op("set_opacity", opacity)

    def orbit(self, theta=0, phi=0):
        return self._op("orbit", theta, phi)

    def zoom(self, factor):
        if factor <= 0:
            raise ValueError("Zoom must be positive")
        return self._op("zoom", factor)

    def set_color(self, color, **kwargs):
        if self.operations:
            raise ValueError("Play color and transforms as separate concurrent animations")
        from .styling import Colorize

        return Colorize(self.target, color, **kwargs)

    def focus(self, region, *, margin=1.25, aspect=16 / 9, follow=True):
        if self.operations:
            raise ValueError("Play focus and camera orbit/zoom in separate clips")
        return Focus(self.target, region, margin=margin, aspect=aspect, follow=follow)

    def bind(self):
        super().bind()
        self.start = self.target.snapshot()
        if hasattr(self.target, "positions"):
            self.center = self.target.positions.mean(0)

    def apply(self, alpha):
        p, s = self.target, self.start
        if "camera" in self.channels:
            p.restore(s)
        if "transform" in self.channels:
            p.position, p.orientation, p.size = s["position"].copy(), s["orientation"].copy(), s["size"]
        if "opacity" in self.channels:
            p.opacity = s["opacity"]
        for name, args in self.operations:
            if name == "rotate":
                old = p.orientation.copy()
                p.orientation = rotation(args[0] * alpha, args[1]) @ old
                p.position += p.size * ((old - p.orientation) @ self.center)
            elif name == "scale":
                factor = args[0] ** alpha
                p.position += p.size * (1 - factor) * (p.orientation @ self.center)
                p.size *= factor
            elif name == "set_opacity":
                p.opacity = (1 - alpha) * s["opacity"] + alpha * args[0]
            elif name == "shift":
                p.shift(args[0] * alpha)
            elif name == "orbit":
                p.orbit(args[0] * alpha, args[1] * alpha)
            elif name == "zoom":
                p.zoom(args[0] ** alpha)
