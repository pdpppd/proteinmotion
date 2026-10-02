"""Deterministic animations: capture at compilation, evaluate at any time."""

import numpy as np

from .math3d import align_coordinates, rotation
from .rates import evaluate, linear, resolve, smooth
from .structure import coordinates
from .trajectory import FrameCache


class Animation:
    channels = frozenset()

    def __init__(self, target, *, rate_func=smooth, easing=None):
        self.target, self.rate_func = target, resolve(rate_func if easing is None else easing)
        self._bound = False

    def during(self, seconds, *, delay=0):
        """Give this animation a duration and start delay, both in seconds."""
        from .timeline import AnimationGroup

        return AnimationGroup(self, durations=[seconds], offsets=[delay])

    def _set_easing(self, easing):
        self.rate_func = resolve(easing)

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
            anchors = [r.morph_atom for r in p.topology.residues if r.morph_atom >= 0]
            if any(r.is_nucleic for r in p.topology.residues) and len(anchors) < 3:
                raise ValueError(
                    "Aligned nucleotide Morph needs at least three C1' anchors; use align=False to interpolate supplied coordinates"
                )
            self.end = coordinates(
                align_coordinates(self.end, self.start, anchors if len(anchors) >= 3 else None)
            )

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

    def __init__(
        self,
        protein,
        trajectory=None,
        *,
        start=0,
        end=None,
        rate_func=linear,
        state_easing=linear,
        easing=None,
    ):
        super().__init__(protein, rate_func=rate_func, easing=easing)
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
        self.state_easing = resolve(state_easing)

    def apply(self, alpha):
        f = self.start + (self.end - self.start) * alpha
        lo = int(np.floor(f))
        hi = min(lo + 1, len(self.trajectory) - 1)
        a, b = self.cache.get(lo), self.cache.get(hi)
        blend = float(self.state_easing(f - lo))
        if not np.isfinite(blend) or not 0 <= blend <= 1:
            raise ValueError("state_easing must return a finite value in [0, 1]")
        self.target.trajectory_frame = float(lo + blend * (hi - lo))
        self.target._trajectory_source = self.trajectory
        self.target._pair(a, b, blend, (id(self.trajectory), lo), (id(self.trajectory), hi))


class Focus(Animation):
    """Ease a camera to a protein/region; optionally follow its center afterward."""

    channels = frozenset({"camera"})
    late = True  # Evaluate after coordinate/transform writers in the same play() call.

    def __init__(
        self, camera, region, *, margin=1.25, aspect=None, follow=True, screen_position=(0.5, 0.5), **kwargs
    ):
        super().__init__(camera, **kwargs)
        self.region, self.margin, self.aspect, self.follow = (
            region,
            margin,
            camera.aspect if aspect is None else aspect,
            follow,
        )
        self.screen_position = screen_position

    def bind(self):
        from .camera import Camera

        super().bind()
        if not isinstance(self.target, Camera):
            raise TypeError("Focus needs a Camera as its first argument")
        self.target.update_tracking()
        self.start = self.target.snapshot()
        fitted = Camera()
        fitted.fov = self.target.fov
        fitted.theta, fitted.phi = self.target.theta, self.target.phi
        fitted.frame(
            self.region, margin=self.margin, aspect=self.aspect, screen_position=self.screen_position
        )
        self.end = fitted.snapshot()

    def apply(self, alpha):
        camera = self.target
        endpoint = self.end["target"]
        if self.follow:
            m = self.region.model_matrix
            points = self.region.positions @ m[:3, :3].T + m[:3, 3]
            endpoint = (points.min(0) + points.max(0)) / 2 + camera.composition_offset(
                self.screen_position, distance=self.end["distance"], aspect=self.aspect
            )
        camera.target = (1 - alpha) * self.start["target"] + alpha * endpoint
        camera.distance = self.start["distance"] * (self.end["distance"] / self.start["distance"]) ** alpha
        camera.radius = (1 - alpha) * self.start["radius"] + alpha * self.end["radius"]
        camera._tracking = self.region if self.follow and alpha >= 1 else None
        camera._screen_position, camera.aspect = tuple(self.screen_position), self.aspect


class FocusPull(Animation):
    """Ease EEVEE lens focus to a protein, region or world point without reframing."""

    channels = frozenset({"lens"})
    late = True

    def __init__(
        self, camera, target, *, chain=None, residues=None, atoms=None, fstop=None, follow=True, **kwargs
    ):
        from .camera import Camera

        if not isinstance(camera, Camera):
            raise TypeError("FocusPull needs a Camera as its first argument")
        super().__init__(camera, **kwargs)
        probe = Camera().set_focus(
            target,
            chain=chain,
            residues=residues,
            atoms=atoms,
            fstop=fstop if fstop is not None else camera.fstop,
            follow=True,
        )
        if not probe.dof:
            raise ValueError("FocusPull needs a focus target")
        self.destination = probe._focus_target or probe.focus_point
        self.follow, self.fstop = follow, fstop

    def bind(self):
        super().bind()
        camera = self.target
        self.start = camera.focus_point if camera.dof else camera.target.copy()
        self.start_fstop = camera.fstop
        self.end_fstop = camera.fstop if self.fstop is None else float(self.fstop)
        self.end = camera._target_point(self.destination)

    def apply(self, alpha):
        camera = self.target
        end = camera._target_point(self.destination) if self.follow else self.end
        camera.dof = True
        camera.fstop = self.start_fstop * (self.end_fstop / self.start_fstop) ** alpha
        camera._focus_point = (1 - alpha) * self.start + alpha * end
        camera._focus_target = (
            self.destination
            if self.follow and alpha >= 1 and hasattr(self.destination, "positions")
            else None
        )


class Reveal(Animation):
    """Open a view-dependent cutaway onto a region, like an iris in front of it.

    Geometry between the camera and the region fades inside a soft window that stays
    aimed at the region as the camera or molecule moves. The region and everything
    behind it stay drawn. Atoms are hidden for visibility only; nothing moves.
    ``window`` scales the window radius relative to the region's bounding sphere,
    ``softness`` is the fraction of the window used for its fading edge, and ``band``
    is the depth in Å over which geometry just in front of the region fades back in.
    """

    channels = frozenset({"cutaway"})
    late = True
    opening = 1.0

    def __init__(
        self,
        camera,
        region=None,
        *,
        window=None,
        softness=None,
        band=None,
        shape=None,
        wall=None,
        rings=None,
        padding=None,
        surface_keep=None,
        rim=None,
        **kwargs,
    ):
        from .camera import Camera

        if not isinstance(camera, Camera):
            raise TypeError("Reveal needs a Camera as its first argument")
        super().__init__(camera, **kwargs)
        for name, value in (("window", window), ("softness", softness), ("band", band)):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if softness is not None and softness > 1:
            raise ValueError("softness must be at most 1")
        if shape not in (None, "cone", "tunnel"):
            raise ValueError("shape must be 'cone' or 'tunnel'")
        self.region, self.window, self.softness, self.band = region, window, softness, band
        self.tunnel = None if shape is None else shape == "tunnel"
        self.wall, self.rings = wall, rings
        for name, value in (
            ("padding", padding),
            ("surface_keep", surface_keep),
            ("wall", wall),
            ("rings", rings),
        ):
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if surface_keep is not None and surface_keep > 1:
            raise ValueError("surface_keep must be in [0, 1]")
        from .math3d import color

        self.padding, self.surface_keep = padding, surface_keep
        self.rim = None if rim is None else np.asarray(color(rim))

    def bind(self):
        super().bind()
        camera = self.target
        if self.region is None:
            if camera._cutaway_target is None:
                raise ValueError("No open cutaway to close; pass a region")
            self.region = camera._cutaway_target
        same = camera._cutaway_target is self.region and camera.cutaway_opening > 0
        self.start = camera.cutaway_opening if same else 0.0
        # An open window on the same region reshapes smoothly; a new one takes its shape at once.
        self.shape = {}
        self.shape_mode = camera.cutaway_tunnel if self.tunnel is None else self.tunnel
        # A tunnel is drilled along the current view and then stays fixed to the molecule,
        # so camera motion shows its walls in perspective.
        self.axis = camera._cutaway_axis if same else None
        if self.shape_mode and self.axis is None:
            protein = getattr(self.region, "protein", self.region)
            m = protein.model_matrix[:3, :3]
            points = self.region.world_positions
            direction = camera.eye - (points.min(0) + points.max(0)) / 2
            local = np.linalg.solve(m, direction)
            self.axis = local / np.linalg.norm(local)
        values = (
            ("window", self.window),
            ("softness", self.softness),
            ("band", self.band),
            ("wall", self.wall),
            ("rings", self.rings),
            ("padding", self.padding),
            ("surface_keep", self.surface_keep),
            ("rim", self.rim),
        )
        for name, value in values:
            current = getattr(camera, f"cutaway_{name}")
            end = current if value is None else value
            self.shape[name] = (current if same else end, end)

    def apply(self, alpha):
        camera = self.target
        camera._cutaway_target = self.region
        camera.cutaway_tunnel = self.shape_mode
        camera._cutaway_axis = self.axis if self.shape_mode else None
        for name, (start, end) in self.shape.items():
            setattr(camera, f"cutaway_{name}", (1 - alpha) * start + alpha * end)
        camera.cutaway_opening = (1 - alpha) * self.start + alpha * self.opening


class Conceal(Reveal):
    """Close the open cutaway."""

    opening = 0.0


class FadeIn(Animation):
    channels = frozenset({"opacity"})

    def bind(self):
        super().bind()
        self.end = self.target.opacity or getattr(self.target, "_visible_opacity", 1.0)

    def apply(self, alpha):
        self.target.opacity = self.end * alpha


class FadeOut(Animation):
    channels = frozenset({"opacity"})

    def bind(self):
        super().bind()
        self.start = self.target.opacity
        if self.start > 0:
            self.target._visible_opacity = self.start

    def apply(self, alpha):
        if self.start > 0:
            self.target._visible_opacity = self.start
        self.target.opacity = self.start * (1 - alpha)


class Representation(Animation):
    channels = frozenset({"representation"})

    def __init__(self, protein, representation, *, bases=None, rate_func=smooth, easing=None, **options):
        from .styles import MolecularStyle

        if isinstance(representation, MolecularStyle):
            options = {**representation.options, **options}
            representation = representation.representation
        super().__init__(protein, rate_func=rate_func, easing=easing)
        self.options, self.name = options, representation
        names = {"cartoon": 0, "ribbon": 1, "ball_and_stick": 2, "surface": 3}
        if representation not in names:
            raise ValueError(f"Choose a representation from {list(names)}")
        self.end = np.eye(4)[names[representation]][:3]
        self.surface_end = float(representation == "surface")
        from .nucleic import base_weights

        self.base_end = None if bases is None else base_weights(bases)
        if bases is not None:
            self.channels |= {"base_style"}

    def bind(self):
        super().bind()
        self.start = self.target.representation.copy()
        self.surface_start = self.target.surface_opacity
        self.base_start = self.target.base_style.copy()
        self.settings = {}
        if self.options:
            before = self.target.snapshot()
            try:
                getattr(self.target, self.name)(**self.options)
                after = self.target.snapshot()
                keys = (
                    "color_scheme",
                    "atom_scale",
                    "bond_radius",
                    "ribbon_width",
                    "surface_options",
                    "backbone_radius",
                    "base_thickness",
                    "base_radius",
                )
                self.settings = {k: after[k] for k in keys}
                if self.base_end is None:
                    self.base_end = after["base_style"]
            finally:
                self.target.restore(before)
            if self.surface_end:
                self.target._surface_options = self.settings["surface_options"]
        if self.surface_end and self.target._surface_options is None:
            from .surface import surface_options

            self.target._surface_options = surface_options(reference=self.target.positions)

        self.surface_options = self.target._surface_options
        # Blend residue colors between two named schemes. Element colors are already
        # blended by the ball-and-stick weight, so changes to or from them switch directly.
        start, end = self.target.color_scheme, self.settings.get("color_scheme")
        self.schemes = None
        if end is not None and str(start) != str(end) and "element" not in (start, end):
            self.schemes = (getattr(start, "end", start), end)

    def apply(self, alpha):
        for key, value in self.settings.items():
            setattr(self.target, "_surface_options" if key == "surface_options" else key, value)
        if self.schemes is not None:
            from .geometry import SchemeBlend

            start, end = self.schemes
            fraction = round(float(alpha), 3)
            self.target.color_scheme = (
                start if fraction <= 0 else end if fraction >= 1 else SchemeBlend(start, end, fraction)
            )
        self.target.representation = (1 - alpha) * self.start + alpha * self.end
        self.target.surface_opacity = (1 - alpha) * self.surface_start + alpha * self.surface_end
        if self.base_end is not None:
            self.target.base_style = (1 - alpha) * self.base_start + alpha * self.base_end
        if self.surface_end:
            self.target._surface_options = self.surface_options


class SecondaryStructure(Animation):
    """Blend the cartoon to new helix, strand and coil states.

    ``assignments="dssp"`` assigns them from the coordinates when the clip starts,
    for example after a Morph or PlayTrajectory; an H/E/C string sets them directly.
    """

    channels = frozenset({"secondary"})

    def __init__(self, protein, assignments="dssp", **kwargs):
        from .secondary import weights

        super().__init__(protein, **kwargs)
        self.dssp = isinstance(assignments, str) and assignments.lower() == "dssp"
        if not self.dssp:
            if len(assignments) != len(protein.topology.residues):
                raise ValueError("Supply one H/E/C code per residue, or 'dssp'")
            weights(assignments)  # validate the codes now
        self.assignments = assignments

    def bind(self):
        from .secondary import assign, weights

        super().bind()
        p = self.target
        self.start = p._secondary
        self.end = weights(assign(p.topology, p.positions) if self.dssp else self.assignments)

    def apply(self, alpha):
        blended = ((1 - alpha) * self.start + alpha * self.end).astype(np.float32)
        blended.flags.writeable = False
        self.target._secondary = blended


class Animate(Animation):
    """Fluent .animate.shift(...).rotate(...).scale(...) or camera.animate.orbit(...)."""

    def __init__(self, target):
        self.subject = target
        super().__init__(getattr(target, "protein", target), rate_func=linear)
        self.operations = []
        self.extras = []
        self.transform_easing = smooth
        self.channels = frozenset()

    def _set_easing(self, easing):
        self.transform_easing = resolve(easing)
        for extra in self.extras:
            extra._set_easing(easing)

    @property
    def requires_linear_timeline(self):
        return any(getattr(extra, "requires_linear_timeline", False) for extra in self.extras)

    def _extra(self, animation):
        conflict = self.channels & animation.channels
        if isinstance(animation, Focus) and not any("camera" in a.channels for a in self.extras):
            conflict -= {"camera"}
        if conflict:
            raise ValueError("Chained animations write the same property; use separate clips")
        self.channels |= animation.channels
        self.extras.append(animation)
        return self

    def _op(self, name, *args):
        camera = hasattr(self.target, "theta")
        valid = {"orbit", "zoom", "depth_cue"} if camera else {"shift", "rotate", "scale", "set_opacity"}
        if name not in valid:
            raise ValueError(f"{name} is not supported for this object")
        if self.subject is not self.target and name != "set_opacity":
            raise ValueError("Transform the parent protein, not a Region")
        channel = "camera" if camera else ("opacity" if name == "set_opacity" else "transform")
        if any(
            channel in a.channels and not (channel == "camera" and isinstance(a, Focus)) for a in self.extras
        ):
            raise ValueError(f"Chained animations both write {channel}; use separate clips")
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

    def set_opacity(self, opacity, **kwargs):
        if self.subject is not self.target or kwargs:
            from .styling import SetOpacity

            return self._extra(SetOpacity(self.subject, opacity, **kwargs))
        if not 0 <= opacity <= 1:
            raise ValueError("Opacity must be in [0, 1]")
        return self._op("set_opacity", opacity)

    def orbit(self, theta=0, phi=0):
        return self._op("orbit", theta, phi)

    def zoom(self, factor):
        if factor <= 0:
            raise ValueError("Zoom must be positive")
        return self._op("zoom", factor)

    def depth_cue(self, strength):
        """Ease the camera's distance fog to ``strength`` (0 disables it)."""
        if not np.isfinite(strength) or strength < 0:
            raise ValueError("depth_cue must be finite and nonnegative")
        return self._op("depth_cue", float(strength))

    def set_color(self, color, **kwargs):
        from .styling import Colorize

        return self._extra(Colorize(self.subject, color, **kwargs))

    def show_atoms(self, **kwargs):
        from .styling import ShowAtoms

        return self._extra(ShowAtoms(self.subject, **kwargs))

    def hide_atoms(self, **kwargs):
        from .styling import HideAtoms

        return self._extra(HideAtoms(self.subject, **kwargs))

    def representation(self, style, **options):
        return self._extra(Representation(self.target, style, **options))

    def focus(self, region, *, margin=1.25, aspect=None, follow=True, screen_position=(0.5, 0.5)):
        return self._extra(
            Focus(
                self.target,
                region,
                margin=margin,
                aspect=aspect,
                follow=follow,
                screen_position=screen_position,
            )
        )

    def frame(self, region, **kwargs):
        """Animate camera framing; use lens_focus for depth of field."""
        return self.focus(region, **kwargs)

    def set_focus(self, target, **kwargs):
        """Create a lens focus pull; play it alongside orbit or zoom animations."""
        return self._extra(FocusPull(self.target, target, **kwargs))

    def lens_focus(self, target, **kwargs):
        return self.set_focus(target, **kwargs)

    def bind(self):
        super().bind()
        self.start = self.target.snapshot()
        if hasattr(self.target, "positions"):
            self.center = self.target.positions.mean(0)
        for extra in self.extras:
            extra.run_time = getattr(self, "run_time", 1.0)
            extra.start_time = getattr(self, "start_time", 0.0)
            extra.bind()

    def apply(self, alpha):
        clock = alpha
        alpha = evaluate(self.transform_easing, alpha)
        p, s = self.target, self.start
        if "camera" in self.channels:
            # A camera movement owns framing, while FocusPull owns the lens.
            for key in ("target", "distance", "theta", "phi", "fov", "depth_cue", "radius", "_tracking"):
                value = s[key]
                setattr(p, key, value.copy() if isinstance(value, np.ndarray) else value)
        if "transform" in self.channels:
            p.position, p.orientation, p.size = s["position"].copy(), s["orientation"].copy(), s["size"]
        if "opacity" in self.channels:
            p.opacity = s["opacity"]
        for extra in self.extras:
            if isinstance(extra, Focus):
                extra.apply(evaluate(extra.rate_func, clock))
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
            elif name == "depth_cue":
                p.depth_cue = (1 - alpha) * s["depth_cue"] + alpha * args[0]
        for extra in self.extras:
            if not isinstance(extra, Focus):
                extra.apply(evaluate(extra.rate_func, clock))
