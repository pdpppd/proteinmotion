"""Thread molecular geometry (or optional wires) into view along seeded flight paths."""

import numpy as np

from .animation import Animation
from .geometry import residue_colors
from .math3d import color as parse_color
from .protein import Protein
from .rates import ease_in_out_sine, linear, smooth
from .structure import Atom, Residue, Topology
from .styling import current_tints


def _catmull(points, samples_per_span):
    """Uniform Catmull–Rom through ``points`` (the cartoon's centerline construction)."""
    pts = np.asarray(points, float)
    if len(pts) < 2:
        return pts.copy()
    padded = np.vstack((pts[0], pts, pts[-1]))
    t = np.linspace(0, 1, samples_per_span, endpoint=False)[:, None]
    out = []
    for i in range(len(pts) - 1):
        a, b, c, d = padded[i : i + 4]
        out.append(
            0.5
            * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t**2 + (-a + 3 * b - 3 * c + d) * t**3)
        )
    out.append(pts[-1:])
    return np.vstack(out)


def _centripetal(points, samples_per_span=24):
    """Centripetal Catmull–Rom: smooth, and free of cusps and self-loops between points."""
    pts = np.asarray(points, float)
    padded = np.vstack((2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]))
    out = []
    for i in range(len(pts) - 1):
        p0, p1, p2, p3 = padded[i : i + 4]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-6) ** 0.5
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-6) ** 0.5
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-6) ** 0.5
        t = np.linspace(t1, t2, samples_per_span, endpoint=False)[:, None]
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    out.append(pts[-1:])
    return np.vstack(out)


def _arc_length(path):
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))))


def _perpendiculars(direction):
    ref = np.array([0.0, 1.0, 0.0]) if abs(direction[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(direction, ref)
    u /= np.linalg.norm(u)
    return u, np.cross(direction, u)


class _Strand:
    """One chain's route: a flight curve that joins the backbone at the C terminus."""

    def __init__(self, backbone, flight, colors):
        self.path = np.vstack((flight, backbone[1:]))
        self.s = _arc_length(self.path)
        self.flight_length = float(_arc_length(flight)[-1])
        self.backbone_s = _arc_length(backbone)
        self.length = float(_arc_length(backbone)[-1])  # The wire is as long as the backbone.
        self.total = float(self.s[-1])
        # Colors belong to the route: the flight takes the C-terminal color.
        self.colors = np.vstack((np.repeat(colors[:1], len(flight), axis=0), colors[1:]))

    def sample(self, head, count):
        """``count`` points along the wire, head first, with the tail up to one wire length behind."""
        tail = max(0.0, head - self.length)
        where = np.linspace(head, tail, count)
        xyz = np.column_stack([np.interp(where, self.s, self.path[:, k]) for k in range(3)])
        rgb = np.column_stack([np.interp(where, self.s, self.colors[:, k]) for k in range(3)])
        return xyz, rgb


class _Wire(Protein):
    """Tube geometry for the threading strands, drawn in the parent protein's frame."""

    def __init__(self, parent, counts, radius):
        atoms, residues, bonds = [], [], []
        for count in counts:
            for j in range(count):
                i = len(atoms)
                atoms.append(Atom("wire", i + 1, "", "WIR", "CA", "C", i))
                residues.append(Residue("wire", i + 1, "", "WIR", i, -1))
                if j:
                    bonds.append((i - 1, i))
        topology = Topology(tuple(atoms), tuple(residues), np.array(bonds, np.uint32).reshape(-1, 2), ())
        super().__init__(topology, np.zeros((len(atoms), 3)))
        self.parent, self.counts, self.wire_radius = parent, list(counts), float(radius)
        self.glow_size, self.glow_heads = 0.0, None  # Local head points and colors for the glow pass.
        # Joint spheres match the tube radius, so bends stay round.
        self.ball_and_stick(atom_scale=1.0, bond_radius=radius)
        self.opacity = 0.0

    def _sync(self):
        parent = self.parent
        self.position = parent.position.copy()
        self.orientation = parent.orientation.copy()
        self.size = parent.size

    def _glow(self):
        """Halo instances (world center, color, radius, intensity) at each wire's head."""
        if self.glow_heads is None or self.glow_size <= 0 or self.opacity <= 0:
            return None
        local, colors, strength = self.glow_heads
        m = self.model_matrix
        rows = np.zeros((len(local), 8), np.float32)
        rows[:, :3] = local @ m[:3, :3].T + m[:3, 3]
        rows[:, 3:6] = colors
        rows[:, 6] = self.glow_size * self.wire_radius * self.size
        rows[:, 7] = strength * self.opacity
        return rows

    def copy(self):
        raise ValueError("Create another Thread animation for an independent wire")


class Thread(Animation):
    """Thread the actual cartoon, ribbon, or ball-and-stick geometry into view.

    Every traced chain gets its own route, entering from a different side of the screen.
    Geometry follows a route through the C terminus toward the N terminus. Residues
    and bound ligands move as rigid groups. The default easing is quintic ``smooth``
    over the full clip; pass a bounded rate function to customize travel.
    Surface representations are unsupported. Motion is illustrative, not dynamics.

    ``mode="wire"`` retains the original wire-to-protein crossfade and defaults to
    sine easing; ``settle`` controls the final crossfade in wire mode only.
    Pass the scene camera so geometry starts
    just off-screen; ``swirl`` scales the corkscrew of the flight and ``seed`` varies the
    paths. ``easing`` maps clip time to head travel (any rate function, such as ``smooth``).
    ``stagger`` delays each chain's start by that fraction of the clip, in chain
    order, so the chains arrive one after another. Set ``aspect`` to the scene's
    width/height for a non-16:9 frame.

    Each head glows: ``glow`` sets the halo radius as a multiple of ``radius`` (0
    disables it), ``glow_color`` its color (default: the chain color blended with warm
    white), and ``glow_brightness`` its intensity.
    """

    channels = frozenset({"opacity"})
    reverse_time = False

    def __init__(
        self,
        protein,
        camera=None,
        *,
        mode="geometry",
        aspect=None,
        radius=0.45,
        settle=None,
        swirl=1.0,
        head=None,
        glow=12.0,
        glow_color=None,
        glow_brightness=1.0,
        easing=None,
        stagger=0.0,
        resolution=None,
        seed=0,
        rate_func=None,
        delay_seconds=None,
        stagger_fraction=None,
        wire_options=None,
    ):
        self._implicit_aspect = aspect is None
        aspect = getattr(camera, "aspect", 16 / 9) if aspect is None else aspect
        if wire_options is not None:
            if mode != "wire":
                raise ValueError("wire_options requires mode='wire'")
            if set(wire_options) - {"settle", "head", "resolution"}:
                raise ValueError("wire_options supports settle, head, and resolution")
            settle = wire_options.get("settle", settle)
            head = wire_options.get("head", head)
            resolution = wire_options.get("resolution", resolution)
        if mode == "geometry" and any(v is not None for v in (settle, head, resolution)):
            import warnings

            warnings.warn(
                "settle, head and resolution affect wire mode only; omit them for geometry threading",
                UserWarning,
                stacklevel=2,
            )
        settle, head, resolution = (
            0.15 if settle is None else settle,
            1.8 if head is None else head,
            0.8 if resolution is None else resolution,
        )
        if delay_seconds is not None and (not np.isfinite(delay_seconds) or delay_seconds < 0):
            raise ValueError("delay_seconds must be finite and nonnegative")
        if stagger_fraction is not None:
            if not np.isfinite(stagger_fraction) or not 0 <= stagger_fraction < 1:
                raise ValueError("stagger_fraction must be in [0, 1)")
            if stagger or delay_seconds is not None:
                raise ValueError("Choose stagger_fraction, delay_seconds, or legacy stagger")
            stagger = stagger_fraction / max(1, len(protein.topology.chains) - 1)
        if delay_seconds is not None and stagger:
            raise ValueError("Choose delay_seconds or legacy stagger")
        self.delay_seconds = delay_seconds
        if not isinstance(protein, Protein):
            raise TypeError("Thread needs a Protein")
        if not protein.topology.chains:
            raise ValueError("Thread needs at least one traced chain")
        if mode not in ("geometry", "wire"):
            raise ValueError("Thread mode must be 'geometry' or 'wire'")
        if protein.surface_opacity > 0:
            raise ValueError(
                "Thread does not support surface representation; use cartoon, ribbon or ball_and_stick"
            )
        if not np.isfinite(aspect) or aspect <= 0:
            raise ValueError("aspect must be finite and positive")
        easing = (smooth if mode == "geometry" else ease_in_out_sine) if easing is None else easing
        for name, value in (("glow", glow), ("glow_brightness", glow_brightness)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(stagger) or stagger < 0:
            raise ValueError("stagger must be finite and nonnegative")
        from .rates import resolve

        easing = resolve(easing)
        for name, value in (("radius", radius), ("head", head), ("resolution", resolution)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= settle < 1:
            raise ValueError("settle must be in [0, 1)")
        if not np.isfinite(swirl) or swirl < 0:
            raise ValueError("swirl must be finite and nonnegative")
        super().__init__(protein, rate_func=rate_func or linear)
        self.mode, self.aspect = mode, float(aspect)
        if mode == "geometry":
            self.channels = self.channels | {"geometry", "representation"}
        self.camera, self.settle, self.swirl, self.seed = camera, float(settle), float(swirl), int(seed)
        self.head_scale, self.resolution = float(head), float(resolution)
        self.easing, self.glow_brightness, self.stagger = easing, float(glow_brightness), float(stagger)
        if (len(protein.topology.chains) - 1) * self.stagger >= 1:
            raise ValueError("stagger is too large: each chain needs part of the threading time")
        self.glow_color = None if glow_color is None else np.asarray(parse_color(glow_color), float)
        # The number of wire points is fixed now; routes follow the coordinates at bind time.
        routes = [self._backbone(protein, chain)[0] for chain in protein.topology.chains]
        counts = [max(8, int(np.ceil(_arc_length(r)[-1] / self.resolution)) + 1) for r in routes]
        if mode == "geometry":
            counts = [1] * len(routes)  # Only head halos; no dense wire mesh.
        self.wire = _Wire(protein, counts, radius)
        self.wire.glow_size = float(glow)
        if mode == "geometry":
            # Reuse the head-halo helper, with no wire geometry drawn.
            self.wire.representation = np.zeros(3)
            self.wire.hide_atoms()

    @property
    def targets(self):
        return (self.target, self.wire)

    @staticmethod
    def _backbone(protein, chain):
        trace = [protein.topology.residues[i].trace_atom for i in chain]
        # The wire enters at the C terminus, so the route runs from the last residue back.
        return _catmull(protein.positions[trace][::-1], 8), np.asarray(chain)[::-1]

    def _prepare(self, scene):
        if self.camera is None:
            self.camera = scene.camera
        if self._implicit_aspect:
            self.aspect = scene.width / scene.height

    def _set_easing(self, easing):
        from .rates import resolve

        self.easing = resolve(easing)

    def bind(self):
        p = self.target
        if p.surface_opacity > 0:
            raise ValueError(
                "Thread does not support surface representation; use cartoon, ribbon or ball_and_stick"
            )
        super().bind()
        if self.delay_seconds is not None:
            self.stagger = self.delay_seconds / getattr(self, "run_time", 1.0)
            if (len(self.target.topology.chains) - 1) * self.stagger >= 1:
                raise ValueError(
                    "delay_seconds leaves no time for chain motion; reduce it or increase run_time"
                )

        self.original = p.snapshot()
        self.end_opacity = p.opacity
        m = p.model_matrix
        world = p.positions @ m[:3, :3].T + m[:3, 3]
        center = world.mean(0)
        radius = float(np.linalg.norm(world - center, axis=1).max())
        if self.camera is not None:
            forward = self.camera.target - self.camera.eye
            forward /= np.linalg.norm(forward)
            # Screen axes: right and up as seen through this camera.
            right = np.cross(forward, [0.0, 1.0, 0.0])
            if np.linalg.norm(right) < 1e-6:
                right = _perpendiculars(forward)[0]
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            depth = float(np.dot(center - self.camera.eye, forward))
            half_height = depth * np.tan(self.camera.fov / 2)
            half_width = half_height * self.aspect
        else:
            forward, right, up = np.array([0, 0, -1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
            half_width = half_height = 2.0 * radius
        # Wires take the colors the cartoon will show, including residue tints.
        tints = current_tints(p)[[r.trace_atom for r in p.topology.residues]]
        palette = residue_colors(p) * (1 - tints[:, 3:4]) + tints[:, :3]
        inverse = np.linalg.inv(m[:3, :3])
        self.strands = []
        routes = [self._backbone(p, chain) for chain in p.topology.chains]
        count = len(routes)
        for k, (route, residues) in enumerate(routes):
            rng = np.random.default_rng(self.seed * 1009 + k)
            # Evenly spaced sides of the screen, starting from the left.
            side = np.pi + 2 * np.pi * k / count + rng.uniform(-0.25, 0.25)
            lateral = right * np.cos(side) + up * np.sin(side)
            # Just past the frame edge in this direction, so the wire enters view promptly.
            edge = min(half_width / max(abs(np.cos(side)), 1e-6), half_height / max(abs(np.sin(side)), 1e-6))
            start = center + lateral * (edge * 1.08 + 4.0) + forward * rng.uniform(-0.4, 0.2) * radius
            entry_world = route[0] @ m[:3, :3].T + m[:3, 3]
            tangent = (route[min(8, len(route) - 1)] - route[0]) @ m[:3, :3].T
            tangent /= max(np.linalg.norm(tangent), 1e-9)
            # Arrive from outside the molecule near the C terminus, turning into the backbone.
            outward = entry_world - center
            outward /= max(np.linalg.norm(outward), 1e-9)
            lead = outward * 0.8 - tangent * 0.2
            lead /= max(np.linalg.norm(lead), 1e-9)
            approach = entry_world + lead * 0.6 * radius
            # A wide swoop around the molecule, turning toward the approach point. Depth
            # varies smoothly so chords between waypoints stay outside the molecule.
            # End the swoop on the C terminus's side of the screen, so the approach does not
            # cross the molecule; about half the strands add an extra loop around it.
            entry_angle = np.arctan2(np.dot(outward, up), np.dot(outward, right))
            sweep = (entry_angle - side + np.pi) % (2 * np.pi) - np.pi
            if rng.uniform() < 0.5:
                sweep += np.sign(sweep or 1.0) * 2 * np.pi
            wobble, phase0 = rng.uniform(0.35, 0.6), rng.uniform(0, 2 * np.pi)
            steps = max(6, int(np.ceil(abs(sweep) / (np.pi / 5))))
            waypoints = [start]
            for frac in np.linspace(0.12, 0.92, steps):
                angle = side + sweep * frac
                ring = right * np.cos(angle) + up * np.sin(angle)
                reach = radius * (1.75 - 0.45 * frac)
                depth_offset = wobble * radius * np.sin(2 * np.pi * frac + phase0)
                waypoints.append(center + ring * reach + forward * depth_offset)
            waypoints += [approach, entry_world]
            # Continue the spline into the backbone for a smooth entry, then cut at the C terminus.
            ahead = [route[min(k, len(route) - 1)] @ m[:3, :3].T + m[:3, 3] for k in (8, 16)]
            flight = _centripetal(np.array(waypoints + ahead), 40)[: 40 * (len(waypoints) - 1) + 1]
            flight = self._corkscrew(flight, rng, radius)
            flight[-1] = entry_world
            # Back to the protein's local frame, where the wire is drawn.
            flight_local = (flight - m[:3, 3]) @ inverse.T
            colors = self._route_colors(palette, route, residues)
            self.strands.append(_Strand(route, flight_local, colors))
        self._combined = None
        if self.mode == "geometry":
            from ._thread_geometry import ThreadGeometry

            self.geometry = ThreadGeometry(p, self.strands)
            # Coordinate interpolation and its visibility clock may be mid-morph
            # at bind time. Freeze that visibility while transporting coordinates.
            controls = p._controls.copy()
            clock = np.clip((p._mix - controls[:, 6]) / np.maximum(controls[:, 7], 1e-8), 0, 1)
            clock = clock * clock * clock * (10 + clock * (-15 + 6 * clock))
            alpha = (1 - clock) * controls[:, 4] + clock * controls[:, 5]
            controls[:, 2] = 0
            controls[:, 4:6] = alpha[:, None]
            controls.flags.writeable = False
            self.geometry_controls = controls

    def _corkscrew(self, flight, rng, radius):
        """Coil around the flight curve on a twist-free frame, fading out before the approach."""
        s = _arc_length(flight)
        u = s / max(s[-1], 1e-9)
        tangent = np.gradient(flight, s, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
        a = np.empty_like(flight)
        a[0] = _perpendiculars(tangent[0])[0]
        for i in range(1, len(flight)):
            # Parallel transport: remove the component along the new tangent.
            v = a[i - 1] - np.dot(a[i - 1], tangent[i]) * tangent[i]
            a[i] = v / max(np.linalg.norm(v), 1e-9)
        b = np.cross(tangent, a)
        turns = rng.uniform(3.0, 5.0)
        amplitude = self.swirl * 0.28 * radius * np.sin(np.pi * np.clip(u / 0.8, 0, 1)) ** 2
        phase = 2 * np.pi * turns * u
        return flight + amplitude[:, None] * (np.cos(phase)[:, None] * a + np.sin(phase)[:, None] * b)

    @staticmethod
    def _route_colors(palette, route, residues):
        # The route is sampled 8 times per residue, from the C terminus to the N terminus.
        index = np.minimum(np.arange(len(route)) // 8, len(residues) - 1)
        return palette[residues[index]]

    def apply(self, alpha):
        alpha = float(alpha)
        if self.reverse_time:
            alpha = 1 - alpha
        if self.mode == "geometry":
            return self._apply_geometry(alpha)
        thread_end = 1 - self.settle
        t = min(alpha / thread_end, 1.0) if thread_end > 0 else 1.0
        fade = 0.0 if alpha <= thread_end else (alpha - thread_end) / max(self.settle, 1e-9)
        fade = fade * fade * (3 - 2 * fade)
        span = 1 - (len(self.strands) - 1) * self.stagger
        points, colors, radii, heads, head_colors, strengths, moving = [], [], [], [], [], [], False
        for k, (strand, count) in enumerate(zip(self.strands, self.wire.counts)):
            local = float(np.clip((t - k * self.stagger) / span, 0.0, 1.0))
            progress = float(self.easing(local))
            if not np.isfinite(progress) or not 0 <= progress <= 1:
                raise ValueError("easing must return a finite value in [0, 1]")
            moving |= progress > 0
            xyz, rgb = strand.sample(progress * strand.total, count)
            heads.append(xyz[0])
            head_colors.append(
                0.45 * rgb[0] + 0.55 * np.array([1.0, 0.95, 0.85])
                if self.glow_color is None
                else self.glow_color
            )
            # The glow is brightest in flight and settles as each head comes to rest.
            strengths.append(0.0 if progress <= 0 else self.glow_brightness * (1.0 - 0.6 * local**6))
            # A brighter, thicker head tapers into the wire.
            taper = np.clip(np.arange(count) / 6.0, 0, 1)
            rgb = rgb * taper[:, None] + (1 - taper[:, None]) * np.array([1.0, 0.97, 0.9])
            r = self.wire.wire_radius * (1 + (self.head_scale - 1) * (1 - taper))
            points.append(xyz)
            colors.append(rgb)
            radii.append(r)
        wire = self.wire
        wire.set_positions(np.vstack(points))
        metadata = np.column_stack((np.vstack(colors), np.concatenate(radii))).astype(np.float32)
        metadata.flags.writeable = False
        wire._metadata_override = metadata
        wire.opacity = 1.0 - fade if moving else 0.0
        wire.glow_heads = (np.array(heads), np.array(head_colors), np.array(strengths))
        self.target.opacity = self.end_opacity * fade

    def _apply_geometry(self, alpha):
        p = self.target
        if p.surface_opacity > 0:
            raise ValueError(
                "Thread does not support surface representation; use cartoon, ribbon or ball_and_stick"
            )
        span = 1 - (len(self.strands) - 1) * self.stagger
        clocks = np.clip((alpha - np.arange(len(self.strands)) * self.stagger) / span, 0, 1)
        xyz, heads, strengths = self.geometry.evaluate(clocks, self.easing)
        if alpha >= 1:
            s = self.original
            p._pair(s["a"], s["b"], s["mix"], s["key_a"], s["key_b"])
            p._controls = s["controls"]
        else:
            p.set_positions(xyz)
            p._controls = self.geometry_controls
        p.opacity = self.end_opacity if alpha > 0 else 0.0
        colors = np.array([strand.colors[-1] for strand in self.strands])
        colors = 0.45 * colors + 0.55 * np.array([1.0, 0.95, 0.85])
        if self.glow_color is not None:
            colors[:] = self.glow_color
        self.wire.glow_heads = (heads, colors, strengths * self.glow_brightness)
        self.wire.opacity = float(0 < alpha < 1)


class Unthread(Thread):
    """Reverse Thread: pull molecular geometry (or optional wires) out of view."""

    reverse_time = True
