"""Nucleotide base meshes with shared native-GPU and EEVEE pose evaluation.

Bases draw with the cartoon and ribbon representations, in one of four shapes (slabs,
rings, sticks, ladder) blended by ``Protein.base_style``. BaseGeometry builds each shape
once, in local frames tied to a few atoms of each base. The native renderer (BaseGPU,
driven by renderer.py) and the EEVEE exporter (BaseGeometry.evaluate, called from
_eevee_geometry.py) pose those frames from the current atom positions, so bases follow
trajectories and morphs without rebuilding meshes. BaseStyle animates the blend.
"""

import warnings

import mapbox_earcut
import numpy as np

from .animation import Animation
from .geometry import residue_colors
from .math3d import normalize
from .styling import current_tints

# Order of the base_style weights and of BaseGeometry.parts.
BASE_STYLES = ("slabs", "rings", "sticks", "ladder")

# Atom names of the phosphate group; the backbone tube already draws it.
_PHOSPHATE = ("P", "OP1", "OP2", "OP3", "O1P", "O2P", "O3P")


def base_weights(style):
    """One-hot weights over BASE_STYLES for a style name; ``"none"`` gives all zeros."""
    if style not in (*BASE_STYLES, "none"):
        raise ValueError(f"Base style must be one of {(*BASE_STYLES, 'none')}")
    return np.zeros(4) if style == "none" else np.eye(4)[BASE_STYLES.index(style)]


class BaseStyle(Animation):
    """Crossfade nucleotide bases between slabs, rings, sticks, ladder, or none.

    Backbone geometry and residue appearance stay on their own animation tracks.
    Bases appear with cartoon/ribbon representations. Durations belong to scene.play().
    """

    channels = frozenset({"base_style"})

    def __init__(self, molecule, style, **kwargs):
        super().__init__(molecule, **kwargs)
        self.end = base_weights(style)

    def bind(self):
        super().bind()
        self.start = self.target.base_style.copy()

    def apply(self, alpha):
        # Crossfade the weights; the renderer scales each shape's opacity by its weight.
        self.target.base_style = (1 - alpha) * self.start + alpha * self.end


def _frame(xyz, ids, mode):
    """A right-handed basis, mirrored exactly by base_vertex in shaders/molecule.wgsl.

    ``ids`` holds three atom indices per vertex and ``mode`` is 0 for planes, 1 for rods.
    Returns the origin and the x, y and z axes, each (vertices, 3).
    """
    # Plane frame: origin at the first atom, x toward the second, z normal to the plane
    # through the third, and y = z × x in the plane.
    origin = xyz[ids[:, 0]]
    x = normalize(xyz[ids[:, 1]] - origin)
    z = normalize(np.cross(x, xyz[ids[:, 2]] - origin))
    y = np.cross(z, x)
    rod = mode > 0
    if rod.any():
        # Rod frame: z is the whole, unnormalized vector from the start atom to the end atom,
        # so local z in [0, 1] runs along the rod. x and y are unit vectors across it, built
        # from world y, or world x when the rod is nearly parallel to y.
        axis = x[rod].copy()
        ref = np.where(np.abs(axis[:, 1:2]) > 0.9, [1, 0, 0], [0, 1, 0])
        x[rod] = normalize(np.cross(axis, ref))
        y[rod] = np.cross(axis, x[rod])
        z[rod] = xyz[ids[rod, 1]] - origin[rod]
    return origin, x, y, z


def _rounded_rectangle(polygon):
    """A rounded rectangle fitted in the actual base plane, around a 2D ring outline."""
    # The principal axes of the ring atoms orient it, 0.18 Å of margin pads it, and its
    # corners are quarter circles of up to 0.28 Å radius, five points each.
    center = polygon.mean(0)
    _, axes = np.linalg.eigh(np.cov(polygon.T))
    rect = (polygon - center) @ axes
    lo, hi = rect.min(0) - 0.18, rect.max(0) + 0.18
    radius = min(0.28, float((hi - lo).min()) / 4)
    outline = []
    for cx, cy, start in (
        (hi[0] - radius, hi[1] - radius, 0),
        (lo[0] + radius, hi[1] - radius, 90),
        (lo[0] + radius, lo[1] + radius, 180),
        (hi[0] - radius, lo[1] + radius, 270),
    ):
        for t in np.deg2rad(np.linspace(start, start + 90, 5)):
            outline.append([cx + radius * np.cos(t), cy + radius * np.sin(t)])
    return np.array(outline) @ axes.T + center


class BaseGeometry:
    """Cache base shapes once; vertex shaders evaluate every trajectory tween.

    Slabs and rings are schematic rigid planes located by three deposited ring atoms.
    Sticks use individual atom endpoints. Missing ring templates fall back to sticks.
    No base pairing or hydrogen bonds are inferred from ladder rods.
    """

    def __init__(self, p):
        # Anything that changes these shapes; see matches().
        self.key = (p.topology, p._base_reference, p.base_thickness, p.base_radius)
        xyz = p._base_reference
        # One (rows, faces, residue of each row) mesh per base style, in BASE_STYLES order.
        self.parts = []
        # Atom indices by name, and the covalent bonds inside each residue.
        groups = [{} for _ in p.topology.residues]
        residue_bonds = [[] for _ in p.topology.residues]
        for i, a in enumerate(p.topology.atoms):
            groups[a.residue_index][a.name.replace("*", "'")] = i
        for a, b in p.topology.bonds:
            ri = p.topology.atoms[a].residue_index
            if p.topology.atoms[b].residue_index == ri:
                residue_bonds[ri].append((int(a), int(b)))
        missing = set()
        for style in BASE_STYLES:
            # _vertex(), _prism() and _rod() append to these lists while a style is built.
            self._rows, self._faces, self._owners = [], [], []
            for ri, r in enumerate(p.topology.residues):
                # Nucleotides off a traced strand (e.g. ATP) draw as ball-and-stick detail.
                if not r.is_nucleic or p.topology.residue_categories[ri] != "polymer":
                    continue
                self._add_base(p, xyz, style, ri, r, groups[ri], residue_bonds[ri], missing)
            rows = np.asarray(self._rows, np.float32).reshape(-1, 14)
            faces = np.asarray(self._faces, np.uint32).reshape(-1, 3)
            self.parts.append((rows, faces, np.array(self._owners, dtype=int)))
        if missing:
            warnings.warn("Incomplete base rings use sticks: " + ", ".join(sorted(missing)), stacklevel=2)
        del self._rows, self._faces, self._owners

    def _add_base(self, p, xyz, style, ri, r, names, bonds, missing):
        """Append one nucleotide's base, in ``style``, to the mesh being built.

        ``names`` maps the residue's atom names to atom indices and ``bonds`` lists its
        internal bonds. Residues that cannot draw slabs or rings are added to ``missing``.
        """
        # Sugar and phosphate atoms are shown by the backbone/connector.
        base_ids = {
            i
            for n, i in names.items()
            if "'" not in n and n not in _PHOSPHATE and p.topology.atoms[i].element != "H"
        }
        if not base_ids:
            return
        # Vertices take their tint and opacity from this atom, normally the trace atom.
        owner = r.trace_atom if r.trace_atom >= 0 else min(base_ids)
        rings = ["N1", "C2", "N3", "C4", "C5", "C6"]
        if "N9" in names:
            # Outer perimeter of fused six/five-membered purine rings.
            rings = ["N1", "C2", "N3", "C4", "N9", "C8", "N7", "C5", "C6"]
        # Slabs and rings need the whole ring, and N1, C2 and C4, which define the plane
        # frame, must not be collinear.
        complete = all(n in names for n in rings)
        anchors = [names[n] for n in ("N1", "C2", "C4")] if complete else None
        if complete:
            a, b, c = xyz[anchors]
            complete = np.linalg.norm(np.cross(b - a, c - a)) > 1e-5
        draw_style = style
        if style in ("slabs", "rings") and not complete:
            missing.add(f"{r.chain}:{r.name}{r.resid}{r.icode}")
            draw_style = "sticks"
        if draw_style == "sticks":
            for a, b in bonds:
                if int(a) in base_ids and int(b) in base_ids:
                    self._rod(a, b, p.base_radius, owner, ri)
        elif draw_style == "ladder":
            # One thick rod from the backbone to the farthest base atom, like a ladder rung.
            if r.trace_atom >= 0:
                end = max(base_ids, key=lambda i: np.linalg.norm(xyz[i] - xyz[owner]))
                self._rod(owner, end, p.base_radius * 1.65, owner, ri)
            return
        else:
            # The ring outline in 2D coordinates of the base plane, measured from N1.
            origin, x, y, _ = _frame(xyz, np.array([anchors]), np.array([0]))
            ring_xyz = xyz[[names[n] for n in rings]] - origin[0]
            polygon = np.column_stack((ring_xyz @ x[0], ring_xyz @ y[0]))
            if draw_style == "slabs":
                polygon = _rounded_rectangle(polygon)
            self._prism(polygon, p.base_thickness, anchors, owner, ri)
        # Locate the glycosidic attachment from a deposited covalent bond.
        # This also handles C-glycosides such as pseudouridine correctly.
        sugar = names.get("C1'")
        if sugar is not None:
            # A rod from the trace atom to C1′, then from C1′ to the base atom bonded to it.
            if owner != sugar:
                self._rod(owner, sugar, p.base_radius, owner, ri)
            for a, b in bonds:
                if a == sugar and int(b) in base_ids:
                    self._rod(a, b, p.base_radius, owner, ri)
                elif b == sugar and int(a) in base_ids:
                    self._rod(b, a, p.base_radius, owner, ri)

    def matches(self, p):
        """True while these meshes still fit ``p``: same topology, reference and base dimensions."""
        return (
            self.key[0] is p.topology
            and self.key[1] is p._base_reference
            and self.key[2:] == (p.base_thickness, p.base_radius)
        )

    def _vertex(self, point, normal, ids, owner, residue, mode):
        """Append one vertex row and return its index."""
        # local XYZ, normal XYZ, palette RGB, owner, frame atom indices, mode.
        # Columns 0:3 and 3:6 are the position and normal in the _frame() basis, 6:9 the
        # palette color (filled by colored_rows), and 9:14 hold uint32 bits: the owner atom
        # (9), the three frame atoms (10:13) and the mode, 0 plane or 1 rod (13).
        # 14 × 4 bytes is the 56-byte vertex stride in BaseGPU.
        row = np.zeros(14, np.float32)
        row[:3], row[3:6] = point, normal
        row[9:14].view(np.uint32)[:] = [owner, *ids, mode]
        self._rows.append(row)
        self._owners.append(residue)
        return len(self._rows) - 1

    def _prism(self, polygon, thickness, ids, owner, residue):
        """A closed slab of ``thickness`` Å over a 2D polygon in the base plane."""
        # Shoelace formula: a negative signed area means a clockwise outline. Make it
        # counterclockwise so caps and side normals face outward.
        area = np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) - polygon[:, 1] * np.roll(polygon[:, 0], -1))
        if area < 0:
            polygon = polygon[::-1].copy()
        # Ear-clipping triangulation of the outline, for the two flat caps.
        triangles = mapbox_earcut.triangulate_float64(
            np.asarray(polygon, np.float64), np.array([len(polygon)], np.uint32)
        ).reshape(-1, 3)
        # Bottom (z = −thickness/2, reversed winding) and top caps.
        for side in (-1, 1):
            offset = len(self._rows)
            for v in polygon:
                self._vertex([*v, side * thickness / 2], [0, 0, side], ids, owner, residue, 0)
            self._faces.extend((triangles[:, ::-1] if side < 0 else triangles) + offset)
        # One quad per outline edge. For a counterclockwise outline, (dy, −dx) points outward.
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            n = normalize([b[1] - a[1], a[0] - b[0], 0])
            j = len(self._rows)
            for pt, z in ((a, -1), (b, -1), (b, 1), (a, 1)):
                self._vertex([*pt, z * thickness / 2], n, ids, owner, residue, 0)
            self._faces.extend(([j, j + 1, j + 2], [j, j + 2, j + 3]))

    def _rod(self, start, end, radius, owner, residue):
        """A capped 12-sided cylinder of ``radius`` Å from atom ``start`` to atom ``end``."""
        # The third frame atom is unused in rod mode; see _frame().
        ids = [start, end, end]
        sides, offset = 12, len(self._rows)
        # Two rings of side vertices, at the start (z = 0) and the end (z = 1).
        for t in (0, 1):
            for angle in np.arange(sides) * 2 * np.pi / sides:
                c, s = np.cos(angle), np.sin(angle)
                self._vertex([radius * c, radius * s, t], [c, s, 0], ids, owner, residue, 1)
        for i in range(sides):
            a, b = offset + i, offset + (i + 1) % sides
            self._faces.extend(([a, b, b + sides], [a, b + sides, a + sides]))
        # Flat end caps: a center vertex and a fan of triangles with their own normals.
        for t, normal in ((0, -1), (1, 1)):
            j = len(self._rows)
            self._vertex([0, 0, t], [0, 0, normal], ids, owner, residue, 1)
            for angle in np.arange(sides) * 2 * np.pi / sides:
                self._vertex(
                    [radius * np.cos(angle), radius * np.sin(angle), t],
                    [0, 0, normal],
                    ids,
                    owner,
                    residue,
                    1,
                )
            for i in range(sides):
                tri = [j, j + 1 + i, j + 1 + (i + 1) % sides]
                self._faces.append(tri if t else tri[::-1])

    def colored_rows(self, p, part):
        """Vertex rows of one base style with the palette color of each vertex's residue."""
        rows, _, residues = self.parts[part]
        rows = rows.copy()
        rows[:, 6:9] = residue_colors(p)[residues]
        return rows

    def evaluate(self, p, part):
        """Pose one base style at the current coordinates, as base_vertex does on the GPU.

        Returns vertices (V, 3) in the protein's own frame, faces (F, 3), colors (V, 3),
        one opacity per face (F,), and unit normals (V, 3).
        """
        rows = self.colored_rows(p, part)
        faces = self.parts[part][1]
        # Unpack the uint32 columns (see _vertex).
        ids = rows[:, 10:13].copy().view(np.uint32)
        mode = rows[:, 13].copy().view(np.uint32)
        owner = rows[:, 9].copy().view(np.uint32)
        origin, x, y, z = _frame(p.positions, ids, mode)
        vertices = origin + x * rows[:, :1] + y * rows[:, 1:2] + z * rows[:, 2:3]
        normals = normalize(x * rows[:, 3:4] + y * rows[:, 4:5] + normalize(z) * rows[:, 5:6])
        # Same blend as tint() in molecule.wgsl: palette × (1 − tint strength) + tint RGB.
        tint = current_tints(p)[owner]
        colors = rows[:, 6:9] * (1 - tint[:, 3:4]) + tint[:, :3]
        # Bases show with cartoons and ribbons only, weighted by this style's blend weight.
        # A face is as transparent as its most transparent vertex.
        alpha = p.atom_opacities[owner] * p.base_style[part] * sum(p.representation[:2])
        return vertices, faces, colors, alpha[faces].min(1), normals


class BaseGPU:
    """GPU vertex and index buffers of one molecule's base meshes, one pair per base style."""

    def __init__(self, renderer, p):
        self.renderer, self.geometry = renderer, BaseGeometry(p)
        self.buffers = []
        # The color scheme baked into the uploaded vertex colors.
        self.color_key = None
        # Every molecule shares one opaque and one transparent pipeline per renderer.
        if not hasattr(renderer, "base_pipelines"):
            # Vertex attributes of base_vertex in molecule.wgsl, in the 56-byte row of _vertex():
            # local position, normal, palette color, owner atom, three frame atoms, mode.
            layout = [
                {
                    "array_stride": 56,
                    "step_mode": "vertex",
                    "attributes": [
                        {"format": fmt, "offset": offset, "shader_location": i}
                        for i, (fmt, offset) in enumerate(
                            (
                                ("float32x3", 0),
                                ("float32x3", 12),
                                ("float32x3", 24),
                                ("uint32", 36),
                                ("uint32x3", 40),
                                ("uint32", 52),
                            )
                        )
                    ],
                }
            ]
            renderer.base_pipelines = {
                t: renderer._pipeline(
                    "base_vertex",
                    "surface_transparent" if t else "surface_fragment",
                    vertex_buffers=layout,
                    transparent=t,
                    # Prisms and rods are closed meshes with outward-facing triangles.
                    cull_mode="back",
                )
                for t in (False, True)
            }
        self.update(p)

    def update(self, p):
        """Rebuild stale meshes, and re-upload vertex rows when the color scheme changes."""
        import wgpu

        changed = not self.geometry.matches(p)
        if changed:
            self.geometry = BaseGeometry(p)
        # Palette colors are baked into the vertex rows. A scheme blend's text includes its
        # fraction, so a running blend uploads new colors on each change.
        if changed or self.color_key != str(p.color_scheme):
            self.close()
            self.draws = []
            for i, (_, faces, _) in enumerate(self.geometry.parts):
                if not len(faces):
                    self.draws.append(None)
                    continue
                vertices = self.renderer.device.create_buffer_with_data(
                    data=self.geometry.colored_rows(p, i), usage=wgpu.BufferUsage.VERTEX
                )
                indices = self.renderer.device.create_buffer_with_data(
                    data=faces, usage=wgpu.BufferUsage.INDEX
                )
                self.buffers.extend((vertices, indices))
                self.draws.append((vertices, indices, faces.size))
            self.color_key = str(p.color_scheme)

    def draw(self, render_pass, bindings, p, transparent):
        """Draw every base style with a nonzero weight."""
        render_pass.set_pipeline(self.renderer.base_pipelines[transparent])
        for i, draw in enumerate(self.draws):
            if draw is None or p.base_style[i] <= 0:
                continue
            vertices, indices, size = draw
            # Bind groups 3 to 6 are the base styles; their object opacity includes the weight.
            render_pass.set_bind_group(1, bindings[3 + i])
            render_pass.set_vertex_buffer(0, vertices)
            render_pass.set_index_buffer(indices, "uint32")
            render_pass.draw_indexed(size)

    def close(self):
        """Release the GPU buffers."""
        for b in self.buffers:
            b.destroy()
        self.buffers = []
