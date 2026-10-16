"""CPU mesh evaluation for Blender. Coordinates remain in world ångströms.

The native renderer builds cartoons, bonds and surface skinning in GPU vertex shaders
(``shaders/molecule.wgsl``). Blender needs explicit triangle meshes, so this module
evaluates the same geometry with numpy: the Catmull–Rom cartoon sweep, atom spheres,
trimmed bond cylinders, skinned molecular surfaces, the view cutaway fade, the depth
tunnel wall, and glow halos. ``eevee.EEVEE`` calls ``MeshExporter.meshes`` for each object
and ``tunnel_meshes`` once per frame, then saves the results for ``_blender_worker.py``.

Each mesh is a dict of arrays:

- ``vertices`` (n, 3) float32, world Å, and ``normals`` (n, 3) float32
- ``faces`` (m, 3) int32 vertex indices
- ``colors`` (n, 3) float32, display RGB in [0, 1]
- ``opacity`` (m,) float32, one value per face in [0, 1]
- optional ``strength`` (n,) glow intensity, ``kind`` (``"glow"``) and ``unlit`` (bool)

Blender gives each distinct opacity its own render layer, so fades here are rounded to
a few levels (eighths for the cutaway, twentieths for the tunnel) to keep that count low.
"""

from functools import lru_cache

import numpy as np

from .geometry import (
    atom_colors,
    atom_metadata,
    residue_colors,
    secondary_weights,
    segments,
    state_data,
)
from .math3d import normalize, perpendiculars
from .styling import atom_weights, current_tints

# Tunnel wall colors, as in the native tunnel shader: light blue at the molecule's outer
# surface, shading to amber at the bottom of the tunnel.
TUNNEL_SURFACE_COLOR = np.array([0.42, 0.62, 0.78])
TUNNEL_DEEP_COLOR = np.array([0.95, 0.73, 0.40])


def _tint(base, tint):
    """Apply premultiplied tints (r, g, b, weight) to base RGB, as tint() in the native shader."""
    return base * (1 - tint[..., 3:4]) + tint[..., :3]


def _smoothstep(edge0, edge1, x):
    """WGSL smoothstep, safe when ``edge0 == edge1``."""
    t = np.clip((x - edge0) / np.maximum(edge1 - edge0, 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def cutaway_amount(points, camera, geometry, tight):
    """Per-point fade of the open cutaway, matching the native shader's cut_to()."""
    # Returns 0 where geometry stays and 1 where it is fully cut away. ``geometry`` is
    # camera.cutaway_geometry(): target center, kept radius, window radius, outer-surface
    # distance (unused here) and tunnel axis, all in world Å. ``tight`` (cartoons, bases,
    # surfaces) carves closer to the center, since these pass in front of a ligand or enclose it.
    center, keep, window, _, axis = geometry
    keep = keep * camera.cutaway_surface_keep if tight else keep
    opening, softness, band = camera.cutaway_opening, camera.cutaway_softness, camera.cutaway_band
    if camera.cutaway_tunnel:
        # Tunnel: a straight cylinder of radius opening × window along the axis. Points
        # farther than the kept radius along the axis are cut; the cut ramps up over ``band`` Å.
        d = points - center
        height = d @ axis
        r = np.linalg.norm(d - height[:, None] * axis, axis=1)
        radius = opening * window
        radial = 1 - _smoothstep(radius * (1 - softness), radius, r)
        return radial * _smoothstep(keep, keep + band, height)
    # Window: a cone from the eye through a disc of radius opening × window at the target,
    # so it is a fixed disc on screen. Points nearer the eye than the kept sphere are cut;
    # they fade back in over ``band`` Å in front of it. t is depth along the view axis.
    eye = camera.eye
    to_target = center - eye
    dist = max(float(np.linalg.norm(to_target)), 1e-4)
    view = to_target / dist
    d = points - eye
    t = d @ view
    front = dist - keep
    aperture = opening * window * np.maximum(t, 0) / dist
    r = np.linalg.norm(d - t[:, None] * view, axis=1)
    radial = 1 - _smoothstep(aperture * (1 - softness), aperture, r)
    return radial * (1 - _smoothstep(front - band, front, t))


def apply_cutaway(mesh, camera, geometry, tight):
    """Fade faces inside the cutaway (in eighths, to limit EEVEE opacity layers) and tint the rim."""
    amount = cutaway_amount(mesh["vertices"].astype(float), camera, geometry, tight)
    if not np.any(amount > 0):
        return mesh
    face = amount[mesh["faces"]].mean(1)
    mesh["opacity"] = (mesh["opacity"] * (1 - np.round(face * 8) / 8)).astype(np.float32)
    # A faint rim where geometry is half faded, at most 40 % rim color, as cutaway_rim() in
    # the native shader. amount × (1 − amount) peaks at 0.25, so 1.6 × 0.25 = 0.4.
    rim = np.clip(amount * (1 - amount) * 1.6, 0, 0.4)[:, None]
    mesh["colors"] = (mesh["colors"] * (1 - rim) + np.asarray(camera.cutaway_rim) * rim).astype(np.float32)
    return mesh


def _depth_color(deep):
    """Tunnel wall color for depth fractions ``deep`` (k, 1): 0 at the outer surface, 1 at the bottom."""
    return (1 - deep) * TUNNEL_SURFACE_COLOR + deep * TUNNEL_DEEP_COLOR


def tunnel_meshes(camera, height_px):
    """The tunnel wall and its depth rings, as the native tunnel pass draws them."""
    geometry = camera.cutaway_geometry()
    if geometry is None or not camera.cutaway_tunnel:
        return []
    center, keep, window, outer, axis = geometry
    # The wall runs from where cartoons and surfaces stop being carved out to the
    # molecule's outer surface; heights are Å from the target center along the axis.
    floor = keep * camera.cutaway_surface_keep
    length = outer - floor
    # The native renderer also skips walls 1 Å long or less.
    if length <= 1:
        return []
    radius = camera.cutaway_opening * window
    u, v = perpendiculars(axis)
    # The same 72 sides and 64 segments as TUNNEL_SIDES and TUNNEL_SEGMENTS in the shader.
    sides, steps = 72, 64
    angle = np.arange(sides) * 2 * np.pi / sides
    ring = np.cos(angle)[:, None] * u + np.sin(angle)[:, None] * v

    def band(h0, h1, scale, count):
        # An open cylinder of radius scale × radius from height h0 to h1, in ``count`` segments.
        heights = np.linspace(h0, h1, count + 1)
        verts = (center + heights[:, None, None] * axis + scale * radius * ring[None]).reshape(-1, 3)
        return verts, _triangles(count + 1, sides), heights

    verts, faces, heights = band(floor, outer, 1.0, steps)
    below = np.repeat(outer - heights, sides)  # Å below the outer surface, per vertex.
    deep = np.clip(below / length, 0, 1)[:, None]
    base = _depth_color(deep)
    normals = np.tile(-ring, (steps + 1, 1))  # The wall is seen from inside.
    eye = camera.eye
    view = eye - verts
    view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-9)
    facing = np.abs(np.sum(normals * view, axis=1))
    # As in the shader: like glass, the wall is most visible at grazing angles. It fades in
    # over 2 Å at both ends, and it fades out between 22 and 4 Å from the camera, so flying
    # down the tunnel does not veil the view.
    ends = _smoothstep(0, 2, below) * _smoothstep(0, 2, length - below)
    near = _smoothstep(4, 22, np.linalg.norm(eye - verts, axis=1))
    alpha = camera.cutaway_wall * (0.35 + 0.65 * (1 - facing)) * ends * near
    pieces = [
        dict(
            vertices=verts.astype(np.float32),
            faces=faces.astype(np.int32),
            colors=base.astype(np.float32),
            # Face opacity in twentieths, to limit EEVEE opacity layers.
            opacity=np.round(alpha[faces].mean(1) * 20) / 20,
            normals=normals.astype(np.float32),
        )
    ]
    # Rings every `rings` Å from the outer surface; every second ring is brighter and wider.
    # The shader draws them a fixed number of pixels wide; here they are thin unlit bands
    # sized from the world width of one pixel at the target's distance.
    pixel = 2 * np.linalg.norm(eye - center) * np.tan(camera.fov / 2) / height_px
    spacing = camera.cutaway_rings
    for k in range(1, int(length // spacing) + 1):
        h = outer - k * spacing
        major = k % 2 == 0
        width = pixel * (4.0 if major else 2.4)
        # Just inside the wall (98.5 % of its radius), so the ring does not z-fight with it.
        rverts, rfaces, _ = band(h - width / 2, h + width / 2, 0.985, 1)
        rbelow = np.full(len(rverts), outer - h)
        rdeep = np.clip(rbelow / length, 0, 1)[:, None]
        rbase = _depth_color(rdeep)
        # Lightened toward white; opacity capped at 0.9 like the shader's wall alpha.
        color = rbase * 0.45 + (0.55 if major else 0.4)
        ralpha = camera.cutaway_wall * 0.6 + (0.6 if major else 0.4)
        pieces.append(
            dict(
                vertices=rverts.astype(np.float32),
                faces=rfaces.astype(np.int32),
                colors=color.astype(np.float32),
                opacity=np.full(len(rfaces), round(min(ralpha, 0.9) * 20) / 20, np.float32),
                normals=np.tile(-ring, (2, 1)).astype(np.float32),
                unlit=np.asarray(True),
            )
        )
    return pieces


def _triangles(rings, sides):
    """Triangles for a tube of ``rings`` × ``sides`` vertices, wrapping around the sides.

    Vertex (ring r, side s) has index r × sides + s. Each quad between rings r and r + 1
    gives two triangles; the result has shape ((rings − 1) × sides × 2, 3).
    """
    # a: (r, s); b: (r + 1, s); c: (r, s + 1) wrapping to side 0; d: (r + 1, s + 1).
    a = (np.arange(rings - 1)[:, None] * sides + np.arange(sides)).ravel()
    b = a + sides
    c = a - a % sides + (a + 1) % sides
    d = c + sides
    return np.stack((np.stack((a, c, d), 1), np.stack((a, d, b), 1)), 1).reshape(-1, 3)


def _atom_guides(p):
    """Per-atom cartoon width axes and twist firmness between the two keyframes, as guide() does.

    Returns guides (atoms, 3), firmness (atoms,), which atoms had their second-keyframe
    guide flipped, and each atom's interpolation progress in [0, 1].
    """
    secondary = secondary_weights(p)
    sa = state_data(p.topology, p._a, secondary=secondary)
    sb = state_data(p.topology, p._b, sa[:, 4:7], secondary)
    # state_data columns 4:7 hold the width axis (guide) and column 7 its firmness.
    ga, gb = sa[:, 4:7], sb[:, 4:7]
    # A guide is an axis, not an arrow. Flip the second keyframe's guide where it opposes
    # the first, so the blend between them never passes through zero.
    flipped = (ga * gb).sum(1) < 0
    gb = np.where(flipped[:, None], -gb, gb)
    progress = p.atom_progress
    guides = ga * (1 - progress[:, None]) + gb * progress[:, None]
    firm = sa[:, 7] * (1 - progress) + sb[:, 7] * progress
    return guides, firm, flipped, progress


def _segment_guide(ids, guides, firm, flipped, progress, t, hold):
    """Width axis and flatness along each segment, as segment_guide() in the native shader.

    ``t`` has shape (1, steps + 1, 1). Returns the guide (segments, steps + 1, 3) and the
    flatness (segments, steps + 1, 1): 1 is a flat ribbon, 0 a round tube.
    """
    first, second = guides[ids[:, 1]], guides[ids[:, 2]]
    firm = firm[ids[:, 1]]
    # Restore the second keyframe's twist of segments that per-atom alignment reversed.
    swap = flipped[ids[:, 1]] != flipped[ids[:, 2]]
    middle = 0.5 * (progress[ids[:, 1]] + progress[ids[:, 2]])
    second = np.where((swap & (middle > 0.5))[:, None], -second, second)
    firm = np.where(swap, firm * _smoothstep(0.1, 0.25, np.abs(middle - 0.5)), firm)
    first, second, firm = first[:, None], second[:, None], firm[:, None, None]
    # A firm segment turns its axis evenly along t. Where the turn's direction is
    # ambiguous, the turn happens mid-segment while the ribbon is briefly round there.
    along = firm * t + (1 - firm) * _smoothstep(0.35, 0.65, t)
    guide = (1 - along) * first + along * second
    guide += hold * (first - guide)
    flat = 1 - (1 - firm) * _smoothstep(0.22, 0.36, t) * _smoothstep(0.22, 0.36, 1 - t)
    flat += hold * (1 - flat)
    return guide, flat


def _cartoon(p, xyz, tint):
    """Cartoon or ribbon tube mesh in local Å: one swept tube per backbone segment.

    Returns (vertices, faces, colors, per-face opacity, normals), or None without segments.
    """
    # segments() rows: 0:4 the four trace-atom indices (uint32 bits) of the Catmull–Rom
    # control points, 4:6 widths and 6:8 thicknesses at the two ends, 8:11 and 12:15 end
    # colors, 16 and 17 chain start/end cap flags, 18 sheet arrow weight.
    rows = segments(p)
    if not len(rows):
        return None
    # Each segment is sampled at steps + 1 points along t, with ``sides`` points around.
    steps, sides = 16, 16
    ids = rows[:, :4].copy().view(np.uint32)
    # Evaluate the same guide interpolation and Catmull–Rom curve as the native shader.
    guides, firm, flipped, progress = _atom_guides(p)
    t = np.linspace(0, 1, steps + 1)[None, :, None]
    a, b, c, d = (xyz[ids[:, i]][:, None] for i in range(4))
    center = 0.5 * (
        2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t * t + (-a + 3 * b - 3 * c + d) * t * t * t
    )
    tangent = normalize(
        0.5 * ((-a + c) + 2 * (2 * a - 5 * b + 4 * c - d) * t + 3 * (-a + 3 * b - 3 * c + d) * t * t)
    )
    # Blend weight between the ribbon (0) and cartoon (1) cross-sections.
    cartoon, ribbon = p.representation[:2]
    mix = cartoon / max(cartoon + ribbon, 1e-8)
    # As segment_guide in the shader: arrowheads keep the strand's axis, and ribbons
    # round where the twist to the next residue is ambiguous.
    hold = np.clip(rows[:, 18, None, None], 0, 1) * mix * (1 - _smoothstep(0.85, 1, t))
    guide, flat = _segment_guide(ids, guides, firm, flipped, progress, t, hold)
    # Cross-section frame: u is the guide made perpendicular to the curve, v completes it.
    u = normalize(guide - tangent * (guide * tangent).sum(-1, keepdims=True))
    v = normalize(np.cross(tangent, u))
    # Elliptical cross-section semi-axes in Å: 0.13 by ribbon_width for a ribbon, the
    # per-residue profile for a cartoon. Where the ribbon is round (flat = 0), the width
    # shrinks to the thickness.
    height = (1 - mix) * 0.13 + mix * ((1 - t) * rows[:, 6, None, None] + t * rows[:, 7, None, None])
    width = (1 - mix) * p.ribbon_width + mix * ((1 - t) * rows[:, 4, None, None] + t * rows[:, 5, None, None])
    width = height + flat * (width - height)
    # Chain ends collapse to a point over the first or last 12 % of the segment, closing the tube.
    cap = np.ones((len(rows), steps + 1, 1))
    for flag, from_end in ((16, t), (17, 1 - t)):
        z = np.clip(from_end / 0.12, 0, 1)
        cap *= np.where(rows[:, flag, None, None] > 0.5, z * z * (3 - 2 * z), 1)
    angle = np.arange(sides) * 2 * np.pi / sides
    offset = u[:, :, None] * np.cos(angle)[None, None, :, None] * width[:, :, None]
    offset += v[:, :, None] * np.sin(angle)[None, None, :, None] * height[:, :, None]
    vertices = (center[:, :, None] + cap[:, :, None] * offset).reshape(-1, 3)
    # The ellipse (cos θ · width, sin θ · height) has its normal along (cos θ / width, sin θ / height).
    normal = normalize(
        u[:, :, None] * np.cos(angle)[None, None, :, None] / width[:, :, None]
        + v[:, :, None] * np.sin(angle)[None, None, :, None] / height[:, :, None]
    ).reshape(-1, 3)
    color_a = _tint(rows[:, 8:11], tint[ids[:, 1]])
    color_b = _tint(rows[:, 12:15], tint[ids[:, 2]])
    colors = np.repeat((1 - t) * color_a[:, None] + t * color_b[:, None], sides, axis=1).reshape(-1, 3)
    # Every segment is its own tube of (steps + 1) × sides vertices.
    template = _triangles(steps + 1, sides)
    faces = (template[None] + np.arange(len(rows))[:, None, None] * (steps + 1) * sides).reshape(-1, 3)
    # One opacity per half-residue strip keeps selection boundaries crisp and avoids
    # introducing hundreds of distinct fade levels across a single backbone segment.
    owner = np.where((np.arange(steps) + 0.5) / steps < 0.5, ids[:, 1, None], ids[:, 2, None])
    # Each strip along t has 2 × sides triangles. The representation weights fade the
    # whole backbone during a change to or from ball-and-stick or surface.
    alpha = np.repeat(p.atom_opacities[owner] * (cartoon + ribbon), sides * 2, axis=1).ravel()
    return vertices, faces, colors, alpha, normal


@lru_cache(maxsize=1)
def _sphere():
    """Unit UV sphere (vertices, faces); the vertices double as outward normals."""
    # UV sphere with shared poles and closed caps: 9 rings of 16 vertices between the poles.
    sides, rings = 16, 10
    phi = np.arange(1, rings) * np.pi / rings
    angle = np.arange(sides) * 2 * np.pi / sides
    vertices = np.stack(
        np.broadcast_arrays(
            np.sin(phi[:, None]) * np.cos(angle),
            np.sin(phi[:, None]) * np.sin(angle),
            np.cos(phi[:, None]) + np.zeros(sides),
        ),
        -1,
    ).reshape(-1, 3)
    vertices = np.vstack((vertices, [0, 0, 1], [0, 0, -1]))
    top, bottom = len(vertices) - 2, len(vertices) - 1
    # Latitude increases from north to south, opposite the sweep direction used
    # by tubes. Reverse the body triangles so geometric and radial normals agree.
    faces = _triangles(rings - 1, sides)[:, ::-1]
    # Triangle fans from the first ring to the north pole and from the last ring to the south pole.
    caps = [[top, j, (j + 1) % sides] for j in range(sides)]
    offset = (rings - 2) * sides
    caps += [[bottom, offset + (j + 1) % sides, offset + j] for j in range(sides)]
    return vertices, np.vstack((faces, caps)).astype(np.int32)


def _atoms(p, xyz, tint, weights):
    """Atom spheres and bond cylinders in local Å, for atoms with ball-and-stick weight > 0.

    Returns a list of (vertices, faces, colors, per-face opacity, normals) pieces: spheres
    first (unless the object draws sticks only), then bonds.
    """
    radii = atom_metadata(p)[:, 3] * p.atom_scale
    colors = _tint(atom_colors(p), tint)
    # Ball-and-stick fraction per atom, including detail atoms over a cartoon.
    opacity = p.atom_opacities * weights
    shown = np.flatnonzero(weights > 0)
    pieces = []
    # Objects such as distance rulers set _draw_atoms = False to draw sticks without spheres.
    if getattr(p, "_draw_atoms", True) and len(shown):
        verts, faces = _sphere()
        n, f = len(verts), len(faces)
        # One scaled copy of the unit sphere per shown atom.
        pieces.append(
            (
                (xyz[shown, None] + radii[shown, None, None] * verts).reshape(-1, 3),
                (faces[None] + np.arange(len(shown))[:, None, None] * n).reshape(-1, 3),
                np.repeat(colors[shown], n, axis=0),
                np.repeat(opacity[shown], f),
                np.tile(verts, (len(shown), 1)),
            )
        )
    bonds = p.topology.bonds
    bonds = bonds[(weights[bonds[:, 0]] > 0) & (weights[bonds[:, 1]] > 0)] if len(bonds) else bonds
    if len(bonds):
        sides = 16  # Points around each bond cylinder.
        a, b = xyz[bonds[:, 0]], xyz[bonds[:, 1]]
        delta = b - a
        length = np.linalg.norm(delta, axis=1)
        direction = normalize(delta)
        # Trim hidden stick interiors at the atom surfaces, also during group fades.
        # A cylinder of radius r_b leaves a sphere of radius r_a at sqrt(r_a² − r_b²) from its center.
        radius = p.bond_radius
        trim = (
            np.sqrt(np.maximum(radii[bonds] ** 2 - radius**2, 0))
            if getattr(p, "_draw_atoms", True)
            else np.zeros((len(bonds), 2))
        )
        exposed = length - trim.sum(1)
        a, b = a + direction * trim[:, :1], b - direction * trim[:, 1:]
        # Cross-section frame from whichever world axis is far from parallel, as in the shader.
        ref = np.where(np.abs(direction[:, 1:2]) > 0.9, [1, 0, 0], [0, 1, 0])
        u = normalize(np.cross(direction, ref))
        v = np.cross(direction, u)
        angle = np.arange(sides) * 2 * np.pi / sides
        normal = u[:, None] * np.cos(angle)[None, :, None] + v[:, None] * np.sin(angle)[None, :, None]
        # (bonds, 2 ends, sides, 3): an open cylinder between the trimmed ends.
        vertices = np.stack((a, b), 1)[:, :, None] + radius * normal[:, None]
        template = _triangles(2, sides)
        faces = (template[None] + np.arange(len(bonds))[:, None, None] * (2 * sides)).reshape(-1, 3)
        # A bond disappears with its less-visible endpoint, and when it is buried entirely.
        alpha = np.minimum(opacity[bonds[:, 0]], opacity[bonds[:, 1]])
        alpha = np.where(exposed > 1e-8, alpha, 0)
        # Color interpolation follows the original bond, including trimmed ends.
        t = np.stack((trim[:, 0] / np.maximum(length, 1e-8), 1 - trim[:, 1] / np.maximum(length, 1e-8)), 1)
        col = (1 - t[..., None]) * colors[bonds[:, 0], None] + t[..., None] * colors[bonds[:, 1], None]
        pieces.append(
            (
                vertices.reshape(-1, 3),
                faces,
                np.repeat(col, sides, axis=1).reshape(-1, 3),
                np.repeat(alpha, len(template)),
                np.repeat(normal[:, None], 2, 1).reshape(-1, 3),
            )
        )
    return pieces


class MeshExporter:
    """Turns scene objects into world-space meshes for Blender.

    Caches molecular surfaces and nucleic-acid base geometry per object between frames.
    """

    def __init__(self):
        self.surfaces = {}  # Object -> (surface options, coordinates at build time, mesh).
        self.bases = {}  # Object -> BaseGeometry.

    def meshes(self, p, camera=None):
        """World-space meshes for one object, with the camera's cutaway and glow applied."""
        geometry = None if camera is None else camera.cutaway_geometry()
        # Mesh objects (density surfaces and slices) export their own world-space mesh.
        if hasattr(p, "_export_mesh"):
            result = p._export_mesh()
            if geometry is not None:
                result = [apply_cutaway(mesh, camera, geometry, tight=False) for mesh in result]
            return result
        return self._molecule(p, camera, geometry)

    def _molecule(self, p, camera, geometry):
        """Meshes for a Protein or NucleicAcid: cartoon, bases, atoms and bonds, surface, glow."""
        from .surface import build_surface

        xyz, tint = p.positions, current_tints(p)
        # Pieces are (vertices, faces, colors, per-face opacity, normals) in local Å.
        pieces = []
        tight = []  # Cartoons, bases and surfaces are carved closer to a cutaway's target.
        if sum(p.representation[:2]) > 0:
            part = _cartoon(p, xyz, tint)
            if part is not None:
                pieces.append(part)
                tight.append(True)
            if any(r.is_nucleic for r in p.topology.residues) and np.any(p.base_style > 0):
                from .nucleic import BaseGeometry

                # Base shapes are rebuilt only when the topology, reference coordinates or base sizes change.
                if p not in self.bases or not self.bases[p].matches(p):
                    self.bases[p] = BaseGeometry(p)
                # One piece per base style with weight (slabs, rings, sticks, ladder).
                for i, weight in enumerate(p.base_style):
                    if weight > 0 and len(self.bases[p].parts[i][1]):
                        pieces.append(self.bases[p].evaluate(p, i))
                        tight.append(True)
        weights = atom_weights(p)
        if np.any(weights > 0):
            atoms = _atoms(p, xyz, tint, weights)
            pieces.extend(atoms)
            tight.extend([False] * len(atoms))
        if p.surface_opacity > 0:
            # Rebuild when the surface options change, or for update="rebuild" when the
            # coordinates change. update="deform" keeps the mesh and skins it below.
            options = p._surface_options
            cached = self.surfaces.get(p)
            if (
                cached is None
                or cached[0] is not options
                or (options.update == "rebuild" and not np.array_equal(cached[1], xyz))
            ):
                reference = options.reference if options.update == "deform" else xyz
                mesh = build_surface(p, options, reference)
                self.surfaces[p] = options, xyz.copy(), mesh
            else:
                mesh = cached[2]
            # Skinning, as on the GPU: each vertex moves with the weighted displacement of its
            # four nearest atoms from the reference coordinates the mesh was built on.
            vertices = mesh.vertices + np.sum(
                (xyz - mesh.reference)[mesh.neighbors] * mesh.weights[..., None], axis=1
            )
            # Each vertex takes the color of its nearest atom (mesh.owners).
            base = (
                atom_metadata(p)[mesh.owners, :3]
                if p.color_scheme == "element"
                else residue_colors(p)[[a.residue_index for a in p.topology.atoms]][mesh.owners]
            )
            col = _tint(base, tint[mesh.owners])
            # A face is as transparent as its most transparent corner.
            alpha = (p.atom_opacities[mesh.owners] * p.surface_opacity)[mesh.faces].min(1)
            # Recompute smooth normals for deformed surfaces.
            normals = np.zeros_like(vertices)
            tri = vertices[mesh.faces]
            face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            for i in range(3):
                np.add.at(normals, mesh.faces[:, i], face_normals)
            pieces.append((vertices, mesh.faces, col, alpha, normalize(normals)))
            tight.append(True)
        # Local to world: model matrix (scale, rotation, translation) on vertices; its
        # linear part on normals, which are then renormalized.
        m = p.model_matrix
        result = []
        for (vertices, faces, colors, alpha, normals), carve in zip(pieces, tight):
            mesh = dict(
                vertices=np.asarray(vertices @ m[:3, :3].T + m[:3, 3], np.float32),
                faces=np.asarray(faces, np.int32),
                colors=np.asarray(colors, np.float32),
                opacity=np.clip(alpha, 0, 1).astype(np.float32),
                normals=np.asarray(normalize(normals @ m[:3, :3].T), np.float32),
            )
            if geometry is not None:
                mesh = apply_cutaway(mesh, camera, geometry, carve)
            result.append(mesh)
        # Thread wires glow at their heads.
        glow = p._glow() if hasattr(p, "_glow") else None
        if glow is not None and len(glow):
            result.append(glow_mesh(glow))
        return result


def glow_mesh(rows):
    """Halo spheres for glow rows (world center, color, radius, intensity)."""
    # Row columns: 0:3 center in world Å, 3:6 RGB, 6 radius in Å, 7 intensity. Opacity
    # stays 1: the glow material adds light, and ``strength`` sets how much.
    verts, faces = _sphere()
    n = len(verts)
    centers, colors, radii, strength = rows[:, :3], rows[:, 3:6], rows[:, 6], rows[:, 7]
    return dict(
        vertices=(centers[:, None] + radii[:, None, None] * verts).reshape(-1, 3).astype(np.float32),
        faces=(faces[None] + np.arange(len(rows))[:, None, None] * n).reshape(-1, 3).astype(np.int32),
        colors=np.repeat(np.clip(colors, 0, 1), n, axis=0).astype(np.float32),
        strength=np.repeat(strength, n).astype(np.float32),
        opacity=np.ones(len(rows) * len(faces), np.float32),
        normals=np.tile(verts, (len(rows), 1)).astype(np.float32),
        kind=np.asarray("glow"),
    )
