"""CPU mesh evaluation for Blender. Coordinates remain in world ångströms."""

from functools import lru_cache

import numpy as np

from .geometry import atom_colors, atom_metadata, residue_colors, segments, state_data
from .math3d import normalize
from .styling import atom_weights, current_tints


def _tint(base, tint):
    return base * (1 - tint[..., 3:4]) + tint[..., :3]


def _smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / np.maximum(edge1 - edge0, 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def cutaway_amount(points, camera, geometry, tight):
    """Per-point fade of the open cutaway, matching the native shader's cut_to()."""
    center, keep, window, _, axis = geometry
    keep = keep * camera.cutaway_surface_keep if tight else keep
    opening, softness, band = camera.cutaway_opening, camera.cutaway_softness, camera.cutaway_band
    if camera.cutaway_tunnel:
        d = points - center
        height = d @ axis
        r = np.linalg.norm(d - height[:, None] * axis, axis=1)
        radius = opening * window
        radial = 1 - _smoothstep(radius * (1 - softness), radius, r)
        return radial * _smoothstep(keep, keep + band, height)
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
    rim = np.clip(amount * (1 - amount) * 1.6, 0, 0.4)[:, None]
    mesh["colors"] = (mesh["colors"] * (1 - rim) + np.asarray(camera.cutaway_rim) * rim).astype(np.float32)
    return mesh


def tunnel_meshes(camera, height_px):
    """The tunnel wall and its depth rings, as the native tunnel pass draws them."""
    geometry = camera.cutaway_geometry()
    if geometry is None or not camera.cutaway_tunnel:
        return []
    center, keep, window, outer, axis = geometry
    floor = keep * camera.cutaway_surface_keep
    length = outer - floor
    if length <= 1:
        return []
    radius = camera.cutaway_opening * window
    u, v = _perpendiculars(axis)
    sides, steps = 72, 64
    angle = np.arange(sides) * 2 * np.pi / sides
    ring = np.cos(angle)[:, None] * u + np.sin(angle)[:, None] * v

    def band(h0, h1, scale, count):
        heights = np.linspace(h0, h1, count + 1)
        verts = (center + heights[:, None, None] * axis + scale * radius * ring[None]).reshape(-1, 3)
        return verts, _triangles(count + 1, sides), heights

    verts, faces, heights = band(floor, outer, 1.0, steps)
    below = np.repeat(outer - heights, sides)
    deep = np.clip(below / length, 0, 1)[:, None]
    base = (1 - deep) * np.array([0.42, 0.62, 0.78]) + deep * np.array([0.95, 0.73, 0.40])
    normals = np.tile(-ring, (steps + 1, 1))
    eye = camera.eye
    view = eye - verts
    view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-9)
    facing = np.abs(np.sum(normals * view, axis=1))
    ends = _smoothstep(0, 2, below) * _smoothstep(0, 2, length - below)
    near = _smoothstep(4, 22, np.linalg.norm(eye - verts, axis=1))
    alpha = camera.cutaway_wall * (0.35 + 0.65 * (1 - facing)) * ends * near
    pieces = [
        dict(
            vertices=verts.astype(np.float32),
            faces=faces.astype(np.int32),
            colors=base.astype(np.float32),
            opacity=np.round(alpha[faces].mean(1) * 20) / 20,
            normals=normals.astype(np.float32),
        )
    ]
    # Rings every `rings` Å from the outer surface; every second ring is brighter and wider.
    pixel = 2 * np.linalg.norm(eye - center) * np.tan(camera.fov / 2) / height_px
    spacing = camera.cutaway_rings
    for k in range(1, int(length // spacing) + 1):
        h = outer - k * spacing
        major = k % 2 == 0
        width = pixel * (4.0 if major else 2.4)
        rverts, rfaces, _ = band(h - width / 2, h + width / 2, 0.985, 1)
        rbelow = np.full(len(rverts), outer - h)
        rdeep = np.clip(rbelow / length, 0, 1)[:, None]
        rbase = (1 - rdeep) * np.array([0.42, 0.62, 0.78]) + rdeep * np.array([0.95, 0.73, 0.40])
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


def _perpendiculars(direction):
    ref = np.array([0.0, 1.0, 0.0]) if abs(direction[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(direction, ref)
    u /= np.linalg.norm(u)
    return u, np.cross(direction, u)


def _triangles(rings, sides):
    a = (np.arange(rings - 1)[:, None] * sides + np.arange(sides)).ravel()
    b = a + sides
    c = a - a % sides + (a + 1) % sides
    d = c + sides
    return np.stack((np.stack((a, c, d), 1), np.stack((a, d, b), 1)), 1).reshape(-1, 3)


def _cartoon(p, xyz, tint):
    rows = segments(p)
    if not len(rows):
        return None
    steps, sides = 16, 16
    ids = rows[:, :4].copy().view(np.uint32)
    # Evaluate the same guide interpolation and Catmull–Rom curve as the native shader.
    ga = state_data(p.topology, p._a)[:, 4:7]
    gb = state_data(p.topology, p._b, ga)[:, 4:7]
    gb = np.where((ga * gb).sum(1)[:, None] < 0, -gb, gb)
    guides = ga * (1 - p.atom_progress[:, None]) + gb * p.atom_progress[:, None]
    t = np.linspace(0, 1, steps + 1)[None, :, None]
    a, b, c, d = (xyz[ids[:, i]][:, None] for i in range(4))
    center = 0.5 * (
        2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t * t + (-a + 3 * b - 3 * c + d) * t * t * t
    )
    tangent = normalize(
        0.5 * ((-a + c) + 2 * (2 * a - 5 * b + 4 * c - d) * t + 3 * (-a + 3 * b - 3 * c + d) * t * t)
    )
    guide = (1 - t) * guides[ids[:, 1]][:, None] + t * guides[ids[:, 2]][:, None]
    u = normalize(guide - tangent * (guide * tangent).sum(-1, keepdims=True))
    v = normalize(np.cross(tangent, u))
    cartoon, ribbon = p.representation[:2]
    mix = cartoon / max(cartoon + ribbon, 1e-8)
    width = (1 - mix) * p.ribbon_width + mix * ((1 - t) * rows[:, 4, None, None] + t * rows[:, 5, None, None])
    height = (1 - mix) * 0.13 + mix * ((1 - t) * rows[:, 6, None, None] + t * rows[:, 7, None, None])
    cap = np.ones((len(rows), steps + 1, 1))
    for flag, progress in ((16, t), (17, 1 - t)):
        z = np.clip(progress / 0.12, 0, 1)
        cap *= np.where(rows[:, flag, None, None] > 0.5, z * z * (3 - 2 * z), 1)
    angle = np.arange(sides) * 2 * np.pi / sides
    offset = u[:, :, None] * np.cos(angle)[None, None, :, None] * width[:, :, None]
    offset += v[:, :, None] * np.sin(angle)[None, None, :, None] * height[:, :, None]
    vertices = (center[:, :, None] + cap[:, :, None] * offset).reshape(-1, 3)
    normal = normalize(
        u[:, :, None] * np.cos(angle)[None, None, :, None] / width[:, :, None]
        + v[:, :, None] * np.sin(angle)[None, None, :, None] / height[:, :, None]
    ).reshape(-1, 3)
    color_a = _tint(rows[:, 8:11], tint[ids[:, 1]])
    color_b = _tint(rows[:, 12:15], tint[ids[:, 2]])
    colors = np.repeat((1 - t) * color_a[:, None] + t * color_b[:, None], sides, axis=1).reshape(-1, 3)
    template = _triangles(steps + 1, sides)
    faces = (template[None] + np.arange(len(rows))[:, None, None] * (steps + 1) * sides).reshape(-1, 3)
    # One opacity per half-residue strip keeps selection boundaries crisp and avoids
    # introducing hundreds of distinct fade levels across a single backbone segment.
    owner = np.where((np.arange(steps) + 0.5) / steps < 0.5, ids[:, 1, None], ids[:, 2, None])
    alpha = np.repeat(p.atom_opacities[owner] * (cartoon + ribbon), sides * 2, axis=1).ravel()
    return vertices, faces, colors, alpha, normal


@lru_cache(maxsize=1)
def _sphere():
    # UV sphere with shared poles and closed caps.
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
    caps = [[top, j, (j + 1) % sides] for j in range(sides)]
    offset = (rings - 2) * sides
    caps += [[bottom, offset + (j + 1) % sides, offset + j] for j in range(sides)]
    return vertices, np.vstack((faces, caps)).astype(np.int32)


def _atoms(p, xyz, tint, weights):
    radii = atom_metadata(p)[:, 3] * p.atom_scale
    colors = _tint(atom_colors(p), tint)
    # Ball-and-stick fraction per atom, including detail atoms over a cartoon.
    opacity = p.atom_opacities * weights
    shown = np.flatnonzero(weights > 0)
    pieces = []
    if getattr(p, "_draw_atoms", True) and len(shown):
        verts, faces = _sphere()
        n, f = len(verts), len(faces)
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
        a, b = xyz[bonds[:, 0]], xyz[bonds[:, 1]]
        delta = b - a
        length = np.linalg.norm(delta, axis=1)
        direction = normalize(delta)
        # Trim hidden stick interiors at the atom surfaces, also during group fades.
        radius = p.bond_radius
        trim = (
            np.sqrt(np.maximum(radii[bonds] ** 2 - radius**2, 0))
            if getattr(p, "_draw_atoms", True)
            else np.zeros((len(bonds), 2))
        )
        exposed = length - trim.sum(1)
        a, b = a + direction * trim[:, :1], b - direction * trim[:, 1:]
        ref = np.where(np.abs(direction[:, 1:2]) > 0.9, [1, 0, 0], [0, 1, 0])
        u = normalize(np.cross(direction, ref))
        v = np.cross(direction, u)
        angle = np.arange(16) * 2 * np.pi / 16
        normal = u[:, None] * np.cos(angle)[None, :, None] + v[:, None] * np.sin(angle)[None, :, None]
        vertices = np.stack((a, b), 1)[:, :, None] + radius * normal[:, None]
        template = _triangles(2, 16)
        faces = (template[None] + np.arange(len(bonds))[:, None, None] * 32).reshape(-1, 3)
        alpha = np.minimum(opacity[bonds[:, 0]], opacity[bonds[:, 1]])
        alpha = np.where(exposed > 1e-8, alpha, 0)
        # Color interpolation follows the original bond, including trimmed ends.
        t = np.stack((trim[:, 0] / np.maximum(length, 1e-8), 1 - trim[:, 1] / np.maximum(length, 1e-8)), 1)
        col = (1 - t[..., None]) * colors[bonds[:, 0], None] + t[..., None] * colors[bonds[:, 1], None]
        pieces.append(
            (
                vertices.reshape(-1, 3),
                faces,
                np.repeat(col, 16, axis=1).reshape(-1, 3),
                np.repeat(alpha, len(template)),
                np.repeat(normal[:, None], 2, 1).reshape(-1, 3),
            )
        )
    return pieces


class MeshExporter:
    def __init__(self):
        self.surfaces = {}
        self.bases = {}

    def meshes(self, p, camera=None):
        """World-space meshes for one object, with the camera's cutaway and glow applied."""
        geometry = None if camera is None else camera.cutaway_geometry()
        if hasattr(p, "_export_mesh"):
            result = p._export_mesh()
            if geometry is not None:
                result = [apply_cutaway(mesh, camera, geometry, tight=False) for mesh in result]
            return result
        return self._molecule(p, camera, geometry)

    def _molecule(self, p, camera, geometry):
        from .surface import build_surface

        xyz, tint = p.positions, current_tints(p)
        pieces = []
        tight = []  # Cartoons, bases and surfaces are carved closer to a cutaway's target.
        if sum(p.representation[:2]) > 0:
            part = _cartoon(p, xyz, tint)
            if part is not None:
                pieces.append(part)
                tight.append(True)
            if any(r.is_nucleic for r in p.topology.residues) and np.any(p.base_style > 0):
                from .nucleic import BaseGeometry

                if p not in self.bases or not self.bases[p].matches(p):
                    self.bases[p] = BaseGeometry(p)
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
            vertices = mesh.vertices + np.sum(
                (xyz - mesh.reference)[mesh.neighbors] * mesh.weights[..., None], axis=1
            )
            base = (
                atom_metadata(p)[mesh.owners, :3]
                if p.color_scheme == "element"
                else residue_colors(p)[[a.residue_index for a in p.topology.atoms]][mesh.owners]
            )
            col = _tint(base, tint[mesh.owners])
            alpha = (p.atom_opacities[mesh.owners] * p.surface_opacity)[mesh.faces].min(1)
            # Recompute smooth normals for deformed surfaces.
            normals = np.zeros_like(vertices)
            tri = vertices[mesh.faces]
            face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            for i in range(3):
                np.add.at(normals, mesh.faces[:, i], face_normals)
            pieces.append((vertices, mesh.faces, col, alpha, normalize(normals)))
            tight.append(True)
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
        glow = p._glow() if hasattr(p, "_glow") else None
        if glow is not None and len(glow):
            result.append(glow_mesh(glow))
        return result


def glow_mesh(rows):
    """Halo spheres for glow rows (world center, color, radius, intensity)."""
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
