"""CPU mesh evaluation for Blender. Coordinates remain in world ångströms."""

from functools import lru_cache

import numpy as np

from .geometry import atom_metadata, residue_colors, segments, state_data
from .math3d import normalize
from .styling import current_tints


def _tint(base, tint):
    return base * (1 - tint[..., 3:4]) + tint[..., :3]


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
    faces = _triangles(rings - 1, sides)
    caps = [[top, j, (j + 1) % sides] for j in range(sides)]
    offset = (rings - 2) * sides
    caps += [[bottom, offset + (j + 1) % sides, offset + j] for j in range(sides)]
    return vertices, np.vstack((faces, caps)).astype(np.int32)


def _atoms(p, xyz, tint):
    meta = atom_metadata(p)
    radii = meta[:, 3] * p.atom_scale
    colors = _tint(meta[:, :3], tint)
    pieces = []
    if getattr(p, "_draw_atoms", True):
        verts, faces = _sphere()
        n, f = len(verts), len(faces)
        pieces.append(
            (
                (xyz[:, None] + radii[:, None, None] * verts).reshape(-1, 3),
                (faces[None] + np.arange(len(xyz))[:, None, None] * n).reshape(-1, 3),
                np.repeat(colors, n, axis=0),
                np.repeat(p.atom_opacities * p.representation[2], f),
                np.tile(verts, (len(xyz), 1)),
            )
        )
    bonds = p.topology.bonds
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
        alpha = np.minimum(p.atom_opacities[bonds[:, 0]], p.atom_opacities[bonds[:, 1]]) * p.representation[2]
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

    def meshes(self, p):
        if hasattr(p, "_export_mesh"):
            return p._export_mesh()
        from .surface import build_surface

        xyz, tint = p.positions, current_tints(p)
        pieces = []
        if sum(p.representation[:2]) > 0:
            part = _cartoon(p, xyz, tint)
            if part is not None:
                pieces.append(part)
        if p.representation[2] > 0:
            pieces.extend(_atoms(p, xyz, tint))
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
            palette = residue_colors(p)
            base = palette[[a.residue_index for a in p.topology.atoms]][mesh.owners]
            col = _tint(base, tint[mesh.owners])
            alpha = (p.atom_opacities[mesh.owners] * p.surface_opacity)[mesh.faces].min(1)
            # Recompute smooth normals for deformed surfaces.
            normals = np.zeros_like(vertices)
            tri = vertices[mesh.faces]
            face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            for i in range(3):
                np.add.at(normals, mesh.faces[:, i], face_normals)
            pieces.append((vertices, mesh.faces, col, alpha, normalize(normals)))
        m = p.model_matrix
        result = []
        for vertices, faces, colors, alpha, normals in pieces:
            result.append(
                dict(
                    vertices=np.asarray(vertices @ m[:3, :3].T + m[:3, 3], np.float32),
                    faces=np.asarray(faces, np.int32),
                    colors=np.asarray(colors, np.float32),
                    opacity=np.clip(alpha, 0, 1).astype(np.float32),
                    normals=np.asarray(normalize(normals @ m[:3, :3].T), np.float32),
                )
            )
        return result
