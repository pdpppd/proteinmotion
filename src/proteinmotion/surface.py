"""Cached molecular meshes and native GPU skinning; voxel SES is an approximation."""

from dataclasses import dataclass

import gemmi
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt, gaussian_filter
from scipy.spatial import cKDTree

from .geometry import atom_metadata, residue_colors
from .structure import coordinates


@dataclass(frozen=True, eq=False)
class SurfaceOptions:
    kind: str
    probe_radius: float
    resolution: float
    update: str
    max_voxels: int
    reference: object = None


def surface_options(
    *, kind="ses", probe_radius=1.4, resolution=0.7, update="rebuild", max_voxels=8_000_000, reference=None
):
    if kind not in ("vdw", "sas", "ses"):
        raise ValueError("Surface kind must be vdw, sas, or ses")
    if not np.isfinite(probe_radius) or probe_radius < 0:
        raise ValueError("probe_radius must be finite and nonnegative")
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be finite and positive")
    if update not in ("deform", "rebuild"):
        raise ValueError("Surface update must be deform or rebuild")
    if not isinstance(max_voxels, int) or max_voxels < 64:
        raise ValueError("max_voxels must be an integer >= 64")
    return SurfaceOptions(
        kind,
        float(probe_radius),
        float(resolution),
        update,
        max_voxels,
        None if reference is None else coordinates(reference),
    )


@dataclass(eq=False)
class SurfaceMesh:
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    owners: np.ndarray
    neighbors: np.ndarray
    weights: np.ndarray
    reference: np.ndarray
    grid_shape: tuple


def build_surface(protein, options=None, xyz=None):
    from skimage.measure import marching_cubes

    options = options or protein._surface_options or surface_options()
    xyz = coordinates(protein.positions if xyz is None else xyz, len(protein.topology.atoms))
    h = options.resolution
    radii = np.array([max(float(gemmi.Element(a.element).vdw_r), 1) for a in protein.topology.atoms])
    probe = options.probe_radius if options.kind != "vdw" else 0
    expanded = radii + probe
    lo = np.floor((xyz - expanded[:, None]).min(0) / h) * h - 3 * h
    hi = np.ceil((xyz + expanded[:, None]).max(0) / h) * h + 3 * h
    shape = tuple((np.ceil((hi - lo) / h).astype(int) + 1).tolist())
    if np.prod(shape, dtype=np.int64) > options.max_voxels:
        raise ValueError(
            f"Surface grid {shape} exceeds max_voxels={options.max_voxels}; increase resolution spacing"
        )
    field = np.full(shape, 4 * h + expanded.max(), np.float32)
    # Each atom visits only its local voxel box, not the full protein grid.
    for point, radius in zip(xyz, expanded):
        first = np.maximum(0, np.floor((point - radius - 2 * h - lo) / h).astype(int))
        last = np.minimum(shape, np.ceil((point + radius + 2 * h - lo) / h).astype(int) + 1)
        axes = [lo[k] + np.arange(first[k], last[k]) * h - point[k] for k in range(3)]
        distance = (
            np.sqrt(axes[0][:, None, None] ** 2 + axes[1][None, :, None] ** 2 + axes[2][None, None, :] ** 2)
            - radius
        )
        slices = tuple(slice(a, b) for a, b in zip(first, last))
        np.minimum(field[slices], distance, out=field[slices])
    if options.kind == "ses" and probe > 0:
        # Dilate by the probe, fill enclosed inaccessible cavities, then erode.
        solid = binary_fill_holes(field <= 0)
        inside_distance = distance_transform_edt(solid, sampling=h) - 0.5 * h
        field = gaussian_filter((probe - inside_distance).astype(np.float32), 0.5)
    vertices, faces, normals, _ = marching_cubes(field, 0, spacing=(h, h, h), allow_degenerate=False)
    vertices = np.asarray(vertices + lo, np.float32)
    # marching_cubes normals face decreasing values; our field is negative inside.
    normals = -np.asarray(normals, np.float32)
    geometric = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]
    )
    if np.sum(geometric * normals[faces].mean(1)) < 0:
        faces = faces[:, ::-1]
    distances, neighbors = cKDTree(xyz).query(vertices, k=min(4, len(xyz)))
    distances, neighbors = distances.reshape(len(vertices), -1), neighbors.reshape(len(vertices), -1)
    weights = 1 / (distances**2 + 4)
    weights /= weights.sum(1, keepdims=True)
    missing = 4 - neighbors.shape[1]
    neighbors = np.pad(neighbors, ((0, 0), (0, missing)), mode="edge")
    weights = np.pad(weights, ((0, 0), (0, missing)))
    return SurfaceMesh(
        vertices,
        np.ascontiguousarray(faces, np.uint32),
        normals,
        neighbors[:, 0].copy(),
        neighbors.astype(np.uint32),
        weights.astype(np.float32),
        xyz,
        shape,
    )


class SurfaceGPU:
    def __init__(self, renderer, protein):
        import wgpu

        self.renderer, self.protein = renderer, protein
        self.buffers = []
        self.key = self.options = None
        self.mesh = None
        if not hasattr(renderer, "surface_layout"):
            renderer.surface_layout = renderer.device.create_bind_group_layout(
                entries=[
                    {
                        "binding": 0,
                        "visibility": wgpu.ShaderStage.VERTEX,
                        "buffer": {"type": "read-only-storage"},
                    }
                ]
            )
            layout = renderer.device.create_pipeline_layout(
                bind_group_layouts=[renderer.camera_layout, renderer.object_layout, renderer.surface_layout]
            )
            formats = [
                ("float32x3", 0),
                ("float32x3", 12),
                ("uint32x4", 24),
                ("float32x4", 40),
                ("float32x3", 56),
                ("uint32", 68),
            ]
            buffers = [
                {
                    "array_stride": 72,
                    "step_mode": "vertex",
                    "attributes": [
                        {"format": fmt, "offset": offset, "shader_location": i}
                        for i, (fmt, offset) in enumerate(formats)
                    ],
                }
            ]
            renderer.surface_pipelines = {
                False: renderer._pipeline(
                    "mesh_vertex",
                    "mesh_fragment",
                    vertex_buffers=buffers,
                    layout_override=layout,
                    cull_mode="back",
                )
            }
            renderer._surface_pipeline_args = (layout, buffers)

    def update(self):

        p = self.protein
        options = p._surface_options
        if options is None:
            options = surface_options(reference=p.positions)
            p._surface_options = options
        key = (p._key_a, p._key_b, p._mix, id(p._controls)) if options.update == "rebuild" else None
        if self.mesh is None or self.options is not options or self.key != key:
            reference = p.positions if options.update == "rebuild" else options.reference
            self.mesh = build_surface(p, options, xyz=reference)
            self.options, self.key = options, key
            self.state_refs = (p._a, p._b, p._controls)
            self._upload()
        elif str(p.color_scheme) != self.color_key:
            self._upload()

    def _upload(self):
        import wgpu

        for buffer in self.buffers:
            buffer.destroy()
        self.buffers = []
        d, mesh, p = self.renderer.device, self.mesh, self.protein
        data = np.zeros((len(mesh.vertices), 18), np.float32)
        data[:, :3], data[:, 3:6] = mesh.vertices, mesh.normals
        data[:, 6:10].view(np.uint32)[:] = mesh.neighbors
        data[:, 10:14] = mesh.weights
        if p.color_scheme == "element":
            data[:, 14:17] = atom_metadata(p)[mesh.owners, :3]
        else:
            residue_ids = np.array([a.residue_index for a in p.topology.atoms])
            data[:, 14:17] = residue_colors(p)[residue_ids[mesh.owners]]
        data[:, 17].view(np.uint32)[:] = mesh.owners
        ref = np.zeros((len(mesh.reference), 4), np.float32)
        ref[:, :3] = mesh.reference
        self.vertices = d.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.VERTEX)
        self.indices = d.create_buffer_with_data(data=mesh.faces, usage=wgpu.BufferUsage.INDEX)
        self.reference = d.create_buffer_with_data(data=ref, usage=wgpu.BufferUsage.STORAGE)
        self.buffers = [self.vertices, self.indices, self.reference]
        self.binding = d.create_bind_group(
            layout=self.renderer.surface_layout,
            entries=[{"binding": 0, "resource": {"buffer": self.reference}}],
        )
        self.color_key = str(p.color_scheme)

    def draw(self, render_pass, binding, transparent):
        r = self.renderer
        # A cutaway exposes the inner walls of the closed surface, so keep back faces.
        cull = "none" if getattr(r, "cutaway_open", False) else "back"
        key = transparent if cull == "back" else (transparent, cull)
        if key not in r.surface_pipelines:
            layout, buffers = r._surface_pipeline_args
            r.surface_pipelines[key] = r._pipeline(
                "mesh_vertex",
                "mesh_transparent" if transparent else "mesh_fragment",
                vertex_buffers=buffers,
                transparent=transparent,
                layout_override=layout,
                cull_mode=cull,
            )
        render_pass.set_pipeline(r.surface_pipelines[key])
        render_pass.set_bind_group(1, binding)
        render_pass.set_bind_group(2, self.binding)
        render_pass.set_vertex_buffer(0, self.vertices)
        render_pass.set_index_buffer(self.indices, "uint32")
        render_pass.draw_indexed(self.mesh.faces.size)

    def close(self):
        for buffer in self.buffers:
            buffer.destroy()
