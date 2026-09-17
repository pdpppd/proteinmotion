"""Shared triangle meshes for density surfaces and slices."""

import numpy as np

from .math3d import rotation


class MeshObject:
    """Internal scene-object interface for cached triangle geometry."""

    def __init__(self, *, opacity=1, follow=None):
        from .protein import Protein

        if follow is not None and not isinstance(follow, Protein):
            raise TypeError("follow must be a Protein or None")
        self.position = np.zeros(3)
        self.orientation = np.eye(3)
        self.size = 1.0
        self.follow = follow
        self.set_opacity(opacity)
        self._mesh_key = self._mesh = None
        self.unlit = False

    def set_opacity(self, opacity):
        if not np.isfinite(opacity) or not 0 <= opacity <= 1:
            raise ValueError("Opacity must be finite and in [0, 1]")
        self.opacity = float(opacity)
        return self

    def shift(self, vector):
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError("shift needs a finite 3-vector")
        self.position += vector
        return self

    def rotate(self, angle, axis=(0, 1, 0)):
        old = self.orientation.copy()
        self.orientation = rotation(angle, axis) @ old
        self.position += self.size * ((old - self.orientation) @ self.positions.mean(0))
        return self

    def scale(self, factor):
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("Scale must be finite and positive")
        self.position += self.size * (1 - factor) * (self.orientation @ self.positions.mean(0))
        self.size *= factor
        return self

    @property
    def model_matrix(self):
        matrix = np.eye(4)
        matrix[:3, :3] = self.size * self.orientation
        matrix[:3, 3] = self.position
        return matrix if self.follow is None else self.follow.model_matrix @ matrix

    def snapshot(self):
        return {
            k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in vars(self).items()
            if k not in ("_mesh_key", "_mesh", "density")
        }

    def restore(self, state):
        for k, v in state.items():
            setattr(self, k, v.copy() if isinstance(v, np.ndarray) else v)

    @property
    def animate(self):
        from .density import DensityAnimate

        return DensityAnimate(self)

    def _export_mesh(self):
        mesh = self.mesh_data()
        if not len(mesh["faces"]):
            return []
        m = self.model_matrix
        return [
            dict(
                vertices=(mesh["vertices"] @ m[:3, :3].T + m[:3, 3]).astype(np.float32),
                faces=mesh["faces"],
                colors=mesh["colors"],
                normals=(mesh["normals"] @ self.orientation.T).astype(np.float32)
                if self.follow is None
                else (mesh["normals"] @ self.orientation.T @ self.follow.orientation.T).astype(np.float32),
                opacity=np.full(len(mesh["faces"]), self.opacity, np.float32),
                unlit=np.asarray(self.unlit),
            )
        ]


class MeshGPU:
    """One persistent vertex/index allocation per mesh; uniforms carry transforms."""

    def __init__(self, renderer, obj):
        import wgpu

        self.renderer, self.obj = renderer, obj
        self.vertices = self.indices = self.data = None
        self.uniform = renderer.device.create_buffer(
            size=112, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self.binding = renderer.device.create_bind_group(
            layout=renderer.mesh_layout, entries=[{"binding": 0, "resource": {"buffer": self.uniform}}]
        )

    def update(self):
        import wgpu

        mesh, r, obj = self.obj.mesh_data(), self.renderer, self.obj
        if mesh is not self.data:
            for b in (self.vertices, self.indices):
                if b is not None:
                    b.destroy()
            self.data = mesh
            if len(mesh["faces"]):
                data = np.column_stack((mesh["vertices"], mesh["normals"], mesh["colors"])).astype(np.float32)
                self.vertices = r.device.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.VERTEX)
                self.indices = r.device.create_buffer_with_data(
                    data=mesh["faces"].astype(np.uint32), usage=wgpu.BufferUsage.INDEX
                )
            else:
                self.vertices = self.indices = None
        u = np.zeros(28, np.float32)
        u[:16] = obj.model_matrix.T.ravel()
        u[17], u[26] = obj.opacity, float(obj.unlit)
        r.device.queue.write_buffer(self.uniform, 0, u)

    def draw(self, render_pass, transparent=False):
        if not self.data["faces"].size:
            return
        r = self.renderer
        render_pass.set_pipeline(r.mesh_pipelines[transparent])
        render_pass.set_bind_group(0, r.camera_group)
        render_pass.set_bind_group(1, self.binding)
        render_pass.set_vertex_buffer(0, self.vertices)
        render_pass.set_index_buffer(self.indices, "uint32")
        render_pass.draw_indexed(self.data["faces"].size)

    def close(self):
        for b in (self.vertices, self.indices, self.uniform):
            if b is not None:
                b.destroy()
