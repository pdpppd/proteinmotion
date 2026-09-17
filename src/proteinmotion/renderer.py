"""Native wgpu renderer. Metal on macOS; no browser or OpenGL context required."""

import sys
from collections import deque
from importlib.resources import files

import numpy as np
import wgpu

from .annotations import Annotation
from .geometry import atom_metadata, segments, state_data, sweep_grid
from .mesh import MeshGPU, MeshObject


class _MoleculeGPU:
    def __init__(self, renderer, protein):
        self.renderer, self.topology = renderer, protein.topology
        d = renderer.device
        self.buffers = []
        self.surface = None

        def storage(data):
            data = np.ascontiguousarray(data)
            if not data.nbytes:
                data = np.zeros(32, np.float32)
            b = d.create_buffer_with_data(
                data=data, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
            )
            self.buffers.append(b)
            return b

        self.states = [storage(np.zeros((len(protein.topology.atoms), 8), np.float32)) for _ in range(2)]
        self.keys = [None, None]
        # Hold cached arrays alive so Python cannot reuse an id for different data.
        self.state_references = [None, None]
        self.metadata = storage(atom_metadata(protein))
        self.metadata_key = id(protein._metadata_override)
        self.metadata_reference = protein._metadata_override
        self.bonds = storage(protein.topology.bonds)
        self.segment_data = segments(protein)
        self.segments = storage(self.segment_data)
        self.controls = storage(protein._controls)
        self.controls_key = id(protein._controls)
        self.controls_reference = protein._controls
        self.has_opacity_controls = bool(np.any(protein._controls[:, 4:6] != 1))
        self.appearance = storage(protein._appearance)
        self.appearance_reference = protein._appearance
        self.has_style_opacity = bool(np.any(protein._appearance[:, 12:14] != 1))
        self.shape_reference = protein._cartoon_scale
        self.color_key = str(protein.color_scheme)
        self.reference = state_data(self.topology, protein._a)[:, 4:7].copy()
        self.uniforms, self.bindings = [], []
        for _ in range(3):  # atom/bond, backbone, surface uniforms
            uniform = d.create_buffer(size=112, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
            self.buffers.append(uniform)
            self.uniforms.append(uniform)
            groups = {}
            for a in range(2):
                for b in range(2):
                    buffers = [
                        uniform,
                        self.states[a],
                        self.states[b],
                        self.metadata,
                        self.bonds,
                        self.segments,
                        self.controls,
                        self.appearance,
                    ]
                    groups[a, b] = d.create_bind_group(
                        layout=renderer.object_layout,
                        entries=[
                            {"binding": i, "resource": {"buffer": buf}} for i, buf in enumerate(buffers)
                        ],
                    )
            self.bindings.append(groups)

    def update(self, protein):
        queue = self.renderer.device.queue
        if protein.topology is not self.topology:
            raise ValueError("Topology changed after GPU upload; set secondary structure before rendering")
        wanted = [protein._key_a, protein._key_b]
        if id(protein._metadata_override) != self.metadata_key:
            queue.write_buffer(self.metadata, 0, atom_metadata(protein))
            self.metadata_key = id(protein._metadata_override)
            self.metadata_reference = protein._metadata_override
        if id(protein._controls) != self.controls_key:
            queue.write_buffer(self.controls, 0, protein._controls)
            self.controls_key = id(protein._controls)
            self.controls_reference = protein._controls
            self.has_opacity_controls = bool(np.any(protein._controls[:, 4:6] != 1))
        if protein._appearance is not self.appearance_reference:
            queue.write_buffer(self.appearance, 0, protein._appearance)
            self.appearance_reference = protein._appearance
            self.has_style_opacity = bool(np.any(protein._appearance[:, 12:14] != 1))
        chosen = []
        for key, coords in zip(wanted, (protein._a, protein._b)):
            if key in self.keys:
                slot = self.keys.index(key)
            else:
                slot = next(
                    i for i, existing in enumerate(self.keys) if existing not in wanted and i not in chosen
                )
                packed = state_data(self.topology, coords, self.reference)
                queue.write_buffer(self.states[slot], 0, packed)
                self.keys[slot] = key
                self.state_references[slot] = coords
            chosen.append(slot)
        if self.color_key != str(protein.color_scheme) or not np.array_equal(
            self.shape_reference, protein._cartoon_scale
        ):
            queue.write_buffer(self.segments, 0, segments(protein))
            self.color_key = str(protein.color_scheme)
            self.shape_reference = protein._cartoon_scale
        cartoon, ribbon, ball = protein.representation
        for i, opacity in enumerate((ball, cartoon + ribbon, protein.surface_opacity)):
            u = np.zeros(28, np.float32)
            u[:16] = protein.model_matrix.T.ravel()
            u[16:20] = [protein._mix, protein.opacity * opacity, protein.atom_scale, protein.bond_radius]
            u[20:24] = [
                cartoon / max(cartoon + ribbon, 1e-8),
                protein.ribbon_width,
                protein.size,
                float(getattr(protein, "_draw_atoms", True)),
            ]
            u[24:28] = [protein._color_mix, protein._opacity_mix, 0, 0]
            queue.write_buffer(self.uniforms[i], 0, u)
        if protein.surface_opacity > 0:
            if self.surface is None:
                from .surface import SurfaceGPU

                self.surface = SurfaceGPU(self.renderer, protein)
            self.surface.update()
        return [groups[tuple(chosen)] for groups in self.bindings]

    def close(self):
        if self.surface is not None:
            self.surface.close()
        for b in self.buffers:
            b.destroy()


class Renderer:
    def __init__(self, width=1920, height=1080, *, msaa=4, require_metal=None, readback_format="rgba"):
        if readback_format not in ("rgba", "nv12"):
            raise ValueError("readback_format must be rgba or nv12")
        if readback_format == "nv12" and (width % 2 or height % 2):
            raise ValueError("NV12 export requires even width and height")
        self.readback_format = readback_format
        self.width, self.height, self.msaa = width, height, msaa
        require_metal = sys.platform == "darwin" if require_metal is None else require_metal
        adapters = wgpu.gpu.enumerate_adapters_sync()
        if require_metal:
            candidates = [a for a in adapters if a.info["backend_type"] == "Metal"]
            if not candidates:
                raise RuntimeError("No Metal adapter found. Run proteinmotion doctor for diagnostics.")
            self.adapter = candidates[0]
        else:
            self.adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.adapter_info = dict(self.adapter.info)
        # State A/B, atom metadata, bonds, segments and per-atom motion/visibility controls.
        self.device = self.adapter.request_device_sync(
            required_limits={"max-storage-buffers-per-shader-stage": 8}
        )
        d = self.device
        self.camera_buffer = d.create_buffer(
            size=112, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self.camera_layout = d.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "uniform"},
                }
            ]
        )
        self.camera_group = d.create_bind_group(
            layout=self.camera_layout, entries=[{"binding": 0, "resource": {"buffer": self.camera_buffer}}]
        )
        self.object_layout = d.create_bind_group_layout(
            entries=[
                {
                    "binding": i,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "uniform" if i == 0 else "read-only-storage"},
                }
                for i in range(8)
            ]
        )
        layout = d.create_pipeline_layout(bind_group_layouts=[self.camera_layout, self.object_layout])
        shader = d.create_shader_module(
            code=files("proteinmotion").joinpath("shaders/molecule.wgsl").read_text()
        )

        def pipeline(
            vertex,
            fragment="surface_fragment",
            vertex_buffers=None,
            transparent=False,
            layout_override=None,
            cull_mode=None,
        ):
            targets = [{"format": "rgba8unorm"}]
            if transparent:
                additive = {"src_factor": "one", "dst_factor": "one", "operation": "add"}
                transmittance = {"src_factor": "zero", "dst_factor": "one-minus-src", "operation": "add"}
                targets = [
                    {"format": "rgba16float", "blend": {"color": additive, "alpha": additive}},
                    {"format": "r16float", "blend": {"color": transmittance, "alpha": transmittance}},
                ]
            return d.create_render_pipeline(
                layout=layout_override or layout,
                vertex={"module": shader, "entry_point": vertex, "buffers": vertex_buffers or []},
                fragment={"module": shader, "entry_point": fragment, "targets": targets},
                # Swept ribbon/bond triangles wind inward. Keep their exterior face once
                # when transparent, rather than accumulating both skins of a closed surface.
                primitive={
                    "topology": "triangle-list",
                    "cull_mode": cull_mode
                    or ("front" if transparent and vertex != "sphere_vertex" else "none"),
                },
                depth_stencil={
                    "format": "depth32float",
                    "depth_write_enabled": not transparent,
                    "depth_compare": "less-equal" if transparent else "less",
                },
                multisample={"count": msaa},
            )

        self.sphere_pipeline = pipeline("sphere_vertex", "sphere_fragment")
        self.bond_pipeline = pipeline("bond_vertex", "bond_fragment")
        self._ribbon_vertex_buffers = [
            {
                "array_stride": 8,
                "step_mode": "vertex",
                "attributes": [{"format": "float32x2", "offset": 0, "shader_location": 0}],
            }
        ]
        self.ribbon_pipeline = pipeline("ribbon_vertex", vertex_buffers=self._ribbon_vertex_buffers)
        self.mesh_layout = d.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "uniform"},
                }
            ]
        )
        mesh_layout = d.create_pipeline_layout(bind_group_layouts=[self.camera_layout, self.mesh_layout])
        mesh_buffers = [
            {
                "array_stride": 36,
                "step_mode": "vertex",
                "attributes": [
                    {"format": "float32x3", "offset": i * 12, "shader_location": i} for i in range(3)
                ],
            }
        ]
        self.mesh_pipelines = {
            transparent: pipeline(
                "density_vertex",
                "density_transparent" if transparent else "density_fragment",
                vertex_buffers=mesh_buffers,
                transparent=transparent,
                layout_override=mesh_layout,
                cull_mode="none",
            )
            for transparent in (False, True)
        }
        self._meshes = {}
        self._pipeline = pipeline
        self._oit_textures = []
        self.transparency_pipelines = None
        verts, indices = sweep_grid()
        self.vertices = d.create_buffer_with_data(data=verts, usage=wgpu.BufferUsage.VERTEX)
        self.indices = d.create_buffer_with_data(data=indices, usage=wgpu.BufferUsage.INDEX)
        self.n_indices = len(indices)
        self.texture = d.create_texture(
            size=(width, height, 1),
            format="rgba8unorm",
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT
            | wgpu.TextureUsage.COPY_SRC
            | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        self.view = self.texture.create_view()
        self.ms_texture = None
        if msaa > 1:
            self.ms_texture = d.create_texture(
                size=(width, height, 1),
                sample_count=msaa,
                format="rgba8unorm",
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            )
        self.ms_view = self.ms_texture.create_view() if self.ms_texture else self.view
        self.depth = d.create_texture(
            size=(width, height, 1),
            sample_count=msaa,
            format="depth32float",
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
        )
        self.depth_view = self.depth.create_view()
        self._molecules = {}
        self._overlay = None
        self._pending, self._free = deque(), []
        self.row_bytes = ((width * 4 + 255) // 256) * 256
        self.readback_rows = height
        self.nv12_buffer = None
        if readback_format == "nv12":
            self.row_bytes = ((width + 3) // 4) * 4
            self.readback_rows = height * 3 // 2
            self.nv12_buffer = d.create_buffer(
                size=self.row_bytes * self.readback_rows,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
            )
            conversion = d.create_shader_module(
                code=files("proteinmotion").joinpath("shaders/nv12.wgsl").read_text()
            )
            self.nv12_pipeline = d.create_compute_pipeline(
                layout="auto", compute={"module": conversion, "entry_point": "convert"}
            )
            self.nv12_group = d.create_bind_group(
                layout=self.nv12_pipeline.get_bind_group_layout(0),
                entries=[
                    {"binding": 0, "resource": self.view},
                    {"binding": 1, "resource": {"buffer": self.nv12_buffer}},
                ],
            )
        self._staging = []
        self._closed = False

    def _create_transparency(self):
        """Allocate/compile OIT only when a scene first contains a fade."""
        d = self.device
        self.transparency_pipelines = (
            self._pipeline("ribbon_vertex", "surface_transparent", self._ribbon_vertex_buffers, True),
            self._pipeline("bond_vertex", "bond_transparent", transparent=True),
            self._pipeline("sphere_vertex", "sphere_transparent", transparent=True),
        )
        for fmt in ("rgba16float", "r16float"):
            self._oit_textures.append(
                d.create_texture(
                    size=(self.width, self.height, 1),
                    sample_count=self.msaa,
                    format=fmt,
                    usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
                )
            )
        self._oit_views = [t.create_view() for t in self._oit_textures]
        code = files("proteinmotion").joinpath("shaders/transparency.wgsl").read_text()
        if self.msaa == 1:
            code = code.replace("texture_multisampled_2d", "texture_2d")
            code = code.replace(", @builtin(sample_index) sample:u32", "").replace("i32(sample)", "0")
        shader = d.create_shader_module(code=code)
        self.composite_pipeline = d.create_render_pipeline(
            layout="auto",
            vertex={"module": shader, "entry_point": "vertex"},
            fragment={
                "module": shader,
                "entry_point": "composite",
                "targets": [
                    {
                        "format": "rgba8unorm",
                        "blend": {
                            "color": {
                                "src_factor": "src-alpha",
                                "dst_factor": "one-minus-src-alpha",
                                "operation": "add",
                            },
                            "alpha": {"src_factor": "zero", "dst_factor": "one", "operation": "add"},
                        },
                    }
                ],
            },
            multisample={"count": self.msaa},
        )
        self.composite_group = d.create_bind_group(
            layout=self.composite_pipeline.get_bind_group_layout(0),
            entries=[{"binding": i, "resource": view} for i, view in enumerate(self._oit_views)],
        )

    def _draw_molecules(self, render_pass, draws, pipelines):
        ribbon, bond, sphere = pipelines
        render_pass.set_bind_group(0, self.camera_group)
        for protein, gpu, bindings in draws:
            if protein.surface_opacity * protein.opacity > 0 and gpu.surface is not None:
                gpu.surface.draw(render_pass, bindings[2], pipelines is self.transparency_pipelines)
            if sum(protein.representation[:2]) * protein.opacity > 0 and len(gpu.segment_data):
                render_pass.set_pipeline(ribbon)
                render_pass.set_bind_group(1, bindings[1])
                render_pass.set_vertex_buffer(0, self.vertices)
                render_pass.set_index_buffer(self.indices, "uint32")
                render_pass.draw_indexed(self.n_indices, len(gpu.segment_data))
            if protein.representation[2] * protein.opacity > 0:
                render_pass.set_bind_group(1, bindings[0])
                if len(protein.topology.bonds):
                    render_pass.set_pipeline(bond)
                    render_pass.draw(12 * 6, len(protein.topology.bonds))
                if getattr(protein, "_draw_atoms", True):
                    render_pass.set_pipeline(sphere)
                    render_pass.draw(6, len(protein.topology.atoms))

    def _commands(self, proteins, camera, background):
        camera.update_tracking()
        d = self.device
        u = np.zeros(28, np.float32)
        u[:16] = camera.matrix(self.width / self.height).T.ravel()
        u[16:20] = [*camera.eye, camera.distance - camera.radius * 1.05]
        u[20:24] = [*background, camera.distance + camera.radius * 1.5]
        u[24] = camera.depth_cue
        d.queue.write_buffer(self.camera_buffer, 0, u)
        draws = []
        mesh_draws = []
        annotations = []
        expanded = []
        for obj in proteins:
            if hasattr(obj, "_geometry_objects"):
                expanded.extend(obj._geometry_objects(camera, self.width, self.height))
            expanded.append(obj)
        for protein in expanded:
            if isinstance(protein, MeshObject):
                if protein not in self._meshes:
                    self._meshes[protein] = MeshGPU(self, protein)
                gpu = self._meshes[protein]
                gpu.update()
                mesh_draws.append(gpu)
                continue
            if isinstance(protein, Annotation):
                annotations.append(protein)
                continue
            if hasattr(protein, "_sync"):
                protein._sync()
            if protein not in self._molecules:
                self._molecules[protein] = _MoleculeGPU(self, protein)
            gpu = self._molecules[protein]
            draws.append((protein, gpu, gpu.update(protein)))
        transparent_draws = [
            draw
            for draw in draws
            if draw[1].has_opacity_controls
            or draw[1].has_style_opacity
            or any(
                0 < draw[0].opacity * fraction < 1
                for fraction in (
                    sum(draw[0].representation[:2]),
                    draw[0].representation[2],
                    draw[0].surface_opacity,
                )
            )
        ]
        has_transparency = bool(transparent_draws) or any(0 < g.obj.opacity < 1 for g in mesh_draws)
        if has_transparency and self.transparency_pipelines is None:
            self._create_transparency()
        encoder = d.create_command_encoder()
        attachment = {
            "view": self.ms_view,
            "resolve_target": self.view
            if self.msaa > 1 and not has_transparency and not annotations
            else None,
            "clear_value": (*map(float, background), 1.0),
            "load_op": "clear",
            "store_op": "store",
        }
        render_pass = encoder.begin_render_pass(
            color_attachments=[attachment],
            depth_stencil_attachment={
                "view": self.depth_view,
                "depth_clear_value": 1.0,
                "depth_load_op": "clear",
                "depth_store_op": "store" if has_transparency else "discard",
            },
        )
        self._draw_molecules(
            render_pass, draws, (self.ribbon_pipeline, self.bond_pipeline, self.sphere_pipeline)
        )
        for gpu in mesh_draws:
            gpu.draw(render_pass)
        render_pass.end()
        if has_transparency:
            trans = encoder.begin_render_pass(
                color_attachments=[
                    {
                        "view": view,
                        "clear_value": (value, value, value, value),
                        "load_op": "clear",
                        "store_op": "store",
                    }
                    for view, value in zip(self._oit_views, (0.0, 1.0))
                ],
                depth_stencil_attachment={"view": self.depth_view, "depth_read_only": True},
            )
            self._draw_molecules(trans, transparent_draws, self.transparency_pipelines)
            for gpu in mesh_draws:
                if 0 < gpu.obj.opacity < 1:
                    gpu.draw(trans, transparent=True)
            trans.end()
            composite = encoder.begin_render_pass(
                color_attachments=[
                    {
                        "view": self.ms_view,
                        "resolve_target": self.view if self.msaa > 1 and not annotations else None,
                        "load_op": "load",
                        "store_op": "store",
                    }
                ]
            )
            composite.set_pipeline(self.composite_pipeline)
            composite.set_bind_group(0, self.composite_group)
            composite.draw(3)
            composite.end()
        if annotations:
            if self._overlay is None:
                from .overlay import OverlayRenderer

                self._overlay = OverlayRenderer(self)
            self._overlay.draw(encoder, annotations, camera)
        return encoder

    def draw(self, proteins, camera, background):
        """GPU-only draw; no pixel readback (used by interactive preview)."""
        self.device.queue.submit([self._commands(proteins, camera, background).finish()])

    def enqueue(self, proteins, camera, background):
        """Submit a frame; return the oldest completed frame when three are in flight."""
        encoder = self._commands(proteins, camera, background)
        if self._free:
            buffer = self._free.pop()
        else:
            buffer = self.device.create_buffer(
                size=self.row_bytes * self.readback_rows,
                usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
            )
            self._staging.append(buffer)
        if self.readback_format == "nv12":
            convert = encoder.begin_compute_pass()
            convert.set_pipeline(self.nv12_pipeline)
            convert.set_bind_group(0, self.nv12_group)
            convert.dispatch_workgroups((self.width + 31) // 32, (self.height + 15) // 16)
            convert.end()
            encoder.copy_buffer_to_buffer(self.nv12_buffer, 0, buffer, 0, self.row_bytes * self.readback_rows)
        else:
            encoder.copy_texture_to_buffer(
                {"texture": self.texture},
                {"buffer": buffer, "bytes_per_row": self.row_bytes, "rows_per_image": self.height},
                (self.width, self.height, 1),
            )
        self.device.queue.submit([encoder.finish()])
        promise = buffer.map_async(wgpu.MapMode.READ)
        self._pending.append((buffer, promise))
        return self._read() if len(self._pending) >= 3 else None

    def _read(self):
        buffer, promise = self._pending.popleft()
        promise.sync_wait()
        raw = np.frombuffer(buffer.read_mapped(copy=False), dtype=np.uint8).reshape(
            self.readback_rows, self.row_bytes
        )
        if self.readback_format == "nv12":
            result = raw[:, : self.width].copy()
        else:
            result = raw[:, : self.width * 4].reshape(self.height, self.width, 4).copy()
        del raw
        buffer.unmap()
        self._free.append(buffer)
        return result

    def drain(self):
        while self._pending:
            yield self._read()

    def render(self, proteins, camera, background):
        if self._pending:
            raise RuntimeError("Drain queued frames before synchronous rendering")
        self.enqueue(proteins, camera, background)
        return self._read()

    def close(self):
        if self._closed:
            return
        for _ in self.drain():
            pass
        for gpu in self._meshes.values():
            gpu.close()
        for molecule in self._molecules.values():
            molecule.close()
        if self._overlay is not None:
            self._overlay.close()
        for resource in [
            self.camera_buffer,
            self.vertices,
            self.indices,
            self.texture,
            self.depth,
            self.ms_texture,
            self.nv12_buffer,
            *self._oit_textures,
            *self._staging,
        ]:
            if resource is not None:
                resource.destroy()
        self.device.destroy()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
