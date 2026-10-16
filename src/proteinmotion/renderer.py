"""Native wgpu renderer: Metal on macOS, Vulkan on Windows/Linux by default.

``Renderer`` turns the objects of one frame (proteins and nucleic acids, density meshes and
annotations) into pixels. ``ProteinScene.render_frame`` and ``ProteinScene.render`` create one
for stills and movies, ``preview`` draws with one into a window, ``StudioRenderer`` (studio.py)
subclasses it for HDR lighting and film effects, and the EEVEE backend keeps one to draw text
overlays.

Every frame is recorded into one command encoder, in this order:

1. Upload the camera uniform and, per molecule, whatever changed since the last frame
   (``_MoleculeGPU.update``). Coordinates live on the GPU as two keyframes and the vertex
   shaders interpolate between them, so a tween between keyframes uploads only uniforms.
2. Opaque pass, with MSAA and a depth buffer: cartoons and ribbons (tubes swept along a spline
   through the trace atoms), bonds (cylinders), atoms (ray-cast sphere impostors), molecular
   surfaces, nucleotide bases and density meshes.
3. Only when something is partly visible: a weighted blended order-independent transparency
   pass into two extra targets, then a pass that composites them over the opaque image.
4. Additive halos at the heads of threading wires, the ``_post_process`` hook (Studio's
   effects), then vector text and panels from overlay.py.
5. With supersampling, average the large image down to the output size.
6. Copy the image into a mappable buffer as RGBA rows, or convert it to NV12 for the video
   encoder first. Up to three frames are in flight, so the CPU encodes one frame while the GPU
   draws the next.

The shaders live in shaders/molecule.wgsl. The uniform layouts packed here must match its
``Camera`` and ``Object`` structs field by field.
"""

from collections import deque
from importlib.resources import files

import numpy as np
import wgpu

from ._gpu import select_adapter
from .annotations import Annotation
from .geometry import packed_metadata, secondary_weights, segments, state_data, sweep_grid
from .mesh import MeshGPU, MeshObject
from .styling import atom_weights

# Uniform block sizes of the structs in shaders/molecule.wgsl. Both structs hold only mat4x4 and
# vec4 fields, so WGSL adds no padding: each vec4 is 16 bytes and a mat4x4 is four of them.
# Camera is one mat4x4 and ten vec4 (Renderer._upload_camera lists the fields).
_CAMERA_UNIFORM_BYTES = 224
# Object is one mat4x4 and three vec4 (_MoleculeGPU._write_uniforms lists the fields).
_OBJECT_UNIFORM_BYTES = 112

# Quads around and along the tunnel wall. molecule.wgsl builds the wall from the vertex index
# alone, with the same TUNNEL_SIDES and TUNNEL_SEGMENTS, so the two must agree.
_TUNNEL_SIDES = 72
_TUNNEL_SEGMENTS = 64


class _MoleculeGPU:
    """GPU buffers and bind groups for one molecule, reused from frame to frame.

    Static data (atom colors and radii, bonds, cartoon segments, per-atom fade controls and
    appearance tracks) is uploaded once and rewritten only when the protein swaps in a new
    array. Coordinates are uploaded once per keyframe into one of two state slots, and the
    shaders interpolate between the slots, so tweening between two keyframes uploads nothing
    but uniforms.

    Most caches are keyed on array identity. Protein code replaces these arrays instead of
    editing them in place, so a new ``id`` means new data. Each cached array is also kept
    referenced here, so Python cannot free it and reuse its id for a different array.
    """

    def __init__(self, renderer, protein):
        self.renderer, self.topology = renderer, protein.topology
        d = renderer.device
        self.buffers = []  # Every buffer made here, destroyed together in close().
        self.surface = None  # SurfaceGPU, made when the molecular surface first shows.
        self.bases = None  # BaseGPU for nucleotide base shapes, made on first use.
        self.has_nucleic = any(r.is_nucleic for r in protein.topology.residues)

        def storage(data):
            data = np.ascontiguousarray(data)
            # WebGPU cannot bind an empty buffer. A molecule without bonds or cartoon
            # segments gets a small zero-filled placeholder instead.
            if not data.nbytes:
                data = np.zeros(32, np.float32)
            b = d.create_buffer_with_data(
                data=data, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
            )
            self.buffers.append(b)
            return b

        # Two keyframe slots of 8 float32 per atom (see geometry.state_data): position xyz,
        # unused, then on trace atoms the cartoon width axis xyz and the firmness of its twist.
        # keys[i] names the keyframe held in slot i.
        self.states = [storage(np.zeros((len(protein.topology.atoms), 8), np.float32)) for _ in range(2)]
        self.keys = [None, None]
        # Hold cached arrays alive so Python cannot reuse an id for different data.
        self.state_references = [None, None]
        # One vec4 per atom: ball-and-stick and detail RGBA8 colors as raw bits, unused, and the
        # display radius (van der Waals, Å).
        self.metadata = storage(packed_metadata(protein))
        self.metadata_key = id(protein._metadata_override)
        self.metadata_reference = protein._metadata_override
        self.metadata_color = str(protein.color_scheme)
        self.bonds = storage(protein.topology.bonds)
        # 80-byte cartoon segment records, one per pair of consecutive residues in a chain.
        self.segment_data = renderer._segment_data(protein)
        self.segments = storage(self.segment_data)
        # Per-atom controls, 8 float32: motion start, span and easing mode, unused, then opacity
        # before and after, fade start and span. Columns 4:6 are the opacity controls.
        self.controls = storage(protein._controls)
        self.controls_key = id(protein._controls)
        self.controls_reference = protein._controls
        self.has_opacity_controls = bool(np.any(protein._controls[:, 4:6] != 1))
        # Per-atom appearance tracks, 24 float32 (see styling.initial_appearance). Columns
        # 12:14 are style opacity before and after, 16:18 the detail weight before and after.
        self.appearance = storage(protein._appearance)
        self.appearance_reference = protein._appearance
        self.has_style_opacity = bool(np.any(protein._appearance[:, 12:14] != 1))
        self.has_detail = bool(np.any(protein._appearance[:, 16:18] > 0))
        self.partial_atoms = False
        self.shape_reference = protein._cartoon_scale
        self.secondary_reference = protein._secondary
        self.color_key = str(protein.color_scheme)
        self.backbone_radius = protein.backbone_radius
        # Width axes from the first keyframe. state_data flips each chain's axes in later
        # keyframes to agree with these in sign, so a ribbon keeps its orientation.
        self.reference = state_data(self.topology, protein._a, secondary=secondary_weights(protein))[
            :, 4:7
        ].copy()
        # One Object uniform per part of the molecule, so each part can carry its own share
        # of the opacity: 0 atoms and bonds, 1 cartoon or ribbon backbone, 2 molecular surface,
        # then 3 to 6 for the four nucleotide base styles (slabs, rings, sticks, ladder).
        # Each part gets four bind groups, one for every way of assigning the two state slots
        # to the shader's state_a and state_b; update() returns the pair that matches the
        # frame's keyframes.
        self.uniforms, self.bindings = [], []
        for _ in range(7 if self.has_nucleic else 3):  # atoms, backbone, surface, four base styles
            uniform = d.create_buffer(
                size=_OBJECT_UNIFORM_BYTES, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
            )
            self.buffers.append(uniform)
            self.uniforms.append(uniform)
            groups = {}
            for a in range(2):
                for b in range(2):
                    # Binding order matches @group(1) in molecule.wgsl.
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
        """Upload what changed since the last frame and return this frame's bind groups.

        Returns one bind group per part (see ``__init__``), each binding the two state slots
        that hold the protein's current keyframes as state_a and state_b.
        """
        queue = self.renderer.device.queue
        if protein.topology is not self.topology:
            raise ValueError("Topology changed after GPU upload; set secondary structure before rendering")
        # Secondary-structure animations assign a new array every frame, often with unchanged
        # values, so compare values as well as identity.
        secondary_changed = protein._secondary is not self.secondary_reference and not np.array_equal(
            protein._secondary, self.secondary_reference
        )
        self._update_atom_tracks(protein, queue, secondary_changed)
        # Growing detail atoms need the transparent pass, unlike a settled cartoon.
        cartoon, ribbon, ball = protein.representation
        weights = atom_weights(protein) if self.has_detail and ball < 1 else None
        self.partial_atoms = (weights is not None and bool(np.any((weights > 0) & (weights < 1)))) or (
            self.has_detail and 0 < protein.opacity < 1
        )
        chosen = self._update_states(protein, queue, secondary_changed)
        self._update_segments(protein, queue, secondary_changed)
        self._write_uniforms(protein, queue)
        self._update_surface_and_bases(protein, cartoon + ribbon)
        return [groups[tuple(chosen)] for groups in self.bindings]

    def _update_atom_tracks(self, protein, queue, secondary_changed):
        """Rewrite atom colors, fade controls and appearance tracks that the protein replaced."""
        # Colors come from an override (rulers, region highlights, threading wires) or from the
        # color scheme. The secondary-structure palette also changes with the assignment.
        if (
            id(protein._metadata_override) != self.metadata_key
            or self.metadata_color != str(protein.color_scheme)
            or (secondary_changed and protein._metadata_override is None)
        ):
            queue.write_buffer(self.metadata, 0, packed_metadata(protein))
            self.metadata_key = id(protein._metadata_override)
            self.metadata_reference = protein._metadata_override
            self.metadata_color = str(protein.color_scheme)
        if id(protein._controls) != self.controls_key:
            queue.write_buffer(self.controls, 0, protein._controls)
            self.controls_key = id(protein._controls)
            self.controls_reference = protein._controls
            self.has_opacity_controls = bool(np.any(protein._controls[:, 4:6] != 1))
        if protein._appearance is not self.appearance_reference:
            queue.write_buffer(self.appearance, 0, protein._appearance)
            self.appearance_reference = protein._appearance
            self.has_style_opacity = bool(np.any(protein._appearance[:, 12:14] != 1))
            self.has_detail = bool(np.any(protein._appearance[:, 16:18] > 0))

    def _update_states(self, protein, queue, secondary_changed):
        """Place the protein's two keyframes in state slots; return the slot of each."""
        wanted = [protein._key_a, protein._key_b]
        if secondary_changed:
            # Guides depend on secondary structure through the expected twist.
            self.keys = [None, None]
        # A keyframe already on the GPU stays in its slot. A new one goes into a slot whose
        # keyframe is no longer wanted and that the other end of the tween did not take.
        # When both ends are the same keyframe, both map to the same slot.
        chosen = []
        for key, coords in zip(wanted, (protein._a, protein._b)):
            if key in self.keys:
                slot = self.keys.index(key)
            else:
                slot = next(
                    i for i, existing in enumerate(self.keys) if existing not in wanted and i not in chosen
                )
                packed = state_data(self.topology, coords, self.reference, secondary_weights(protein))
                queue.write_buffer(self.states[slot], 0, packed)
                self.keys[slot] = key
                self.state_references[slot] = coords
            chosen.append(slot)
        return chosen

    def _update_segments(self, protein, queue, secondary_changed):
        """Rebuild cartoon segments when their colors, widths or secondary structure change."""
        if (
            self.color_key != str(protein.color_scheme)
            or self.backbone_radius != protein.backbone_radius
            or not np.array_equal(self.shape_reference, protein._cartoon_scale)
            or secondary_changed
        ):
            queue.write_buffer(self.segments, 0, self.renderer._segment_data(protein))
            self.color_key = str(protein.color_scheme)
            self.shape_reference = protein._cartoon_scale
            self.secondary_reference = protein._secondary
            self.backbone_radius = protein.backbone_radius

    def _write_uniforms(self, protein, queue):
        """Write the Object uniform of every part. These change every frame of a tween."""
        cartoon, ribbon, ball = protein.representation
        # Each part's share of the protein's opacity. The atom part uses 1, because the shader
        # applies the ball-and-stick fraction per atom (see stick_opacity).
        fractions = [1.0, cartoon + ribbon, protein.surface_opacity]
        if self.has_nucleic:
            fractions.extend((cartoon + ribbon) * protein.base_style)
        for i, opacity in enumerate(fractions):
            # struct Object in molecule.wgsl, 28 float32. Matrices are transposed because
            # WGSL stores them by column.
            #   [0:16]  model matrix
            #   [16:20] params: keyframe tween, opacity of this part, atom scale, bond radius (Å)
            #   [20:24] style: cartoon share of cartoon + ribbon in [0, 1], ribbon width (Å),
            #           object scale, 1 to draw atom spheres (rulers draw bonds only)
            #   [24:28] appearance: color, opacity and detail clocks, ball-and-stick fraction
            u = np.zeros(28, np.float32)
            u[:16] = protein.model_matrix.T.ravel()
            u[16:20] = [protein._mix, protein.opacity * opacity, protein.atom_scale, protein.bond_radius]
            u[20:24] = [
                cartoon / max(cartoon + ribbon, 1e-8),
                protein.ribbon_width,
                protein.size,
                float(getattr(protein, "_draw_atoms", True)),
            ]
            u[24:28] = [protein._color_mix, protein._opacity_mix, protein._detail_mix, ball]
            queue.write_buffer(self.uniforms[i], 0, u)

    def _update_surface_and_bases(self, protein, backbone):
        """Create or refresh the surface mesh and base shapes once they become visible.

        ``backbone`` is the cartoon plus ribbon fraction, which also scales the bases.
        """
        # Studio's density mode ray-marches its own surface (studio._GridSurface) instead.
        if protein.surface_opacity > 0 and not getattr(self.renderer, "gpu_surfaces", False):
            if self.surface is None:
                from .surface import SurfaceGPU

                self.surface = SurfaceGPU(self.renderer, protein)
            self.surface.update()
        if self.has_nucleic and backbone > 0 and np.any(protein.base_style > 0):
            if self.bases is None:
                from .nucleic import BaseGPU

                self.bases = BaseGPU(self.renderer, protein)
            else:
                self.bases.update(protein)

    def close(self):
        if self.bases is not None:
            self.bases.close()
        if self.surface is not None:
            self.surface.close()
        for b in self.buffers:
            b.destroy()


class Renderer:
    """Reusable native GPU renderer for scene objects.

    ``render`` returns one RGBA frame. ``enqueue`` and ``drain`` keep up to three frames in
    flight for movie export, and ``draw`` renders without copying pixels back (as the preview
    window does). Use it in a ``with`` block, or call ``close``, to release GPU memory.
    """

    # Subclasses can render the scene into another format and post-process it.
    SCENE_FORMAT = "rgba8unorm"
    # Keep the depth buffer after the opaque pass even when no later pass of this class reads
    # it. Studio samples it for ambient occlusion.
    KEEP_DEPTH = False
    # Reversed-Z maps the far plane to depth 0 and the near plane to 1. With float depth,
    # precision is then nearly uniform with distance instead of concentrated near the camera.
    REVERSED_Z = False

    def _depth_test(self, transparent=False):
        """Depth comparison: nearer fragments pass. Passes that only read depth accept ties."""
        if self.REVERSED_Z:
            return "greater-equal" if transparent else "greater"
        return "less-equal" if transparent else "less"

    def _segment_data(self, protein):
        """Cartoon segment records for the ribbon pass. Studio's classic cartoon overrides this."""
        return segments(protein)

    def _sweep_grid(self):
        """(u, v) vertices and triangle indices of one segment's tube: u along it, v around it."""
        return sweep_grid()

    def _shader_code(self):
        """WGSL source of the scene shaders. Studio edits it by string search, so keep names."""
        return files("proteinmotion").joinpath("shaders/molecule.wgsl").read_text()

    def _setup_scene_targets(self):
        """Scene passes render into these views; the base renderer draws straight to the output."""
        self.scene_ms_view, self.scene_view = self.ms_view, self.view

    def _scene_final(self, annotations):
        """True when the scene pass resolves straight into the image that is read back."""
        return not annotations

    def _post_process(self, encoder, camera, background, annotations):
        """Hook between the scene and the overlay passes."""

    def _prepare(self, encoder, draws):
        """Hook for GPU work (such as compute passes) before the scene passes."""

    def _draw_extra(self, render_pass, draws, transparent):
        """Hook for additional scene geometry in the opaque or transparent pass."""

    def __init__(
        self,
        width=1920,
        height=1080,
        *,
        msaa=4,
        supersampling=1,
        require_metal=None,
        readback_format="rgba",
        backend=None,
        adapter_name=None,
    ):
        if readback_format not in ("rgba", "nv12"):
            raise ValueError("readback_format must be rgba or nv12")
        # NV12 stores one chroma sample per 2 × 2 pixel block.
        if readback_format == "nv12" and (width % 2 or height % 2):
            raise ValueError("NV12 export requires even width and height")
        if supersampling not in (1, 2, 3, 4):
            raise ValueError("supersampling must be 1, 2, 3, or 4")
        # WebGPU guarantees only 1 and 4 samples for the render target formats. Check here,
        # before a GPU is opened, rather than fail later inside pipeline creation.
        if msaa not in (1, 4):
            raise ValueError("msaa must be 1 or 4")
        self.readback_format = readback_format
        # Frames are drawn at supersampling times the output size and averaged down, so
        # every scene target, overlay and post-processing pass uses the larger size.
        self.supersampling = supersampling
        self.output_width, self.output_height = width, height
        self.width, self.height, self.msaa = width * supersampling, height * supersampling, msaa
        self._open_device(backend, adapter_name, require_metal)
        # Each step uses what the earlier ones made: pipelines need the layouts, and the NV12
        # pass reads the output texture. Subclass hooks run inside these steps (_shader_code,
        # _sweep_grid, _setup_scene_targets), so a subclass sets its options before calling
        # this __init__.
        self._create_layouts()
        self._create_scene_pipelines()
        # One (u, v) grid of a swept tube, drawn once per cartoon segment by instancing.
        d = self.device
        verts, indices = self._sweep_grid()
        self.vertices = d.create_buffer_with_data(data=verts, usage=wgpu.BufferUsage.VERTEX)
        self.indices = d.create_buffer_with_data(data=indices, usage=wgpu.BufferUsage.INDEX)
        self.n_indices = len(indices)
        self._create_targets()
        self._molecules = {}  # Protein → _MoleculeGPU
        self._overlay = None  # OverlayRenderer, made for the first frame with annotations.
        # Readback buffers of frames in flight (with their map promises), and free ones to reuse.
        self._pending, self._free = deque(), []
        self._create_readback()
        self._staging = []  # Every readback buffer ever made, for close().
        self._closed = False

    def _open_device(self, backend, adapter_name, require_metal):
        """Choose a GPU and open a device on it."""
        self.adapter = select_adapter(backend=backend, adapter_name=adapter_name, require_metal=require_metal)
        self.adapter_info = dict(self.adapter.info)
        largest = self.adapter.limits["max-texture-dimension-2d"]
        if max(self.width, self.height) > largest:
            raise ValueError(
                f"{self.output_width}×{self.output_height} with supersampling={self.supersampling} "
                f"exceeds this GPU's {largest}-pixel textures"
            )
        # The object bind group holds seven storage buffers (state A/B, atom metadata, bonds,
        # segments, per-atom motion/visibility controls and appearance tracks), and molecular
        # surfaces bind one more for their reference vertices.
        self.device = self.adapter.request_device_sync(
            required_limits={"max-storage-buffers-per-shader-stage": 8}
        )

    def _create_layouts(self):
        """Camera uniform with its bind group (group 0), and the per-object layouts (group 1)."""
        d = self.device
        self.camera_buffer = d.create_buffer(
            size=_CAMERA_UNIFORM_BYTES, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self.cutaway_open = False
        self.draw_geometry_objects = True  # EEVEE's overlay pass draws only 2D annotation parts.
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
        # Molecules: the Object uniform, then the seven storage buffers of _MoleculeGPU.
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

    def _create_scene_pipelines(self):
        """Compile the scene shader and the opaque pipelines for molecules and meshes."""
        d = self.device
        self._molecule_pipeline_layout = d.create_pipeline_layout(
            bind_group_layouts=[self.camera_layout, self.object_layout]
        )
        self._shader = d.create_shader_module(code=self._shader_code())
        self._glow_pipeline = self._glow_buffer = None
        self.sphere_pipeline = self._pipeline("sphere_vertex", "sphere_fragment")
        self.bond_pipeline = self._pipeline("bond_vertex", "bond_fragment")
        # The ribbon pass reads the (u, v) of the swept tube grid from a vertex buffer.
        self._ribbon_vertex_buffers = [
            {
                "array_stride": 8,
                "step_mode": "vertex",
                "attributes": [{"format": "float32x2", "offset": 0, "shader_location": 0}],
            }
        ]
        self.ribbon_pipeline = self._pipeline("ribbon_vertex", vertex_buffers=self._ribbon_vertex_buffers)
        # Density meshes have no per-atom data: only an Object uniform in group 1.
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
        # Mesh vertices are 9 float32: position, normal and RGB color.
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
            transparent: self._pipeline(
                "density_vertex",
                "density_transparent" if transparent else "density_fragment",
                vertex_buffers=mesh_buffers,
                transparent=transparent,
                layout_override=mesh_layout,
                cull_mode="none",
            )
            for transparent in (False, True)
        }
        self._meshes = {}  # MeshObject → MeshGPU
        # Transparency targets and pipelines are made on the first frame that needs them.
        self._oit_textures = []
        self.transparency_pipelines = None

    def _pipeline(
        self,
        vertex,
        fragment="surface_fragment",
        vertex_buffers=None,
        transparent=False,
        layout_override=None,
        cull_mode=None,
    ):
        """Render pipeline for the scene shader, opaque or for the transparency pass.

        surface.py and nucleic.py also build their pipelines through this method.
        """
        targets = [{"format": self.SCENE_FORMAT}]
        if transparent:
            # Weighted blended OIT: add up weighted premultiplied colors in the first target,
            # and multiply (1 − alpha) of every layer into the second ("revealage").
            additive = {"src_factor": "one", "dst_factor": "one", "operation": "add"}
            transmittance = {"src_factor": "zero", "dst_factor": "one-minus-src", "operation": "add"}
            targets = [
                {"format": "rgba16float", "blend": {"color": additive, "alpha": additive}},
                {"format": "r16float", "blend": {"color": transmittance, "alpha": transmittance}},
            ]
        return self.device.create_render_pipeline(
            layout=layout_override or self._molecule_pipeline_layout,
            vertex={"module": self._shader, "entry_point": vertex, "buffers": vertex_buffers or []},
            fragment={"module": self._shader, "entry_point": fragment, "targets": targets},
            # Swept ribbon/bond triangles wind inward. Keep their exterior face once
            # when transparent, rather than accumulating both skins of a closed surface.
            primitive={
                "topology": "triangle-list",
                "cull_mode": cull_mode or ("front" if transparent and vertex != "sphere_vertex" else "none"),
            },
            # The transparency pass tests against the opaque depth but never writes it.
            depth_stencil={
                "format": "depth32float",
                "depth_write_enabled": not transparent,
                "depth_compare": self._depth_test(transparent),
            },
            multisample={"count": self.msaa},
        )

    def _create_targets(self):
        """Allocate the color, multisample and depth textures the scene is drawn into."""
        d = self.device
        # The single-sample image at the drawing size. MSAA resolves into it, overlays draw
        # on it, and without supersampling it is also the image that is read back.
        self.texture = d.create_texture(
            size=(self.width, self.height, 1),
            format="rgba8unorm",
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT
            | wgpu.TextureUsage.COPY_SRC
            | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        self.view = self.texture.create_view()
        self.output_texture, self.output_view = self.texture, self.view
        if self.supersampling > 1:
            self._create_downsample()
        self.ms_texture = None
        if self.msaa > 1:
            self.ms_texture = d.create_texture(
                size=(self.width, self.height, 1),
                sample_count=self.msaa,
                format="rgba8unorm",
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            )
        self.ms_view = self.ms_texture.create_view() if self.ms_texture else self.view
        self.depth = d.create_texture(
            size=(self.width, self.height, 1),
            sample_count=self.msaa,
            format="depth32float",
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        self.depth_view = self.depth.create_view()
        self._setup_scene_targets()

    def _create_readback(self):
        """Size the CPU readback rows and, for NV12, compile the RGBA-to-NV12 compute pass."""
        d = self.device
        # copy_texture_to_buffer needs each row to start at a multiple of 256 bytes.
        self.row_bytes = ((self.output_width * 4 + 255) // 256) * 256
        self.readback_rows = self.output_height
        self.nv12_buffer = None
        if self.readback_format == "nv12":
            # The shader writes whole 32-bit words of four luma bytes, so rows are padded to a
            # multiple of 4 bytes. A full-height luma plane is followed by a half-height plane
            # of interleaved U and V.
            self.row_bytes = ((self.output_width + 3) // 4) * 4
            self.readback_rows = self.output_height * 3 // 2
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
                    {"binding": 0, "resource": self.output_view},
                    {"binding": 1, "resource": {"buffer": self.nv12_buffer}},
                ],
            )

    def _create_downsample(self):
        """Output-size texture and the pass that averages supersampled pixels into it."""
        d = self.device
        self.output_texture = d.create_texture(
            size=(self.output_width, self.output_height, 1),
            format="rgba8unorm",
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT
            | wgpu.TextureUsage.COPY_SRC
            | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        self.output_view = self.output_texture.create_view()
        # Write the supersampling factor into the shader in place of its FACTOR placeholder.
        code = files("proteinmotion").joinpath("shaders/downsample.wgsl").read_text()
        shader = d.create_shader_module(code=code.replace("FACTOR", str(self.supersampling)))
        self.downsample_pipeline = d.create_render_pipeline(
            layout="auto",
            vertex={"module": shader, "entry_point": "vertex"},
            fragment={"module": shader, "entry_point": "fragment", "targets": [{"format": "rgba8unorm"}]},
            primitive={"topology": "triangle-list"},
        )
        self.downsample_group = d.create_bind_group(
            layout=self.downsample_pipeline.get_bind_group_layout(0),
            entries=[{"binding": 0, "resource": self.view}],
        )

    def _downsample(self, encoder):
        """Average the drawn image into the output texture, one output pixel per fragment."""
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.output_view,
                    "load_op": "clear",
                    "store_op": "store",
                    "clear_value": (0, 0, 0, 0),
                }
            ]
        )
        render_pass.set_pipeline(self.downsample_pipeline)
        render_pass.set_bind_group(0, self.downsample_group)
        render_pass.draw(3)  # One triangle that covers the whole screen.
        render_pass.end()

    def _create_transparency(self):
        """Allocate/compile OIT only when a scene first contains a fade."""
        d = self.device
        # Same order as the opaque pipelines passed to _draw_molecules: ribbon, bond, sphere.
        self.transparency_pipelines = (
            self._pipeline("ribbon_vertex", "surface_transparent", self._ribbon_vertex_buffers, True),
            self._pipeline("bond_vertex", "bond_transparent", transparent=True),
            self._pipeline("sphere_vertex", "sphere_transparent", transparent=True),
        )
        # Accumulated color and weight, then revealage, multisampled like the scene.
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
            # Without MSAA the targets are plain textures and there is only sample 0.
            code = code.replace("texture_multisampled_2d", "texture_2d")
            code = code.replace(", @builtin(sample_index) sample:u32", "").replace("i32(sample)", "0")
        shader = d.create_shader_module(code=code)
        # The composite blends the averaged transparent color over the opaque image with
        # alpha = 1 − revealage, and leaves the image's own alpha unchanged.
        self.composite_pipeline = d.create_render_pipeline(
            layout="auto",
            vertex={"module": shader, "entry_point": "vertex"},
            fragment={
                "module": shader,
                "entry_point": "composite",
                "targets": [
                    {
                        "format": self.SCENE_FORMAT,
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
        """Draw bases, surfaces, cartoons, bonds and atoms with (ribbon, bond, sphere) pipelines.

        Each draw's bind groups are indexed as in _MoleculeGPU: 0 atoms and bonds, 1 backbone,
        2 surface, 3 onwards base styles.
        """
        ribbon, bond, sphere = pipelines
        transparent = pipelines is self.transparency_pipelines
        render_pass.set_bind_group(0, self.camera_group)
        for protein, gpu, bindings in draws:
            backbone = sum(protein.representation[:2])
            if backbone * protein.opacity > 0 and gpu.bases is not None:
                gpu.bases.draw(render_pass, bindings, protein, transparent)
            if protein.surface_opacity * protein.opacity > 0 and gpu.surface is not None:
                gpu.surface.draw(render_pass, bindings[2], transparent)
            if backbone * protein.opacity > 0 and len(gpu.segment_data):
                # One instance of the tube grid per cartoon segment.
                render_pass.set_pipeline(ribbon)
                render_pass.set_bind_group(1, bindings[1])
                render_pass.set_vertex_buffer(0, self.vertices)
                render_pass.set_index_buffer(self.indices, "uint32")
                render_pass.draw_indexed(self.n_indices, len(gpu.segment_data))
            # Detail atoms (ligands, ions, side chains) draw over any representation.
            if (protein.representation[2] > 0 or gpu.has_detail) and protein.opacity > 0:
                render_pass.set_bind_group(1, bindings[0])
                if len(protein.topology.bonds):
                    # A 12-sided cylinder per bond, two triangles (six vertices) per side.
                    render_pass.set_pipeline(bond)
                    render_pass.draw(12 * 6, len(protein.topology.bonds))
                if getattr(protein, "_draw_atoms", True):
                    # One camera-facing quad per atom; the fragment shader ray-casts the sphere.
                    render_pass.set_pipeline(sphere)
                    render_pass.draw(6, len(protein.topology.atoms))

    def _commands(self, proteins, camera, background):
        """Record one frame's GPU work into a new command encoder and return it, unsubmitted."""
        self._upload_camera(camera, background)
        draws, mesh_draws, annotations, expanded = self._collect_draws(proteins, camera)
        transparent_draws = [draw for draw in draws if self._needs_transparency(draw[0], draw[1])]
        has_transparency = bool(transparent_draws) or any(0 < g.obj.opacity < 1 for g in mesh_draws)
        glow = self._glow_instances(expanded)
        if has_transparency and self.transparency_pipelines is None:
            self._create_transparency()
        encoder = self.device.create_command_encoder()
        self._prepare(encoder, draws)
        # Only the last pass that draws into the multisampled scene resolves it into the
        # single-sample image. Later passes need the depth buffer, so keep it for them.
        self._draw_opaque(
            encoder,
            draws,
            mesh_draws,
            background,
            resolve=self.msaa > 1
            and not has_transparency
            and self._scene_final(annotations)
            and glow is None,
            keep_depth=has_transparency or glow is not None or self.KEEP_DEPTH,
        )
        if has_transparency:
            self._draw_transparent(
                encoder,
                transparent_draws,
                mesh_draws,
                resolve=self.msaa > 1 and self._scene_final(annotations) and glow is None,
            )
        if glow is not None:
            self._draw_glow(encoder, glow, resolve=self.msaa > 1 and self._scene_final(annotations))
        self._post_process(encoder, camera, background, annotations)
        if annotations:
            if self._overlay is None:
                from .overlay import OverlayRenderer

                self._overlay = OverlayRenderer(self)
            self._overlay.draw(encoder, annotations, camera)
        return encoder

    def _upload_camera(self, camera, background):
        """Write the camera uniform and remember this frame's view-projection matrix.

        Also sets ``cutaway_open`` and ``tunnel_wall``, which the later passes read.
        """
        camera.update_tracking()
        d = self.device
        # struct Camera in molecule.wgsl, 56 float32. Distances are in Å.
        #   [0:16]  view-projection matrix, transposed because WGSL stores matrices by column
        #   [16:20] eye position, fog start distance
        #   [20:24] background RGB, fog end distance
        #   [24:28] depth cue strength, cutaway rim color RGB
        #   [28:32] cutaway target center, radius kept around it for atoms, bonds and meshes
        #   [32:36] cutaway opening in [0, 1], window radius at the target, edge softness, depth band
        #   [36:40] radius kept around the target for cartoons, ribbons, bases and surfaces
        #   [40:44] tunnel mode, outer-surface distance from the target, wall opacity, ring spacing
        #   [44:48] tunnel direction from the target
        #   [48:56] camera right and up vectors, for lights that follow the view
        u = np.zeros(56, np.float32)
        self._last_vp = camera.matrix(self.width / self.height)
        if self.REVERSED_Z:
            # Replace clip z by w − z, so the depth after division becomes 1 − depth.
            flip = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 1], [0, 0, 0, 1]], float)
            self._last_vp = flip @ self._last_vp
        self._last_camera = camera
        u[:16] = self._last_vp.T.ravel()
        # Camera basis, for lights that follow the view.
        forward = camera.target - camera.eye
        forward /= max(np.linalg.norm(forward), 1e-9)
        # Right is perpendicular to world up (+y). Looking straight up or down, fall back to +x.
        right = np.cross(forward, [0.0, 1.0, 0.0])
        right = right / np.linalg.norm(right) if np.linalg.norm(right) > 1e-6 else np.array([1.0, 0, 0])
        u[48:51], u[52:55] = right, np.cross(right, forward)
        # Fog runs from just in front of the molecule's bounding sphere to past its far side.
        # EEVEE uses the same range.
        u[16:20] = [*camera.eye, camera.distance - camera.radius * 1.05]
        u[20:24] = [*background, camera.distance + camera.radius * 1.5]
        u[24] = camera.depth_cue
        cutaway = camera.cutaway_geometry()
        self.cutaway_open = cutaway is not None
        if cutaway is not None:
            center, keep, window, outer, axis = cutaway
            u[44:47] = axis
            u[25:28] = camera.cutaway_rim
            u[28:32] = [*center, keep]
            u[32:36] = [camera.cutaway_opening, window, camera.cutaway_softness, camera.cutaway_band]
            u[36] = keep * camera.cutaway_surface_keep
            u[40:44] = [float(camera.cutaway_tunnel), outer, camera.cutaway_wall, camera.cutaway_rings]
        # Draw a tunnel wall only when the tunnel reaches more than 1 Å past the radius where
        # surfaces stop being carved.
        self.tunnel_wall = (
            cutaway is not None
            and camera.cutaway_tunnel
            and cutaway[3] > cutaway[1] * camera.cutaway_surface_keep + 1
        )
        d.queue.write_buffer(self.camera_buffer, 0, u)

    def _collect_draws(self, proteins, camera):
        """Sort a frame's objects into molecule draws, mesh draws and annotations.

        Returns ``(draws, mesh_draws, annotations, expanded)``. Each draw is
        ``(protein, _MoleculeGPU, bind groups)``, with its GPU buffers already updated.
        ``expanded`` is the frame's objects plus the 3D parts that annotations add.
        """
        draws = []
        mesh_draws = []
        annotations = []
        expanded = []
        for obj in proteins:
            # Annotations such as Distance add 3D parts (ruler cylinders) that are drawn in the
            # scene, where molecules can hide them. The annotation itself stays in the list for
            # its 2D parts.
            if hasattr(obj, "_geometry_objects") and self.draw_geometry_objects:
                expanded.extend(obj._geometry_objects(camera, self.width, self.height))
            expanded.append(obj)
        for obj in expanded:
            if isinstance(obj, MeshObject):
                if obj not in self._meshes:
                    self._meshes[obj] = MeshGPU(self, obj)
                gpu = self._meshes[obj]
                gpu.update()
                mesh_draws.append(gpu)
                continue
            if isinstance(obj, Annotation):
                annotations.append(obj)
                continue
            # Helper objects (threading wires, region highlights) follow their parent here.
            if hasattr(obj, "_sync"):
                obj._sync()
            if obj not in self._molecules:
                self._molecules[obj] = _MoleculeGPU(self, obj)
            gpu = self._molecules[obj]
            draws.append((obj, gpu, gpu.update(obj)))
        return draws, mesh_draws, annotations, expanded

    def _needs_transparency(self, protein, gpu):
        """True when some part of this molecule is partly visible and needs the OIT pass."""
        # An open cutaway fades geometry near the window, so every molecule may be partial.
        if gpu.has_opacity_controls or gpu.has_style_opacity or gpu.partial_atoms or self.cutaway_open:
            return True
        backbone = sum(protein.representation[:2])
        fractions = (
            backbone,
            protein.representation[2],
            protein.surface_opacity,
            *(backbone * protein.base_style if gpu.has_nucleic else ()),
        )
        return any(0 < protein.opacity * fraction < 1 for fraction in fractions)

    @staticmethod
    def _glow_instances(objects):
        """Stack the halo rows of every object that glows, or None when nothing does.

        Each row is 8 float32: world center, RGB color, radius and intensity.
        """
        glows = [
            g for obj in objects if hasattr(obj, "_glow") for g in [obj._glow()] if g is not None and len(g)
        ]
        return np.ascontiguousarray(np.concatenate(glows), np.float32) if glows else None

    def _draw_opaque(self, encoder, draws, mesh_draws, background, *, resolve, keep_depth):
        """Clear the scene and depth, then draw every fully opaque fragment."""
        attachment = {
            "view": self.scene_ms_view,
            "resolve_target": self.scene_view if resolve else None,
            "clear_value": (*map(float, background), 1.0),
            "load_op": "clear",
            "store_op": "store",
        }
        render_pass = encoder.begin_render_pass(
            color_attachments=[attachment],
            depth_stencil_attachment={
                "view": self.depth_view,
                "depth_clear_value": 0.0 if self.REVERSED_Z else 1.0,
                "depth_load_op": "clear",
                "depth_store_op": "store" if keep_depth else "discard",
            },
        )
        self._draw_molecules(
            render_pass, draws, (self.ribbon_pipeline, self.bond_pipeline, self.sphere_pipeline)
        )
        self._draw_extra(render_pass, draws, False)
        for gpu in mesh_draws:
            gpu.draw(render_pass)
        render_pass.end()

    def _draw_transparent(self, encoder, transparent_draws, mesh_draws, *, resolve):
        """Accumulate partly visible fragments (weighted blended OIT), then composite them."""
        d = self.device
        # Accumulation starts at 0 and revealage at 1 (nothing in front of the background).
        # Depth is read-only: opaque geometry hides transparent fragments behind it, and
        # transparent fragments never hide each other.
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
        self._draw_extra(trans, transparent_draws, True)
        if self.tunnel_wall:
            if not hasattr(self, "tunnel_pipeline"):
                # The wall is procedural: it reads only the camera uniform.
                layout = d.create_pipeline_layout(bind_group_layouts=[self.camera_layout])
                self.tunnel_pipeline = self._pipeline(
                    "tunnel_vertex",
                    "tunnel_transparent",
                    transparent=True,
                    layout_override=layout,
                    cull_mode="none",
                )
            trans.set_pipeline(self.tunnel_pipeline)
            trans.set_bind_group(0, self.camera_group)
            trans.draw(_TUNNEL_SIDES * _TUNNEL_SEGMENTS * 6)
        for gpu in mesh_draws:
            if 0 < gpu.obj.opacity < 1:
                gpu.draw(trans, transparent=True)
        trans.end()
        composite = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.scene_ms_view,
                    "resolve_target": self.scene_view if resolve else None,
                    "load_op": "load",
                    "store_op": "store",
                }
            ]
        )
        composite.set_pipeline(self.composite_pipeline)
        composite.set_bind_group(0, self.composite_group)
        composite.draw(3)  # One triangle that covers the whole screen.
        composite.end()

    def _draw_glow(self, encoder, glow, *, resolve):
        """Additive halos after molecules and transparency; depth-tested, never depth-writing."""
        d = self.device
        if self._glow_pipeline is None:
            layout = d.create_pipeline_layout(bind_group_layouts=[self.camera_layout])
            # Add light to the color and leave the image's alpha unchanged.
            additive = {"src_factor": "one", "dst_factor": "one", "operation": "add"}
            keep = {"src_factor": "zero", "dst_factor": "one", "operation": "add"}
            self._glow_pipeline = d.create_render_pipeline(
                layout=layout,
                vertex={
                    "module": self._shader,
                    "entry_point": "glow_vertex",
                    # One instance per halo row: center xyz, color RGB, (radius, intensity).
                    "buffers": [
                        {
                            "array_stride": 32,
                            "step_mode": "instance",
                            "attributes": [
                                {"format": "float32x3", "offset": 0, "shader_location": 0},
                                {"format": "float32x3", "offset": 12, "shader_location": 1},
                                {"format": "float32x2", "offset": 24, "shader_location": 2},
                            ],
                        }
                    ],
                },
                fragment={
                    "module": self._shader,
                    "entry_point": "glow_fragment",
                    "targets": [{"format": self.SCENE_FORMAT, "blend": {"color": additive, "alpha": keep}}],
                },
                primitive={"topology": "triangle-list", "cull_mode": "none"},
                depth_stencil={
                    "format": "depth32float",
                    "depth_write_enabled": False,
                    "depth_compare": self._depth_test(True),
                },
                multisample={"count": self.msaa},
            )
        # Grow the instance buffer only when a frame has more halos than it holds.
        if self._glow_buffer is None or self._glow_buffer.size < glow.nbytes:
            if self._glow_buffer is not None:
                self._glow_buffer.destroy()
            self._glow_buffer = d.create_buffer(
                size=max(glow.nbytes, 1024), usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST
            )
        d.queue.write_buffer(self._glow_buffer, 0, glow)
        glow_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.scene_ms_view,
                    "resolve_target": self.scene_view if resolve else None,
                    "load_op": "load",
                    "store_op": "store",
                }
            ],
            depth_stencil_attachment={"view": self.depth_view, "depth_read_only": True},
        )
        glow_pass.set_pipeline(self._glow_pipeline)
        glow_pass.set_bind_group(0, self.camera_group)
        glow_pass.set_vertex_buffer(0, self._glow_buffer)
        glow_pass.draw(6, len(glow))  # One camera-facing quad per halo.
        glow_pass.end()

    def draw(self, proteins, camera, background):
        """GPU-only draw; no pixel readback (used by interactive preview)."""
        self.device.queue.submit([self._commands(proteins, camera, background).finish()])

    def enqueue(self, proteins, camera, background):
        """Submit a frame; return the oldest completed frame when three are in flight."""
        encoder = self._commands(proteins, camera, background)
        if self.supersampling > 1:
            self._downsample(encoder)
        # Reuse a readback buffer that _read has unmapped, or make another one.
        if self._free:
            buffer = self._free.pop()
        else:
            buffer = self.device.create_buffer(
                size=self.row_bytes * self.readback_rows,
                usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
            )
            self._staging.append(buffer)
        if self.readback_format == "nv12":
            # Each invocation packs a 4 × 2 pixel tile and a workgroup is 8 × 8 invocations,
            # so one workgroup covers 32 × 16 output pixels.
            convert = encoder.begin_compute_pass()
            convert.set_pipeline(self.nv12_pipeline)
            convert.set_bind_group(0, self.nv12_group)
            convert.dispatch_workgroups((self.output_width + 31) // 32, (self.output_height + 15) // 16)
            convert.end()
            encoder.copy_buffer_to_buffer(self.nv12_buffer, 0, buffer, 0, self.row_bytes * self.readback_rows)
        else:
            encoder.copy_texture_to_buffer(
                {"texture": self.output_texture},
                {"buffer": buffer, "bytes_per_row": self.row_bytes, "rows_per_image": self.output_height},
                (self.output_width, self.output_height, 1),
            )
        self.device.queue.submit([encoder.finish()])
        promise = buffer.map_async(wgpu.MapMode.READ)
        self._pending.append((buffer, promise))
        # Three frames in flight let the GPU draw ahead while the caller encodes video.
        return self._read() if len(self._pending) >= 3 else None

    def _read(self):
        """Wait for the oldest frame in flight and return its pixels without row padding.

        RGBA frames are (height, width, 4) uint8; NV12 frames are (height × 3 / 2, width) uint8.
        """
        buffer, promise = self._pending.popleft()
        promise.sync_wait()
        raw = np.frombuffer(buffer.read_mapped(copy=False), dtype=np.uint8).reshape(
            self.readback_rows, self.row_bytes
        )
        width, height = self.output_width, self.output_height
        # Copy out of the mapped memory, so the buffer can be unmapped and reused.
        if self.readback_format == "nv12":
            result = raw[:, :width].copy()
        else:
            result = raw[:, : width * 4].reshape(height, width, 4).copy()
        del raw
        buffer.unmap()
        self._free.append(buffer)
        return result

    def drain(self):
        """Yield the frames still in flight, oldest first."""
        while self._pending:
            yield self._read()

    def render(self, proteins, camera, background):
        """Draw one frame and return its pixels as an RGBA or NV12 uint8 array."""
        # _read returns the oldest frame in flight, so earlier frames would come back instead.
        if self._pending:
            raise RuntimeError("Drain queued frames before synchronous rendering")
        self.enqueue(proteins, camera, background)
        return self._read()

    def close(self):
        """Release GPU buffers and renderer resources. Calling it again does nothing."""
        if self._closed:
            return
        # Wait for frames in flight before destroying the buffers they are copied into.
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
            # Without supersampling the output texture is the drawn texture; destroy it once.
            self.output_texture if self.output_texture is not self.texture else None,
            self.depth,
            self.ms_texture,
            self.nv12_buffer,
            self._glow_buffer,
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
