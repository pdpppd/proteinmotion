"""Studio renderer: lighting/material presets, ambient occlusion, and global film effects.

StudioRenderer draws the same geometry as Renderer into an HDR buffer with physically based
lighting (a warm key, cool fill, and rim light that follow the camera, plus hemisphere
ambient), then adds ambient occlusion, bloom, warm halation, highlight roll-off, and
luminance-based monochrome grain. Overlays are composited after the effects.
"""

import re
from importlib.resources import files

import numpy as np
import wgpu

from .looks import LIGHTING_PRESETS as LIGHTING_PRESETS
from .looks import MATERIAL_PRESETS as MATERIAL_PRESETS
from .looks import StudioLook
from .renderer import Renderer

STUDIO_SHADE = """
fn ggx(n_h: f32, rough: f32) -> f32 {
    let a = rough * rough;
    let a2 = a * a;
    let d = n_h * n_h * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * d * d);
}
fn visibility(n_v: f32, n_l: f32, rough: f32) -> f32 {
    let k = (rough + 1.0) * (rough + 1.0) / 8.0;
    return 0.25 / ((n_v * (1.0 - k) + k) * (n_l * (1.0 - k) + k));
}
fn schlick(cos_theta: f32, f0: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}
fn studio_light(n: vec3f, v: vec3f, l: vec3f, albedo: vec3f, radiance: vec3f) -> vec3f {
    let n_l = max(dot(n, l), 0.0);
    // Slight wrap keeps the terminator soft on small spheres and thin tubes.
    let wrap = max((dot(n, l) + 0.18) / 1.18, 0.0);
    if wrap <= 0.0 { return vec3f(0.0); }
    let h = normalize(l + v);
    let n_v = max(dot(n, v), 1e-3);
    let n_h = max(dot(n, h), 0.0);
    let v_h = max(dot(v, h), 0.0);
    let rough = STUDIO_ROUGHNESS;
    let spec = ggx(n_h, rough) * visibility(n_v, n_l, rough) * schlick(v_h, 0.04) * STUDIO_SPECULAR;
    let coat_rough = STUDIO_COAT_ROUGHNESS;
    let coat = ggx(n_h, coat_rough) * visibility(n_v, n_l, coat_rough) * schlick(v_h, 0.04) * STUDIO_CLEARCOAT;
    return (albedo / 3.14159265 * wrap + vec3f(spec + coat) * n_l) * radiance;
}

fn shade(p: vec3<f32>, norm: vec3<f32>, base: vec3<f32>) -> vec4<f32> {
    let v = safe_normal(camera.eye_fog_start.xyz-p);
    var n = safe_normal(norm);
    if dot(n,v)<0.0 { n = -n; }
    let right = camera.view_right.xyz;
    let up = camera.view_up.xyz;
    let back = safe_normal(cross(right, up));
    let albedo = pow(max(base, vec3f(0.0)), vec3f(2.2));
    let key_d = STUDIO_KEY_DIRECTION;
    let fill_d = STUDIO_FILL_DIRECTION;
    let rim_d = STUDIO_RIM_DIRECTION;
    let key = safe_normal(key_d.x*right + key_d.y*up + key_d.z*back);
    let fill = safe_normal(fill_d.x*right + fill_d.y*up + fill_d.z*back);
    let rim = safe_normal(rim_d.x*right + rim_d.y*up + rim_d.z*back);
    var c = studio_light(n, v, key, albedo, STUDIO_KEY_COLOR * STUDIO_KEY);
    c += studio_light(n, v, fill, albedo, STUDIO_FILL_COLOR * STUDIO_FILL);
    c += studio_light(n, v, rim, albedo, STUDIO_RIM_COLOR * STUDIO_RIM);
    // Rim: a Fresnel-shaped band of colored light along silhouettes, stronger toward the rim light.
    let facing = 1.0 - max(dot(n, v), 0.0);
    let rim_band = pow(facing, 3.0) * (0.35 + 0.65 * max(dot(n, rim) * 0.5 + 0.5, 0.0));
    c += STUDIO_RIM_COLOR * rim_band * STUDIO_RIM * 0.12 * (0.4 + albedo);
    // Hemisphere ambient plus a faint Fresnel sheen from the environment.
    let sky = mix(STUDIO_GROUND, STUDIO_SKY, 0.5 + 0.5 * dot(n, up));
    c += albedo * sky * STUDIO_AMBIENT;
    c += sky * schlick(max(dot(n, v), 0.0), 0.04) * 0.5 * STUDIO_AMBIENT * STUDIO_SPECULAR;
    var display = pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.2));
    display = cutaway_rim(display, p);
    let d = distance(camera.eye_fog_start.xyz,p);
    let fog = smoothstep(camera.eye_fog_start.w,camera.background_fog_end.w,d)*camera.lighting.x;
    display = mix(display,camera.background_fog_end.xyz,fog);
    return vec4<f32>(display,1.0);
}
"""


class StudioRenderer(Renderer):
    """Renderer with studio lighting, ambient occlusion, bloom, and highlight roll-off.

    A drop-in replacement for Renderer: pass it to ``scene.render_frame(renderer=...)`` or
    use ``scene.render(..., renderer="studio")``. Adjust the look with ``look=StudioLook(...)``.
    """

    SCENE_FORMAT = "rgba16float"
    KEEP_DEPTH = True
    REVERSED_Z = True
    BLOOM_LEVELS = 6

    def __init__(self, width=1920, height=1080, *, lighting="studio", material="glossy", look=None, **kwargs):
        if look is not None and (lighting, material) != ("studio", "glossy"):
            raise ValueError("Pass lighting and material inside StudioLook when look is given")
        if look is not None and not isinstance(look, StudioLook):
            raise TypeError("look must be a StudioLook")
        self.look = (look or StudioLook(lighting=lighting, material=material)).validate()
        # Molecular mode shares native's VDW/SAS/SES meshes and GPU deformation.
        # Density mode retains the original GPU splat / ray-march implementation.
        self.gpu_surfaces = self.look.surface_mode == "density"
        self._classic_cartoon = self.look.cartoon_style == "classic"
        self.time = 0.0
        super().__init__(width, height, **kwargs)
        self._create_post()

    def set_time(self, time):
        """Set deterministic grain time. Scene stills and movies call this automatically."""
        if not np.isfinite(time) or time < 0:
            raise ValueError("Render time must be finite and nonnegative")
        self.time = float(time)

    def _segment_data(self, protein):
        if self._classic_cartoon:
            from .geometry import studio_cartoon_segments

            return studio_cartoon_segments(protein)
        return super()._segment_data(protein)

    def _sweep_grid(self):
        if self._classic_cartoon:
            from .geometry import sweep_grid

            return sweep_grid(steps=20, sides=24)
        return super()._sweep_grid()

    def _shader_code(self):
        code = super()._shader_code()
        if self._classic_cartoon:
            start = code.index("@vertex fn ribbon_vertex(")
            end = code.index("\n}\n", start) + 3
            body = code[code.index(" {", start) : end].replace("object.style.x", "cartoon")
            helper = "fn ribbon_sweep(uv:vec2f, instance:u32, cartoon:f32) -> Surface" + body
            cartoon = files("proteinmotion").joinpath("shaders/studio_cartoon.wgsl").read_text()
            code = code[:start] + helper + cartoon + code[end:]
        code += files("proteinmotion").joinpath("shaders/studio_surface.wgsl").read_text()
        start = code.index("fn shade(p: vec3<f32>")
        end = code.index("\n}\n", start) + 3
        look = self.look
        studio = STUDIO_SHADE

        def vec(values):
            return "vec3f({:.5f}, {:.5f}, {:.5f})".format(*values)

        # Longer names first, so STUDIO_KEY does not replace part of STUDIO_KEY_COLOR.
        for name, value in (
            ("STUDIO_KEY_DIRECTION", vec(look.key_direction)),
            ("STUDIO_FILL_DIRECTION", vec(look.fill_direction)),
            ("STUDIO_RIM_DIRECTION", vec(look.rim_direction)),
            ("STUDIO_KEY_COLOR", vec(look.key_color)),
            ("STUDIO_FILL_COLOR", vec(look.fill_color)),
            ("STUDIO_RIM_COLOR", vec(look.rim_color)),
            ("STUDIO_SKY", vec(look.sky)),
            ("STUDIO_GROUND", vec(look.ground)),
            ("STUDIO_COAT_ROUGHNESS", look.clearcoat_roughness),
            ("STUDIO_ROUGHNESS", look.roughness),
            ("STUDIO_SPECULAR", look.specular),
            ("STUDIO_CLEARCOAT", look.clearcoat),
            ("STUDIO_KEY", look.key),
            ("STUDIO_FILL", look.fill),
            ("STUDIO_RIM", look.rim),
            ("STUDIO_AMBIENT", look.ambient),
        ):
            studio = re.sub(
                rf"\b{name}\b", value if isinstance(value, str) else f"{float(value):.5f}", studio
            )
        code = code[:start] + studio + code[end:]
        # Order-independent transparency: as opacity approaches 1, weight nearer layers much
        # more strongly, so a fading object converges to its opaque appearance instead of
        # averaging in the layers behind and snapping when it becomes opaque.
        old = "let weight=(0.1+8.0*pow(1.0-relative_depth,3.0))*(0.1+0.9*alpha*alpha*alpha);"
        if code.count(old) != 1:
            raise RuntimeError("Studio transparency weight patch did not match")
        new = (
            "let steep=3.0+45.0*pow(alpha,4.0);"
            "let weight=exp(clamp(-steep*(relative_depth-0.5),-10.0,10.0))*(0.1+0.9*alpha*alpha*alpha);"
        )
        return code.replace(old, new)

    def _setup_scene_targets(self):
        d, w, h = self.device, self.width, self.height
        usage = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING
        self.hdr = d.create_texture(size=(w, h, 1), format="rgba16float", usage=usage)
        self.hdr_view = self.hdr.create_view()
        self.hdr_ms = None
        if self.msaa > 1:
            self.hdr_ms = d.create_texture(
                size=(w, h, 1), sample_count=self.msaa, format="rgba16float", usage=usage
            )
        self.scene_ms_view = self.hdr_ms.create_view() if self.hdr_ms else self.hdr_view
        self.scene_view = self.hdr_view

    def _scene_final(self, annotations):
        return True  # The scene always resolves into the HDR buffer for post-processing.

    def _create_post(self):
        d = self.device
        code = files("proteinmotion").joinpath("shaders/studio_post.wgsl").read_text()
        if self.msaa == 1:
            code = code.replace("texture_depth_multisampled_2d", "texture_depth_2d")
            code = code.replace("texture_multisampled_2d<f32>", "texture_2d<f32>")
        shader = d.create_shader_module(code=code)
        self.post_buffer = d.create_buffer(
            size=4 * 64, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self.sampler = d.create_sampler(
            mag_filter="linear",
            min_filter="linear",
            address_mode_u="clamp-to-edge",
            address_mode_v="clamp-to-edge",
        )
        self.post_layout = d.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT, "buffer": {"type": "uniform"}},
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "depth", "multisampled": self.msaa > 1},
                },
                {"binding": 2, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": {"sample_type": "float"}},
                {"binding": 3, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": {"sample_type": "float"}},
                {"binding": 4, "visibility": wgpu.ShaderStage.FRAGMENT, "sampler": {"type": "filtering"}},
                {
                    "binding": 5,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "depth", "multisampled": self.msaa > 1},
                },
                {
                    "binding": 6,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "unfilterable-float", "multisampled": self.msaa > 1},
                },
            ]
        )
        layout = d.create_pipeline_layout(bind_group_layouts=[self.post_layout])

        def pipeline(entry, fmt, *, additive=False, samples=1):
            target = {"format": fmt}
            if additive:
                one = {"src_factor": "one", "dst_factor": "one", "operation": "add"}
                target["blend"] = {"color": one, "alpha": one}
            return d.create_render_pipeline(
                layout=layout,
                vertex={"module": shader, "entry_point": "fullscreen"},
                fragment={"module": shader, "entry_point": entry, "targets": [target]},
                primitive={"topology": "triangle-list"},
                multisample={"count": samples},
            )

        usage = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING
        self.ao = d.create_texture(size=(self.width, self.height, 1), format="rg16float", usage=usage)
        self.ao_view = self.ao.create_view()
        # Front layer: nearest fragment of any opacity, drawn without multisampling.
        size = (self.width, self.height, 1)
        # Same multisampling as the scene, so the front layer's sample 0 matches the solid
        # layer's exactly wherever both see the same surface.
        self.front_depth = d.create_texture(
            size=size, format="depth32float", usage=usage, sample_count=self.msaa
        )
        self.front_alpha = d.create_texture(size=size, format="r16float", usage=usage, sample_count=self.msaa)
        self.front_depth_view = self.front_depth.create_view()
        self.front_alpha_view = self.front_alpha.create_view()
        self.bloom_levels = []
        w, h = self.width, self.height
        for _ in range(self.BLOOM_LEVELS):
            w, h = max(1, w // 2), max(1, h // 2)
            tex = d.create_texture(size=(w, h, 1), format="rgba16float", usage=usage)
            self.bloom_levels.append((tex, tex.create_view()))
        self.ao_pipeline = pipeline("ambient_occlusion", "rg16float")
        self.accum = d.create_texture(size=size, format="rgba16float", usage=usage)
        self.accum_view = self.accum.create_view()
        self.accumulate_pipeline = pipeline("accumulate", "rgba16float", additive=True)
        self.copy_pipeline = pipeline("copy_color", "rgba16float")
        self.prefilter_pipeline = pipeline("bloom_prefilter", "rgba16float")
        self.down_pipeline = pipeline("bloom_down", "rgba16float")
        self.up_pipeline = pipeline("bloom_up", "rgba16float", additive=True)
        self.add_bloom_pipeline = pipeline("add_bloom", "rgba16float", additive=True)
        self.halation_prefilter_pipeline = pipeline("halation_prefilter", "rgba16float")
        self.halation_horizontal_pipeline = pipeline("halation_horizontal", "rgba16float")
        self.halation_vertical_pipeline = pipeline("halation_vertical", "rgba16float")
        self.add_halation_pipeline = pipeline("add_halation", "rgba16float", additive=True)
        self.halation_textures = [
            d.create_texture(
                size=(max(1, self.width // 2), max(1, self.height // 2), 1), format="rgba16float", usage=usage
            )
            for _ in range(3)
        ]
        self.halation_views = [texture.create_view() for texture in self.halation_textures]
        self.composite_post = pipeline("composite", "rgba8unorm", samples=self.msaa)
        depth_view = self.depth.create_view()

        def group(color_view, aux_view):
            return d.create_bind_group(
                layout=self.post_layout,
                entries=[
                    {"binding": 0, "resource": {"buffer": self.post_buffer}},
                    {"binding": 1, "resource": depth_view},
                    {"binding": 2, "resource": color_view},
                    {"binding": 3, "resource": aux_view},
                    {"binding": 4, "resource": self.sampler},
                    {"binding": 5, "resource": self.front_depth_view},
                    {"binding": 6, "resource": self.front_alpha_view},
                ],
            )

        views = [view for _, view in self.bloom_levels]
        self.group_prefilter = group(self.hdr_view, self.ao_view)
        self.group_down = [group(self.hdr_view, views[i]) for i in range(len(views) - 1)]
        self.group_up = [group(self.hdr_view, views[i + 1]) for i in range(len(views) - 1)]
        # A pass cannot read the texture it renders into, so these bind other textures.
        self.group_add = group(self.ao_view, views[0])
        self.group_ao = group(self.hdr_view, views[-1])
        self.group_composite = group(self.hdr_view, self.ao_view)
        self.group_accumulate = group(self.hdr_view, self.ao_view)
        self.group_copy = group(self.ao_view, self.accum_view)
        source, horizontal, blurred = self.halation_views
        self.group_halation_prefilter = group(self.hdr_view, self.ao_view)
        self.group_halation_horizontal = group(self.ao_view, source)
        self.group_halation_vertical = group(self.ao_view, horizontal)
        self.group_halation_add = group(source, blurred)
        self._post_resources = [
            self.post_buffer,
            self.ao,
            self.front_depth,
            self.front_alpha,
            self.accum,
            *self.halation_textures,
            *[tex for tex, _ in self.bloom_levels],
        ]
        self._create_grid_pipelines()
        self._create_front_pipelines()

    # Front layer ------------------------------------------------------------------------------

    def _front_shader_code(self):
        """The scene shader, changed to draw any visible fragment opaquely and output its opacity."""
        code = "var<private> frag_alpha: f32 = 1.0;\n" + self._shader_code()
        replacements = [
            (
                "return select(alpha,1.0,alpha>=0.999999);",
                "frag_alpha = alpha; return select(0.0,1.0,alpha>=0.5);",
                3,
            ),
            ("return vec4<f32>(display,1.0);", "return vec4<f32>(frag_alpha,0.0,0.0,1.0);", 1),
            (
                "if grid.params.x * rgba.a < 0.999 { discard; }",
                "frag_alpha = grid.params.x * rgba.a; if frag_alpha < 0.5 { discard; }",
                1,
            ),
        ]
        for old, new, count in replacements:
            if code.count(old) != count:
                raise RuntimeError(f"Studio front-layer shader patch did not match: {old}")
            code = code.replace(old, new)
        return code

    def _create_front_pipelines(self):
        d = self.device
        self._front_shader = d.create_shader_module(code=self._front_shader_code())
        molecule = d.create_pipeline_layout(bind_group_layouts=[self.camera_layout, self.object_layout])
        empty = d.create_bind_group_layout(entries=[])
        grid = d.create_pipeline_layout(
            bind_group_layouts=[self.camera_layout, empty, empty, self.grid_layout]
        )

        self.front_ribbon = self._front_pipeline(
            "ribbon_vertex", "surface_fragment", molecule, self._ribbon_vertex_buffers
        )
        self.front_bond = self._front_pipeline("bond_vertex", "bond_fragment", molecule)
        self.front_sphere = self._front_pipeline("sphere_vertex", "sphere_fragment", molecule)
        self.front_grid = self._front_pipeline("grid_vertex", "grid_fragment", grid)
        self.front_mesh = None  # SurfaceGPU creates the shared mesh layout on first use.

    def _front_pipeline(self, vertex, fragment, layout, buffers=None):
        return self.device.create_render_pipeline(
            layout=layout,
            vertex={"module": self._front_shader, "entry_point": vertex, "buffers": buffers or []},
            fragment={
                "module": self._front_shader,
                "entry_point": fragment,
                "targets": [{"format": "r16float"}],
            },
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            depth_stencil={
                "format": "depth32float",
                "depth_write_enabled": True,
                "depth_compare": self._depth_test(),
            },
            multisample={"count": self.msaa},
        )

    def _draw_front(self, encoder):
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.front_alpha_view,
                    "clear_value": (0, 0, 0, 0),
                    "load_op": "clear",
                    "store_op": "store",
                }
            ],
            depth_stencil_attachment={
                "view": self.front_depth_view,
                "depth_clear_value": 0.0 if self.REVERSED_Z else 1.0,
                "depth_load_op": "clear",
                "depth_store_op": "store",
            },
        )
        render_pass.set_bind_group(0, self.camera_group)
        for protein, gpu, bindings in self._frame_draws:
            if protein.opacity <= 0:
                continue
            if sum(protein.representation[:2]) > 0 and len(gpu.segment_data):
                render_pass.set_pipeline(self.front_ribbon)
                render_pass.set_bind_group(1, bindings[1])
                render_pass.set_vertex_buffer(0, self.vertices)
                render_pass.set_index_buffer(self.indices, "uint32")
                render_pass.draw_indexed(self.n_indices, len(gpu.segment_data))
            if protein.representation[2] > 0 or gpu.has_detail:
                render_pass.set_bind_group(1, bindings[0])
                if len(protein.topology.bonds):
                    render_pass.set_pipeline(self.front_bond)
                    render_pass.draw(12 * 6, len(protein.topology.bonds))
                if getattr(protein, "_draw_atoms", True):
                    render_pass.set_pipeline(self.front_sphere)
                    render_pass.draw(6, len(protein.topology.atoms))
            if gpu.surface is not None and protein.surface_opacity > 0:
                if self.front_mesh is None:
                    layout, buffers = self._surface_pipeline_args
                    self.front_mesh = self._front_pipeline("mesh_vertex", "mesh_fragment", layout, buffers)
                render_pass.set_pipeline(self.front_mesh)
                render_pass.set_bind_group(1, bindings[2])
                render_pass.set_bind_group(2, gpu.surface.binding)
                render_pass.set_vertex_buffer(0, gpu.surface.vertices)
                render_pass.set_index_buffer(gpu.surface.indices, "uint32")
                render_pass.draw_indexed(gpu.surface.mesh.faces.size)
            grid = self._grids.get(protein)
            if grid is not None and protein.surface_opacity > 0:
                render_pass.set_pipeline(self.front_grid)
                render_pass.set_bind_group(1, self._empty_group)
                render_pass.set_bind_group(2, self._empty_group)
                render_pass.set_bind_group(3, grid.group)
                render_pass.draw(3)
        render_pass.end()

    # Optional artistic GPU density surfaces --------------------------------------------------

    def _create_grid_pipelines(self):
        d = self.device
        storage = wgpu.ShaderStage.COMPUTE
        self.splat_layout = d.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": storage, "buffer": {"type": "uniform"}},
                {"binding": 1, "visibility": storage, "buffer": {"type": "read-only-storage"}},
                {"binding": 2, "visibility": storage, "buffer": {"type": "read-only-storage"}},
                {"binding": 3, "visibility": storage, "buffer": {"type": "storage"}},
                {
                    "binding": 4,
                    "visibility": storage,
                    "storage_texture": {
                        "access": "write-only",
                        "format": "rgba16float",
                        "view_dimension": "3d",
                    },
                },
                {
                    "binding": 5,
                    "visibility": storage,
                    "storage_texture": {
                        "access": "write-only",
                        "format": "rgba16float",
                        "view_dimension": "3d",
                    },
                },
            ]
        )
        compute = d.create_shader_module(
            code=files("proteinmotion").joinpath("shaders/studio_splat.wgsl").read_text()
        )
        layout = d.create_pipeline_layout(bind_group_layouts=[self.splat_layout])
        self.splat_pipeline = d.create_compute_pipeline(
            layout=layout, compute={"module": compute, "entry_point": "splat_atoms"}
        )
        self.resolve_pipeline = d.create_compute_pipeline(
            layout=layout, compute={"module": compute, "entry_point": "resolve_grid"}
        )
        visible = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT
        self.grid_layout = d.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": visible, "buffer": {"type": "uniform"}},
                {
                    "binding": 1,
                    "visibility": visible,
                    "texture": {"sample_type": "float", "view_dimension": "3d"},
                },
                {
                    "binding": 2,
                    "visibility": visible,
                    "texture": {"sample_type": "float", "view_dimension": "3d"},
                },
                {"binding": 3, "visibility": visible, "sampler": {"type": "filtering"}},
            ]
        )
        empty = d.create_bind_group_layout(entries=[])
        self._empty_group = d.create_bind_group(layout=empty, entries=[])
        grid_layout = d.create_pipeline_layout(
            bind_group_layouts=[self.camera_layout, empty, empty, self.grid_layout]
        )
        additive = {"src_factor": "one", "dst_factor": "one", "operation": "add"}
        transmittance = {"src_factor": "zero", "dst_factor": "one-minus-src", "operation": "add"}
        self.grid_pipelines = {}
        for transparent in (False, True):
            targets = (
                [
                    {"format": "rgba16float", "blend": {"color": additive, "alpha": additive}},
                    {"format": "r16float", "blend": {"color": transmittance, "alpha": transmittance}},
                ]
                if transparent
                else [{"format": self.SCENE_FORMAT}]
            )
            self.grid_pipelines[transparent] = d.create_render_pipeline(
                layout=grid_layout,
                vertex={"module": self._shader, "entry_point": "grid_vertex"},
                fragment={
                    "module": self._shader,
                    "entry_point": "grid_transparent" if transparent else "grid_fragment",
                    "targets": targets,
                },
                primitive={"topology": "triangle-list", "cull_mode": "none"},
                depth_stencil={
                    "format": "depth32float",
                    "depth_write_enabled": not transparent,
                    "depth_compare": self._depth_test(transparent),
                },
                multisample={"count": self.msaa},
            )
        self._grids = {}

    def _prepare(self, encoder, draws):
        self._frame_draws = draws
        live = set()
        for protein, _, _ in draws:
            if self.gpu_surfaces and protein.surface_opacity * protein.opacity > 0:
                grid = self._grids.get(protein)
                if grid is None:
                    grid = self._grids[protein] = _GridSurface(self, protein)
                grid.update(encoder)
                live.add(protein)
        for protein in [p for p in self._grids if p not in live]:
            self._grids.pop(protein).close()

    def _draw_extra(self, render_pass, draws, transparent):
        for protein, _, _ in draws:
            grid = self._grids.get(protein)
            if grid is None or protein.surface_opacity * protein.opacity <= 0:
                continue
            render_pass.set_pipeline(self.grid_pipelines[transparent])
            render_pass.set_bind_group(0, self.camera_group)
            render_pass.set_bind_group(1, self._empty_group)
            render_pass.set_bind_group(2, self._empty_group)
            render_pass.set_bind_group(3, grid.group)
            render_pass.draw(3)

    def _fullscreen(self, encoder, pipeline, group, view, *, load=False, resolve=None):
        attachment = {"view": view, "load_op": "load" if load else "clear", "store_op": "store"}
        if not load:
            attachment["clear_value"] = (0.0, 0.0, 0.0, 0.0)
        if resolve is not None:
            attachment["resolve_target"] = resolve
        render_pass = encoder.begin_render_pass(color_attachments=[attachment])
        render_pass.set_pipeline(pipeline)
        render_pass.set_bind_group(0, group)
        render_pass.draw(3)
        render_pass.end()

    def _post_process(self, encoder, camera, background, annotations):
        look = self.look
        vp = self._last_vp
        u = np.zeros(64, np.float32)
        u[:16] = vp.T.ravel()
        u[16:32] = np.linalg.inv(vp).T.ravel()
        radius = float(np.clip(camera.radius * 0.14, 3.5, 12.0))
        u[32:36] = [*camera.eye, radius]
        u[36:40] = [look.ambient_occlusion, look.bloom, look.bloom_threshold, look.vignette]
        u[40:44] = [self.width, self.height, 1 / self.width, 1 / self.height]
        u[44:48] = [*background, look.knee]
        # The front layer is only needed when something is partly transparent.
        fading = any(
            gpu.has_opacity_controls
            or gpu.has_style_opacity
            or gpu.partial_atoms
            or 0 < protein.opacity < 1
            or 0 < protein.surface_opacity < 1
            or any(0 < f < 1 for f in (sum(protein.representation[:2]), protein.representation[2]))
            for protein, gpu, _ in getattr(self, "_frame_draws", ())
        )
        u[48] = float(fading and look.ambient_occlusion > 0)
        u[49] = self._layer_weight
        tick = int(np.floor(self.time * look.grain_fps + 1e-6)) % (2**32)
        u[50:52].view(np.uint32)[:] = [tick, look.grain_seed]
        u[52:56] = [look.grain, look.grain_size, look.bloom_radius, look.halation]
        u[56:60] = [look.halation_threshold, look.halation_radius, 0, 0]
        u[60:64] = [*look.halation_color, 0]
        self.device.queue.write_buffer(self.post_buffer, 0, u)
        # Every frame takes the same path: occlusion is applied per layer, then bloom and the
        # final roll-off run once on the accumulated image, whether or not anything fades.
        if look.ambient_occlusion > 0:
            if fading:
                self._draw_front(encoder)
            self._fullscreen(encoder, self.ao_pipeline, self.group_ao, self.ao_view)
        self._fullscreen(
            encoder,
            self.accumulate_pipeline,
            self.group_accumulate,
            self.accum_view,
            load=not self._layer_first,
        )
        if not self._layer_final:
            return
        self._fullscreen(encoder, self.copy_pipeline, self.group_copy, self.hdr_view)
        # Extract halation before bloom; neither effect feeds back into its own source.
        if look.halation > 0:
            source, horizontal, blurred = self.halation_views
            self._fullscreen(encoder, self.halation_prefilter_pipeline, self.group_halation_prefilter, source)
            self._fullscreen(
                encoder, self.halation_horizontal_pipeline, self.group_halation_horizontal, horizontal
            )
            self._fullscreen(encoder, self.halation_vertical_pipeline, self.group_halation_vertical, blurred)
        views = [view for _, view in self.bloom_levels]
        if look.bloom > 0:
            self._fullscreen(encoder, self.prefilter_pipeline, self.group_prefilter, views[0])
            for i in range(len(views) - 1):
                self._fullscreen(encoder, self.down_pipeline, self.group_down[i], views[i + 1])
            for i in reversed(range(len(views) - 1)):
                self._fullscreen(encoder, self.up_pipeline, self.group_up[i], views[i], load=True)
            self._fullscreen(encoder, self.add_bloom_pipeline, self.group_add, self.hdr_view, load=True)
        if look.halation > 0:
            self._fullscreen(
                encoder, self.add_halation_pipeline, self.group_halation_add, self.hdr_view, load=True
            )
        resolve = self.view if self.msaa > 1 and not annotations else None
        self._fullscreen(encoder, self.composite_post, self.group_composite, self.ms_view, resolve=resolve)

    # Fades --------------------------------------------------------------------------------------

    def _commands(self, proteins, camera, background):
        """Render whole-object fades as blends of solid renders, not transparency.

        Transparency blending never exactly matches solid drawing, so an object that fades
        through it jumps when it becomes opaque. Instead, with fading opacities
        u1 < … < uk, layer j draws every object whose opacity is at least uj as solid, and
        the layers are added with weights uj − uj−1 (and 1 − uk for the solid-only layer).
        One fading object therefore gives exactly a·(scene with it) + (1 − a)·(scene without
        it), continuous at both ends and with correct occlusion in each layer.
        """
        from .annotations import Annotation
        from .protein import Protein

        objects = list(proteins)
        molecules = [p for p in objects if isinstance(p, Protein)]
        saved = [(p, p.opacity, p.representation.copy()) for p in molecules]
        level = {id(p): round(float(o), 5) for p, o, _ in saved}
        try:
            for p in molecules:
                # A surface crossfade keeps the other representations solid underneath: the
                # surface encloses them, so it fades in or out over them without a dip.
                total = float(np.sum(p.representation))
                if p.surface_opacity > 0 and 0 < total < 1:
                    p.representation = p.representation / total
            levels = sorted({round(float(o), 5) for _, o, _ in saved if 0 < o < 1})
            thresholds = [2.0, *reversed(levels)]  # Solid-only layer first, then lower thresholds.
            weights = [1 - (levels[-1] if levels else 0.0)]
            weights += [u - (levels[i - 1] if i else 0.0) for i, u in reversed(list(enumerate(levels)))]
            encoder = None
            for index, (threshold, weight) in enumerate(zip(thresholds, weights)):
                for p, opacity, _ in saved:
                    if 0 < opacity < 1:
                        p.opacity = 1.0 if level[id(p)] >= threshold else 0.0
                final = index == len(thresholds) - 1
                # Text overlays are drawn once, over the final image.
                layer = [
                    o
                    for o in objects
                    if (final or not isinstance(o, Annotation))
                    and (not isinstance(o, Protein) or o.opacity > 0)
                ]
                self._layer_weight, self._layer_first, self._layer_final = weight, index == 0, final
                encoder = super()._commands(layer, camera, background)
                if not final:
                    self.device.queue.submit([encoder.finish()])
            return encoder
        finally:
            for p, opacity, representation in saved:
                p.opacity, p.representation = opacity, representation

    def close(self):
        if not self._closed:
            for grid in self._grids.values():
                grid.close()
            for resource in (*self._post_resources, self.hdr, self.hdr_ms):
                if resource is not None:
                    resource.destroy()
        super().close()


class _GridSurface:
    """Per-protein density grid: splatted by compute shaders, ray-marched in the scene pass."""

    def __init__(self, renderer, protein):
        from .geometry import atom_metadata

        self.renderer, self.protein = renderer, protein
        self.device = renderer.device
        self.radii = np.asarray(atom_metadata(protein)[:, 3], np.float32)
        n = len(protein.topology.atoms)
        usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        self.atoms = self.device.create_buffer(size=max(16 * n, 16), usage=usage)
        self.colors = self.device.create_buffer(size=max(16 * n, 16), usage=usage)
        self.splat_uniform = self.device.create_buffer(
            size=48, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self.grid_uniform = self.device.create_buffer(
            size=256, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self.sampler = self.device.create_sampler(
            mag_filter="linear",
            min_filter="linear",
            address_mode_u="clamp-to-edge",
            address_mode_v="clamp-to-edge",
            address_mode_w="clamp-to-edge",
        )
        self.dims = None
        self.spacing = None
        self.key = None
        self.textures = []

    def _allocate(self, dims):
        d = self.device
        for resource in self.textures:
            resource.destroy()
        count = int(np.prod(dims))
        self.accum = d.create_buffer(
            size=count * 5 * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        )
        usage = (
            wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_SRC
        )
        size = (int(dims[0]), int(dims[1]), int(dims[2]))
        self.density = d.create_texture(size=size, dimension="3d", format="rgba16float", usage=usage)
        self.color = d.create_texture(size=size, dimension="3d", format="rgba16float", usage=usage)
        self.textures = [self.accum, self.density, self.color]
        density_view, color_view = self.density.create_view(), self.color.create_view()
        self.splat_group = d.create_bind_group(
            layout=self.renderer.splat_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self.splat_uniform}},
                {"binding": 1, "resource": {"buffer": self.atoms}},
                {"binding": 2, "resource": {"buffer": self.colors}},
                {"binding": 3, "resource": {"buffer": self.accum}},
                {"binding": 4, "resource": density_view},
                {"binding": 5, "resource": color_view},
            ],
        )
        self.group = d.create_bind_group(
            layout=self.renderer.grid_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self.grid_uniform}},
                {"binding": 1, "resource": density_view},
                {"binding": 2, "resource": color_view},
                {"binding": 3, "resource": self.sampler},
            ],
        )
        self.dims = np.asarray(dims)

    def _colors(self):
        from .geometry import atom_metadata, residue_colors
        from .styling import current_tints

        p = self.protein
        if p.color_scheme == "element":
            base = atom_metadata(p)[:, :3]
        else:
            base = residue_colors(p)[[a.residue_index for a in p.topology.atoms]]
        tint = current_tints(p)
        rgb = base * (1 - tint[:, 3:4]) + tint[:, :3]
        opacity = p.atom_opacities / max(p.opacity, 1e-6)
        # Ligands, ions and waters are drawn as ball-and-stick; leaving them out of the
        # surface keeps them visible in their pockets, as in most molecular figures.
        opacity = np.where(p.topology.untraced_atoms, 0.0, opacity)
        return np.column_stack((rgb, opacity)).astype(np.float32)

    def update(self, encoder):
        p, look = self.protein, self.renderer.look
        key = (
            p._key_a,
            p._key_b,
            p._mix,
            id(p._controls),
            id(p._appearance),
            p._color_mix,
            p._opacity_mix,
            str(p.color_scheme),
        )
        xyz = np.asarray(p.positions, np.float32)
        options = p._surface_options
        resolution = options.resolution if options is not None else 0.7
        pad = float(self.radii.max()) + look.surface_inflation + 3 * resolution + 1.0
        lo, hi = xyz.min(0) - pad, xyz.max(0) + pad
        spacing = self.spacing if self.dims is not None else None
        needed = None if spacing is None else np.ceil((hi - lo) / spacing).astype(int) + 1
        if spacing is None or np.any(needed > self.dims) or np.prod(needed) < 0.4 * np.prod(self.dims):
            # Choose the spacing and allocate with slack only when the molecule outgrows the
            # grid. A fixed spacing and a lattice fixed in the molecule's frame keep a moving
            # surface from being resampled differently every frame, which would shimmer.
            slack = 1.15
            spacing = max(resolution, float((np.prod((hi - lo) * slack) / look.surface_voxels) ** (1 / 3)))
            needed = np.ceil((hi - lo) / spacing).astype(int) + 1
            self.spacing = spacing
            self._allocate(np.ceil(needed * slack).astype(int) + 2)
            self.key = None
        center = (lo + hi) / 2
        grid_min = np.floor((center - (self.dims - 1) * spacing / 2) / spacing) * spacing
        m = p.model_matrix
        vp = self.renderer._last_vp
        u = np.zeros(64, np.float32)
        u[0:16] = m.T.ravel()
        u[16:32] = np.linalg.inv(m).T.ravel()
        u[32:48] = np.linalg.inv(vp).T.ravel()
        u[48:52] = [*grid_min, spacing]
        u[52:56] = [*self.dims, 1.0]
        camera = self.renderer._last_camera
        aspect = self.renderer.width / self.renderer.height
        # Surfaces are carved down to the target's center plane, so a ligand in the target
        # sits in front of the pocket's back wall instead of behind its front wall.
        keep = -camera.cutaway_band
        u[56:60] = [p.surface_opacity, np.tan(camera.fov / 2), aspect, keep]
        self.device.queue.write_buffer(self.grid_uniform, 0, u)
        key = (*key, spacing, tuple(np.round(grid_min, 4)))
        if key == self.key:
            return
        self.key = key
        atoms = np.column_stack((xyz, self.radii)).astype(np.float32)
        queue = self.device.queue
        queue.write_buffer(self.atoms, 0, atoms)
        queue.write_buffer(self.colors, 0, self._colors())
        s = np.zeros(12, np.float32)
        s[0:4] = [*grid_min, spacing]
        s[4:8].view(np.uint32)[:] = [*self.dims, len(atoms)]
        s[8:12] = [look.surface_inflation, look.surface_blobbiness, 0, 0]
        queue.write_buffer(self.splat_uniform, 0, s)
        encoder.clear_buffer(self.accum)
        compute = encoder.begin_compute_pass()
        compute.set_bind_group(0, self.splat_group)
        compute.set_pipeline(self.renderer.splat_pipeline)
        compute.dispatch_workgroups((len(atoms) + 63) // 64)
        compute.set_pipeline(self.renderer.resolve_pipeline)
        nx, ny, nz = (int(v) for v in self.dims)
        compute.dispatch_workgroups((nx + 3) // 4, (ny + 3) // 4, (nz + 3) // 4)
        compute.end()

    def close(self):
        for resource in (*self.textures, self.atoms, self.colors, self.splat_uniform, self.grid_uniform):
            resource.destroy()
