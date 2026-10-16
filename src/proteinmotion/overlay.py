"""Native vector overlays. Glyph buffers persist; only small uniforms change during Write.

Renderer creates one OverlayRenderer the first time a frame contains annotations and
calls draw() after the scene, transparency and post-processing passes. draw() asks each
annotation for its layout (pixels of the render target, supersampling included; see
``annotations.py``) and draws it over the finished image in three kinds of draws:

- panel triangles, the filled shapes of plots (``shaders/panel.wgsl``);
- leader lines, drawn as round-ended segments by the stroke pipeline of ``shaders/text.wgsl``;
- glyphs, whose cached outline and fill meshes go through both pipelines of ``text.wgsl``.

GPU buffers are cached per annotation, per leader and per Text, so a glyph mesh is
uploaded once and Write animates it through a 64-byte uniform. The EEVEE renderer reuses
a native Renderer, and so this module, to draw annotations over Blender frames.
"""

from importlib.resources import files

import numpy as np
import wgpu


class _Drawing:
    """GPU buffers for one placed Text or one leader: fill vertices, stroke segments and a uniform.

    ``fill`` rows are (x, y, glyph index) float32, matching the 12-byte vertex layout of
    the fill pipeline. ``strokes`` rows are 8 float32 (32 bytes), matching ``Segment`` in
    text.wgsl: x0, y0, x1, y1, then the start and end of the segment as fractions of its
    glyph's outline length, the glyph index, and one unused padding value.
    ``dynamic`` strokes (leaders) are rewritten every frame; glyph strokes never change.
    """

    def __init__(self, overlay, fill, strokes, *, dynamic=False):
        self.overlay = overlay
        device = overlay.renderer.device
        self.fill_count, self.stroke_count = len(fill), len(strokes)
        # Keep at least one row so neither buffer is zero-sized; the counts above hold the
        # real sizes and draw() skips empty meshes.
        self.fill = device.create_buffer_with_data(
            data=fill if len(fill) else np.zeros((1, 3), np.float32), usage=wgpu.BufferUsage.VERTEX
        )
        self.capacity = max(1, len(strokes))
        self.strokes = device.create_buffer_with_data(
            data=strokes if len(strokes) else np.zeros((1, 8), np.float32),
            usage=wgpu.BufferUsage.STORAGE | (wgpu.BufferUsage.COPY_DST if dynamic else 0),
        )
        # 16 float32 values, the Uniforms struct of text.wgsl (filled by update()).
        self.uniform = device.create_buffer(
            size=64, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self._bind()

    def _bind(self):
        # Binding 0 is the uniform and binding 1 the stroke segments, as in text.wgsl.
        self.binding = self.overlay.renderer.device.create_bind_group(
            layout=self.overlay.layout,
            entries=[
                {"binding": i, "resource": {"buffer": buffer}}
                for i, buffer in enumerate((self.uniform, self.strokes))
            ],
        )

    def update_strokes(self, strokes):
        """Upload new stroke segments (N, 8), growing the buffer when they no longer fit."""
        self.stroke_count = len(strokes)
        if len(strokes) > self.capacity:
            # Grow to the next power of two so a line that lengthens frame by frame, such as
            # a revealed trace, reallocates only a few times. The bind group points at the
            # old buffer, so it is rebuilt.
            self.strokes.destroy()
            self.capacity = 1 << (len(strokes) - 1).bit_length()
            self.strokes = self.overlay.renderer.device.create_buffer(
                size=self.capacity * 32, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
            )
            self._bind()
        if len(strokes):
            self.overlay.renderer.device.queue.write_buffer(self.strokes, 0, strokes)

    def update(self, origin, scale, opacity, progress, lag, count, offset, reverse, stroke_width, color):
        """Write this frame's uniform and decide which pipelines draw.

        ``origin`` is the pixel position of the mesh's (0, 0) and ``scale`` converts mesh
        units to pixels. ``progress`` is the Write progress in [0, 1], ``lag`` the
        lag ratio, ``count`` the annotation's glyph count and ``offset`` the rank of this
        mesh's first glyph. ``stroke_width`` is in mesh units; ``color`` is RGB.
        """
        renderer = self.overlay.renderer
        # Same order as the Uniforms struct in text.wgsl; the color gets alpha 1.
        data = np.array(
            [
                renderer.width,
                renderer.height,
                *origin,
                scale,
                opacity,
                progress,
                lag,
                count,
                offset,
                float(reverse),
                stroke_width,
                *color,
                1.0,
            ],
            np.float32,
        )
        renderer.device.queue.write_buffer(self.uniform, 0, data)
        # In text.wgsl each glyph draws its outline during the first half of its progress,
        # then fades in its fill while the outline thins to nothing. So fills can show only
        # once writing has started, and outlines only while it is under way.
        self.draw_fill = progress > 0 and opacity > 0
        self.draw_stroke = 0 < progress < 1 and opacity > 0

    def draw(self, render_pass):
        render_pass.set_bind_group(0, self.binding)
        if self.fill_count and self.draw_fill:
            render_pass.set_pipeline(self.overlay.fill_pipeline)
            render_pass.set_vertex_buffer(0, self.fill)
            render_pass.draw(self.fill_count)
        if self.stroke_count and self.draw_stroke:
            # One instance per segment; the vertex shader expands each into a 6-vertex quad.
            render_pass.set_pipeline(self.overlay.stroke_pipeline)
            render_pass.draw(6, self.stroke_count)

    def close(self):
        for buffer in (self.fill, self.strokes, self.uniform):
            buffer.destroy()


class OverlayRenderer:
    """Draws annotation layouts over a Renderer's finished frame, caching their GPU buffers."""

    def __init__(self, renderer):
        self.renderer = renderer
        # GPU caches: _Drawing by (annotation, Text), _Drawing by (annotation, leader index),
        # and _PanelDrawing by annotation. Entries live until close(), except that a Text's
        # entry is replaced when its glyphs change.
        self.text, self.leaders, self.panels = {}, {}, {}
        device = renderer.device
        self.layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "uniform"},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": "read-only-storage"},
                },
            ]
        )
        layout = device.create_pipeline_layout(bind_group_layouts=[self.layout])
        shader = device.create_shader_module(
            code=files("proteinmotion").joinpath("shaders/text.wgsl").read_text()
        )
        # Ordinary "over" blending for color. Alpha keeps the destination value, so drawing
        # overlays never changes the frame's alpha channel.
        blend = {
            "color": {"src_factor": "src-alpha", "dst_factor": "one-minus-src-alpha", "operation": "add"},
            "alpha": {"src_factor": "zero", "dst_factor": "one", "operation": "add"},
        }

        # The fill and stroke pipelines share the shader and the bind group layout; their
        # entry points are named fill_* and stroke_*. Both draw into the renderer's
        # multisampled target, so they use its sample count.
        def pipeline(kind, buffers):
            return device.create_render_pipeline(
                layout=layout,
                vertex={"module": shader, "entry_point": f"{kind}_vertex", "buffers": buffers},
                fragment={
                    "module": shader,
                    "entry_point": f"{kind}_fragment",
                    "targets": [{"format": "rgba8unorm", "blend": blend}],
                },
                primitive={"topology": "triangle-list"},
                multisample={"count": renderer.msaa},
            )

        # Fill vertices are (x, y, glyph index) float32.
        self.fill_pipeline = pipeline(
            "fill",
            [
                {
                    "array_stride": 12,
                    "step_mode": "vertex",
                    "attributes": [{"format": "float32x3", "offset": 0, "shader_location": 0}],
                }
            ],
        )
        # Strokes have no vertex buffer: the shader reads its segment from storage by instance index.
        self.stroke_pipeline = pipeline("stroke", [])
        panel_shader = device.create_shader_module(
            code=files("proteinmotion").joinpath("shaders/panel.wgsl").read_text()
        )
        # Panel vertices are 24 bytes: pixel x, y (float32x2) then r, g, b, a (float32x4).
        self.panel_pipeline = device.create_render_pipeline(
            layout="auto",
            vertex={
                "module": panel_shader,
                "entry_point": "vertex",
                "buffers": [
                    {
                        "array_stride": 24,
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x2", "offset": 0, "shader_location": 0},
                            {"format": "float32x4", "offset": 8, "shader_location": 1},
                        ],
                    }
                ],
            },
            fragment={
                "module": panel_shader,
                "entry_point": "fragment",
                "targets": [{"format": "rgba8unorm", "blend": blend}],
            },
            primitive={"topology": "triangle-list"},
            multisample={"count": renderer.msaa},
        )

    def draw(self, encoder, annotations, camera):
        """Lay out every annotation for this frame and record one render pass that draws them."""
        renderer = self.renderer
        line_draws, text_draws, panel_draws = [], [], []
        for root in annotations:
            layout = root.layout(camera, renderer.width, renderer.height)
            # Filled triangles: one vertex buffer per annotation, rewritten only when they change.
            if layout.triangles is not None:
                if root not in self.panels:
                    self.panels[root] = _PanelDrawing(self)
                panel = self.panels[root]
                panel.update(layout.triangles, root.opacity)
                panel_draws.append(panel)
            for index, leader in enumerate(layout.leaders):
                key = (root, index)
                # A polyline becomes consecutive point pairs; (N, 2, 2) input is already pairs.
                points = leader.points
                pairs = points if points.ndim == 3 else np.stack((points[:-1], points[1:]), axis=1)
                # Every segment counts as a whole outline (fractions 0 to 1) of glyph 0, so it
                # is drawn in full. The column layout is described on _Drawing.
                segments = np.zeros((len(pairs), 8), np.float32)
                segments[:, :4] = pairs.reshape(-1, 4)
                segments[:, 5] = 1
                if key not in self.leaders:
                    self.leaders[key] = _Drawing(self, np.empty((0, 3), np.float32), segments, dynamic=True)
                gpu = self.leaders[key]
                gpu.update_strokes(segments)
                # Leader points are already pixels, so origin (0, 0) and scale 1. Progress 0.5
                # is the point in text.wgsl where a stroke is complete and still at full width.
                gpu.update(
                    (0, 0), 1, root.opacity * leader.opacity, 0.5, 0, 1, 0, False, leader.width, leader.color
                )
                line_draws.append(gpu)
            for placement in layout.text:
                text = placement.text
                # Each placement owns its uniform, even when it shares cached glyph geometry.
                key = (root, text)
                # When set_text() has replaced the glyph mesh, the old buffers are stale.
                if key in self.text and self.text[key].geometry is not text.geometry:
                    self.text.pop(key).close()
                if key not in self.text:
                    self.text[key] = _Drawing(self, text.geometry.fill, text.geometry.strokes)
                    self.text[key].geometry = text.geometry
                gpu = self.text[key]
                # A Text that is itself the annotation already contributed its opacity as root.
                opacity = root.opacity * placement.opacity * (text.opacity if text is not root else 1)
                # Glyph meshes are in 1080p design pixels; height / 1080 converts them to pixels.
                gpu.update(
                    placement.origin,
                    renderer.height / 1080,
                    opacity,
                    root.text_progress,
                    root._lag_ratio,
                    root.glyph_count,
                    placement.glyph_offset,
                    root._reverse,
                    root._stroke_width,
                    text.color,
                )
                text_draws.append(gpu)
        # Draw over the image already in the multisampled target and resolve it into the
        # output view when multisampling.
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": renderer.ms_view,
                    "resolve_target": renderer.view if renderer.msaa > 1 else None,
                    "load_op": "load",
                    "store_op": "store",
                }
            ]
        )
        # All panels first, then all lines, then all text, so text is never hidden behind a
        # plot or a leader from another annotation.
        for gpu in (*panel_draws, *line_draws, *text_draws):
            gpu.draw(render_pass)
        render_pass.end()

    def close(self):
        for gpu in (*self.text.values(), *self.leaders.values(), *self.panels.values()):
            gpu.close()


class _PanelDrawing:
    """A growable vertex buffer of filled plot triangles and the uniform for panel.wgsl."""

    def __init__(self, overlay):
        self.overlay, self.capacity, self.vertices, self.data = overlay, 0, None, None
        device = overlay.renderer.device
        # Viewport width, height, annotation opacity and one padding float.
        self.uniform = device.create_buffer(
            size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self.binding = device.create_bind_group(
            layout=overlay.panel_pipeline.get_bind_group_layout(0),
            entries=[{"binding": 0, "resource": {"buffer": self.uniform}}],
        )

    def update(self, data, opacity):
        """Upload triangle vertices (N, 6) when they changed and set the panel opacity."""
        renderer = self.overlay.renderer
        self.count = len(data)
        # Grow to the next power of two, as _Drawing does for strokes.
        if self.count > self.capacity:
            if self.vertices is not None:
                self.vertices.destroy()
            self.capacity = 1 << (self.count - 1).bit_length()
            self.vertices = renderer.device.create_buffer(
                size=self.capacity * 24, usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST
            )
            self.data = None
        # A static plot produces identical triangles every frame; skip the upload then.
        if self.count and (self.data is None or not np.array_equal(data, self.data)):
            renderer.device.queue.write_buffer(self.vertices, 0, data)
            self.data = data
        renderer.device.queue.write_buffer(
            self.uniform, 0, np.array([renderer.width, renderer.height, opacity, 0], np.float32)
        )

    def draw(self, render_pass):
        if self.count:
            render_pass.set_pipeline(self.overlay.panel_pipeline)
            render_pass.set_bind_group(0, self.binding)
            render_pass.set_vertex_buffer(0, self.vertices)
            render_pass.draw(self.count)

    def close(self):
        if self.vertices is not None:
            self.vertices.destroy()
        self.uniform.destroy()
