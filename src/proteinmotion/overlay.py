"""Native vector overlays. Glyph buffers persist; only small uniforms change during Write."""

from importlib.resources import files

import numpy as np
import wgpu


class _Drawing:
    def __init__(self, overlay, fill, strokes, *, dynamic=False):
        self.overlay = overlay
        d = overlay.renderer.device
        self.fill_count, self.stroke_count = len(fill), len(strokes)
        self.fill = d.create_buffer_with_data(
            data=fill if len(fill) else np.zeros((1, 3), np.float32), usage=wgpu.BufferUsage.VERTEX
        )
        self.capacity = max(1, len(strokes))
        self.strokes = d.create_buffer_with_data(
            data=strokes if len(strokes) else np.zeros((1, 8), np.float32),
            usage=wgpu.BufferUsage.STORAGE | (wgpu.BufferUsage.COPY_DST if dynamic else 0),
        )
        self.uniform = d.create_buffer(size=64, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._bind()

    def _bind(self):
        self.binding = self.overlay.renderer.device.create_bind_group(
            layout=self.overlay.layout,
            entries=[
                {"binding": i, "resource": {"buffer": buffer}}
                for i, buffer in enumerate((self.uniform, self.strokes))
            ],
        )

    def update_strokes(self, strokes):
        self.stroke_count = len(strokes)
        if len(strokes) > self.capacity:
            self.strokes.destroy()
            self.capacity = 1 << (len(strokes) - 1).bit_length()
            self.strokes = self.overlay.renderer.device.create_buffer(
                size=self.capacity * 32, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
            )
            self._bind()
        if len(strokes):
            self.overlay.renderer.device.queue.write_buffer(self.strokes, 0, strokes)

    def update(self, origin, scale, opacity, progress, lag, count, offset, reverse, stroke_width, color):
        r = self.overlay.renderer
        data = np.array(
            [
                r.width,
                r.height,
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
        r.device.queue.write_buffer(self.uniform, 0, data)
        self.draw_fill = progress > 0 and opacity > 0
        self.draw_stroke = 0 < progress < 1 and opacity > 0

    def draw(self, render_pass):
        render_pass.set_bind_group(0, self.binding)
        if self.fill_count and self.draw_fill:
            render_pass.set_pipeline(self.overlay.fill_pipeline)
            render_pass.set_vertex_buffer(0, self.fill)
            render_pass.draw(self.fill_count)
        if self.stroke_count and self.draw_stroke:
            render_pass.set_pipeline(self.overlay.stroke_pipeline)
            render_pass.draw(6, self.stroke_count)

    def close(self):
        for buffer in (self.fill, self.strokes, self.uniform):
            buffer.destroy()


class OverlayRenderer:
    def __init__(self, renderer):
        self.renderer = renderer
        self.text, self.leaders, self.panels = {}, {}, {}
        d = renderer.device
        self.layout = d.create_bind_group_layout(
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
        layout = d.create_pipeline_layout(bind_group_layouts=[self.layout])
        shader = d.create_shader_module(code=files("proteinmotion").joinpath("shaders/text.wgsl").read_text())
        blend = {
            "color": {"src_factor": "src-alpha", "dst_factor": "one-minus-src-alpha", "operation": "add"},
            "alpha": {"src_factor": "zero", "dst_factor": "one", "operation": "add"},
        }

        def pipeline(kind, buffers):
            return d.create_render_pipeline(
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
        self.stroke_pipeline = pipeline("stroke", [])
        panel_shader = d.create_shader_module(
            code=files("proteinmotion").joinpath("shaders/panel.wgsl").read_text()
        )
        self.panel_pipeline = d.create_render_pipeline(
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
        r = self.renderer
        line_draws, text_draws, panel_draws = [], [], []
        for root in annotations:
            layout = root.layout(camera, r.width, r.height)
            if layout.triangles is not None:
                if root not in self.panels:
                    self.panels[root] = _PanelDrawing(self)
                panel = self.panels[root]
                panel.update(layout.triangles, root.opacity)
                panel_draws.append(panel)
            for index, leader in enumerate(layout.leaders):
                key = (root, index)
                points = leader.points
                pairs = points if points.ndim == 3 else np.stack((points[:-1], points[1:]), axis=1)
                segments = np.zeros((len(pairs), 8), np.float32)
                segments[:, :4] = pairs.reshape(-1, 4)
                segments[:, 5] = 1
                if key not in self.leaders:
                    self.leaders[key] = _Drawing(self, np.empty((0, 3), np.float32), segments, dynamic=True)
                gpu = self.leaders[key]
                gpu.update_strokes(segments)
                gpu.update(
                    (0, 0), 1, root.opacity * leader.opacity, 0.5, 0, 1, 0, False, leader.width, leader.color
                )
                line_draws.append(gpu)
            for placement in layout.text:
                text = placement.text
                # Each placement owns its uniform, even when it shares cached glyph geometry.
                key = (root, text)
                if key in self.text and self.text[key].geometry is not text.geometry:
                    self.text.pop(key).close()
                if key not in self.text:
                    self.text[key] = _Drawing(self, text.geometry.fill, text.geometry.strokes)
                    self.text[key].geometry = text.geometry
                gpu = self.text[key]
                opacity = root.opacity * placement.opacity * (text.opacity if text is not root else 1)
                gpu.update(
                    placement.origin,
                    r.height / 1080,
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
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": r.ms_view,
                    "resolve_target": r.view if r.msaa > 1 else None,
                    "load_op": "load",
                    "store_op": "store",
                }
            ]
        )
        for gpu in (*panel_draws, *line_draws, *text_draws):
            gpu.draw(render_pass)
        render_pass.end()

    def close(self):
        for gpu in (*self.text.values(), *self.leaders.values(), *self.panels.values()):
            gpu.close()


class _PanelDrawing:
    def __init__(self, overlay):
        self.overlay, self.capacity, self.vertices, self.data = overlay, 0, None, None
        d = overlay.renderer.device
        self.uniform = d.create_buffer(size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.binding = d.create_bind_group(
            layout=overlay.panel_pipeline.get_bind_group_layout(0),
            entries=[{"binding": 0, "resource": {"buffer": self.uniform}}],
        )

    def update(self, data, opacity):
        r = self.overlay.renderer
        self.count = len(data)
        if self.count > self.capacity:
            if self.vertices is not None:
                self.vertices.destroy()
            self.capacity = 1 << (self.count - 1).bit_length()
            self.vertices = r.device.create_buffer(
                size=self.capacity * 24, usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST
            )
            self.data = None
        if self.count and (self.data is None or not np.array_equal(data, self.data)):
            r.device.queue.write_buffer(self.vertices, 0, data)
            self.data = data
        r.device.queue.write_buffer(self.uniform, 0, np.array([r.width, r.height, opacity, 0], np.float32))

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
