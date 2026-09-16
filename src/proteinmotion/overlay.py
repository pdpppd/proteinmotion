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
        self.text, self.leaders = {}, {}
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

    def draw(self, encoder, annotations, camera):
        r = self.renderer
        line_draws, text_draws = [], []
        for root in annotations:
            layout = root.layout(camera, r.width, r.height)
            for index, leader in enumerate(layout.leaders):
                key = (root, index)
                points = leader.points
                segments = np.zeros((max(0, len(points) - 1), 8), np.float32)
                segments[:, :2], segments[:, 2:4] = points[:-1], points[1:]
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
                if key not in self.text:
                    self.text[key] = _Drawing(self, text.geometry.fill, text.geometry.strokes)
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
        for gpu in (*line_draws, *text_draws):
            gpu.draw(render_pass)
        render_pass.end()

    def close(self):
        for gpu in (*self.text.values(), *self.leaders.values()):
            gpu.close()
