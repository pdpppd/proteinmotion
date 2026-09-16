"""Native interactive preview: GPU texture → swapchain, without CPU readback."""

import time

import numpy as np

BLIT = """
@group(0) @binding(0) var image:texture_2d<f32>;
@group(0) @binding(1) var image_sampler:sampler;
struct V { @builtin(position) p:vec4<f32>, @location(0) uv:vec2<f32> };
@vertex fn vertex(@builtin(vertex_index) i:u32) -> V {
    let xy=array<vec2<f32>,3>(vec2<f32>(-1.,-1.),vec2<f32>(3.,-1.),vec2<f32>(-1.,3.));
    var out:V;
    out.p=vec4<f32>(xy[i],0.,1.);
    out.uv=vec2<f32>((xy[i].x+1.)*0.5,1.-(xy[i].y+1.)*0.5);
    return out;
}
@fragment fn fragment(v:V) -> @location(0) vec4<f32> {
    return textureSample(image,image_sampler,v.uv);
}
"""


class PreviewControls:
    def __init__(self, duration):
        self.duration, self.time = duration, 0.0
        self.playing = False
        self.theta, self.phi, self.zoom = 0.0, 0.0, 1.0
        self.drag = None
        self.close_requested = False

    def event(self, event):
        kind = event["event_type"]
        if kind == "pointer_down" and event.get("button") == 1:
            self.drag = (event["x"], event["y"])
        elif kind == "pointer_up":
            self.drag = None
        elif kind == "pointer_move" and self.drag is not None:
            self.theta -= (event["x"] - self.drag[0]) * 0.008
            self.phi += (event["y"] - self.drag[1]) * 0.008
            self.drag = (event["x"], event["y"])
        elif kind == "wheel":
            self.zoom = float(np.clip(self.zoom * np.exp(-event["dy"] * 0.0015), 0.15, 8.0))
        elif kind == "key_down":
            key = event["key"]
            if key in (" ", "Space"):
                self.playing = not self.playing
            elif key in ("ArrowLeft", "ArrowRight"):
                self.time = float(
                    np.clip(self.time + (-0.25 if key == "ArrowLeft" else 0.25), 0, self.duration)
                )
            elif key == "Home":
                self.time = 0.0
            elif key.lower() == "r":
                self.theta, self.phi, self.zoom = 0.0, 0.0, 1.0
            elif key == "Escape":
                self.close_requested = True

    def advance(self, delta):
        if self.playing and self.duration:
            self.time = (self.time + delta) % self.duration


def preview(scene, *, close_after=None):
    try:
        from rendercanvas.glfw import RenderCanvas, loop
    except ImportError as exc:
        raise ImportError('Install native preview with pip install "proteinmotion[preview]"') from exc
    from .renderer import Renderer

    scene.build()
    canvas = RenderCanvas(
        title="ProteinMotion · drag to orbit · Space to play · arrows to seek",
        size=(min(scene.width, 1440), min(scene.height, 900)),
        update_mode="continuous",
        max_fps=60,
    )
    renderer = Renderer(scene.width, scene.height, msaa=scene.msaa)
    context = canvas.get_wgpu_context()
    fmt = context.get_preferred_format(renderer.adapter)
    # The offscreen texture contains display-referred colors, so avoid a second sRGB encoding.
    if fmt.endswith("-srgb"):
        fmt = fmt[:-5]
    context.configure(device=renderer.device, format=fmt)
    shader = renderer.device.create_shader_module(code=BLIT)
    pipeline = renderer.device.create_render_pipeline(
        layout="auto",
        vertex={"module": shader, "entry_point": "vertex"},
        fragment={"module": shader, "entry_point": "fragment", "targets": [{"format": fmt}]},
    )
    group = renderer.device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": renderer.view},
            {
                "binding": 1,
                "resource": renderer.device.create_sampler(mag_filter="linear", min_filter="linear"),
            },
        ],
    )
    controls = PreviewControls(scene.duration)
    born = last = time.perf_counter()
    errors = []

    def draw():
        nonlocal last
        try:
            now = time.perf_counter()
            controls.advance(now - last)
            last = now
            if controls.close_requested or (close_after is not None and now - born > close_after):
                canvas.close()
                return
            proteins = scene.seek(controls.time)
            scene.camera.orbit(controls.theta, controls.phi).zoom(controls.zoom)
            renderer.draw(proteins, scene.camera, scene.background)
            encoder = renderer.device.create_command_encoder()
            rp = encoder.begin_render_pass(
                color_attachments=[
                    {
                        "view": context.get_current_texture().create_view(),
                        "load_op": "clear",
                        "store_op": "store",
                        "clear_value": (0, 0, 0, 1),
                    }
                ]
            )
            rp.set_pipeline(pipeline)
            rp.set_bind_group(0, group)
            rp.draw(3)
            rp.end()
            renderer.device.queue.submit([encoder.finish()])
        except Exception as exc:
            errors.append(exc)
            canvas.close()

    canvas.add_event_handler(
        controls.event, "pointer_down", "pointer_up", "pointer_move", "wheel", "key_down"
    )
    canvas.request_draw(draw)
    try:
        loop.run()
    finally:
        context.unconfigure()
        renderer.close()
        canvas.close()
    if errors:
        raise errors[0]
