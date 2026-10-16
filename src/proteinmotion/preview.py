"""Native interactive preview: GPU texture → swapchain, without CPU readback.

``ProteinScene.preview()`` calls ``preview`` here. It opens a GLFW window (rendercanvas), and
on every redraw seeks the scene, draws the frame with a native or Studio ``Renderer`` into its
offscreen texture, and copies that texture to the window with a one-triangle blit pass.
``PreviewControls`` turns mouse and keyboard events into playback time and camera offsets.
"""

import time

import numpy as np

# Blit shader: one triangle that covers the whole window samples the renderer's image.
BLIT = """
@group(0) @binding(0) var image:texture_2d<f32>;
@group(0) @binding(1) var image_sampler:sampler;
struct V { @builtin(position) p:vec4<f32>, @location(0) uv:vec2<f32> };
// Corners (-1, -1), (3, -1), (-1, 3) give a triangle that contains the whole clip square.
// Texture v runs down the image, clip y runs up, so v is flipped.
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
    """Playback time and camera offsets driven by preview window events.

    Drag with the left button to orbit, scroll to zoom, Space to play or pause, the left and
    right arrows to step 0.25 s, Home to rewind, R to reset the view and Escape to close.
    ``theta`` and ``phi`` (radians) and ``zoom`` (a factor) apply on top of the scene's camera.
    """

    def __init__(self, duration):
        self.duration, self.time = duration, 0.0
        self.playing = False
        self.theta, self.phi, self.zoom = 0.0, 0.0, 1.0
        self.drag = None  # Last pointer position during a left-button drag.
        self.close_requested = False

    def event(self, event):
        """Update the state from one rendercanvas event dict."""
        kind = event["event_type"]
        if kind == "pointer_down" and event.get("button") == 1:
            self.drag = (event["x"], event["y"])
        elif kind == "pointer_up":
            self.drag = None
        elif kind == "pointer_move" and self.drag is not None:
            # 0.008 rad per pixel: a drag across a 400-pixel window turns about 180°.
            self.theta -= (event["x"] - self.drag[0]) * 0.008
            self.phi += (event["y"] - self.drag[1]) * 0.008
            self.drag = (event["x"], event["y"])
        elif kind == "wheel":
            # Exponential, so equal scroll steps give equal zoom ratios; kept in [0.15, 8].
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
        """Move playback forward by ``delta`` seconds while playing, looping at the end."""
        if self.playing and self.duration:
            self.time = (self.time + delta) % self.duration


def preview(scene, *, close_after=None, renderer=None, look=None):
    """Open an interactive window that plays ``scene`` on the GPU.

    ``renderer`` is "native" or "studio" (the scene's setting when None), and ``look`` a
    StudioLook for the studio renderer. ``close_after`` closes the window after that many
    seconds. See ``PreviewControls`` for the mouse and keyboard controls.
    """
    backend, look, _ = scene._render_settings(renderer, look, None)
    if backend not in ("native", "studio"):
        raise ValueError(
            "Interactive preview supports renderer='native' or 'studio'; use render_frame for EEVEE"
        )
    if look is not None and backend != "studio":
        raise ValueError("StudioLook settings require renderer='studio'")
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
    options = {}
    if backend == "studio":
        from .studio import StudioRenderer as Renderer

        options["look"] = look
    # The renderer draws at the scene's size; the blit scales its image to the window.
    renderer = Renderer(scene.width, scene.height, msaa=scene.msaa, **options)
    context = canvas.get_wgpu_context()
    fmt = context.get_preferred_format(renderer.adapter)
    # The offscreen texture contains display-referred colors, so avoid a second sRGB encoding.
    if fmt.endswith("-srgb"):
        fmt = fmt[:-5]
    context.configure(device=renderer.device, format=fmt)
    pipeline, group = _blit_pipeline(renderer, fmt)
    controls = PreviewControls(scene.duration)
    born = last = time.perf_counter()
    errors = []

    def draw():
        nonlocal last
        # rendercanvas logs errors from a draw callback and keeps drawing. Keep the first
        # error instead, close the window, and raise it after the loop ends.
        try:
            now = time.perf_counter()
            controls.advance(now - last)
            last = now
            if controls.close_requested or (close_after is not None and now - born > close_after):
                canvas.close()
                return
            # seek() restores the timeline's camera; the viewer's orbit and zoom go on top.
            proteins = scene.seek(controls.time)
            scene.camera.orbit(controls.theta, controls.phi).zoom(controls.zoom)
            if hasattr(renderer, "set_time"):
                renderer.set_time(controls.time)
            renderer.draw(proteins, scene.camera, scene.background)
            _blit(renderer, context, pipeline, group)
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


def _blit_pipeline(renderer, fmt):
    """Pipeline and bind group that copy ``renderer.view`` to a window of format ``fmt``."""
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
    return pipeline, group


def _blit(renderer, context, pipeline, group):
    """Draw the renderer's latest image into the window's current swapchain texture."""
    encoder = renderer.device.create_command_encoder()
    render_pass = encoder.begin_render_pass(
        color_attachments=[
            {
                "view": context.get_current_texture().create_view(),
                "load_op": "clear",
                "store_op": "store",
                "clear_value": (0, 0, 0, 1),
            }
        ]
    )
    render_pass.set_pipeline(pipeline)
    render_pass.set_bind_group(0, group)
    render_pass.draw(3)
    render_pass.end()
    renderer.device.queue.submit([encoder.finish()])
