"""Manim-style authoring with a seekable, deterministic timeline."""

import math
from pathlib import Path

import numpy as np

from .annotations import Annotation
from .camera import Camera
from .math3d import color
from .mesh import MeshObject
from .protein import Protein


class ProteinScene:
    renderer = "native"
    look = None
    eevee = None

    def __init__(
        self,
        *,
        width=1920,
        height=1080,
        fps=30,
        background="#0b1220",
        msaa=4,
        renderer=None,
        look=None,
        eevee=None,
    ):
        if not isinstance(width, int) or not isinstance(height, int) or width < 1 or height < 1:
            raise ValueError("Image dimensions must be positive integers")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive")
        if msaa not in (1, 4):
            raise ValueError("msaa must be 1 or 4")
        self.width, self.height, self.fps, self.msaa = width, height, float(fps), msaa
        self.background = color(background)
        self.renderer = renderer if renderer is not None else type(self).renderer
        self.look = look if look is not None else type(self).look
        self.eevee = eevee if eevee is not None else type(self).eevee
        self.camera = Camera()
        self.camera.aspect = width / height
        self.duration = 0.0
        self._objects, self._clips = [], []
        self._built, self._building = False, False
        self._camera_base = None
        self._author_states = {}
        self._view_states = None

    def _remember(self):
        self._author_states = {id(p): p.snapshot() for p, _, _ in self._objects}
        self._author_states[id(self.camera)] = self.camera.snapshot()
        self._view_states = None

    def _capture_edits(self):
        from .timeline import RestoreState, same_state

        edits = []
        for target in [p for p, _, _ in self._objects] + [self.camera]:
            state = target.snapshot()
            previous = self._author_states.get(id(target))
            if self._view_states is not None and id(target) in self._view_states and previous is not None:
                viewed = self._view_states[id(target)]
                changes = {
                    key: value
                    for key, value in state.items()
                    if key not in viewed or not same_state(value, viewed[key])
                }
                state = {**previous, **changes}
                target.restore(state)
            if previous is not None and not same_state(state, previous):
                edits.append(RestoreState(target, state))
        if edits:
            self._clips.append((self.duration, 0.0, tuple(edits), None))
        self._remember()

    def construct(self):
        """Override this method, using add(), play(), wait(), and camera.frame()."""

    def add(self, *proteins):
        for p in proteins:
            if not isinstance(p, (Protein, Annotation, MeshObject)):
                raise TypeError("Only Protein, density mesh, or annotation objects can be added")
            if any(item[0] is p for item in self._objects):
                raise ValueError("Object is already in this scene")
            self._objects.append((p, self.duration, p.snapshot()))
            self._author_states[id(p)] = p.snapshot()
        return self

    def play(self, *animations, run_time=None, rate_func=None, easing=None):
        from .rates import resolve
        from .timeline import AnimationGroup

        if any(isinstance(a, AnimationGroup) for a in animations):
            animations = (AnimationGroup(*animations),)
        run_time = (
            (getattr(animations[0], "duration", 1.0) if animations else 1.0) if run_time is None else run_time
        )
        if easing is not None and rate_func is not None:
            raise ValueError("Use easing or its legacy clip-clock override rate_func, not both")
        if easing is not None:
            for animation in animations:
                animation._set_easing(easing)
        rate_func = resolve(rate_func) if rate_func is not None else None
        if not np.isfinite(run_time) or run_time <= 0 or not animations:
            raise ValueError("play() needs animations and a positive run_time")
        self._capture_edits()
        if self._camera_base is None:
            self._camera_base = self.camera.snapshot()
        used = set()
        for anim in animations:
            if hasattr(anim, "_prepare"):
                anim._prepare(self)
            if rate_func is not None and hasattr(anim, "transform_easing"):
                from .rates import linear

                anim.transform_easing = linear
            if getattr(anim, "requires_linear_timeline", False) and rate_func not in (None, anim.rate_func):
                raise ValueError(
                    "Staggered animations need a linear clip clock for delays in seconds; use their per-residue easing"
                )
            for target in anim.targets:
                if isinstance(target, (Protein, Annotation, MeshObject)) and not any(
                    p is target for p, _, _ in self._objects
                ):
                    self.add(target)
                for channel in anim.channels:
                    key = (id(target), channel)
                    if key in used:
                        raise ValueError(
                            f"Concurrent animations both write {channel}; combine or sequence them"
                        )
                    used.add(key)
        for anim in animations:
            anim.run_time = float(run_time)
            anim.start_time = self.duration
            anim.bind()
        ordered = tuple(sorted(animations, key=lambda a: bool(getattr(a, "late", False))))
        self._clips.append((self.duration, float(run_time), ordered, rate_func))
        for anim in ordered:
            anim.apply((rate_func or anim.rate_func)(1.0))
        self.camera.update_tracking()
        self.duration += float(run_time)
        self._remember()
        return self

    def focus(
        self, target, *, run_time=1.5, margin=1.25, follow=True, rate_func=None, screen_position=(0.5, 0.5)
    ):
        """Animate the camera to a selected region using this scene's aspect ratio."""
        from .animation import Focus

        return self.play(
            Focus(
                self.camera,
                target,
                margin=margin,
                aspect=self.width / self.height,
                follow=follow,
                screen_position=screen_position,
            ),
            run_time=run_time,
            rate_func=rate_func,
        )

    def wait(self, duration=1.0):
        if not np.isfinite(duration) or duration < 0:
            raise ValueError("Wait duration must be finite and nonnegative")
        self._capture_edits()
        if self._camera_base is None:
            self._camera_base = self.camera.snapshot()
        self.duration += float(duration)
        return self

    def build(self):
        if not self._built:
            self._building = True
            try:
                self.construct()
                self._capture_edits()
                self._camera_base = self._camera_base or self.camera.snapshot()
                self._built = True
            finally:
                self._building = False
        return self

    def seek(self, time):
        self.build()
        if not np.isfinite(time):
            raise ValueError("Time must be finite")
        time = float(np.clip(time, 0, self.duration))
        for p, _, state in self._objects:
            p.restore(state)
        self.camera.restore(self._camera_base)
        for start, duration, animations, override in self._clips:
            if time < start:
                break
            alpha = min((time - start) / duration, 1.0) if duration else 1.0
            for anim in animations:
                eased = float((override or anim.rate_func)(alpha))
                if not np.isfinite(eased) or not 0 <= eased <= 1:
                    raise ValueError("Rate functions must return a finite value in [0, 1]")
                anim.apply(eased)
        self.camera.update_tracking()
        visible = [p for p, start, _ in self._objects if start <= time and p.opacity > 0]
        for p in visible:
            if hasattr(p, "_set_time"):
                p._set_time(time)
            if hasattr(p, "_sync"):
                p._sync()
        self._view_states = {id(p): p.snapshot() for p, _, _ in self._objects}
        self._view_states[id(self.camera)] = self.camera.snapshot()
        return visible

    def render_frame(self, time=0.0, *, output=None, renderer=None, eevee=None, look=None):
        """Render a still. Use renderer='studio', look=StudioLook(...) for studio presets."""
        from .renderer import Renderer

        renderer, look, eevee = self._render_settings(renderer, look, eevee)

        if eevee is not None and renderer != "eevee":
            raise ValueError("EEVEE settings require renderer='eevee'")
        if look is not None and renderer != "studio":
            raise ValueError(
                "StudioLook settings require renderer='studio'; configure existing renderers directly"
            )
        own = renderer is None or isinstance(renderer, str)
        if renderer is None or renderer == "native":
            renderer = Renderer(self.width, self.height, msaa=self.msaa)
        elif renderer == "studio":
            from .studio import StudioRenderer

            renderer = StudioRenderer(self.width, self.height, msaa=self.msaa, look=look)
        elif renderer == "eevee":
            from .eevee import EEVEE

            renderer = EEVEE(self.width, self.height, options=eevee)
        elif isinstance(renderer, str):
            raise ValueError("renderer must be 'native', 'studio', 'eevee', or a renderer instance")
        try:
            objects = self.seek(time)
            if hasattr(renderer, "set_time"):
                renderer.set_time(float(np.clip(time, 0, self.duration)))
            pixels = renderer.render(objects, self.camera, self.background)
            if output is not None:
                from PIL import Image

                Path(output).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pixels).save(output)
            return pixels
        finally:
            if own:
                renderer.close()

    def render(
        self, output, *, codec="auto", bitrate="20M", progress=True, renderer=None, eevee=None, look=None
    ):
        """Export a movie. Studio lighting/material/effects are configured with look=StudioLook(...)."""
        import platform
        import time

        from .renderer import Renderer
        from .video import VideoWriter

        self.build()
        renderer, look, eevee = self._render_settings(renderer, look, eevee)
        if renderer not in ("native", "studio", "eevee"):
            raise ValueError("renderer must be 'native', 'studio', or 'eevee'")
        if renderer != "eevee" and eevee is not None:
            raise ValueError("EEVEE settings require renderer='eevee'")
        if look is not None and renderer != "studio":
            raise ValueError("StudioLook settings require renderer='studio'")
        count = max(1, math.ceil(self.duration * self.fps))
        platform_name = platform.system()
        start = time.perf_counter()
        if renderer == "eevee":
            from .eevee import EEVEE

            with EEVEE(self.width, self.height, options=eevee) as backend:
                with VideoWriter(
                    output, self.width, self.height, self.fps, codec=codec, bitrate=bitrate
                ) as writer:
                    if progress:
                        print(
                            f"Platform: {platform_name}; renderer: EEVEE "
                            f"({backend.adapter_info.get('backend', 'default')}); encoder: {writer.codec}",
                            flush=True,
                        )
                    for i in range(count):
                        writer.write(backend.render(self.seek(i / self.fps), self.camera, self.background))
                        if progress and i % max(1, int(self.fps)) == 0:
                            print(f"\rEEVEE: {i + 1}/{count} frames", end="", flush=True)
                adapter = backend.adapter_info
            elapsed = time.perf_counter() - start
            if progress:
                print(f"\rEEVEE: rendered {count} frames in {elapsed:.2f}s -> {output}")
            return {
                "frames": count,
                "seconds": elapsed,
                "fps": count / elapsed,
                "adapter": adapter,
                "codec": writer.codec,
                "platform": platform_name,
                "output": str(output),
            }
        options = {}
        if renderer == "studio":
            from .studio import StudioRenderer as Renderer

            options["look"] = look
        with Renderer(self.width, self.height, msaa=self.msaa, readback_format="nv12", **options) as renderer:
            with VideoWriter(
                output, self.width, self.height, self.fps, codec=codec, bitrate=bitrate, pixel_format="nv12"
            ) as writer:
                if progress:
                    print(
                        f"Platform: {platform_name}; GPU: {renderer.adapter_info['device']} "
                        f"({renderer.adapter_info['backend_type']}); encoder: {writer.codec}",
                        flush=True,
                    )
                # Triple-buffered GPU readback overlaps rendering and hardware encoding.
                for i in range(count):
                    if hasattr(renderer, "set_time"):
                        renderer.set_time(i / self.fps)
                    pixels = renderer.enqueue(self.seek(i / self.fps), self.camera, self.background)
                    if pixels is not None:
                        writer.write(pixels)
                    if progress and (i % max(1, int(self.fps * 2)) == 0):
                        print(f"\rRendering {i + 1}/{count} frames", end="", flush=True)
                for pixels in renderer.drain():
                    writer.write(pixels)
            adapter = renderer.adapter_info
        elapsed = time.perf_counter() - start
        if progress:
            print(f"\rRendered {count} frames in {elapsed:.2f}s ({count / elapsed:.1f} fps) -> {output}")
        return {
            "frames": count,
            "seconds": elapsed,
            "fps": count / elapsed,
            "adapter": adapter,
            "codec": writer.codec,
            "platform": platform_name,
            "output": str(output),
        }

    def _render_settings(self, renderer, look, eevee):
        renderer = self.renderer if renderer is None else renderer
        look = self.look if look is None and renderer == "studio" else look
        eevee = self.eevee if eevee is None and renderer == "eevee" else eevee
        return renderer, look, eevee

    def preview(self, *, renderer=None, look=None, close_after=None):
        from .preview import preview

        return preview(self, renderer=renderer, look=look, close_after=close_after)


Scene = ProteinScene
