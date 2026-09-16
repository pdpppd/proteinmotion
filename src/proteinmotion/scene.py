"""Manim-style authoring with a seekable, deterministic timeline."""

import math
from pathlib import Path

import numpy as np

from .annotations import Annotation
from .camera import Camera
from .math3d import color
from .protein import Protein


class ProteinScene:
    def __init__(self, *, width=1920, height=1080, fps=30, background="#0b1220", msaa=4):
        if not isinstance(width, int) or not isinstance(height, int) or width < 1 or height < 1:
            raise ValueError("Image dimensions must be positive integers")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive")
        if msaa not in (1, 4):
            raise ValueError("msaa must be 1 or 4")
        self.width, self.height, self.fps, self.msaa = width, height, float(fps), msaa
        self.background = color(background)
        self.camera = Camera()
        self.duration = 0.0
        self._objects, self._clips = [], []
        self._built, self._building = False, False
        self._camera_base = None

    def construct(self):
        """Override this method, using add(), play(), wait(), and camera.frame()."""

    def add(self, *proteins):
        for p in proteins:
            if not isinstance(p, (Protein, Annotation)):
                raise TypeError("Only Protein or annotation objects can be added")
            if any(item[0] is p for item in self._objects):
                raise ValueError("Object is already in this scene")
            self._objects.append((p, self.duration, p.snapshot()))
        return self

    def play(self, *animations, run_time=1.0, rate_func=None):
        if not np.isfinite(run_time) or run_time <= 0 or not animations:
            raise ValueError("play() needs animations and a positive run_time")
        if self._camera_base is None:
            self._camera_base = self.camera.snapshot()
        used = set()
        for anim in animations:
            if getattr(anim, "requires_linear_timeline", False) and rate_func not in (None, anim.rate_func):
                raise ValueError(
                    "Staggered animations need a linear clip clock for delays in seconds; use their per-residue easing"
                )
            for target in anim.targets:
                if isinstance(target, (Protein, Annotation)) and not any(
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
        return self

    def focus(self, target, *, run_time=1.5, margin=1.25, follow=True, rate_func=None):
        """Animate the camera to a selected region using this scene's aspect ratio."""
        from .animation import Focus

        return self.play(
            Focus(self.camera, target, margin=margin, aspect=self.width / self.height, follow=follow),
            run_time=run_time,
            rate_func=rate_func,
        )

    def wait(self, duration=1.0):
        if not np.isfinite(duration) or duration < 0:
            raise ValueError("Wait duration must be finite and nonnegative")
        if self._camera_base is None:
            self._camera_base = self.camera.snapshot()
        self.duration += float(duration)
        return self

    def build(self):
        if not self._built:
            self._building = True
            try:
                self.construct()
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
            alpha = min((time - start) / duration, 1.0)
            for anim in animations:
                eased = float((override or anim.rate_func)(alpha))
                if not np.isfinite(eased) or not 0 <= eased <= 1:
                    raise ValueError("Rate functions must return a finite value in [0, 1]")
                anim.apply(eased)
        self.camera.update_tracking()
        visible = [p for p, start, _ in self._objects if start <= time and p.opacity > 0]
        for p in visible:
            if hasattr(p, "_sync"):
                p._sync()
        return visible

    def render_frame(self, time=0.0, *, output=None, renderer=None):
        from .renderer import Renderer

        own = renderer is None
        renderer = renderer or Renderer(self.width, self.height, msaa=self.msaa)
        try:
            pixels = renderer.render(self.seek(time), self.camera, self.background)
            if output is not None:
                from PIL import Image

                Path(output).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pixels).save(output)
            return pixels
        finally:
            if own:
                renderer.close()

    def render(self, output, *, codec="auto", bitrate="20M", progress=True):
        import time

        from .renderer import Renderer
        from .video import VideoWriter

        self.build()
        count = max(1, math.ceil(self.duration * self.fps))
        start = time.perf_counter()
        with Renderer(self.width, self.height, msaa=self.msaa, readback_format="nv12") as renderer:
            with VideoWriter(
                output, self.width, self.height, self.fps, codec=codec, bitrate=bitrate, pixel_format="nv12"
            ) as writer:
                # Triple-buffered GPU readback overlaps rendering and hardware encoding.
                for i in range(count):
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
            print(f"\rRendered {count} frames in {elapsed:.2f}s ({count / elapsed:.1f} fps) → {output}")
        return {
            "frames": count,
            "seconds": elapsed,
            "fps": count / elapsed,
            "adapter": adapter,
            "codec": writer.codec,
            "output": str(output),
        }

    def preview(self):
        from .preview import preview

        return preview(self)


Scene = ProteinScene
