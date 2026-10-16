"""Manim-style authoring with a seekable, deterministic timeline.

ProteinScene records a movie as a list of clips. add(), play() and wait(), called from
construct() or directly, build it up: each clip is a start time, a duration, the
animations it runs, and an optional rate_func override. Direct edits made between those
calls, such as ``protein.set_opacity(0.5)``, become zero-length clips. seek(time)
restores every object to its state when it was added and replays the clips up to
``time``, so frames can be evaluated in any order. render() and render_frame() seek once
per frame and pass the visible objects to the native, Studio or EEVEE renderer.
"""

import math
from pathlib import Path

import numpy as np

from .annotations import Annotation
from .camera import Camera
from .math3d import color
from .mesh import MeshObject
from .protein import Protein


class ProteinScene:
    """A protein movie defined by add(), play() and wait() calls, replayable at any time.

    Subclass it and override construct(), or call the methods on an instance. Set
    ``renderer`` ("native", "studio" or "eevee"), ``look`` and ``eevee`` on the subclass
    or in the constructor, so the command line, preview, stills and movies agree.
    """

    # Class-level defaults; constructor arguments of the same names override them.
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
        supersampling=None,
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
        if supersampling not in (None, 1, 2, 3, 4):
            raise ValueError("supersampling must be 1, 2, 3, 4, or None for automatic")
        self.width, self.height, self.fps, self.msaa = width, height, float(fps), msaa
        self.supersampling = supersampling
        self.background = color(background)
        self.renderer = renderer if renderer is not None else type(self).renderer
        self.look = look if look is not None else type(self).look
        self.eevee = eevee if eevee is not None else type(self).eevee
        self.camera = Camera()
        self.camera.aspect = width / height
        # Authoring time in seconds: where the next add(), play() or wait() lands.
        self.duration = 0.0
        # Timeline state:
        #   _objects        (object, time added, snapshot when added), in the order added
        #   _clips          (start, duration, animations in evaluation order, rate_func
        #                   override or None), in time order; edits have duration 0
        #   _camera_base    camera snapshot where the timeline starts
        #   _author_states  id(object) -> snapshot after the last authoring call, which is
        #                   the state the next play() starts from
        #   _view_states    id(object) -> snapshot right after the last seek(), or None when
        #                   no seek() happened since the last authoring call
        self._objects, self._clips = [], []
        self._built = False  # construct() has run
        self._camera_base = None
        self._author_states = {}
        self._view_states = None

    def _remember(self):
        """Take the current state of every object as the authoring state, and forget any seek."""
        self._author_states = {id(p): p.snapshot() for p, _, _ in self._objects}
        self._author_states[id(self.camera)] = self.camera.snapshot()
        self._view_states = None

    def _capture_edits(self):
        """Record changes made directly to objects since the last authoring call as a clip."""
        from .timeline import RestoreState, same_state

        edits = []
        for target in [p for p, _, _ in self._objects] + [self.camera]:
            state = target.snapshot()
            # None for the camera before the first play() or wait(): its setup until then
            # is the timeline's starting state, not an edit.
            previous = self._author_states.get(id(target))
            if self._view_states is not None and id(target) in self._view_states and previous is not None:
                # A seek() since the last call left the object in its replayed state at that
                # time, not at the end of the timeline. Keep only what changed after the seek,
                # layer it on the authoring state, and put the object back in that state.
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
        """Add proteins, density meshes or annotations at the current time."""
        for p in proteins:
            if not isinstance(p, (Protein, Annotation, MeshObject)):
                raise TypeError("Only Protein, density mesh, or annotation objects can be added")
            if any(item[0] is p for item in self._objects):
                raise ValueError("Object is already in this scene")
            # seek() restores this snapshot first and hides the object before this time.
            self._objects.append((p, self.duration, p.snapshot()))
            self._author_states[id(p)] = p.snapshot()
        return self

    def play(self, *animations, run_time=None, rate_func=None, easing=None):
        """Run animations together for ``run_time`` seconds, then advance the timeline."""
        from .rates import resolve
        from .timeline import AnimationGroup, runs_late

        # Timed actions (from .during()) share one clock, so wrap the whole call in one group.
        if any(isinstance(a, AnimationGroup) for a in animations):
            animations = (AnimationGroup(*animations),)
        # By default a group lasts its own length, and anything else 1 s.
        run_time = (
            (getattr(animations[0], "duration", 1.0) if animations else 1.0) if run_time is None else run_time
        )
        # easing goes to each animation; rate_func is a legacy override of the whole clip's clock.
        if easing is not None and rate_func is not None:
            raise ValueError("Use easing or its legacy clip-clock override rate_func, not both")
        if easing is not None:
            for animation in animations:
                animation._set_easing(easing)
        rate_func = resolve(rate_func) if rate_func is not None else None
        if not np.isfinite(run_time) or run_time <= 0 or not animations:
            raise ValueError("play() needs animations and a positive run_time")
        # Edits made since the last call go on the timeline just before this clip.
        self._capture_edits()
        if self._camera_base is None:
            self._camera_base = self.camera.snapshot()
        used = set()
        for anim in animations:
            # Let the animation read scene settings, such as the camera, before it binds.
            if hasattr(anim, "_prepare"):
                anim._prepare(self)
            # With a clip-wide rate_func, a fluent chain takes it instead of its own easing,
            # for the operations and for every chained animation, so each eases once, as it
            # would on its own.
            if rate_func is not None and hasattr(anim, "transform_easing"):
                from .rates import linear

                anim.transform_easing = linear
                for extra in anim.extras:
                    extra.rate_func = linear
            # Delays scheduled in seconds would be distorted by a warped clip clock.
            if getattr(anim, "requires_linear_timeline", False) and rate_func not in (None, anim.rate_func):
                raise ValueError(
                    "Staggered animations need a linear clip clock for delays in seconds; use their per-residue easing"
                )
            for target in anim.targets:
                # Objects an animation touches join the scene now.
                if isinstance(target, (Protein, Annotation, MeshObject)) and not any(
                    p is target for p, _, _ in self._objects
                ):
                    self.add(target)
                # No two animations in one call may write the same channel of the same object.
                for channel in anim.channels:
                    key = (id(target), channel)
                    if key in used:
                        raise ValueError(
                            f"Concurrent animations both write {channel}; combine or sequence them"
                        )
                    used.add(key)
        # Bind every animation from the same starting state, before any is applied.
        for anim in animations:
            anim.run_time = float(run_time)
            anim.start_time = self.duration
            anim.bind()
        # Camera animations run after the others, so they see this frame's molecule.
        ordered = tuple(sorted(animations, key=runs_late))
        self._clips.append((self.duration, float(run_time), ordered, rate_func))
        # Leave every object at the end of the clip, where the next authoring call starts.
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
        """Hold the current state for ``duration`` seconds."""
        if not np.isfinite(duration) or duration < 0:
            raise ValueError("Wait duration must be finite and nonnegative")
        self._capture_edits()
        if self._camera_base is None:
            self._camera_base = self.camera.snapshot()
        self.duration += float(duration)
        return self

    def build(self):
        """Run construct() once and record any edits made after its last play() or wait()."""
        if not self._built:
            self.construct()
            self._capture_edits()
            # Without any play() or wait(), the timeline starts from the camera as set up.
            self._camera_base = self._camera_base or self.camera.snapshot()
            self._built = True
        return self

    def seek(self, time):
        """Put every object in its state at ``time`` seconds and return the visible ones."""
        self.build()
        if not np.isfinite(time):
            raise ValueError("Time must be finite")
        time = float(np.clip(time, 0, self.duration))
        # Start from each object as it was added and the camera where the timeline starts.
        for p, _, state in self._objects:
            p.restore(state)
        self.camera.restore(self._camera_base)
        # Replay clips in order: finished clips at their end, the current one part-way, and
        # zero-length edit clips in full. Clips are stored in time order, so stop at the
        # first that starts later.
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
        # Objects added by now and not fully transparent are drawn. Time-aware annotations
        # (plots) read the time, and derived objects (highlights, thread wires) follow
        # their parents.
        visible = [p for p, start, _ in self._objects if start <= time and p.opacity > 0]
        for p in visible:
            if hasattr(p, "_set_time"):
                p._set_time(time)
            if hasattr(p, "_sync"):
                p._sync()
        # Remember the replayed state, so _capture_edits() can tell edits made after this
        # seek from the replay itself.
        self._view_states = {id(p): p.snapshot() for p, _, _ in self._objects}
        self._view_states[id(self.camera)] = self.camera.snapshot()
        return visible

    def _supersampling(self):
        """Native and Studio frames render at 2× up to 1920 × 1080 by default, then average down."""
        if self.supersampling is not None:
            return self.supersampling
        return 2 if self.width * self.height <= 1920 * 1080 else 1

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
        # A renderer created here is closed afterwards; an instance passed in stays open
        # for the caller to reuse. Native and Studio draw at _supersampling() times the
        # output size and average down; EEVEE takes its supersampling from its options.
        own = renderer is None or isinstance(renderer, str)
        if renderer is None or renderer == "native":
            renderer = Renderer(self.width, self.height, msaa=self.msaa, supersampling=self._supersampling())
        elif renderer == "studio":
            from .studio import StudioRenderer

            renderer = StudioRenderer(
                self.width, self.height, msaa=self.msaa, supersampling=self._supersampling(), look=look
            )
        elif renderer == "eevee":
            from .eevee import EEVEE

            renderer = EEVEE(self.width, self.height, options=eevee)
        elif isinstance(renderer, str):
            raise ValueError("renderer must be 'native', 'studio', 'eevee', or a renderer instance")
        try:
            objects = self.seek(time)
            # Studio film grain follows the scene time, clamped as seek() clamps it.
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

        self.build()
        renderer, look, eevee = self._render_settings(renderer, look, eevee)
        if renderer not in ("native", "studio", "eevee"):
            raise ValueError("renderer must be 'native', 'studio', or 'eevee'")
        if renderer != "eevee" and eevee is not None:
            raise ValueError("EEVEE settings require renderer='eevee'")
        if look is not None and renderer != "studio":
            raise ValueError("StudioLook settings require renderer='studio'")
        # Frame i shows time i / fps; even an empty timeline gets one frame.
        count = max(1, math.ceil(self.duration * self.fps))
        platform_name = platform.system()
        start = time.perf_counter()
        if renderer == "eevee":
            return self._render_eevee_movie(
                output, count, start, platform_name, codec, bitrate, progress, eevee
            )
        return self._render_gpu_movie(
            output, count, start, platform_name, codec, bitrate, progress, renderer, look
        )

    def _render_eevee_movie(self, output, count, start, platform_name, codec, bitrate, progress, eevee):
        """Encode ``count`` frames rendered by Blender EEVEE; return render()'s summary."""
        import time

        from .eevee import EEVEE
        from .video import VideoWriter

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
        return self._movie_summary(count, elapsed, adapter, writer.codec, platform_name, output)

    def _render_gpu_movie(
        self, output, count, start, platform_name, codec, bitrate, progress, renderer, look
    ):
        """Encode ``count`` frames from the native or Studio GPU renderer; return render()'s summary."""
        import time

        from .renderer import Renderer
        from .video import VideoWriter

        options = {}
        if renderer == "studio":
            from .studio import StudioRenderer as Renderer

            options["look"] = look
        # Frames are read back from the GPU as NV12, the encoder's input format.
        with Renderer(
            self.width,
            self.height,
            msaa=self.msaa,
            supersampling=self._supersampling(),
            readback_format="nv12",
            **options,
        ) as renderer:
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
                # enqueue() returns the oldest finished frame once three are in flight, and
                # drain() returns the rest at the end.
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
        return self._movie_summary(count, elapsed, adapter, writer.codec, platform_name, output)

    @staticmethod
    def _movie_summary(count, elapsed, adapter, codec, platform_name, output):
        """The dictionary render() returns; ``fps`` is render throughput, not the movie's rate."""
        return {
            "frames": count,
            "seconds": elapsed,
            "fps": count / elapsed,
            "adapter": adapter,
            "codec": codec,
            "platform": platform_name,
            "output": str(output),
        }

    def _render_settings(self, renderer, look, eevee):
        """Fill in unset renderer, look and EEVEE options from the scene's own settings.

        The scene's look applies only to the studio renderer, and its EEVEE options only to eevee.
        """
        renderer = self.renderer if renderer is None else renderer
        look = self.look if look is None and renderer == "studio" else look
        eevee = self.eevee if eevee is None and renderer == "eevee" else eevee
        return renderer, look, eevee

    def preview(self, *, renderer=None, look=None, close_after=None):
        """Open an interactive window that plays this scene on the GPU."""
        from .preview import preview

        return preview(self, renderer=renderer, look=look, close_after=close_after)


Scene = ProteinScene
