"""Animation composition and deterministic authoring state changes."""

import numpy as np

from .animation import Animation
from .rates import evaluate, linear


def same_state(a, b):
    if a is b:
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, type(a)):
        return len(a) == len(b) and all(same_state(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and isinstance(a, (str, int, float, bool, type(None))) and a == b


class RestoreState(Animation):
    def __init__(self, target, state):
        super().__init__(target, rate_func=linear)
        self.state = state

    def apply(self, alpha):
        self.target.restore(self.state)


class AnimationGroup(Animation):
    """Play actions together, with optional durations and offsets in seconds.

    ``scene.play(move.during(2), caption.during(1, delay=0.5))`` is shorthand
    for a group. An explicit scene run_time scales the complete group's timing.
    Overlapping writes to the same target/channel are rejected before binding.
    """

    def __init__(self, *animations, durations=None, offsets=None):
        if not animations:
            raise ValueError("AnimationGroup needs at least one animation")
        super().__init__(animations[0].target, rate_func=linear)
        self.animations = tuple(animations)
        durations = [getattr(a, "duration", 1.0) for a in animations] if durations is None else durations
        offsets = [0.0] * len(animations) if offsets is None else offsets
        if len(durations) != len(animations) or len(offsets) != len(animations):
            raise ValueError("Provide one duration and offset per animation")
        self.durations, self.offsets = np.asarray(durations, float), np.asarray(offsets, float)
        if (
            self.durations.ndim != 1
            or self.offsets.ndim != 1
            or not np.isfinite(self.durations).all()
            or (self.durations <= 0).any()
            or not np.isfinite(self.offsets).all()
            or (self.offsets < 0).any()
        ):
            raise ValueError("Durations must be positive and offsets nonnegative finite seconds")
        self.duration = float(np.max(self.offsets + self.durations))
        self.channels = frozenset().union(*(a.channels for a in animations))

    def _prepare(self, scene):
        for animation in self.animations:
            if hasattr(animation, "_prepare"):
                animation._prepare(scene)

    def _set_easing(self, easing):
        for animation in self.animations:
            animation._set_easing(easing)

    @property
    def targets(self):
        return tuple({id(t): t for a in self.animations for t in a.targets}.values())

    def bind(self):
        super().bind()
        scale = getattr(self, "run_time", self.duration) / self.duration

        def flatten(group, origin, factor):
            for offset, duration, animation in zip(group.offsets, group.durations, group.animations):
                start, span = origin + float(offset * factor), float(duration * factor)
                if isinstance(animation, AnimationGroup):
                    if animation._bound:
                        raise ValueError("Animation objects may only be played once; create a new animation")
                    animation._bound = True
                    yield from flatten(animation, start, span / animation.duration)
                else:
                    yield start, span, animation

        self._schedule = sorted(
            [(start, span, i, a) for i, (start, span, a) in enumerate(flatten(self, 0, scale))],
            key=lambda item: (item[0], bool(getattr(item[3], "late", False)), item[2]),
        )
        writes = {}
        for offset, duration, _, animation in self._schedule:
            for target in animation.targets:
                for channel in animation.channels:
                    key = (id(target), channel)
                    if any(
                        offset < end - 1e-10 and offset + duration > start + 1e-10
                        for start, end in writes.get(key, ())
                    ):
                        raise ValueError(
                            f"Concurrent animations both write {channel}; combine or sequence them"
                        )
                    writes.setdefault(key, []).append((offset, offset + duration))
        states = [(t, t.snapshot()) for t in self.targets]
        bound = []
        try:
            for offset, duration, _, animation in self._schedule:
                for target, state in states:
                    target.restore(state)
                self._apply_at(offset, bound)
                animation.run_time = duration
                animation.start_time = getattr(self, "start_time", 0.0) + offset
                animation.bind()
                bound.append((offset, duration, 0, animation))
        finally:
            for target, state in states:
                target.restore(state)

    @staticmethod
    def _apply_at(time, schedule):
        from .camera import Camera

        # Establish entrance states before delayed actions start. Do this only for
        # first writers, otherwise a later move would overwrite an earlier one.
        seen = set()
        for start, _, _, animation in schedule:
            keys = {(id(target), channel) for target in animation.targets for channel in animation.channels}
            if time < start and not keys & seen:
                animation.apply(evaluate(animation.rate_func, 0))
            seen.update(keys)
        # Evaluate molecular motion before camera tracks, while retaining the
        # chronological order of successive frame/orbit/zoom actions.
        ordered = sorted(
            schedule,
            key=lambda item: (
                bool(getattr(item[3], "late", False)) or any(isinstance(t, Camera) for t in item[3].targets),
                item[0],
                item[2],
            ),
        )
        for start, duration, _, animation in ordered:
            if time >= start:
                animation.apply(evaluate(animation.rate_func, min((time - start) / duration, 1)))

    def apply(self, alpha):
        self._apply_at(float(alpha) * getattr(self, "run_time", self.duration), self._schedule)
