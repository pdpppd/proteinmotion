"""Animation composition and deterministic authoring state changes.

AnimationGroup runs several animations inside one play() clip, each with its own duration
and start offset in seconds; ``Animation.during()`` builds one. RestoreState fills the
zero-length clips that ProteinScene records when a direct setter, such as
``protein.set_opacity(0.5)``, changes an object between play() calls, and same_state()
is how the scene tells whether anything changed.
"""

import numpy as np

from .animation import Animation
from .rates import evaluate, linear


def same_state(a, b):
    """True when two snapshot values are equal.

    Arrays, dicts, lists and tuples compare by content, and str, int, float, bool and None
    by value when their types match. Any other object is equal only to itself, so a doubtful
    comparison reports a change; the scene then records a redundant edit, which is harmless.
    """
    if a is b:
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, type(a)):
        return len(a) == len(b) and all(same_state(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and isinstance(a, (str, int, float, bool, type(None))) and a == b


def runs_late(animation):
    """True when ``animation`` runs after the other animations of its clip.

    These are animations that set ``late``, such as Focus, and any that change a Camera,
    such as a ``camera.animate`` chain. A camera track then reads where the molecule is
    in the same frame. play() and AnimationGroup both order animations by this rule, so
    the order of play()'s arguments does not change the result.
    """
    from .camera import Camera

    return bool(getattr(animation, "late", False)) or any(isinstance(t, Camera) for t in animation.targets)


class RestoreState(Animation):
    """Put ``target`` back into a saved ``state`` (a ``target.snapshot()`` dict) at any alpha."""

    # ProteinScene._capture_edits() records one per edited object, in a zero-length clip at
    # the time of the edit. It declares no channels, so it never conflicts with an animation.

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
        # The group's clock is linear time; each child eases its own local progress.
        super().__init__(animations[0].target, rate_func=linear)
        self.animations = tuple(animations)
        # A nested group lasts its own length by default; any other child lasts 1 s.
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
        # The group ends when its last child ends; play() uses this as the default run_time.
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
        # Every child's targets, once each, in first-seen order.
        return tuple({id(t): t for a in self.animations for t in a.targets}.values())

    def bind(self):
        super().bind()
        # An explicit play(run_time=...) stretches or compresses every offset and duration alike.
        scale = getattr(self, "run_time", self.duration) / self.duration

        def flatten(group, origin, factor):
            # Yield (start, span, animation) in seconds from the start of the outermost group,
            # expanding nested groups. A nested group is never bound itself, so mark it bound
            # here to keep the play-once rule.
            for offset, duration, animation in zip(group.offsets, group.durations, group.animations):
                start, span = origin + float(offset * factor), float(duration * factor)
                if isinstance(animation, AnimationGroup):
                    if animation._bound:
                        raise ValueError("Animation objects may only be played once; create a new animation")
                    animation._bound = True
                    yield from flatten(animation, start, span / animation.duration)
                else:
                    yield start, span, animation

        # Schedule rows are (start, span, index, animation), ordered by start. At equal starts,
        # late (camera) animations go after the others, then the original order decides.
        self._schedule = sorted(
            [(start, span, i, a) for i, (start, span, a) in enumerate(flatten(self, 0, scale))],
            key=lambda item: (item[0], runs_late(item[3]), item[2]),
        )
        # Reject two children that write the same channel of the same object at the same
        # time. The 1e-10 s tolerance lets one child start exactly where another ends.
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
        # Bind each child from the state at its own start time: restore the group's starting
        # state, replay the children bound so far up to this offset, then bind. A delayed move
        # therefore starts where an earlier move in the same group left the object.
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
                # Index 0 for every row: the stable sort in _apply_at keeps schedule order.
                bound.append((offset, duration, 0, animation))
        finally:
            # Leave the targets as bind() found them.
            for target, state in states:
                target.restore(state)

    @staticmethod
    def _apply_at(time, schedule):
        """Evaluate schedule rows (start, span, index, animation) at ``time`` seconds into the group."""
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
        ordered = sorted(schedule, key=lambda item: (runs_late(item[3]), item[0], item[2]))
        # Children that have started run at their local progress; finished ones hold at 1.
        for start, duration, _, animation in ordered:
            if time >= start:
                animation.apply(evaluate(animation.rate_func, min((time - start) / duration, 1)))

    def apply(self, alpha):
        # alpha is the clip progress (linear unless play() overrides rate_func), so this is
        # the time in seconds since the group started.
        self._apply_at(float(alpha) * getattr(self, "run_time", self.duration), self._schedule)
